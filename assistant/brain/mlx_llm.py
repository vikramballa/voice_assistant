"""The MLX engine for the slow brain: qwen3 4B (4-bit) in this process, on the Mac GPU. ~0.8 s per plan.

Two tricks make it fast and reliable:

1. A grammar. Ollama pretty-prints its JSON (~35 tokens per plan); here a small character-level state
   machine (Grammar) only lets the model write compact JSON with real action names (~15-20 tokens,
   about half the time). Without it the model wrote shapes like "media_play_pause()" or "arg".
   Only the opening '{"steps":[{"action":"' is given to the model; forcing later fixed parts
   ourselves tokenizes them unlike the model would, and confuses it.

2. A saved prompt cache. Reading the 1.7k-token system prompt costs ~5 s on the M4, so its KV cache is
   saved to models/llm_cache/ (~250 MB) and loaded at startup. The file name is a hash of model + prompt,
   so a changed prompt (e.g. a new app installed) builds a new one once.
"""

import hashlib
import json
import os
import re

from assistant.config import MODELS_DIR
from assistant.gpu import MLX_LOCK

MODEL = os.environ.get("LAYA_MLX_MODEL", "mlx-community/Qwen3-4B-4bit")
STEPS = "steps"   # {"steps":[{"action":"<ACTION>","args":["..",..]},..]}
REPLY = "reply"   # {"reply":".."}
TOP_K = 20        # the best allowed token among the model's top 20 is taken (greedy)
WIDE_K = 3000     # ...or among the top 3000 if none of the 20 fits
PREFILL_CHUNK = 512
PROMPT_CACHE_DIR = MODELS_DIR / "llm_cache"

_mlx: dict = {}  # model, tokenizer, token texts, special token ids, one KV cache per system prompt


class Grammar:
    """Character-level state machine over the allowed JSON. feed(text) -> False if not allowed.
    `actions` are the only action names allowed (for the STEPS grammar)."""

    def __init__(self, kind: str, actions):
        self.actions = actions
        self.literal = '{"steps":[{"action":"' if kind == STEPS else '{"reply":"'
        self.state = "literal"
        self.after_literal = "action" if kind == STEPS else "string"
        self.after_string = "after_arg" if kind == STEPS else "end_reply"
        self.name = ""
        self.escaped = False

    def feed(self, text: str) -> bool:
        return all(self._char(c) for c in text)

    def closing(self) -> str | None:
        """The shortest text that finishes the JSON from here (when the model wants to stop early)."""
        steps = self.after_string == "after_arg"
        return {"string": '"]}]}' if steps else '"}', "after_arg": "]}]}", "args_start": "]}]}",
                "after_step": "]}", "end_reply": "}"}.get(self.state)

    def copy(self) -> "Grammar":
        g = Grammar.__new__(Grammar)
        g.__dict__.update(self.__dict__)
        return g

    def _lit(self, text: str, then: str) -> None:
        self.literal, self.after_literal, self.state = text, then, "literal"

    def _char(self, c: str) -> bool:
        s = self.state
        if s == "literal":  # fixed text: exactly these characters
            if not self.literal.startswith(c):
                return False
            self.literal = self.literal[1:]
            if not self.literal:
                self.state = self.after_literal
            return True
        if s == "action":  # an action name, one of self.actions
            if c == '"':
                if self.name not in self.actions:
                    return False
                self.name = ""
                self._lit(',"args":[', "args_start")
                return True
            if not any(a.startswith(self.name + c) for a in self.actions):
                return False
            self.name += c
            return True
        if s == "string":  # any text on one line, with \-escapes
            if self.escaped:
                self.escaped = False
            elif c == "\\":
                self.escaped = True
            elif c == '"':
                self.state = self.after_string
            elif c in "\n\r":
                return False
            return True
        if s == "args_start":
            if c == '"':
                self.state = "string"
            elif c == "]":
                self._lit("}", "after_step")
            else:
                return False
            return True
        if s == "after_arg":
            if c == ",":
                self._lit('"', "string")
            elif c == "]":
                self._lit("}", "after_step")
            else:
                return False
            return True
        if s == "after_step":
            if c == ",":
                self._lit('{"action":"', "action")
            elif c == "]":
                self._lit("}", "done")
            else:
                return False
            return True
        if s == "end_reply":
            if c != "}":
                return False
            self.state = "done"
            return True
        return False  # done: nothing more


def complete(messages: list[dict], grammar: str, max_tokens: int, actions) -> str:
    """Greedy decoding that only takes tokens the grammar allows. The KV cache for the part of
    the prompt that didn't change (the long system prompt) is reused from the last call."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

    with MLX_LOCK:
        if not _mlx:
            _load()
        model, tok, texts, special = _mlx["model"], _mlx["tok"], _mlx["texts"], _mlx["special"]
        system = messages[0]["content"]
        if system not in _mlx["caches"]:
            _mlx["caches"][system] = (make_prompt_cache(model), [])
        cache, old_ids = _mlx["caches"][system]
        g = Grammar(grammar, actions)
        opening = g.literal
        g.feed(opening)
        ids = tok.encode(tok.apply_chat_template(messages, add_generation_prompt=True,
                                                 enable_thinking=False, tokenize=False) + opening)
        same = 0
        for a, b in zip(old_ids, ids[:-1]):  # always feed at least one new token
            if a != b:
                break
            same += 1
        trim_prompt_cache(cache, cache[0].offset - same)
        _mlx["caches"][system] = (cache, ids)

        out: list[int] = []
        feed = ids[same:]
        # Read the prompt without the output layer: next-token scores are only needed for the last token
        # (all 1.7k positions x 151k vocab = 1 GB and ~5 s at startup).
        for i in range(0, len(feed) - 1, PREFILL_CHUNK):
            model(mx.array([feed[i:min(i + PREFILL_CHUNK, len(feed) - 1)]]), cache=cache)
            mx.eval([c.state for c in cache])
        feed = feed[-1:]
        while g.state != "done" and len(out) < max_tokens:
            logits = model(mx.array([feed]), cache=cache)[0, -1]
            if int(mx.argmax(logits).item()) in special and g.closing():
                # The model wants to stop (e.g. it ended a string with a curly ” instead of "): close the JSON.
                ending = g.closing()
                g.feed(ending)
                out += tok.encode(ending)
                break
            t = None
            for k in (TOP_K, WIDE_K):  # usually the first try; rarely the model's favourites are all invalid
                top = mx.argpartition(-logits, k)[:k]
                for cand in top[mx.argsort(-logits[top])].tolist():
                    if cand in special:  # "<|im_end|>" inside a string (after a curly ” it thought closed it)
                        continue
                    trial = g.copy()
                    if texts[cand] and trial.feed(texts[cand]):
                        g, t = trial, cand
                        break
                if t is not None:
                    break
            if t is None and g.state == "literal":
                # Stuck inside fixed text ('"arg' and the model won't write the 's'): write it for the model.
                forced = tok.encode(g.literal)
                g.feed(g.literal)
                out += forced
                feed = forced
                continue
            if t is None:
                break  # nothing allowed at all: give up (the JSON won't parse, the caller handles it)
            out.append(t)
            feed = [t]
        trim_prompt_cache(cache, cache[0].offset - len(ids))  # keep only the prompt for next time
    return opening + tok.decode(out)


def prime(system: str, actions) -> None:
    """Have the KV cache for this system prompt ready: from disk, or computed once and saved."""
    from mlx_lm.models.cache import load_prompt_cache, save_prompt_cache, trim_prompt_cache

    with MLX_LOCK:
        if not _mlx:
            _load()
        tok = _mlx["tok"]
        name = hashlib.sha1(f"{MODEL}\n{system}".encode()).hexdigest()[:16]
        path = PROMPT_CACHE_DIR / f"{name}.safetensors"
        if path.exists():
            cache, meta = load_prompt_cache(str(path), return_metadata=True)
            _mlx["caches"][system] = (cache, json.loads(meta["ids"]))
            return
        # the tokens every prompt with this system message starts with
        a, b = ([*tok.encode(tok.apply_chat_template([{"role": "system", "content": system},
                                                      {"role": "user", "content": u}], tokenize=False))]
                for u in ("a", "b"))
        prefix = a[:next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)]
        complete([{"role": "system", "content": system}, {"role": "user", "content": "a"}], STEPS, 1, actions)
        cache, _ = _mlx["caches"][system]
        trim_prompt_cache(cache, cache[0].offset - len(prefix))
        _mlx["caches"][system] = (cache, prefix)
        PROMPT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for old in PROMPT_CACHE_DIR.glob("*.safetensors"):  # prompt or model changed: old caches are useless
            old.unlink()
        save_prompt_cache(str(path), cache, {"ids": json.dumps(prefix)})


def _load() -> None:
    from mlx_lm import load
    model, tok = load(MODEL)
    texts = [tok.decode([i]) for i in range(len(tok.get_vocab()))]
    _mlx.update(model=model, tok=tok, caches={}, texts=texts,
                special={i for i, t in enumerate(texts) if re.fullmatch(r"<\|[^|]+\|>", t)})
