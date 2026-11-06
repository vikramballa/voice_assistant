"""The slow brain: a local LLM (qwen3 4B) turns a sentence into steps, when the fast brain isn't sure.

    plan(text) -> ask the LLM for {"steps": [...]} -> repair() checks them against the words -> steps

The model can't answer in free text: a grammar (MLX) or a JSON schema (Ollama) holds it to the plan
shape. Two engines, same model and same test scores (2026-09-27):
    LAYA_BACKEND=mlx     (default) in this process via mlx_llm, ~0.8 s
    LAYA_BACKEND=ollama  the Ollama app, ~1.4-2.1 s; also the automatic fallback if MLX can't load

Test from the terminal:
    python -m assistant.brain.slow "open chrome and search youtube for neural networks"
"""

import json
import os
import re
import sys
import time

from assistant.actions import ACTIONS
from assistant.language.sentence import clean

from . import mlx_llm
from .prompts import CHAT, REPLY_SCHEMA, SCHEMA, SCREEN, system_prompt
from .repair import repair

OLLAMA_MODEL = os.environ.get("LAYA_LLM", "qwen3:4b")
backend = os.environ.get("LAYA_BACKEND", "mlx")  # switched to "ollama" by warm_up() if MLX fails

CALC_WITHOUT_NUMBERS = re.compile(r"\b(?:calculat(?!ors?\b)\w*|operations?|math|maths|sums?|arithmetic)\b", re.IGNORECASE)
DIDNT_CATCH = "Sorry, I didn't catch that"
RESULTS = ("Results:\n{}\nGo on with my request using these results: the next steps, or answer(...) with what I "
           "asked for in one or two short spoken sentences. Don't repeat a step that already ran.")


def plan(text: str, history: list[tuple[str, list[dict]]] | None = None, browser: str | None = None,
         seen: list[tuple[list[dict], list[str]]] | None = None) -> list[dict]:
    """Steps for `text`. history: the last few (command, steps), so "what about Delhi?" can follow
    "weather in Mumbai". browser: the one in use, for when the words don't name one.
    seen: steps of this task that already ran and what they returned (mail, calendar), to plan what's next."""
    text = clean(text)
    earlier = []
    for said, did in history or []:
        earlier += [{"role": "user", "content": said}, {"role": "assistant", "content": _as_json(did)}]
    messages = [{"role": "system", "content": system_prompt()}, *earlier, {"role": "user", "content": text}]
    for did, results in seen or []:
        found = "\n".join(f"{s['action']}: {r}" for s, r in zip(did, results))
        messages += [{"role": "assistant", "content": _as_json(did)}, {"role": "user", "content": RESULTS.format(found)}]
    if backend == "mlx":
        steps = _steps_from(mlx_llm.complete(messages, mlx_llm.STEPS, 200, ACTIONS))
    else:
        steps = json.loads(_ollama(messages, SCHEMA, {"temperature": 0}))["steps"]
    fixed = repair(steps, text, browser)
    if CALC_WITHOUT_NUMBERS.search(text) and not re.search(r"\d", text) \
            and not any(s["action"] in ("calculate", "answer") for s in fixed):
        fixed.append({"action": "answer", "args": ["What should I calculate?"]})  # "open calculator and do math"
    if steps and not fixed:  # every step was nonsense for these words: just talk instead
        fixed = [{"action": "answer", "args": [chat_answer(text)]}]
    return fixed


def chat_answer(text: str) -> str:
    """A plain spoken reply. It still goes through a {"reply"} schema: without one, qwen3 leaks its reasoning."""
    messages = [{"role": "system", "content": CHAT}, {"role": "user", "content": text}]
    if backend == "mlx":
        content = mlx_llm.complete(messages, mlx_llm.REPLY, 120, ACTIONS)
    else:
        content = _ollama(messages, REPLY_SCHEMA, {"temperature": 0, "num_predict": 120})
    try:
        return json.loads(content, strict=False)["reply"].strip() or DIDNT_CATCH
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        return DIDNT_CATCH


SCREEN_CHARS = 5000  # ~1.3k tokens of screen text: enough for a chat list or an error page, read in ~0.5 s


def answer_about_screen(question: str, screen: str, where: str = "the screen") -> str:
    """A spoken answer to `question` from the text on the screen (actions.screen.look_at_screen uses it)."""
    content = f"Screen ({where}):\n{screen[:SCREEN_CHARS]}\n\nQuestion: {question}"
    messages = [{"role": "system", "content": SCREEN}, {"role": "user", "content": content}]
    if backend == "mlx":
        reply = mlx_llm.complete(messages, mlx_llm.REPLY, 160, ACTIONS)
    else:
        reply = _ollama(messages, REPLY_SCHEMA, {"temperature": 0, "num_predict": 160})
    try:
        return json.loads(reply, strict=False)["reply"].strip() or "I couldn't read the screen"
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        return "I couldn't read the screen"


def warm_up() -> None:
    """Load the model and the long system prompt at startup so the first command isn't slow."""
    global backend
    system = system_prompt()
    if backend == "mlx":
        try:
            mlx_llm.prime(system, ACTIONS)
            mlx_llm.complete([{"role": "system", "content": system}, {"role": "user", "content": "hi"}],
                             mlx_llm.STEPS, 1, ACTIONS)
            return
        except Exception as e:  # missing package / model download failed: Ollama still works
            print(f"  (MLX LLM unavailable, using Ollama: {str(e).splitlines()[0][:100]})")
            backend = "ollama"
    _ollama([{"role": "system", "content": system}, {"role": "user", "content": "hi"}], None, {"num_predict": 1})


def _ollama(messages: list[dict], schema: dict | None, options: dict) -> str:
    """One chat call to the Ollama app. keep_alive=-1 keeps the model loaded between commands."""
    import ollama
    return ollama.chat(model=OLLAMA_MODEL, messages=messages, format=schema, think=False, keep_alive=-1,
                       options=options).message.content


def _as_json(steps: list[dict]) -> str:
    return json.dumps({"steps": steps}, separators=(",", ":"))


def _steps_from(content: str) -> list[dict]:
    """Steps from the model's JSON; args may come back as a list, a dict or a single value."""
    try:
        steps = json.loads(content, strict=False)["steps"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return []
    out = []
    for step in steps if isinstance(steps, list) else []:
        if not isinstance(step, dict) or step.get("action") not in ACTIONS:
            continue
        args = step.get("args", [])
        args = list(args.values()) if isinstance(args, dict) else args if isinstance(args, list) else [args]
        out.append({"action": step["action"], "args": ["" if a is None else str(a) for a in args]})
    return out


if __name__ == "__main__":
    warm_up()
    t = time.perf_counter()
    print(plan(" ".join(sys.argv[1:])))
    print(f"({(time.perf_counter() - t) * 1000:.0f} ms)")
