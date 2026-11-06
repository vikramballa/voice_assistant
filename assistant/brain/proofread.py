"""Auto-correction of dictated text: Whisper writes words that sound like what was said but don't fit
("I'm a developer. And a hackle." -> "And a hacker."). The LLM proposes a fix with the note so far as
context; a fix is kept only when every changed word SOUNDS like the word heard (phonetic.sound_alike),
so the meaning can never be rewritten.

    correct(text, before)  -> (text, [(heard, fixed), ...])
"""

import difflib
import json
import re

from assistant.language.phonetic import sound_alike

SOUND_ALIKE = 0.55      # a changed word must sound at least this much like the one heard ("hackle"/"hacker" 0.75)
MAX_CHANGED = 0.4       # ...and at most this share of the words may change
MAX_TOKENS = 80
SMALL_WORDS = {"a", "an", "the", "i", "to", "of", "and", "in", "on", "from", "for", "is", "am", "are"}

PROOFREAD = """You fix speech-to-text mistakes in dictated text.
The text was heard by a speech recognizer, so some words may be replaced by words that SOUND similar but make no sense
(e.g. "by milk" -> "buy milk", "and a hackle" -> "and a hacker", "I scream" -> "ice cream").
Rules:
- Only replace words that are clearly misheard, with words that sound alike and fit the context.
- Keep everything else exactly: same words, same order, same meaning, same punctuation. Never add new ideas.
- Never answer or comment on the text. If nothing is wrong, return it unchanged.
Reply as JSON: {"reply": "<the corrected text>"}"""


def _words(text: str) -> list[str]:
    return re.findall(r"[\w']+", text.lower())


def believable(heard: str, fixed: str) -> list[tuple[str, str]] | None:
    """The word changes from `heard` to `fixed` when each one sounds like what was heard, else None."""
    a, b = _words(heard), _words(fixed)
    if not a or not b:
        return None
    changes, changed = [], 0
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        old, new = " ".join(a[i1:i2]), " ".join(b[j1:j2])
        if op == "insert" and set(b[j1:j2]) <= SMALL_WORDS and j2 - j1 == 1:
            pass  # "I from Delhi" -> "I'm from Delhi" style: one small word
        elif op == "delete" or op == "insert" or abs((i2 - i1) - (j2 - j1)) > 1 or sound_alike(old, new) < SOUND_ALIKE:
            return None
        changes.append((old, new))
        changed += max(i2 - i1, j2 - j1)
    if changed > max(1, MAX_CHANGED * len(a)):
        return None
    return changes


def _last_sentences(fixed: str, text: str) -> str:
    """The LLM sometimes repeats the earlier text: keep only as many sentences as were dictated."""
    split = lambda t: re.findall(r"[^.?!]+[.?!]*", t.strip())
    mine, got = split(text), split(fixed)
    return "".join(got[-len(mine):]).strip() if len(got) > len(mine) and mine else fixed


def correct(text: str, before: str = "") -> tuple[str, list[tuple[str, str]]]:
    """(the text with misheard words fixed, the fixes). The text itself when unsure or the LLM fails."""
    if len(_words(text)) < 2:
        return text, []
    from . import mlx_llm, slow
    if slow.backend != "mlx":
        return text, []  # Ollama: ~1.5 s per sentence is too slow for dictation
    context = f"Earlier text, for context only (do not repeat it): {before[-300:].strip()}\n\n" if before.strip() else ""
    messages = [{"role": "system", "content": PROOFREAD}, {"role": "user", "content": f"{context}Fix only this: {text}"}]
    try:
        fixed = json.loads(mlx_llm.complete(messages, mlx_llm.REPLY, MAX_TOKENS, slow.ACTIONS), strict=False)["reply"].strip()
    except Exception:
        return text, []
    fixed = _last_sentences(fixed, text)
    changes = believable(text, fixed)
    if not changes:
        return text, []
    if text[-1:] in ".?!" and fixed[-1:] not in ".?!":
        fixed += text[-1]
    return fixed, changes
