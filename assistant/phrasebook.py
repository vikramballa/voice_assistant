"""Phrases the user taught Laya by correcting it: "open not" -> (Laya fails) -> "no, I meant open notes".
The next time the exact same words are heard, the corrected steps run at once (no model).

This is how Laya learns from use, safely:
  - only from an explicit correction ("no, ...", "I meant ...", see dialogue.correction), never by itself
  - only the exact words (normalized), so nothing else changes
  - never answers or sums (they go stale), never phrases that point back ("close it", "do that again")
  - a later correction of the same words replaces the old one
  - it's a small JSON file the user can read; forget one or all from the terminal:

    python -m assistant.phrasebook                      # list what was learned
    python -m assistant.phrasebook forget "open not"
    python -m assistant.phrasebook clear

The evals (train/evaluate.py, evals/) call the brains directly, so their scores never include this memory.
"""

import json
import re
import sys
import threading
import time

from assistant.config import MODELS_DIR
from assistant.language.sentence import clean, strip_wake
from assistant.language.words import normalize

PATH = MODELS_DIR / "phrasebook.json"   # personal, like the voiceprint (models/ is not in git)
# The answer depends on when it is asked, or on what came before: never remembered.
NOT_LEARNED = ("answer", "calculate", "look_at_screen", "look_through_camera")
POINTS_BACK = re.compile(r"\b(?:it|that|this|them|those|again|instead|same)\b")

_lock = threading.Lock()


def key(text: str) -> str:
    """The words that must match exactly: "Hey Laya, Open Not." -> "open not"."""
    return normalize(clean(strip_wake(text) or text))


def lookup(text: str) -> list[dict] | None:
    """The steps the user taught for these exact words, or None."""
    entry = _load().get(key(text))
    return [dict(step, args=list(step["args"])) for step in entry["steps"]] if entry else None


def can_learn(text: str, steps: list[dict]) -> bool:
    k = key(text)
    return bool(k) and bool(steps) and not POINTS_BACK.search(k) \
        and not any(s["action"] in NOT_LEARNED for s in steps)


def learn(text: str, steps: list[dict]) -> bool:
    """Remember `steps` for these words (replacing an older lesson). False if they shouldn't be learned."""
    if not can_learn(text, steps):
        return False
    with _lock:
        book = _load()
        book[key(text)] = {"steps": steps, "learned": time.strftime("%Y-%m-%d %H:%M:%S")}
        _save(book)
    return True


def forget(text: str) -> bool:
    with _lock:
        book = _load()
        if book.pop(key(text), None) is None:
            return False
        _save(book)
    return True


def entries() -> dict[str, dict]:
    return _load()


def _load() -> dict[str, dict]:
    try:
        return json.loads(PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save(book: dict) -> None:
    PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(book, indent=1))
    tmp.replace(PATH)  # never a half-written file


if __name__ == "__main__":
    command, words = (sys.argv[1] if len(sys.argv) > 1 else "list"), " ".join(sys.argv[2:])
    if command == "forget":
        print("forgotten" if forget(words) else f"nothing learned for {key(words)!r}")
    elif command == "clear":
        PATH.unlink(missing_ok=True)
        print("cleared")
    else:
        book = entries()
        for said, entry in book.items():
            shown = ", ".join(f"{s['action']}{tuple(s['args'])}" for s in entry["steps"])
            print(f"  {said!r:40} -> {shown}   ({entry['learned']})")
        print(f"{len(book)} learned phrase(s) in {PATH}")
