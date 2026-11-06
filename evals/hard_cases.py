"""Tricky commands through the real decision pipeline (Laya, then the LLM if Laya
hands off), nothing executed on the Mac. Many are real commands from the logs
that used to fail. Not a pytest file: it loads the LLM and takes ~30 s.

Run from the project folder:
    python evals/hard_cases.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from assistant.brain import fast, slow  # noqa: E402
from assistant.language.dialogue import follow_up_question  # noqa: E402

ANY = None  # any args
ASKS = []   # unfinished: Laya should ask what's missing

# (command, list of acceptable results). A result is a list of (action, args or ANY).
CASES = [
    ("Please open Gemini in my Google account", [[("open_website", ["gemini.google.com", "chrome"])]]),
    ("Open YouTube", [[("open_website", ["youtube.com", "chrome"])]]),
    ("Open Braille", [[("open_app", ["Brave Browser"])]]),
    ("volume up", [[("volume_up", [])]]),
    ("Host", [[("answer", ANY)]]),
    ("Search neural networks in...", [[("web_search", ["neural networks", "chrome"])]]),
    ("Read the containers what are there", [[("answer", ANY)], [("open_app", ["Docker"])]]),
    ("tell me a joke", [[("answer", ANY)]]),
    ("what's the capital of france", [[("answer", ANY)]]),
    ("who are you", [[("answer", ANY)]]),
    ("what time is it", [[("tell_time", ["time"])]]),
    ("what's the date today", [[("tell_time", ["date"])]]),
    ("how much battery do I have", [[("battery_status", [])]]),
    ("turn on dark mode", [[("dark_mode", ["on"])]]),
    ("open notion", [[("open_website", ["notion.so", "chrome"])]]),
    ("open google docs", [[("open_website", ["docs.google.com", "chrome"])]]),
    ("go to my google account", [[("open_website", ["myaccount.google.com", "chrome"])]]),
    ("open whatsapp in chrome", [[("open_website", ["web.whatsapp.com", "chrome"])]]),
    ("open whatsapp", [[("open_app", ["WhatsApp"])]]),
    ("open vs code", [[("answer", ANY)]]),
    ("open youtube and play arijit singh songs", [[("youtube_search", ANY)]]),
    ("play arijit singh songs", [[("youtube_search", ANY)]]),
    ("open brave and search for pizza", [[("web_search", ["pizza", "brave"])]]),
    ("close whatsapp", [[("close_app", ["WhatsApp"])]]),
    ("open calender and google holidays in india",
     [[("open_app", ["Calendar"]), ("web_search", ["holidays in india", "chrome"])]]),
    ("send a message to mom", ASKS),
    ("make it a bit louder", [[("volume_up", [])]]),
    ("turn the volume up to 80", [[("set_volume", ["80"])]]),
    ("open the downloads folder and take a screenshot", [[("open_folder", ["downloads"]), ("screenshot", [])]]),
]


def matches(steps: list[dict], want: list[tuple]) -> bool:
    return len(steps) == len(want) and all(
        s["action"] == a and (args is ANY or s["args"] == args) for s, (a, args) in zip(steps, want))


def main() -> None:
    fast.warm_up()
    slow.warm_up()
    passed = 0
    for text, options in CASES:
        question = follow_up_question(text)
        t = time.perf_counter()
        steps, _ = fast.decide(text)
        path = "laya"
        if steps is None:
            steps, path = slow.plan(text), "llm"
        ms = (time.perf_counter() - t) * 1000
        ok = bool(question) if options is ASKS else not question and any(matches(steps, o) for o in options)
        passed += ok
        shown = ", ".join(f"{s['action']}{tuple(s['args'])}" for s in steps) or "nothing"
        print(f"{'ok  ' if ok else 'FAIL'} {path:4} {ms:5.0f} ms  {text!r:50} -> {shown}")
    print(f"\n{passed}/{len(CASES)} passed")


if __name__ == "__main__":
    main()
