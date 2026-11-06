"""Turn real commands the LLM had to handle into Laya training examples.

Every time Laya wasn't sure, the LLM decided instead and the assistant logged it.
Those are exactly the phrasings Laya should learn, and the LLM's answer is
the label (the "teacher"). Single-step, error-free decisions are added to
train/learned.jsonl; the LLM finding nothing to do becomes "other".

Better still are the user's own corrections ("no, I meant open notes", logged with "corrects"): the words
that went wrong get the label of what the user meant, and the LLM's label for them is not used.
Laya's own replies heard back through the mic ("that makes 171", "opening website") are skipped.

Then retrain:
    python train/learn_from_logs.py     # review what gets added
    python train/make_dataset.py        # merges learned.jsonl in
    python train/finetune.py
    python train/evaluate.py
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from assistant import logbook  # noqa: E402
from assistant.brain.fast import INTENTS  # noqa: E402
from assistant.language.sentence import split_commands  # noqa: E402
from assistant.language.words import normalize  # noqa: E402

LEARNED = ROOT / "train" / "learned.jsonl"
# The start of Laya's own replies (actions/*.py): such a logged "command" was our voice coming back.
OUR_REPLY = re.compile(r"^(?:opening|searching (?:google|youtube)|that makes|volume \d|muted|unmuted|typed|closed|clicked|"
                       r"scrolled|locking|done|next|previous|screenshot saved|timer set|i'll remind|added a reminder|"
                       r"it's \d|today is|battery is|sorry|i'm|i can|i couldn't|i need|i don't|\S+ isn't (?:installed|open)|"
                       r"[\d.]+ (?:times|plus|minus|divided by|to the power of) [\d.]+ is)\b")


def intent_of(steps: list[dict]) -> str | None:
    """Laya's label for a one-step plan (None if Laya has no such option)."""
    intent = steps[0]["action"] if steps else "other"
    if intent in ("answer", "dark_mode"):  # LLM-only actions: for Laya these are "other"
        intent = "other"
    return intent if intent in INTENTS else None


def main() -> None:
    known = set()
    if LEARNED.exists():
        known = {json.loads(l)["text"] for l in LEARNED.read_text().splitlines() if l.strip()}
    new = []
    everything = logbook.entries()
    # 1. The user's corrections: the best labels there are.
    corrected = set()
    for entry in everything:
        if entry.get("learned") and len(entry["steps"]) == 1:
            text, intent = normalize(entry["corrects"]), intent_of(entry["steps"])
            corrected.add(text)
            if intent and text not in known:
                known.add(text)
                new.append({"text": text, "intent": intent})
    # 2. What the LLM decided when Laya wasn't sure.
    for entry in everything:
        if entry["path"] != "llm" or entry.get("error"):
            continue
        text = normalize(entry["text"])
        steps = entry["steps"]
        if len(steps) > 1 or len(split_commands(text)) > 1:
            continue  # multi-part: splitting handles these, nothing to learn per sentence
        if text in corrected or OUR_REPLY.match(text):
            continue  # the user said the LLM got it wrong / our own voice heard back
        intent = intent_of(steps)
        if intent and text not in known:
            known.add(text)
            new.append({"text": text, "intent": intent})
    with LEARNED.open("a") as f:
        for row in new:
            f.write(json.dumps(row) + "\n")
            print(f"  + {row['intent']:16s} {row['text']}")
    print(f"added {len(new)} examples to {LEARNED} (edit that file to fix any wrong labels)")


if __name__ == "__main__":
    main()
