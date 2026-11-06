"""Measure the fast brain on train/test_set.jsonl.

Prints:
  - intent accuracy on single commands (Laya's top choice vs the label)
  - how accuracy and fast-path coverage change with the confidence threshold
  - fast-path results at the current threshold: how many commands it handled,
    and whether any handled command ran the WRONG steps (the number that matters most)

Test rows are either single commands {"intent", "args"} or multi-part commands
{"intent": "multi", "steps": [...]}. "args"/"steps" null means the fast path
must NOT handle it (it should go to the LLM).

Run from the project folder:
    python train/evaluate.py
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from assistant.brain import fast as brain_fast  # noqa: E402


def expected_steps(row: dict) -> list[dict] | None:
    if row["intent"] == "multi":
        return row["steps"]
    if row["args"] is None:
        return None
    return [{"action": row["intent"], "args": row["args"]}]


rows = [json.loads(l) for l in (ROOT / "train" / "test_set.jsonl").read_text().splitlines() if l.strip()]
brain_fast.warm_up()

results, times = [], []
for r in rows:
    t = time.perf_counter()
    steps, info = brain_fast.decide(r["text"])
    times.append((time.perf_counter() - t) * 1000)
    results.append((r, steps, info))

single = [(r, i["clauses"][0]) for r, s, i in results if r["intent"] != "multi"]
correct = sum(c["intent"] == r["intent"] for r, c in single)
print(f"intent accuracy (single commands): {correct}/{len(single)} = {correct / len(single):.0%}")
print(f"latency: median {sorted(times)[len(times) // 2]:.0f} ms, max {max(times):.0f} ms\n")

print("threshold  accuracy-above  coverage")
for th in (0.5, 0.6, 0.7, 0.8, 0.9):
    above = [(r, c) for r, c in single if c["confidence"] >= th]
    acc = sum(c["intent"] == r["intent"] for r, c in above) / len(above) if above else 0
    print(f"  {th:.1f}       {acc:6.0%} ({len(above):2d})     {len(above) / len(single):5.0%}")

fast = [(r, s, i) for r, s, i in results if s is not None]
wrong = [(r, s, i) for r, s, i in fast if s != expected_steps(r)]
should_be_fast = [r for r, s, i in results if expected_steps(r) is not None]
print(f"\nfast path at threshold {brain_fast.CONFIDENCE}: handled {len(fast)}/{len(results)} "
      f"(of {len(should_be_fast)} it could), WRONG {len(wrong)}, rest go to the LLM")
for r, s, i in wrong:
    print(f"  WRONG  {r['text']!r}: got {s}, want {expected_steps(r)}")

print("\ncommands sent to the LLM:")
for r, s, i in results:
    if s is None:
        cl = ", ".join(f"{c['intent']} {c['confidence']:.2f}" for c in i["clauses"])
        print(f"  {i.get('handoff', ''):16s} [{cl}]  {r['text']}")
