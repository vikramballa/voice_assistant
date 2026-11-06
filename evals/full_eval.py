"""Whole decision pipeline (rules -> Laya -> LLM fallback) on a labelled file. Nothing is executed.

Row formats (jsonl):
  {"text", "intent": "multi", "steps": [...]}          final steps must equal these
  {"text", "intent": "multi", "steps": null, "llm_action": "answer"}
                                                       Laya must hand off, the LLM's first step must be this action
                                                       ("ask" = we must ask a question back instead)
  {"text", "intent", "args"}                           test_set.jsonl style (args null = must go to the LLM)

Run from the project folder:
    python evals/full_eval.py                     # train/dev_set.jsonl
    python evals/full_eval.py train/test_set.jsonl
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from assistant.brain import fast, slow  # noqa: E402
from assistant.language.dialogue import follow_up_question  # noqa: E402


def expected(row: dict) -> list[dict] | None:
    if row["intent"] == "multi":
        return row["steps"]
    return None if row["args"] is None else [{"action": row["intent"], "args": row["args"]}]


def main() -> None:
    path = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "train/dev_set.jsonl")
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    fast.warm_up()
    slow.warm_up()
    ok = fast_count = fast_wrong = llm_calls = 0
    fails = []
    for row in rows:
        text, want = row["text"], expected(row)
        t = time.perf_counter()
        question = follow_up_question(text)
        steps, info, route = None, {}, "ask"
        if not question:
            steps, info = fast.decide(text)
            route = "laya"
            if steps is None:
                steps, route = slow.plan(text), "llm"
                llm_calls += 1
            else:
                fast_count += 1
        ms = (time.perf_counter() - t) * 1000
        if want is not None:
            good = route != "ask" and steps == want
            if route == "laya" and steps != want:
                fast_wrong += 1
        elif "llm_action" in row:
            if row["llm_action"] == "ask":
                good = route == "ask"
            else:
                good = route == "llm" and bool(steps) and steps[0]["action"] == row["llm_action"]
        else:  # test_set row that must go to the LLM; any LLM answer is accepted
            good = route != "laya"
        ok += good
        if not good:
            shown = ", ".join(f"{s['action']}{tuple(s['args'])}" for s in steps or []) or route
            clauses = ", ".join(f"{c['intent']} {c['confidence']:.2f}" for c in info.get("clauses", []))
            fails.append(f"  FAIL {route:4} {ms:5.0f} ms  {text!r:48} -> {shown}   want {want or row.get('llm_action')}"
                         f"   [laya: {clauses}]")
    print("\n".join(fails))
    print(f"\n{path.name}: {ok}/{len(rows)} right = {ok / len(rows):.0%} | fast path {fast_count} "
          f"(WRONG {fast_wrong}) | LLM {llm_calls}")


if __name__ == "__main__":
    main()
