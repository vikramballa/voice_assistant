"""Run one action from the terminal: python -m assistant.actions <action> [args...]"""

import sys
import time

from . import ACTIONS, run_action

if len(sys.argv) < 2 or sys.argv[1] not in ACTIONS:
    print("usage: python -m assistant.actions <action> [args...]")
    print("actions:", ", ".join(ACTIONS))
    sys.exit(1)
start = time.perf_counter()
print(run_action(sys.argv[1], *sys.argv[2:]))
print(f"({(time.perf_counter() - start) * 1000:.0f} ms)")
