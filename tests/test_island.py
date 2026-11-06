"""The island window process: it takes updates, survives bad input, and quits with the assistant."""

import time

from assistant.mac import island


def test_island_runs_and_quits_with_main(monkeypatch):
    monkeypatch.setattr(island, "_proc", None)  # restored after the test
    island.start()
    proc = island._proc
    try:
        time.sleep(1.5)  # app launch
        for state, text in [("listening", ""), ("heard", "open Safari"), ("thinking", ""),
                            ("reply", "Opening Safari"), ("error", "x" * 500), ("hide", "")]:
            island.show(state, text)
            time.sleep(0.2)
        proc.stdin.write("not json\n")  # must be ignored, not crash
        proc.stdin.flush()
        time.sleep(0.5)
        assert proc.poll() is None, "island crashed"
    finally:
        proc.stdin.close()  # what happens when the assistant exits
    assert proc.wait(timeout=5) == 0


def test_show_is_safe_without_a_window(monkeypatch):
    monkeypatch.setattr(island, "_proc", None)
    island.show("reply", "hi")  # no window: must do nothing
