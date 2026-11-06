"""Start Laya:  python -m assistant [--ptt | --text] [--debug] [--fast] [--no-ui] [--say] [--anyone]
(python main.py does the same).

Exiting (Ctrl+C, or an error) always goes through _exit(): the voice thread is stopped and its sound
stream closed first, then the process ends with os._exit, skipping Python's normal teardown. That teardown
shuts down PortAudio, MLX and torch while their own threads may still be running inside them, which
crashed Python with a segmentation fault (2026-09-29). Nothing is lost by skipping it: the command log
commits each row immediately.

Startup runs offline: every model is already on disk, and Hugging Face's update check waited 20-30 s per
model on a bad network, so "loading models..." looked frozen for minutes (2026-10-03). Only when a model is
missing does Laya start again online to download it.
"""

import os
import sys
import traceback

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # before anything imports huggingface_hub (it reads this once)

from assistant import app, modes  # noqa: E402
from assistant.config import parse_args  # noqa: E402
from assistant.mac import island, permissions  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    code = 0
    try:
        run(parse_args(argv))
    except KeyboardInterrupt:
        pass
    except SystemExit as e:  # --help, bad arguments
        code = e.code if isinstance(e.code, int) else 0
    except BaseException:
        traceback.print_exc()
        code = 1
    _exit(code)


def run(options) -> None:
    if os.environ["HF_HUB_OFFLINE"] == "1" and (missing := app.missing_models()):
        print(f"downloading {', '.join(missing)}...")
        os.environ["HF_HUB_OFFLINE"] = "0"
        sys.stdout.flush()
        os.execv(sys.executable, sys.orig_argv)  # the same command again, now online
    if options.ui:
        island.start()
    print("loading models...")
    app.warm_up(use_mic=options.mode != "text", fast_end=options.fast, use_say=options.say)
    assistant = app.Assistant()
    assistant.say(permissions.check() or "Ready")
    if options.mode == "text":
        modes.run_text(assistant)
    elif options.mode == "ptt":
        modes.run_push_to_talk(assistant, fast_end=options.fast)
    else:
        modes.WakeWordLoop(assistant, debug=options.debug, fast_end=options.fast,
                           owner_only=not options.anyone).run()


def _exit(code: int) -> None:
    """Stop our threads that touch devices, then end the process without Python's teardown (see above)."""
    from assistant.audio import speaker
    try:
        speaker.shutdown()
        island.stop()
    finally:
        print("\nbye")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)


if __name__ == "__main__":
    main()
