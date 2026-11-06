"""Where things live on disk, and the command-line options.

Tuning numbers (how long a pause ends a sentence, how sure Laya must be, ...) are not here:
each one sits at the top of the module that uses it, next to the code it tunes.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"        # downloaded / cached model files (Smart Turn, the LLM's prompt cache)
LOGS_DIR = ROOT / "logs"            # the command log (logbook.py)
TRAIN_DIR = ROOT / "train"          # training data and the fine-tuned Laya checkpoint


@dataclass(frozen=True)
class Options:
    """How this run of the assistant behaves (set once at startup from the command line)."""
    mode: str = "wake"      # "wake": say "Hey Laya" | "ptt": press Enter, then speak | "text": type commands
    debug: bool = False     # print every sound the mic picked up
    fast: bool = False      # end listening at a short pause when the command sounds finished (~0.25 s sooner)
    ui: bool = True         # show the Siri-style island at the top of the screen
    say: bool = False       # classic macOS `say` voice instead of Kokoro
    anyone: bool = False    # wake-word mode: accept every voice, not only the enrolled owner's (voice_id)


def parse_args(argv: list[str] | None = None) -> Options:
    parser = argparse.ArgumentParser(prog="python main.py", description="Laya: fast, local voice control for your Mac.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--ptt", action="store_true", help="push-to-talk: press Enter, then speak")
    mode.add_argument("--text", action="store_true", help="type commands instead of speaking (no mic)")
    parser.add_argument("--debug", action="store_true", help="also show every sound the mic picked up")
    parser.add_argument("--fast", action="store_true",
                        help="reply ~0.25 s sooner on short pauses (may cut a sentence at a pause)")
    parser.add_argument("--no-ui", action="store_true", help="no Siri-style island at the top of the screen")
    parser.add_argument("--say", action="store_true", help="classic macOS voice instead of Kokoro")
    parser.add_argument("--anyone", action="store_true",
                        help="accept commands from every voice (default: only the enrolled owner's, if enrolled)")
    args = parser.parse_args(argv)
    return Options(mode="text" if args.text else "ptt" if args.ptt else "wake",
                   debug=args.debug, fast=args.fast, ui=not args.no_ui, say=args.say,
                   anyone=args.anyone)
