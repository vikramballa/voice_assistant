"""End-to-end voice test without a microphone.

macOS `say` speaks each test command into an audio file (3 different
accents). The audio goes through the real pipeline - VAD segmentation,
Whisper, wake word, Laya, LLM fallback - and the resulting steps are
compared with what should happen. Nothing is executed on the Mac.

Run from the project folder:
    python evals/voice_benchmark.py
    python evals/voice_benchmark.py --fast   # with early end-of-speech (Smart Turn)
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from assistant.audio import listen, transcribe  # noqa: E402
from assistant.brain import fast, slow  # noqa: E402
from assistant.language import dialogue  # noqa: E402
from assistant.language.sentence import continuation, strip_wake  # noqa: E402
from assistant.modes import fast_end_policy  # noqa: E402
from speech import as_stream, synthesize  # noqa: E402

VOICES = ["Samantha", "Daniel", "Aman"]

# (spoken text, expected steps). None = must be ignored (no wake word).
CASES = [
    ("Hey Laya, open Chrome and search YouTube for neural networks",
     [{"action": "youtube_search", "args": ["neural networks", "chrome"]}]),
    ("Hey Laya, open Safari", [{"action": "open_app", "args": ["Safari"]}]),
    ("Hey Laya, google the weather in Hyderabad", [{"action": "web_search", "args": ["the weather in hyderabad", "chrome"]}]),
    ("Hey Laya, set the volume to 40", [{"action": "set_volume", "args": ["40"]}]),
    ("Hey Laya, take a screenshot and lock my Mac",
     [{"action": "screenshot", "args": []}, {"action": "lock_screen", "args": []}]),
    ("Hey Laya, skip this song", [{"action": "next_track", "args": []}]),
    ("Hey Laya, open my downloads folder", [{"action": "open_folder", "args": ["downloads"]}]),
    ("Hey Laya, quit Chrome", [{"action": "close_app", "args": ["Google Chrome"]}]),
    ("Hey Laya, go to github.com", [{"action": "open_website", "args": ["github.com", "chrome"]}]),
    ("Hey Laya, play a video about black holes", [{"action": "youtube_search", "args": ["black holes", "chrome"]}]),
    ("Hey Laya, mute", [{"action": "mute", "args": []}]),
    ("Hey Laya, open notes and type buy milk",
     [{"action": "open_app", "args": ["Notes"]}, {"action": "type_text", "args": ["buy milk"]}]),
    ("I told my friend to open Safari", None),
    # " | " = the speaker pauses 0.45 s there (mid-sentence pause: must not lose the second half)
    ("Hey Laya, open Notes | and type buy milk",
     [{"action": "open_app", "args": ["Notes"]}, {"action": "type_text", "args": ["buy milk"]}]),
    ("Hey Laya, search YouTube for | lofi music", [{"action": "youtube_search", "args": ["lofi music", "chrome"]}]),
    ("Hey Laya, set the volume to | 30", [{"action": "set_volume", "args": ["30"]}]),
    ("Can you turn the volume up a bit", None),
]


def main() -> None:
    print("loading models...")
    fast.warm_up()
    slow.warm_up()
    transcribe.warm_up()
    # --fast: measure the early end-of-speech too
    speculative = transcribe.SpeculativeTranscriber(fast_end_policy(lambda: True) if "--fast" in sys.argv else None)
    early_count = 0

    ok = total = 0
    stt_times, laya_times, llm_times = [], [], []
    for spoken, expected in CASES:
        for voice in VOICES:
            total += 1
            utts = list(listen.segment(as_stream(synthesize(spoken, voice)), early_end=speculative.on_pause))
            # Like modes.WakeWordLoop: a wake-word command, then "and ..." continuations.
            heard, commands = [], []
            for u in utts:
                text, timing = speculative.transcribe(u)
                early_count += bool(timing.get("early"))
                stt_times.append(timing["stt_ms"])
                heard.append(text)
                c = strip_wake(text)
                if c is None and commands and (dialogue.trails_off(commands[-1]) or
                                               dialogue.follow_up_question(commands[-1])):
                    commands[-1] = dialogue.complete(commands[-1], text)  # "set the volume to..." + "30"
                    continue
                if c is None and commands:
                    c = continuation(text)
                if c:
                    commands.append(c)
            heard = " / ".join(heard)
            command = commands[0] if commands else None

            path, steps = "-", None
            for c in commands:
                t = time.perf_counter()
                part, _ = fast.decide(c)
                laya_times.append((time.perf_counter() - t) * 1000)
                path = "laya" if path != "llm" else path
                if part is None:
                    t = time.perf_counter()
                    part = slow.plan(c)
                    llm_times.append((time.perf_counter() - t) * 1000)
                    path = "llm"
                steps = (steps or []) + part
            passed = (steps == expected) if expected is not None else command is None
            ok += passed
            mark = "PASS" if passed else "FAIL"
            got = ", ".join(f"{s['action']}{tuple(s['args'])}" for s in steps) if steps else "ignored"
            print(f"{mark} {voice:8s} {path:4s} {len(utts)} utt  heard {heard!r} -> {got}")

    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else 0  # noqa: E731
    print(f"\npassed {ok}/{total}")
    print(f"whisper median {med(stt_times):.0f} ms | laya median {med(laya_times):.0f} ms | "
          f"llm median {med(llm_times):.0f} ms ({len(llm_times)}/{len(laya_times)} commands needed it)")
    wait = listen.EARLY_SECONDS * 1000 if early_count > len(stt_times) / 2 else listen.SILENCE_SECONDS * 1000
    print(f"{early_count}/{len(stt_times)} utterances ended early (after {listen.EARLY_SECONDS * 1000:.0f} ms of "
          f"silence instead of {listen.SILENCE_SECONDS * 1000:.0f} ms)")
    print(f"typical response after you stop talking: {wait:.0f} ms silence wait + "
          f"{med(stt_times):.0f} ms whisper + {med(laya_times):.0f} ms laya "
          f"= {wait + med(stt_times) + med(laya_times):.0f} ms")


if __name__ == "__main__":
    main()
