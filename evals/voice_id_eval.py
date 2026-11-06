"""How well does voice_id tell the owner from other people? (not pytest; ~1-2 min, nothing executed)

Each of 14 macOS `say` voices takes a turn as the "owner": 10 enrollment phrases, then 8 new phrases
(3 of them one word) must pass, while the other voices and Laya's own Kokoro voice must fail.
All audio is mixed into the recorded room noise (voice 5x the room level) and padded like the
listener does (0.3 s before, 0.55 s after).

    .venv/bin/python evals/voice_id_eval.py 2>/dev/null
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from assistant.audio import voice_id  # noqa: E402
from tests.speech import synthesize  # noqa: E402

VOICES = ["Samantha", "Daniel", "Karen", "Moira", "Rishi", "Tessa", "Fred", "Ralph", "Albert", "Kathy",
          "Eddy (English (US))", "Reed (English (UK))", "Grandpa (English (US))", "Aman"]
SAME_VOICE = {"Eddy (English (US))", "Reed (English (UK))"}  # one voice, two accents: not counted against each other
TEST = ["Hey Laya, open Brave and search for cooking videos", "Lock the screen", "Mute", "Stop", "Next",
        "Scroll down a little", "Open GitHub dot com", "Remind me to call mom at six"]
LAYA_TEST = ["Opening YouTube in Brave", "I can help you with bear", "Done", "Search for what?",
             "The volume is now at fifty percent"]
FAMILIES = [{"Eddy (English (US))", "Reed (English (UK))", "Grandpa (English (US))"}, {"Rishi", "Aman"}]
ROOM = np.load(Path(__file__).resolve().parent.parent / "tests/data/room_noise.npy").astype(np.float32)
RATIO = 5.0
rng = np.random.default_rng(0)


def in_room(speech: np.ndarray) -> np.ndarray:
    sr = voice_id.SAMPLE_RATE
    padded = np.concatenate([np.zeros(int(0.3 * sr)), speech, np.zeros(int(0.55 * sr))]).astype(np.float32)
    start = rng.integers(0, len(ROOM) - len(padded))
    noise = ROOM[start:start + len(padded)]
    rms = lambda a: np.sqrt(np.mean(a ** 2)) + 1e-9  # noqa: E731
    return (padded / rms(speech) * rms(noise) * RATIO + noise).astype(np.float32)


def main() -> None:
    from scipy.signal import resample_poly
    from assistant.audio import speaker
    print("making audio...")
    enroll = {v: [in_room(synthesize(p, v)) for p in voice_id.PHRASES] for v in VOICES}
    test = {v: [in_room(synthesize(p, v)) for p in TEST] for v in VOICES}
    speaker.warm_up()
    test["Kokoro"] = [in_room(resample_poly(speaker.voice_sample(t), 2, 3).astype(np.float32)) for t in LAYA_TEST]
    laya = voice_id.laya_voice()

    owner_ok = {"long": [0, 0], "short": [0, 0]}
    others = [0, 0]
    kokoro = [0, 0]
    scores, t, sweep = [], 0.0, []
    for owner in VOICES:
        samples = [voice_id.embedding(a) for a in enroll[owner]]
        voice = {"owner": voice_id._unit(np.mean(samples, axis=0)), "laya": laya}
        for speaker_name, audios in test.items():
            if speaker_name != owner and {speaker_name, owner} <= SAME_VOICE:
                continue
            for phrase, audio in zip(TEST if speaker_name != "Kokoro" else LAYA_TEST, audios):
                t0 = time.perf_counter()
                ok, score = voice_id.check(audio, voice)
                t += time.perf_counter() - t0
                scores.append(score)
                family = speaker_name != owner and any({speaker_name, owner} <= f for f in FAMILIES)
                kind = "short" if len(phrase.split()) == 1 else "long"
                sweep.append((speaker_name == owner, speaker_name == "Kokoro", family, kind, score))
                if speaker_name == owner:
                    kind = "short" if len(phrase.split()) == 1 else "long"
                    owner_ok[kind][0] += ok
                    owner_ok[kind][1] += 1
                    if not ok:
                        print(f"  missed owner {owner[:10]:10s} {phrase!r} ({score:.2f})")
                elif speaker_name == "Kokoro":
                    kokoro[0] += ok
                    kokoro[1] += 1
                else:
                    others[0] += ok
                    others[1] += 1
                    if ok:
                        print(f"  let in {speaker_name[:10]:10s} as {owner[:10]:10s} {phrase!r} ({score:.2f})")
    pct = lambda a: f"{a[0]}/{a[1]} ({100 * a[0] / a[1]:.0f}%)"  # noqa: E731
    print(f"\nowner accepted, sentences:  {pct(owner_ok['long'])}")
    print(f"owner accepted, one word:   {pct(owner_ok['short'])}")
    print(f"other voices let in:        {pct(others)}")
    print(f"Laya's own voice let in:    {pct(kokoro)}")
    print("\nthreshold sweep (similarity only)   owner long / owner short / others long / others short"
          " / others excl. same-synth families")
    for th in np.arange(0.25, 0.71, 0.05):
        rate = lambda rows: f"{100 * np.mean([r[4] >= th for r in rows]):5.1f}%"  # noqa: E731
        own = [r for r in sweep if r[0]]
        oth = [r for r in sweep if not r[0] and not r[1]]
        print(f"  {th:.2f}  {rate([r for r in own if r[3] == 'long'])}  {rate([r for r in own if r[3] == 'short'])}"
              f"  {rate([r for r in oth if r[3] == 'long'])}  {rate([r for r in oth if r[3] == 'short'])}"
              f"  {rate([r for r in oth if not r[2]])}")
    print(f"check time: {1000 * t / len(scores):.1f} ms per utterance")


if __name__ == "__main__":
    main()
    import os
    sys.stdout.flush()
    os._exit(0)
