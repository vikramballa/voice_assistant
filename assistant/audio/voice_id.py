"""Whose voice is it? Only the enrolled owner may give commands (anyone may still say "stop").

A speaker model (NVIDIA TitaNet-small, Apache 2.0, 40 MB ONNX run by sherpa-onnx on the CPU, ~8 ms)
turns an utterance into 192 numbers that describe the voice, not the words. Enrollment averages ~10 of
your phrases into a voiceprint; every later utterance is compared with it (cosine similarity, 0-1).
Laya's own reply voice (Kokoro) is saved next to it, so our reply heard back through the mic loses
against the voiceprint even when it is close.

Chosen 2026-09-29 over TitaNet-large, WeSpeaker CAM++/ResNet34 and 3D-Speaker CAM++ on 14 `say` voices +
Kokoro in the room noise: small had the lowest error and was the fastest (see evals/voice_id_eval.py).

    enrolled()        is there a voiceprint?
    check(audio)      (is it the owner?, similarity)
    python -m assistant.audio.voice_id enroll   record the voiceprint (10 phrases, ~30 s)
    python -m assistant.audio.voice_id add      add phrases (e.g. with another microphone)
    python -m assistant.audio.voice_id test     say anything, see the similarity
"""

import sys
import threading
from datetime import date

import numpy as np

from assistant.config import MODELS_DIR

MODEL = MODELS_DIR / "speaker-titanet-small.onnx"
URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
       "speaker-recongition-models/nemo_en_titanet_small.onnx")
VOICEPRINT = MODELS_DIR / "voiceprint.npz"   # personal: models/ is not in git
SAMPLE_RATE = 16000
THRESHOLD = 0.50        # the owner at least this similar (1 s or longer); eval: owner 100%, others 3.6%
SHORT_THRESHOLD = 0.30  # one-word commands ("mute", "next") carry less voice; owner 90%, others 13%
SHORT_SECONDS = 1.0     # speech shorter than this (without the silence around it) counts as short
LAYA_MARGIN = 0.05      # must also be this much closer to the owner than to our own reply voice
MIN_SECONDS = 0.5       # enrollment phrases shorter than this are asked again (0.8 rejected clear
                        # "Play the next song": its loud part is only ~0.6 s; a cough is < 0.3 s)

PHRASES = ["Hey Laya, open Chrome", "Search YouTube for neural networks", "Turn the volume up a little",
           "What's the weather like today", "Open my downloads folder", "Play the next song",
           "Set a timer for ten minutes", "Close Safari please", "What is twenty five times four",
           "Take a screenshot and then lock the screen"]
LAYA_PHRASES = ["Opening Chrome", "Sorry, that didn't work", "I can help you with that. What should I do?",
                "The answer is six hundred and twenty five", "Okay"]

_extractor = None
_print: dict | None = None
_lock = threading.Lock()   # sherpa streams are not shared between threads


def _load():
    global _extractor
    if _extractor is None:
        import sherpa_onnx
        if not MODEL.exists():
            from urllib.request import urlretrieve
            MODEL.parent.mkdir(exist_ok=True)
            urlretrieve(URL, MODEL)
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(MODEL), num_threads=2)
        _extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
    return _extractor


def embedding(audio: np.ndarray) -> np.ndarray:
    """The voice of 16 kHz float audio as a unit-length vector."""
    extractor = _load()
    with _lock:
        stream = extractor.create_stream()
        stream.accept_waveform(SAMPLE_RATE, np.ascontiguousarray(audio, dtype=np.float32))
        stream.input_finished()
        vector = np.asarray(extractor.compute(stream), dtype=np.float32)
    return vector / (np.linalg.norm(vector) + 1e-9)


def voiceprint() -> dict | None:
    """{"owner": unit vector, "laya": unit vector or None, "samples": all enrollment vectors, "date"}."""
    global _print
    if _print is None and VOICEPRINT.exists():
        data = np.load(VOICEPRINT)
        laya = data["laya"] if "laya" in data and data["laya"].size else None
        _print = {"owner": _unit(data["samples"].mean(axis=0)), "laya": laya,
                  "samples": data["samples"], "date": str(data["date"])}
    return _print


def enrolled() -> bool:
    return voiceprint() is not None


def check(audio: np.ndarray, voice: dict | None = None) -> tuple[bool, float]:
    """(is it the owner?, similarity to the owner). Everyone passes when nobody is enrolled."""
    voice = voice or voiceprint()
    if voice is None:
        return True, 1.0
    vector = embedding(audio)
    score = float(vector @ voice["owner"])
    needed = SHORT_THRESHOLD if speech_seconds(audio) < SHORT_SECONDS else THRESHOLD
    if voice["laya"] is not None and score < float(vector @ voice["laya"]) + LAYA_MARGIN:
        return False, score
    return score >= needed, score


def speech_seconds(audio: np.ndarray) -> float:
    """Roughly how long someone talks: 20 ms frames louder than a third of the loudest ones."""
    frame = SAMPLE_RATE // 50
    n = len(audio) // frame
    if n == 0:
        return 0.0
    rms = np.sqrt((audio[:n * frame].reshape(n, frame) ** 2).mean(axis=1))
    return float((rms > np.quantile(rms, 0.95) / 3).sum()) / 50


def save(samples: list[np.ndarray], laya: np.ndarray | None) -> None:
    global _print
    MODELS_DIR.mkdir(exist_ok=True)
    np.savez(VOICEPRINT, samples=np.stack(samples), laya=laya if laya is not None else np.zeros(0),
             date=str(date.today()))
    _print = None


def laya_voice() -> np.ndarray | None:
    """Our own reply voice (Kokoro) as a unit vector, or None when Kokoro isn't available."""
    from scipy.signal import resample_poly
    from assistant.audio import speaker
    speaker.warm_up()
    audios = [speaker.voice_sample(t) for t in LAYA_PHRASES]
    if audios[0] is None:
        return None
    return _unit(np.mean([embedding(resample_poly(a, 2, 3).astype(np.float32)) for a in audios], axis=0))


def warm_up() -> None:
    embedding(np.zeros(SAMPLE_RATE, dtype=np.float32))
    voiceprint()


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


def _record(phrases: list[str]) -> list[np.ndarray]:
    from assistant.audio import listen
    listen.warm_up()
    samples = []
    for i, phrase in enumerate(phrases, 1):
        while True:
            print(f'  {i}/{len(phrases)}  say: "{phrase}"')
            audio = listen.listen_once(start_timeout=8)
            if audio is not None and speech_seconds(audio) >= MIN_SECONDS:
                samples.append(audio)
                break
            heard = speech_seconds(audio) if audio is not None else 0.0
            print(f"     didn't hear enough ({heard:.1f} s of speech, need {MIN_SECONDS}) - please say it again")
    return samples


def _enroll(add: bool) -> None:
    old = list(voiceprint()["samples"]) if add and enrolled() else []
    print("Speak in your normal voice, where you usually use Laya (room noise is fine).")
    vectors = [embedding(a) for a in _record(PHRASES if not add else PHRASES[:5])]
    samples = old + vectors
    owner = _unit(np.mean(samples, axis=0))
    # How alike your own phrases are: each one against the average of the others.
    alike = [float(v @ _unit(np.mean([w for j, w in enumerate(samples) if j != i], axis=0)))
             for i, v in enumerate(samples)]
    print("  saving Laya's own voice next to yours...")
    laya = laya_voice()
    save(samples, laya)
    print(f"\nSaved {len(samples)} phrases to {VOICEPRINT.name}.")
    print(f"  your phrases vs your voiceprint: lowest {min(alike):.2f}, average {np.mean(alike):.2f}"
          f" (needed: {THRESHOLD})")
    if laya is not None:
        print(f"  Laya's voice vs your voiceprint: {float(laya @ owner):.2f} (lower is better)")
    if min(alike) < THRESHOLD:
        print("  Some phrases were unlike the others (noise? another voice?). Run enroll again if Laya ignores you.")


def _test() -> None:
    from assistant.audio import listen
    if not enrolled():
        print("Nobody enrolled yet: python -m assistant.audio.voice_id enroll")
        return
    listen.warm_up()
    print("Say anything (Ctrl+C to stop)...")
    try:
        for audio in listen.utterances():
            ok, score = check(audio)
            print(f"  {'YOU' if ok else 'not you'}  similarity {score:.2f}  ({speech_seconds(audio):.1f} s of speech)")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "test"
    if command in ("enroll", "add"):
        _enroll(add=command == "add")
    elif command == "test":
        _test()
    else:
        print(__doc__)
    import os
    sys.stdout.flush()
    os._exit(0)  # the speaker/mic threads must not meet Python's teardown (see assistant.__main__)
