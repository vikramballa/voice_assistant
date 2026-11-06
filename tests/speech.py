"""Test speech without a microphone: macOS `say` speaks a phrase into audio, cut into mic-sized chunks.
Used by test_listen.py and evals/voice_benchmark.py."""

import subprocess
import tempfile
import wave

import numpy as np

from assistant.audio import listen

PAUSE_SECONDS = 0.45  # " | " in a phrase = the speaker pauses this long there


def synthesize(text: str, voice: str) -> np.ndarray:
    """16 kHz float32 audio of `voice` saying `text`; " | " in the text is a mid-sentence pause."""
    if " | " in text:
        gap = np.zeros(int(PAUSE_SECONDS * listen.SAMPLE_RATE), dtype=np.float32)
        parts = [_say(p, voice) for p in text.split(" | ")]
        return np.concatenate([x for p in parts for x in (p, gap)][:-1])
    return _say(text, voice)


def as_stream(speech: np.ndarray) -> list[np.ndarray]:
    """Quiet room noise, the speech, then silence - cut into mic-sized chunks."""
    rng = np.random.default_rng(0)
    noise = lambda n: (rng.standard_normal(n) * 0.003).astype(np.float32)  # noqa: E731
    sr = listen.SAMPLE_RATE
    audio = np.concatenate([noise(sr), speech + noise(len(speech)), noise(int(1.5 * sr))])
    n = len(audio) // listen.CHUNK
    return [audio[i * listen.CHUNK:(i + 1) * listen.CHUNK] for i in range(n)]


def _say(text: str, voice: str) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        subprocess.run(["say", "-v", voice, "-o", f.name, "--data-format=LEI16@16000", text], check=True)
        with wave.open(f.name) as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768
