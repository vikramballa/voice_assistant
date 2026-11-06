"""The speaker check with the real model (TitaNet-small, ~10 ms per utterance) on `say` voices."""

import numpy as np
import pytest

from assistant.audio import voice_id
from tests.speech import synthesize

ROOM = np.load("tests/data/room_noise.npy").astype(np.float32)


def in_room(speech: np.ndarray, ratio: float = 5.0) -> np.ndarray:
    """0.3 s before and 0.55 s after, like the listener; voice `ratio` times the room level."""
    sr = voice_id.SAMPLE_RATE
    padded = np.concatenate([np.zeros(int(0.3 * sr)), speech, np.zeros(int(0.55 * sr))]).astype(np.float32)
    noise = ROOM[:len(padded)]
    rms = lambda a: np.sqrt(np.mean(a ** 2))  # noqa: E731
    return (padded / rms(speech) * rms(noise) * ratio + noise).astype(np.float32)


@pytest.fixture(scope="module")
def samantha(tmp_path_factory):
    """Samantha enrolled as the owner (saved and loaded back like the real voiceprint)."""
    path = tmp_path_factory.mktemp("voice") / "voiceprint.npz"
    old = voice_id.VOICEPRINT
    voice_id.VOICEPRINT, voice_id._print = path, None
    samples = [voice_id.embedding(in_room(synthesize(p, "Samantha"))) for p in voice_id.PHRASES]
    voice_id.save(samples, laya=None)
    yield voice_id.voiceprint()
    voice_id.VOICEPRINT, voice_id._print = old, None


def test_voiceprint_is_saved_and_loaded(samantha):
    assert samantha["samples"].shape == (10, 192)
    assert abs(np.linalg.norm(samantha["owner"]) - 1) < 1e-5 and samantha["laya"] is None


def test_owner_passes_other_voices_do_not(samantha):
    for text in ["Hey Laya, open Brave and search for cooking videos", "Remind me to call mom at six"]:
        assert voice_id.check(in_room(synthesize(text, "Samantha")), samantha)[0], text
        # Clear cases; how often similar voices get in is measured by evals/voice_id_eval.py (Moira scores 0.55 here).
        for other in ["Daniel", "Fred", "Kathy"]:
            ok, score = voice_id.check(in_room(synthesize(text, other)), samantha)
            assert not ok, (other, text, score)


def test_our_own_reply_voice_loses_against_the_owner(samantha):
    """With Laya's voice saved next to the owner's, audio closer to Laya's voice is never the owner."""
    karen = voice_id.embedding(in_room(synthesize("Opening Chrome for you", "Karen")))
    voice = dict(samantha, laya=karen)  # stand-in for Kokoro: no MLX model needed in this test
    assert not voice_id.check(in_room(synthesize("The answer is six hundred", "Karen")), voice)[0]
    assert voice_id.check(in_room(synthesize("Open GitHub dot com", "Samantha")), voice)[0]


def test_speech_seconds_ignores_the_silence_around_it():
    sr = voice_id.SAMPLE_RATE
    tone = np.sin(np.arange(sr) * 2 * np.pi * 220 / sr).astype(np.float32) * 0.3
    audio = np.concatenate([np.zeros(sr // 2), tone, np.zeros(sr)]).astype(np.float32)
    assert 0.9 <= voice_id.speech_seconds(audio) <= 1.1
    assert voice_id.speech_seconds(np.zeros(10, dtype=np.float32)) == 0.0


def test_short_enrollment_phrases_are_long_enough():
    """"Play the next song" was asked again 7 times (2026-10-02): its loud part is only ~0.6 s."""
    from tests.speech import synthesize
    for voice in ("Samantha", "Daniel"):
        for phrase in voice_id.PHRASES:
            assert voice_id.speech_seconds(synthesize(phrase, voice)) >= voice_id.MIN_SECONDS, (voice, phrase)
