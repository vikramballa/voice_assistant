"""Speech detection and transcription on `say` audio and a real recording of a noisy room."""

import itertools
from pathlib import Path

import numpy as np

from assistant.audio import listen, transcribe
from assistant.language import dialogue
from assistant.audio.transcribe import SpeculativeTranscriber
from speech import as_stream, synthesize


def test_noise_only_gives_no_utterance():
    rng = np.random.default_rng(1)
    noise = (rng.standard_normal(listen.SAMPLE_RATE * 3) * 0.01).astype(np.float32)
    chunks = [noise[i:i + listen.CHUNK] for i in range(0, len(noise) - listen.CHUNK, listen.CHUNK)]
    assert list(listen.segment(chunks)) == []


def test_one_command_gives_one_utterance():
    speech = synthesize("Hey Laya, open Chrome and search YouTube for neural networks", "Samantha")
    utts = list(listen.segment(as_stream(speech)))
    assert len(utts) == 1
    # The utterance must contain (nearly) all of the speech, not cut the start or end.
    assert len(utts[0]) >= len(speech) * 0.9


def test_on_start_called_once_per_utterance():
    speech = synthesize("Hey Laya, open Safari", "Samantha")
    starts = []
    utts = list(listen.segment(as_stream(speech), on_start=lambda: starts.append(1)))
    assert len(utts) == 1 and len(starts) == 1


def test_transcript_is_prepared_during_the_final_pause():
    speculative = SpeculativeTranscriber()
    speech = synthesize("Hey Laya, open Safari", "Samantha")
    (utt,) = listen.segment(as_stream(speech), early_end=speculative.on_pause)
    text, timings = speculative.transcribe(utt)
    assert timings.get("early") and "safari" in text.lower()


def test_start_timeout_stops_waiting():
    silence = [np.zeros(listen.CHUNK, dtype=np.float32)] * 1000
    assert list(listen.segment(iter(silence), start_timeout=1.0)) == []


def _room_with(speech: np.ndarray, loudness: float) -> list[np.ndarray]:
    """A real recording of a noisy room (fan + people talking), with speech mixed in."""
    room = np.load(Path(__file__).parent / "data" / "room_noise.npy").astype(np.float32) / 32768
    floor = np.median([np.sqrt(np.mean(room[i:i + listen.CHUNK] ** 2)) for i in range(0, len(room), listen.CHUNK)])
    sr = listen.SAMPLE_RATE
    audio = room[:len(speech) + 5 * sr].copy()
    audio[sr:sr + len(speech)] += speech / np.sqrt(np.mean(speech ** 2)) * floor * loudness
    return [audio[i:i + listen.CHUNK] for i in range(0, len(audio) - listen.CHUNK, listen.CHUNK)]


def test_command_in_noisy_room_ends_and_is_understood():
    # Before the loudness check, Silero called this room "speech" non-stop and the utterance never ended.
    speech = synthesize("Hey Laya, open Safari", "Samantha")
    utts = list(listen.segment(_room_with(speech, loudness=5)))
    assert utts, "utterance never ended"
    assert len(utts[0]) < len(speech) + 2 * listen.SAMPLE_RATE
    assert "open safari" in transcribe.transcribe(utts[0]).lower()


def test_whisper_loops_are_dropped(monkeypatch):
    monkeypatch.setattr(transcribe.mlx_whisper, "transcribe", lambda *a, **k: {"text": "No, " * 60})
    assert transcribe.transcribe(np.zeros(listen.SAMPLE_RATE, dtype=np.float32)) == ""


def test_dead_microphone_warns(monkeypatch, capsys):
    class SilentMic:  # what macOS delivers with the lid closed or no mic permission
        def __init__(self, callback, blocksize, **_):
            self.callback, self.blocksize = callback, blocksize

        def __enter__(self):
            for _ in range(listen.DEAD_MIC_CHUNKS + 1):
                self.callback(np.zeros((self.blocksize, 1), dtype=np.float32), self.blocksize, None, None)

        def __exit__(self, *_):
            pass

    monkeypatch.setattr(listen.sd, "InputStream", SilentMic)
    for _ in itertools.islice(listen._mic_chunks(), listen.DEAD_MIC_CHUNKS + 1):
        pass
    assert "pure silence" in capsys.readouterr().out


def test_a_pause_inside_a_command_does_not_cut_it():
    # 2026-10-03: "open ChatGPT ... in Brave browser" was cut at the pause and opened the ChatGPT app.
    sr = listen.SAMPLE_RATE
    first = synthesize("Hey Laya, open ChatGPT", "Samantha")
    rest = synthesize("in Brave browser", "Samantha")
    pause = np.zeros(int(0.9 * sr), dtype=np.float32)
    audio = np.concatenate([first, pause, rest])
    assert len(list(listen.segment(as_stream(audio)))) == 2  # without patience: two commands
    speculative = SpeculativeTranscriber(may_go_on=dialogue.may_go_on)
    (utt,) = listen.segment(as_stream(audio), early_end=speculative.on_pause)
    text, _ = speculative.transcribe(utt)
    assert "brave" in text.lower() and "chatgpt" in text.lower().replace(" ", "")


def test_a_whole_command_is_not_kept_waiting():
    speculative = SpeculativeTranscriber(may_go_on=dialogue.may_go_on)
    speech = synthesize("Hey Laya, scroll down", "Samantha")
    stream = as_stream(speech)
    (utt,) = listen.segment(stream, early_end=speculative.on_pause)
    silence_after = len(utt) - len(speech)
    assert silence_after < (listen.SILENCE_SECONDS + 0.4) * listen.SAMPLE_RATE
