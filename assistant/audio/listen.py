"""The ears: microphone -> speech detection -> one audio array per thing said.

Silero VAD (a tiny speech-detection model, ~1 ms per 32 ms chunk on CPU) decides when you start and
stop talking. Silero also calls fans, AC and far-away voices "speech", so a loudness gate is added:
speech must be louder than the room (the median level while nobody talks to us), and an utterance
ends when the voice drops well below its own peak.

    utterances()  endless stream of utterances, for wake-word mode
    listen_once() one utterance, for push-to-talk mode
    segment()     the same cutting logic on any list of chunks (no microphone needed: used by the tests)

Test from the terminal:
    python -m assistant.audio.listen
"""

import queue
import time
from collections import deque
from functools import cache

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
CHUNK = 512                  # 32 ms, the size Silero expects at 16 kHz
START_PROB = 0.5             # speech starts above this
END_PROB = 0.35              # ...and counts as silence below this
START_RATIO = 1.8            # start: this many times louder than the room
END_RATIO = 1.4              # still talking: this many times louder than the room...
PEAK_FRACTION = 0.45         # ...and at least this fraction of the utterance's loudest part
MIN_FLOOR = 0.002            # room level used in a perfectly silent room
ROOM_SECONDS = 3             # how much recent idle audio defines the room level
SILENCE_SECONDS = 0.7        # end of utterance after this much silence (0.55 cut "open ChatGPT | in Brave")
PATIENT_SECONDS = 1.4        # ...or this much, when the words so far may go on (early_end says so)
EARLY_SECONDS = 0.29         # a short pause: early_end() is asked here (speculative transcription, --fast)
PRE_ROLL_SECONDS = 0.3       # keep audio from just before speech started
MIN_SPEECH_SECONDS = 0.25    # ignore clicks and coughs
MAX_SECONDS = 8              # commands are short; never record forever in a noisy room
DEAD_MIC_CHUNKS = int(2 * SAMPLE_RATE / CHUNK)  # warn after 2 s of exact zeros

captured_at = 0.0  # when the newest chunk handed out was recorded (utterances can wait in the queue)


@cache
def _vad():
    import torch
    from silero_vad import load_silero_vad
    torch.set_num_threads(1)
    return load_silero_vad()


def _speech_prob(chunk: np.ndarray) -> float:
    import torch
    return _vad()(torch.from_numpy(chunk), SAMPLE_RATE).item()


def segment(chunks, start_timeout: float | None = None, early_end=None, on_start=None):
    """Cut a stream of CHUNK-sized audio blocks into utterances (a generator of float32 arrays).
    start_timeout: stop after this many seconds (of audio) without speech.
    early_end(audio) is asked once per short pause: True ends the utterance right there, a number of
    seconds waits that long for the pause to end it ("open ChatGPT" ... "in Brave"), else the usual wait.
    on_start() is called when speech begins (must return quickly)."""
    per_second = SAMPLE_RATE / CHUNK
    pre_roll: deque = deque(maxlen=int(PRE_ROLL_SECONDS * per_second))
    room: deque = deque(maxlen=int(ROOM_SECONDS * per_second))  # idle loudness history
    recent: deque = deque(maxlen=4)                              # ~130 ms loudness smoothing
    silence_limit = int(SILENCE_SECONDS * per_second)
    early_limit = int(EARLY_SECONDS * per_second)
    max_chunks = int(MAX_SECONDS * per_second)
    min_chunks = int(MIN_SPEECH_SECONDS * per_second)
    timeout_chunks = int(start_timeout * per_second) if start_timeout else None

    vad = _vad()
    vad.reset_states()
    waited = 0
    speech: list[np.ndarray] = []
    speech_chunks = silent = 0
    limit = silence_limit  # silence that ends this pause
    peak = floor = 0.0
    for chunk in chunks:
        prob = _speech_prob(chunk)
        recent.append(float(np.sqrt(np.mean(chunk ** 2))))
        level = sum(recent) / len(recent)
        if not speech:  # waiting for speech to start
            floor = max(float(np.median(room)) if room else MIN_FLOOR, MIN_FLOOR)
            if prob >= START_PROB and level >= START_RATIO * floor:
                speech = list(pre_roll) + [chunk]
                speech_chunks, silent, peak = 1, 0, level
                if on_start is not None:
                    on_start()
            else:
                pre_roll.append(chunk)
                room.append(recent[-1])
                waited += 1
                if timeout_chunks and waited > timeout_chunks:
                    return
            continue
        speech.append(chunk)  # inside an utterance: is the person still talking?
        peak = max(peak, level)
        if prob >= END_PROB and level >= max(END_RATIO * floor, PEAK_FRACTION * peak):
            speech_chunks += 1
            silent = 0
            limit = silence_limit
        else:
            silent += 1
        done_early = False
        if early_end is not None and silent == early_limit and speech_chunks >= min_chunks:
            decision = early_end(np.concatenate(speech))
            done_early = decision is True
            if not isinstance(decision, bool) and isinstance(decision, (int, float)) and decision > 0:
                limit = max(silence_limit, int(decision * per_second))
        if done_early or silent >= limit or len(speech) >= max_chunks:
            if speech_chunks >= min_chunks:
                yield np.concatenate(speech)
            speech, waited, limit = [], 0, silence_limit
            pre_roll.clear()
            recent.clear()
            vad.reset_states()


def _mic_chunks():
    """CHUNK-sized blocks from the default microphone, forever (sets captured_at for each)."""
    global captured_at
    blocks: queue.Queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        blocks.put((time.time(), indata[:, 0].copy()))

    dead = 0  # chunks in a row of exact digital silence
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=CHUNK, callback=callback):
        while True:
            captured_at, chunk = blocks.get()
            dead = dead + 1 if not chunk.any() else 0
            if dead == DEAD_MIC_CHUNKS:
                print(f"\n  !! The microphone ({sd.query_devices(kind='input')['name']}) is sending pure silence.\n"
                      "  !! Open the MacBook lid (the built-in mic is off while it's closed), or pick another\n"
                      "  !! input in System Settings > Sound, or allow Microphone for this terminal in\n"
                      "  !! System Settings > Privacy & Security.", flush=True)
            yield chunk


def utterances(start_timeout: float | None = None, early_end=None, on_start=None):
    """Yield one float32 audio array per thing said into the microphone."""
    yield from segment(_mic_chunks(), start_timeout, early_end, on_start)


def listen_once(start_timeout: float = 5.0, early_end=None, on_start=None) -> np.ndarray | None:
    gen = utterances(start_timeout, early_end, on_start)
    try:
        return next(gen, None)
    finally:
        gen.close()  # releases the microphone


def warm_up() -> None:
    _vad()


if __name__ == "__main__":
    from .transcribe import transcribe
    from .transcribe import warm_up as whisper_warm_up
    print("loading whisper...")
    whisper_warm_up()
    print("speak (Ctrl+C to stop)")
    for audio in utterances():
        t = time.perf_counter()
        text = transcribe(audio)
        print(f"{text!r}  ({len(audio) / SAMPLE_RATE:.1f}s audio, transcribed in {(time.perf_counter() - t) * 1000:.0f} ms)")
