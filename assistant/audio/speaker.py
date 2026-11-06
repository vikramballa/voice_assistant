"""The mouth: speak short replies with Kokoro-82M (natural voice, MLX on the Mac GPU),
falling back to macOS `say` if Kokoro can't load. Non-blocking: a new reply interrupts the one playing.

Rules that keep the sound device safe (each one fixed a real crash):
  - Only the worker thread ever calls PortAudio for output. Opening, stopping and closing one stream
    from two threads freed it twice and crashed Python (2026-09-28). speak() only queues the text and
    raises _latest; the worker notices and stops the old reply.
  - The worker keeps ONE output stream open while replies come (closed after IDLE_CLOSE_SECONDS),
    instead of opening and closing one per reply: fewer chances to collide with macOS rebuilding the
    audio device (mute, headphones). A device error closes it; the next reply opens a fresh one.
  - shutdown() stops the worker and closes the stream BEFORE Python exits. Otherwise Python's exit
    shuts PortAudio down while the worker is still writing: segmentation fault (2026-09-29).

Test from the terminal:
    python -m assistant.audio.speaker "Opening Safari"
"""

import queue
import subprocess
import sys
import threading
import time

from assistant.gpu import MLX_LOCK

KOKORO_MODEL = "mlx-community/Kokoro-82M-bf16"
VOICE = "af_heart"   # warm American English voice; others: af_bella, bf_emma, am_michael
SPEED = 1.15
RATE = 24000
BLOCK = 1200               # samples written per step (50 ms): how fast an interrupt takes effect
IDLE_CLOSE_SECONDS = 30    # close the output stream after this long without replies
RECENT_REPLIES = 20        # how many recent replies are kept for the echo check
ECHO_TAIL = 0.5            # a reply still counts as heard this long after its last sound (buffer + room echo)

last_text = ""       # the most recent reply
_spans: list[list] = []   # [start, end, text] of recent replies (end = inf while playing)
_model = None
_jobs: queue.Queue = queue.Queue()
_worker: threading.Thread | None = None
_latest = 0          # number of the newest reply; the worker stops playing when it isn't the newest any more
_queued = False      # a reply is waiting for the worker (counts as "speaking" for the echo check)
_STOP = None         # the job that tells the worker to finish


def warm_up(use_say: bool = False) -> None:
    """Load Kokoro (~1 s) and speak one word silently so the first reply is fast.
    use_say (--say): keep the classic macOS voice instead."""
    global _model
    if use_say:
        return
    try:
        from mlx_audio.tts.utils import load_model
        with MLX_LOCK:
            model = load_model(KOKORO_MODEL)
            list(model.generate(text="Ready", voice=VOICE, speed=SPEED))
        _model = model
    except Exception as e:  # no internet on first run, missing package...: `say` still works
        print(f"  (Kokoro voice unavailable, using macOS say: {e})")


def speak(text: str, wait: bool = False) -> None:
    """Say `text` (interrupting any reply still playing). Returns at once unless wait=True."""
    global _worker, last_text, _latest, _queued
    if not text.strip():
        return
    last_text = text
    if _worker is None:
        _worker = threading.Thread(target=_run, name="speaker", daemon=True)
        _worker.start()
    _latest += 1  # interrupts the current reply (the worker stops it; never touch its stream here)
    _queued = True
    done = threading.Event()
    _jobs.put((text, done, _latest))
    if wait:
        done.wait()


def shutdown(timeout: float = 2.0) -> None:
    """Stop the reply that is playing, close the sound stream and end the worker thread. Call before exiting."""
    global _latest, _worker
    if _worker is None:
        return
    _latest += 1
    _jobs.put(_STOP)
    _worker.join(timeout)
    _worker = None


def spoke_during(t0: float, t1: float) -> bool:
    """True if a reply was playing at some moment between t0 and t1 (or one is waiting to be played)."""
    return _queued or any(start <= t1 and end + ECHO_TAIL >= t0 for start, end, _ in _spans)


def said_during(t0: float, t1: float) -> list[str]:
    """The texts of the replies playing (or waiting) between t0 and t1: what the mic may have heard."""
    texts = [text for start, end, text in _spans if start <= t1 and end + ECHO_TAIL >= t0]
    return texts + ([last_text] if _queued and last_text not in texts else [])


# --- the worker thread (the only code that touches the output device) ---------------------------------

def _run() -> None:
    stream = None
    while True:
        try:
            job = _jobs.get(timeout=IDLE_CLOSE_SECONDS)
        except queue.Empty:
            stream = _close(stream)  # quiet for a while: release the sound device
            continue
        while job is not _STOP and not _jobs.empty():  # only the newest reply matters
            job[1].set()
            job = _jobs.get()
        if job is _STOP:
            break
        text, done, number = job
        try:
            stream = _play(text, number, stream)
        finally:
            done.set()
    _close(stream)


def _play(text: str, number: int, stream):
    """Say one reply; returns the (possibly new) output stream."""
    global _queued
    span = [time.time(), float("inf"), text]
    _spans.append(span)
    del _spans[:-RECENT_REPLIES]
    if number == _latest:
        _queued = False
    try:
        if _model is None:
            _say(text, number)
        else:
            audio = _synthesize(text)
            if number == _latest:  # not already replaced by a newer reply while synthesizing
                stream = _stream_out(audio, number, stream)
    except Exception as e:
        print(f"  (speak failed: {e})")
    finally:
        span[1] = time.time()
    return stream


def voice_sample(text: str):
    """Our reply voice saying `text` (at RATE), without playing it, or None without Kokoro.
    Used by voice_id enrollment."""
    return None if _model is None else _synthesize(text)


def _synthesize(text: str):
    import numpy as np
    with MLX_LOCK:
        return np.concatenate([np.asarray(r.audio, dtype=np.float32).reshape(-1)
                               for r in _model.generate(text=text, voice=VOICE, speed=SPEED)])


def _stream_out(audio, number: int, stream):
    """Write the audio in small blocks; stop at once when a newer reply arrives. A device error (the output
    device changed) closes the stream and retries once on a fresh one."""
    import sounddevice as sd
    for attempt in range(2):
        try:
            if stream is None:
                stream = sd.OutputStream(samplerate=RATE, channels=1, dtype="float32")
                stream.start()
            for i in range(0, len(audio), BLOCK):
                if number != _latest:
                    stream.abort()  # drop what is still buffered, the new reply starts right away
                    stream.start()
                    return stream
                stream.write(audio[i:i + BLOCK])
            return stream
        except sd.PortAudioError as e:
            stream = _close(stream)
            if attempt:
                raise
            print(f"  (sound device changed, reopening: {e})")
    return stream


def _close(stream):
    if stream is not None:
        try:
            stream.abort()
            stream.close()
        except Exception:
            pass
    return None


def _say(text: str, number: int) -> None:
    """macOS `say`, stopped when a newer reply arrives."""
    proc = subprocess.Popen(["say", "-r", "210", text])
    while proc.poll() is None:
        if number != _latest:
            proc.terminate()
            break
        time.sleep(0.05)


if __name__ == "__main__":
    warm_up()
    t = time.perf_counter()
    speak(" ".join(sys.argv[1:]) or "Hello, I'm Laya.", wait=True)
    print(f"({(time.perf_counter() - t) * 1000:.0f} ms including playback)")
    shutdown()
