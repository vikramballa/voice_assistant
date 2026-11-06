"""Audio -> text with Whisper small.en on the Mac GPU (MLX), ~250 ms per command.

SpeculativeTranscriber starts Whisper already at a short pause, so when that pause turns out to be the
end of the utterance, the text is ready the moment listening stops.
"""

import time
import zlib
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable

import mlx_whisper
import numpy as np

from assistant.gpu import MLX_LOCK

from . import listen

WHISPER_MODEL = "mlx-community/whisper-small.en-mlx"  # large-v3-turbo was 4x slower and no more accurate here
MAX_TOKENS = 64  # a command is a few words; stops Whisper looping for seconds on noise
# Whisper spells unusual words better when they appear in its prompt.
VOCAB_HINT = ("Hey Laya, open Chrome and search YouTube for neural networks. Open YouTube in Brave. Open Safari, "
              "Finder, Calculator, Spotify, VS Code, System Settings. Go to GitHub, Gmail, ChatGPT. Google the weather. "
              "Volume up. Take a screenshot. Scroll down. Zoom in. Reset zoom. What is on my screen? Stop.")
# While the user writes a note, app names in the prompt pull plain speech toward them ("I'm from..." was heard
# as "I'm the Chrome."): the note's own last words are the better prompt (Whisper continues the context).
DICTATION_HINT = "Hello, this is my note."
HINT_CHARS = 200
_hint = VOCAB_HINT


def set_dictation(text_so_far: str | None) -> None:
    """None: commands (app-name hint). A string (maybe ""): the user is writing; the note so far is the hint."""
    global _hint
    _hint = VOCAB_HINT if text_so_far is None else (text_so_far.strip()[-HINT_CHARS:] or DICTATION_HINT)


def transcribe(audio: np.ndarray) -> str:
    with MLX_LOCK:
        result = mlx_whisper.transcribe(audio, path_or_hf_repo=WHISPER_MODEL, language="en",
                                        initial_prompt=_hint, condition_on_previous_text=False,
                                        temperature=0.0, sample_len=MAX_TOKENS)
    text = result["text"].strip()
    # On noise Whisper can loop ("no, no, no, ..."); such text compresses extremely well.
    if len(text) > 40 and len(zlib.compress(text.encode())) < len(text) / 2.4:
        return ""
    return text


def warm_up() -> None:
    """Load Whisper once at startup so the first command isn't slow."""
    transcribe(np.zeros(listen.SAMPLE_RATE, dtype=np.float32))


# end_early(audio, text) -> bool: may end the utterance right at a short pause. `text()` waits for the
# speculative transcript of `audio`, so a policy that only needs the voice doesn't pay for Whisper.
EndEarly = Callable[[np.ndarray, Callable[[], str]], bool]


class SpeculativeTranscriber:
    """At every short pause (listen.EARLY_SECONDS) Whisper starts in the background. If the pause becomes
    the end of the utterance, the transcript is already done: ~0 ms wait instead of ~250 ms. Nothing is
    cut off: listening still waits for the full pause, unless `end_early` says the command is finished.

        speculative = SpeculativeTranscriber()
        audio = listen.listen_once(early_end=speculative.on_pause)
        text, timings = speculative.transcribe(audio)
    """

    def __init__(self, end_early: EndEarly | None = None, may_go_on: Callable[[str], bool] | None = None):
        self._end_early = end_early
        self._may_go_on = may_go_on  # text -> True: wait listen.PATIENT_SECONDS at this pause
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._audio: np.ndarray | None = None  # the audio at the last short pause...
        self._job: Future | None = None        # ...and its transcript being made

    def on_pause(self, audio: np.ndarray) -> bool:
        """listen.segment's early_end callback."""
        self._audio, self._job = audio, self._pool.submit(transcribe, audio)
        if self._end_early is not None and self._end_early(audio, self._job.result):
            return True
        if self._may_go_on is not None and self._may_go_on(self._job.result()):
            return listen.PATIENT_SECONDS  # "open ChatGPT" ... "in Brave": don't cut the sentence at this pause
        return False

    def transcribe(self, audio: np.ndarray) -> tuple[str, dict]:
        """(text, {"stt_ms": time waited, "early": True if the speculative transcript was used})."""
        cached, job = self._audio, self._job
        self._audio = self._job = None
        start = time.perf_counter()
        # Usable if the utterance ended in that pause: after it came only the rest of the silence.
        tail = int((listen.PATIENT_SECONDS - listen.EARLY_SECONDS) * listen.SAMPLE_RATE) + listen.CHUNK
        early = (cached is not None and 0 <= len(audio) - len(cached) <= tail
                 and np.array_equal(audio[:len(cached)], cached))
        text = job.result() if early else transcribe(audio)
        return text, {"stt_ms": round((time.perf_counter() - start) * 1000), **({"early": True} if early else {})}
