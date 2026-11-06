"""Has the user finished talking? Smart Turn v3.2 (pipecat-ai, BSD-2, 8 MB ONNX, ~15 ms on CPU)
listens to the intonation of the last 8 s of audio and gives P(finished).

Used with --fast only, together with language.dialogue.looks_complete (the words), to stop listening
after a short pause instead of always waiting the full listen.SILENCE_SECONDS.
"""

import numpy as np

from assistant.config import MODELS_DIR

MODEL = MODELS_DIR / "smart-turn-v3.2-cpu.onnx"
URL = "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx"
SAMPLE_RATE = 16000
SECONDS = 8

_session = None
_features = None


def _load() -> None:
    global _session, _features
    import onnxruntime as ort
    from transformers import WhisperFeatureExtractor
    if not MODEL.exists():
        from urllib.request import urlretrieve
        MODEL.parent.mkdir(exist_ok=True)
        urlretrieve(URL, MODEL)
    options = ort.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 2
    _session = ort.InferenceSession(str(MODEL), sess_options=options)
    _features = WhisperFeatureExtractor(chunk_length=SECONDS)


def finished_probability(audio: np.ndarray) -> float:
    if _session is None:
        _load()
    audio = audio[-SECONDS * SAMPLE_RATE:]
    feats = _features(audio, sampling_rate=SAMPLE_RATE, return_tensors="np", padding="max_length",
                      max_length=SECONDS * SAMPLE_RATE, truncation=True, do_normalize=True)
    return float(_session.run(None, {"input_features": feats.input_features.astype(np.float32)})[0][0].item())


def warm_up() -> None:
    finished_probability(np.zeros(SAMPLE_RATE, dtype=np.float32))
