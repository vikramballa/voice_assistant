"""One lock for work on the Mac GPU through MLX (Whisper, Kokoro, the LLM, the vision model).

They run in different threads, and two MLX jobs at once from two threads is not safe.
Re-entrant, so a function holding it can call another one that takes it too.
"""

import threading

MLX_LOCK = threading.RLock()
