"""Keeping an eye on the screen: "tell me when Build succeeded shows up".

The app the user is working in is read (Apple OCR, ~0.1 s) every WATCH_SECONDS in the background, until
the text appears (chime + notification + spoken), the time is up, or "stop watching". Only that app's window
is read, never the terminal Laya runs in (it shows the command itself, which would match at once).
One watch at a time; a new one replaces the old.
"""

import re
import threading

from .errors import Failed
from .macos import accessibility
from .reminders import announce
from .screen import window_shot

WATCH_SECONDS = 3
WATCH_MINUTES = 30
_watch: dict = {}  # "stop": the threading.Event of the running watch, "text": what it waits for


def watch_screen(text: str, minutes: str = "") -> str:
    """Tell the user when `text` shows up in the app they are working in."""
    from assistant.mac import vision
    text = text.strip().strip("'\"“”").strip()
    if not text:
        raise Failed("What should I watch for?")
    ax = accessibility()
    app = ax.target_app()
    if app is None:
        raise Failed("I can't tell which app to watch")
    vision.discard(window_shot(ax, app)[0])  # no Screen Recording permission: say so now, not in 30 minutes
    stop_watching()
    stop = threading.Event()
    _watch.update(stop=stop, text=text)
    limit = int(minutes) if re.fullmatch(r"\d+", str(minutes).strip()) else WATCH_MINUTES
    threading.Thread(target=_keep_watching, args=(ax, app, text, limit, stop), daemon=True).start()
    return f"Okay, I'll tell you when {text} shows up in {app.localizedName()}"


def stop_watching() -> str:
    """Stop the running watch, if any."""
    stop = _watch.pop("stop", None)
    if stop is None:
        return "I wasn't watching anything"
    stop.set()
    return f"Okay, I stopped watching for {_watch.pop('text')}"


def _keep_watching(ax, app, text: str, minutes: int, stop: threading.Event) -> None:
    from assistant.mac import vision
    found = False
    for _ in range(int(minutes * 60 / WATCH_SECONDS)):
        if stop.wait(WATCH_SECONDS):
            return
        if app.isTerminated():
            break
        try:
            image, _ = window_shot(ax, app)
        except Exception:  # the window is gone for a moment (minimized, between spaces): look again next time
            continue
        try:
            found = vision.find_text(image, text) is not None
        finally:
            vision.discard(image)
        if found:
            break
    if _watch.get("stop") is stop:  # still this watch (not stopped or replaced meanwhile)
        stop_watching()
        announce(f"{text} is on the screen now" if found else f"I stopped watching for {text}. It didn't show up")
