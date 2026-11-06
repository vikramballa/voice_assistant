"""Apps and folders: open, quit, wait until an app is in front."""

import os
import subprocess
import time
from pathlib import Path

from .errors import Failed
from .macos import osascript, run
from .web import BROWSERS

FOLDERS = {
    "downloads": "~/Downloads", "documents": "~/Documents", "desktop": "~/Desktop",
    "pictures": "~/Pictures", "music": "~/Music", "movies": "~/Movies", "home": "~",
    "applications": "/Applications",
}


def open_app(name: str) -> str:
    app = BROWSERS.get(name.lower(), name)
    try:
        run(["open", "-a", app])
    except subprocess.CalledProcessError:
        raise Failed(f"I couldn't find an app called {app}") from None
    return f"Opening {app}"


def close_app(name: str) -> str:
    app = BROWSERS.get(name.lower(), name).replace('"', "")
    # Checking first: "tell application X to quit" would launch X if it isn't running.
    if osascript(f'application "{app}" is running') != "true":
        return f"{app} isn't open"
    osascript(f'tell application "{app}" to quit')
    return f"Closed {app}"


def wait_for_app(name: str, timeout: float = 4.0) -> None:
    """Block until `name` is the frontmost app, so a following type_text lands in it."""
    from assistant.mac.accessibility import front_app
    want = BROWSERS.get(name.lower(), name).lower()
    deadline = time.time() + timeout
    while time.time() < deadline:
        front = front_app()
        # The process name can differ from the app's file name ("Code" for Visual Studio Code.app): check both.
        names = {(front.localizedName() or "").lower(), Path(front.bundleURL().path()).stem.lower()} \
            if front is not None and front.bundleURL() is not None else set()
        if want in names:
            time.sleep(0.3)  # window is up; give it a moment to accept keystrokes
            return
        time.sleep(0.1)


def open_folder(name: str) -> str:
    path = FOLDERS.get(name.lower(), name)
    run(["open", os.path.expanduser(path)])
    return f"Opening {name}"
