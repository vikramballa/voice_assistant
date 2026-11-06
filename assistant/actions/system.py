"""The Mac itself: screenshot, lock, dark mode, battery."""

import os
import re
import subprocess
import time

from assistant.mac.permissions import screen_recording_message

from .errors import NeedsPermission
from .macos import osascript, run


def screenshot() -> str:
    path = os.path.expanduser(f"~/Desktop/Screenshot {time.strftime('%Y-%m-%d at %H.%M.%S')}.png")
    try:
        run(["screencapture", "-x", path])
    except subprocess.CalledProcessError as e:
        if b"could not create image" in (e.stderr or b""):
            raise NeedsPermission(screen_recording_message()) from None
        raise
    return "Screenshot saved to Desktop"


def lock_screen() -> str:
    try:
        osascript('tell application "System Events" to keystroke "q" using {control down, command down}')
    except NeedsPermission:
        run(["pmset", "displaysleepnow"])  # no permission needed; locks when a password is required after sleep
    return "Locking"


def dark_mode(state: str = "toggle") -> str:
    value = {"on": "true", "off": "false"}.get(state.lower(), "not dark mode")
    osascript(f'tell application "System Events" to tell appearance preferences to set dark mode to {value}')
    on = osascript('tell application "System Events" to tell appearance preferences to get dark mode') == "true"
    return "Dark mode on" if on else "Light mode on"


def battery_status() -> str:
    out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    level = re.search(r"(\d+)%", out)
    if not level:
        return "I can't read the battery on this Mac"
    state = re.search(r"%;\s*([a-z ]+);", out)
    state = state.group(1) if state else ""
    extra = "and charging" if state == "charging" else "and fully charged" if state in ("charged", "finishing charge") \
        else "on power" if "AC Power" in out else ""
    return f"Battery is at {level.group(1)} percent {extra}".strip()
