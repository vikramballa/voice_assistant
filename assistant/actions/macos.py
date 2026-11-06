"""Small helpers every action module uses to talk to macOS: run a command, run AppleScript,
get the Accessibility API (or a clear permission error)."""

import subprocess
from datetime import datetime

from assistant.mac.permissions import accessibility_message

from .errors import Failed, NeedsPermission


def run(cmd: list[str]) -> None:
    """Run a command; raises CalledProcessError if it fails."""
    subprocess.run(cmd, check=True, capture_output=True)


def osascript(script: str, timeout: float = 20) -> str:
    """Run AppleScript and return what it printed. Permission problems become NeedsPermission / Failed."""
    try:
        out = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:  # usually a macOS permission dialog nobody answered
        raise Failed("macOS is asking for permission on screen. Click Allow, then try again") from None
    if out.returncode:
        if "not allowed" in out.stderr or "(1002)" in out.stderr or "assistive access" in out.stderr:
            raise NeedsPermission(accessibility_message())
        raise subprocess.CalledProcessError(out.returncode, "osascript", out.stdout, out.stderr)
    return out.stdout.strip()


def tell_app(app: str, script: str, timeout: float = 30) -> str:
    """AppleScript that drives another app (Mail, Calendar, Reminders...): a clear message when the user
    hasn't allowed us to control it yet."""
    try:
        return osascript(script, timeout)
    except subprocess.CalledProcessError as e:
        if "-1743" in (e.stderr or ""):  # the user said no to "Terminal wants to control <app>"
            raise Failed(f"I need permission to use {app}. Allow it in System Settings, Privacy and "
                         "Security, Automation") from None
        raise Failed(f"{app} didn't do that: {(e.stderr or '').strip()[-80:]}") from None


def quoted(text: str) -> str:
    """Text as an AppleScript string literal."""
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def date_script(name: str, when: datetime) -> str:
    """AppleScript lines that set variable `name` to `when` (independent of the Mac's date format)."""
    return (f"set {name} to current date\nset day of {name} to 1\nset year of {name} to {when.year}\n"
            f"set month of {name} to {when.month}\nset day of {name} to {when.day}\n"
            f"set time of {name} to {when.hour * 3600 + when.minute * 60}\n")


def accessibility():
    """The accessibility module, after checking that macOS lets us use it."""
    from assistant.mac import accessibility as ax
    if not ax.trusted():
        raise NeedsPermission(accessibility_message())
    return ax
