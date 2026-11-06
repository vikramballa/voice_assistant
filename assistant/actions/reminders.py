"""Timers (in this process: lost if Laya quits) and reminders (Apple Reminders: synced to the iPhone)."""

import re
import subprocess
import threading
from datetime import date, datetime, timedelta

from assistant.language.spoken_math import words_to_numbers

from .errors import Failed
from .macos import date_script, quoted, tell_app


def set_timer(amount: str, label: str = "") -> str:
    seconds = _seconds(amount)
    if not seconds:
        raise Failed("How long should the timer be?")
    label = label.strip()
    _say_later(seconds, f"Time's up{': ' + label if label else ''}")
    return f"Timer set for {_duration_words(seconds)}"


def add_reminder(text: str, when: str = "") -> str:
    """A reminder in Apple's Reminders app, with an alert at `when` if given."""
    text = text.strip()
    if not text:
        raise Failed("What should I remind you about?")
    due = parse_when(when) if when.strip() else None
    if when.strip() and due is None:
        raise Failed(f"I didn't understand the time {when}")
    props = f"name:{quoted(text)}" + (", due date:d, remind me date:d" if due else "")
    tell_app("Reminders", (date_script("d", due) if due else "")
             + f'tell application "Reminders" to make new reminder with properties {{{props}}}')
    if due:
        on = "" if due.date() == date.today() else f" on {due.strftime('%A')}"
        return f"I'll remind you to {text} at {due.strftime('%-I:%M %p')}{on}"
    return f"Added a reminder to {text}"


def _seconds(amount: str) -> int | None:
    """"5 minutes", "90 seconds", "1 hour 30 minutes", "10" (minutes) -> seconds."""
    text = amount.lower().replace("half an hour", "30 minutes").replace("an hour and a half", "90 minutes")
    text = words_to_numbers(text.replace("an hour", "1 hour").replace("a minute", "1 minute"))
    parts = re.findall(r"(\d+(?:\.\d+)?)\s*(h|hours?|hrs?|m|min|mins|minutes?|s|sec|secs|seconds?)?", text)
    total = 0.0
    for number, unit in parts:
        unit = unit or "minutes"
        total += float(number) * (3600 if unit.startswith("h") else 1 if unit.startswith("s") else 60)
    return int(total) if total > 0 else None


def _duration_words(seconds: int) -> str:
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    words = [f"{v} {n}{'s' if v != 1 else ''}" for v, n in ((h, "hour"), (m, "minute"), (s, "second")) if v]
    return " and ".join(words)


def parse_when(when: str) -> datetime | None:
    """"in 20 minutes", "at 6 pm", "18:30", "tomorrow at 9 am" -> datetime, or None."""
    text = when.lower().strip()
    now = datetime.now()
    if text.startswith("in "):
        seconds = _seconds(text[3:])
        return now + timedelta(seconds=seconds) if seconds else None
    text = text.replace("noon", "12 pm").replace("midnight", "12 am")
    m = re.search(r"(\d{1,2})(?::|\.)?(\d{2})?\s*(am|pm|a\.m\.|p\.m\.)?", words_to_numbers(text))
    if not m:
        if re.search(r"\btomorrow\b", text):  # "tomorrow": in the morning
            return (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
        return None
    hour, minute, half = int(m.group(1)), int(m.group(2) or 0), (m.group(3) or "").replace(".", "")
    if half == "pm" and hour < 12:
        hour += 12
    if half == "am" and hour == 12:
        hour = 0
    if hour > 23 or minute > 59:
        return None
    due = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if "tomorrow" in text:
        due += timedelta(days=1)
    elif due <= now:  # "at 9" when it's already 10 -> tomorrow (or tonight for a bare hour)
        due += timedelta(hours=12) if not half and hour < 12 and due + timedelta(hours=12) > now else timedelta(days=1)
    return due


def announce(message: str) -> None:
    """Get the user's attention later, from a background thread: a chime, a notification, and the words spoken."""
    from assistant.audio.speaker import speak
    subprocess.run(["afplay", "/System/Library/Sounds/Glass.aiff"])
    shown = message.replace("\\", "").replace('"', "'")  # a quote in the label would end the AppleScript text
    subprocess.run(["osascript", "-e", f'display notification "{shown}" with title "Laya"'])
    speak(message)


def _say_later(seconds: int, message: str) -> None:
    threading.Timer(seconds, announce, args=(message,)).start()
