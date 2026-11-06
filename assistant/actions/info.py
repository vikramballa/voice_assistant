"""Actions whose whole point is the sentence they return: an answer, the time, the weather."""

import re
import time
from urllib.parse import quote
from urllib.request import urlopen

from .web import web_search


def answer(text: str) -> str:
    """Say something back (the LLM's answer to a question, or why it can't help)."""
    return text


def tell_time(what: str = "time") -> str:
    if what == "date":
        return time.strftime("Today is %A, %B ") + str(int(time.strftime("%d")))
    return time.strftime("It's %-I:%M %p")


def weather(place: str = "") -> str:
    """Current weather from wttr.in (free, no key; needs internet). Empty place = where the Mac is.
    If that fails, a Google search for the weather instead."""
    place = place.strip()
    try:
        with urlopen(f"https://wttr.in/{quote(place)}?format=%C,+%t,+feels+like+%f", timeout=4) as r:
            now = r.read().decode().strip()
    except OSError:
        now = ""
    if not now or "unknown" in now.lower() or "<" in now:
        web_search(f"weather {place}".strip())
        return "I couldn't get the weather, so I searched for it"
    now = re.sub(r"\+?(-?\d+)°C", r"\1 degrees", now.replace("  ", " ").replace(" ,", ","))
    return f"{place.title() + ': ' if place else ''}{now}"
