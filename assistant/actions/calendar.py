"""Apple Calendar (with whatever accounts the user added there: iCloud, Google...)."""

from datetime import datetime, timedelta

from .errors import Failed
from .macos import date_script, quoted, tell_app
from .reminders import parse_when

EVENTS_SCRIPT = '''{start}{end}tell application "Calendar"
  set out to ""
  repeat with c in calendars
    repeat with e in (every event of c whose start date >= dayStart and start date < dayEnd)
      set s to start date of e
      set out to out & ((hours of s) * 60 + (minutes of s)) & tab & (summary of e) & linefeed
    end repeat
  end repeat
  return out
end tell'''


def calendar_events(day: str = "today") -> str:
    """What's on the calendar today / tomorrow / on a weekday, in time order."""
    start = _day(day)
    out = tell_app("Calendar", EVENTS_SCRIPT.format(start=date_script("dayStart", start),
                                                   end=date_script("dayEnd", start + timedelta(days=1))))
    events = sorted((int(minutes), title) for minutes, title in
                    (line.split("\t", 1) for line in out.splitlines() if "\t" in line))
    if not events:
        return f"Nothing on your calendar {day}"
    return "\n".join(f"{datetime(2000, 1, 1, m // 60, m % 60):%-I:%M %p} {title}" for m, title in events)


def add_event(title: str, when: str, minutes: str = "30") -> str:
    start = parse_when(when)
    if start is None:
        raise Failed(f"When should {title} be?")
    end = start + timedelta(minutes=int(minutes) if minutes.isdigit() else 30)
    tell_app("Calendar", date_script("s", start) + date_script("e", end) + f'''tell application "Calendar"
  tell (first calendar whose writable is true)
    make new event at end of events with properties {{summary:{quoted(title)}, start date:s, end date:e}}
  end tell
end tell''')
    return f"Added {title} at {start:%-I:%M %p}" + ("" if start.date() == datetime.now().date() else f" on {start:%A}")


def _day(day: str) -> datetime:
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    words = day.lower()
    if "tomorrow" in words:
        return today + timedelta(days=1)
    for offset in range(7):
        if (today + timedelta(days=offset)).strftime("%A").lower() in words:
            return today + timedelta(days=offset)
    return today
