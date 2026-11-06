"""The hands: everything Laya can do on the Mac.

Every action is a plain function: simple string arguments in, a short sentence to speak out.
Both brains produce the same kind of plan, a list of steps like
    [{"action": "youtube_search", "args": ["neural networks", "chrome"]}]
and run_action() looks the name up in ACTIONS and calls it.

An action that can't be done raises errors.Failed with a message for the user.

Adding an action: write the function in the fitting module, add it to ACTIONS below, describe it in
brain/prompts.py, and (if Laya should learn it) see "Extending Laya" in README.md.

Test from the terminal:
    python -m assistant.actions youtube_search "neural networks"
    python -m assistant.actions open_app Safari
"""

from .agent import do_on_screen
from .apps import close_app, open_app, open_folder, wait_for_app
from .calculator import calculate
from .calendar import add_event, calendar_events
from .camera import look_through_camera
from .errors import Failed, NeedsPermission
from .in_app import clear_text, press_keys, scroll, type_text, write_note
from .info import answer, tell_time, weather
from .media import media_play_pause, mute, next_track, previous_track, set_volume, unmute, volume_down, volume_up
from .messaging import read_mail, send_email, send_message
from .reminders import add_reminder, set_timer
from .screen import click, look_at_screen
from .system import battery_status, dark_mode, lock_screen, screenshot
from .watch import stop_watching, watch_screen
from .web import open_website, switch_tab, web_search, youtube_search

# Name -> function. The brains pick a name from here and fill in the arguments.
ACTIONS = {
    "open_app": open_app,
    "close_app": close_app,
    "web_search": web_search,
    "youtube_search": youtube_search,
    "open_website": open_website,
    "switch_tab": switch_tab,  # rules only: "go to the ChatGPT tab" (an open tab, by name)
    "open_folder": open_folder,
    "media_play_pause": media_play_pause,
    "next_track": next_track,
    "previous_track": previous_track,
    "set_volume": set_volume,
    "volume_up": volume_up,
    "volume_down": volume_down,
    "mute": mute,
    "unmute": unmute,  # rules only (not in the LLM prompt, which stays unchanged); Laya reads "unmute" as mute
    "type_text": type_text,
    "write_note": write_note,
    "clear_text": clear_text,  # rules only: "remove that", "clear the chat box", "delete the last word"  # rules only, like unmute ("write a note saying ...", "write X in notes")
    "screenshot": screenshot,
    "lock_screen": lock_screen,
    "dark_mode": dark_mode,
    "set_timer": set_timer,
    "add_reminder": add_reminder,
    "calendar_events": calendar_events,
    "add_event": add_event,
    "read_mail": read_mail,
    "send_email": send_email,
    "send_message": send_message,
    "tell_time": tell_time,
    "weather": weather,
    "battery_status": battery_status,
    "answer": answer,
    "calculate": calculate,
    "click": click,
    "press_keys": press_keys,
    "scroll": scroll,
    "look_at_screen": look_at_screen,
    "do_on_screen": do_on_screen,  # a task done on screen step by step by the GUI-agent model (agent.py)
    "look_through_camera": look_through_camera,
    "watch_screen": watch_screen,
    "stop_watching": stop_watching,  # rules only: "stop watching"
}
# Actions that open something in a browser; their second argument is the browser.
WEB_ACTIONS = ("open_website", "web_search", "youtube_search")

__all__ = ["ACTIONS", "WEB_ACTIONS", "Failed", "NeedsPermission", "run_action", "wait_for_app"]


def run_action(name: str, *args) -> str:
    return ACTIONS[name](*args)
