"""Doing a task on the screen step by step, the way a person would: look, do one thing, look again.

    do_on_screen("search for running shoes and open the first result")

Each look: a screenshot of the app the user works in -> the GUI-agent model (vision.next_actions) picks the
next action(s) -> the safety rules below check each one -> it is done with the same hands as click / type /
keys / scroll. ~5 s a look (the model reads a 1024 px screenshot each time).

Safety: the user stays in charge.
  - at most MAX_STEPS steps; wanting the step it just did again means finished (the model never says so
    itself; measured 2026-10-27); scrolling the same way STUCK times in a row means stuck
  - moving the mouse stops it at once (the mic isn't listened to while a command runs, so "stop" can't)
  - never pays: a click on buy / pay / order / checkout... always stops, that click is the user's
  - a click on delete / send / submit / sign out... only when the user's own words asked for it
  - never types passwords or codes; presses only SAFE_KEYS (never quit, close, shut down...)
  - never acts in the terminal Laya runs in (target_app); every screenshot is deleted after use
"""

import math
import re
import time

from .errors import Failed
from .in_app import press_keys, scroll, type_text
from .macos import accessibility
from .screen import bring_forward, on_screen, window_shot

_POINT = re.compile(r".*\((\d+\s*,\s*\d+)\).*")
MAX_STEPS = 8
STUCK = 3
STEP_SECONDS = 1.0   # after the model's actions, the page reacts before the next look
ACTION_SECONDS = 0.3  # between two actions of one answer (a clicked box takes the focus)
MAX_WAIT = 3.0       # the model's own wait(duration), capped
MOVED = 20           # points the pointer may move before it counts as the user taking over
SCROLL_SHARE = 0.6   # one scroll step = this much of the window
MONEY = re.compile(r"\b(?:buy|pay|payment|purchase|order|checkout|check\s+out|transfer|donate|bid|book\s+now)\b", re.I)
RISKY = re.compile(r"\b(?:delete|remove|trash|erase|discard|send|submit|post|publish|share|install|uninstall|"
                   r"sign\s+out|log\s+out|subscribe|unsubscribe|accept|agree|confirm|allow|reset|format|cart|basket)\b", re.I)
SECRET = re.compile(r"\b(?:password|passcode|pin|otp|one[\s-]time\s+code|cvv|cvc|card\s+number|security\s+code)\b", re.I)
SAID_SECRET = re.compile(SECRET.pattern + r"\s+(?:is\s+|as\s+|of\s+)?(\S+)", re.I)  # "password hunter2" -> hunter2
SAFE_KEYS = {"enter", "return", "escape", "esc", "tab", "shift+tab", "space", "up", "down", "left", "right",
             "page up", "page down", "pageup", "pagedown", "home", "end", "backspace", "delete",
             "cmd+f", "cmd+l", "cmd+a", "cmd+c", "cmd+v", "cmd+z", "cmd+[", "cmd+]"}


def do_on_screen(task: str) -> str:
    """Do `task` in the app the user is working in by looking at the screen and acting, a step at a time."""
    from assistant.mac import island, vision
    ax = accessibility()
    done, field = [], ""
    at = ax.pointer()
    while len(done) < MAX_STEPS:
        app = bring_forward(ax)
        image, bounds = window_shot(ax, app)
        try:
            actions = vision.next_actions(image, task, done)
            nears = [vision.text_at(image, a["spot"]) if a["spot"] else "" for a in actions]
        except Exception:
            raise Failed("I couldn't load the screen agent model") from None
        finally:
            vision.discard(image)
        if not actions:
            return _stopped("I lost track of what to do next", done)
        for action, near in zip(actions, nears):  # usually one; sometimes "click the box, type shoes"
            name, args = action["name"], action["args"]
            action["said"] = said = _describe(name, args, near)
            if name == "finish":
                return _finished(done)
            if name in ("stop", "call_user"):
                return _stopped(args.get("reason") or "I couldn't do that on this screen", done)
            if said in done[-2:] and name not in ("scroll", "wait"):
                return _finished(done)  # it never says "finished" itself: wanting a step again means it's done
            if done[-STUCK:] == [said] * STUCK:
                return _stopped("I seem to be stuck", done)
            problem = _unsafe(action, near, field, task, ax, app)
            if problem:
                return _stopped(problem, done)
            if math.dist(at, ax.pointer()) > MOVED:  # checked last: the model took seconds, the user may have moved
                return _stopped("You moved the mouse, so I stopped", done)
            print(f"  [screen step {len(done) + 1}: {said}]")
            island.show("thinking", f"{len(done) + 1}. {said}")
            _do(ax, action, bounds)
            done.append(said)
            field = f"{said} {near}" if name in ("click", "doubleclick") else field
            time.sleep(ACTION_SECONDS)
            at = ax.pointer()
            if len(done) == MAX_STEPS:
                break
        time.sleep(STEP_SECONDS)
        at = ax.pointer()
    return _stopped(f"I stopped after {MAX_STEPS} steps", done)


def _describe(name: str, args: dict, near: str) -> str:
    """What a step does, in words: shown on the island and given back to the model as its history."""
    target = " ".join(near.split()[:6]) or "the spot at ({})".format(_POINT.sub(r"\1", args.get("start_box", "")))
    return {"click": f"Click {target}", "doubleclick": f"Double-click {target}",
            "type": f"Type {args.get('content', '').strip()}", "hotkey": f"Press {args.get('key', '')}",
            "scroll": f"Scroll {args.get('direction', 'down')}", "wait": "Wait"}.get(name, name).strip()


def _finished(done: list[str]) -> str:
    if not done:
        return "It looks done already"
    steps = [s[0].lower() + s[1:] for s in done[-4:]]
    return "Done: " + (steps[0] if len(steps) == 1 else ", ".join(steps[:-1]) + " and " + steps[-1])


def _stopped(why: str, done: list[str]) -> str:
    return why if not done else f"{why}, after {len(done)} step{'s' if len(done) > 1 else ''}"


def _unsafe(action: dict, near: str, field: str, task: str, ax, app) -> str | None:
    """Why this action must not be done on its own, or None. `near` is the text where it would click,
    `field` the last click's description and text (the box being typed into: a password box?)."""
    name, args = action["name"], action["args"]
    words = f"{action['said']} {near}"
    if name in ("click", "doubleclick"):
        if action["spot"] is None:
            return "I couldn't tell where to click"
        label = " ".join(near.split()[:6]) or "that"
        if MONEY.search(words):
            return f"I stopped before clicking {label}. Payments are yours to make"
        asked = {" ".join(w.lower().split()) for w in RISKY.findall(task)}
        if {" ".join(w.lower().split()) for w in RISKY.findall(words)} - asked:
            return f'I stopped before clicking {label}. Say "click {label}" if you want me to'
    elif name == "type":  # a password box: by its label, by Accessibility, or the very words said after "password"
        told = [value.strip(".,!?'\"") for value in SAID_SECRET.findall(task)]
        if SECRET.search(f"{words} {field}") or any(v and v in args.get("content", "") for v in told) \
                or _secure_field(ax, app):
            return "I don't type passwords or codes. Please type that one yourself"
    elif name == "hotkey":
        if _key(args.get("key", "")) is None:
            return f"I stopped: it wanted the keys {args.get('key', '')}, which I don't press on my own"
    elif name not in ("scroll", "wait"):
        return f"I can't {name.replace('_', ' ')} on my own yet"
    return None


def _key(key: str) -> str | None:
    """The model's key in press_keys' spelling, if it is a safe one ("Command+C" -> "cmd+c")."""
    key = re.sub(r"\s*\+\s*", "+", key.lower().strip()).replace("command", "cmd").replace("⌘", "cmd+")
    for spelled in (key, key.replace("ctrl+", "cmd+")):  # a model trained partly on Windows writes ctrl+c
        if spelled in SAFE_KEYS:
            return spelled
    return None


def _secure_field(ax, app) -> bool:
    """A password box has focus (in apps that show their boxes to Accessibility)."""
    focused = ax.attr(ax.app_element(app), "AXFocusedUIElement") if app is not None else None
    return focused is not None and ax.attr(focused, "AXSubrole") == "AXSecureTextField"


def _do(ax, action: dict, bounds) -> None:
    name, args, spot = action["name"], action["args"], action["spot"]
    if name in ("click", "doubleclick"):
        ax.click_at(*on_screen(bounds, spot), clicks=2 if name == "doubleclick" else 1)
    elif name == "type":
        type_text(args.get("content", ""))
    elif name == "hotkey":
        press_keys(_key(args["key"]))
    elif name == "wait":
        time.sleep(min(float(re.sub(r"[^\d.]", "", args.get("duration", "")) or 1), MAX_WAIT))
    elif name == "scroll":
        direction = (args.get("direction") or "down").split()[0].lower()
        sideways = direction in ("left", "right")
        if bounds and spot:  # aimed where the model looked (a side panel scrolls on its own)
            pixels = SCROLL_SHARE * bounds[2 if sideways else 3] * (-1 if direction in ("up", "left") else 1)
            ax.scroll(on_screen(bounds, spot), **{"right" if sideways else "down": pixels})
        else:
            scroll(direction)
