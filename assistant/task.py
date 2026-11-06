"""Carrying out a plan the way a person does a task: run the steps; when a step looked something up (mail,
the calendar), read what came back and decide the next steps; never send anything without a yes.

    run(steps, plan_next, say)  -> Outcome (what ran, what to say, what waits for a yes)
    confirmation(step)          -> "Should I email Rahul: "I'll join at 5"?"
    describe(steps)             -> "open Visual Studio Code and play music"
"""

from dataclasses import dataclass, field
from typing import Callable

from assistant.actions import Failed, run_action, wait_for_app

# Steps that bring back information to act on; said out loud while the next steps are planned.
LOOKUPS = {"read_mail": "Checking your mail", "calendar_events": "Checking your calendar"}
CONFIRM = {"send_email": 'Should I email {0}: "{2}"?', "send_message": 'Should I text {0}: "{1}"?'}
SAYING = {"open_app": "open {0}", "open_website": "open {0}", "close_app": "close {0}", "web_search": "search for {0}",
          "youtube_search": "search YouTube for {0}", "media_play_pause": "play music", "set_volume": "set the volume to {0}",
          "type_text": "type {0}", "press_keys": "press {0}", "switch_tab": "go to the {0} tab"}
MAX_ROUNDS = 4  # look, act, look again, answer: enough for "reply to Rahul's last email"

PlanNext = Callable[[list[tuple[list[dict], list[str]]]], list[dict]]


@dataclass
class Outcome:
    ran: list[dict] = field(default_factory=list)      # every step that ran, in order
    last: list[dict] = field(default_factory=list)     # the last round's steps: the reply is about these
    replies: list[str] = field(default_factory=list)   # ...and what each of them returned
    error: str | None = None
    error_is_message: bool = False                     # the error is a sentence meant for the user
    waiting: list[dict] = field(default_factory=list)  # steps that need the user's yes before they run


def run(steps: list[dict], plan_next: PlanNext | None = None, say: Callable[[str], None] | None = None,
        confirmed: bool = False) -> Outcome:
    """Run the steps. After a lookup, `plan_next(seen)` (the LLM) decides what comes next from the results.
    A step that sends something stops the run: it waits in `waiting` unless `confirmed`."""
    out, seen = Outcome(), []
    for _ in range(MAX_ROUNDS):
        if plan_next:  # steps planned after a lookup were guesses: re-planned once its result is known
            steps = steps[:next((i + 1 for i, s in enumerate(steps) if s["action"] in LOOKUPS), len(steps))]
        stop = len(steps) if confirmed else next((i for i, s in enumerate(steps) if s["action"] in CONFIRM), len(steps))
        now, out.waiting = steps[:stop], steps[stop:]
        if plan_next and say and any(s["action"] in LOOKUPS for s in now):
            say(next(LOOKUPS[s["action"]] for s in now if s["action"] in LOOKUPS))
        out.replies, out.error, out.error_is_message = execute(now)
        out.ran += now[:len(out.replies)]
        out.last = now
        if out.error or out.waiting or not plan_next or not any(s["action"] in LOOKUPS for s in now):
            return out
        seen.append((now, out.replies))
        steps, confirmed = plan_next(seen), False
        if not steps or steps == now:
            return out
    return out


def execute(steps: list[dict]) -> tuple[list[str], str | None, bool]:
    """Run the steps in order, stopping at the first failure.
    Returns (one reply per step that ran, error or None, True if the error is a sentence meant for the user)."""
    replies = []
    for i, step in enumerate(steps):
        try:
            replies.append(run_action(step["action"], *step["args"]))
            if step["action"] == "open_app" and i + 1 < len(steps):
                wait_for_app(step["args"][0])  # let a just-opened app come to the front before typing into it
        except Failed as e:  # includes NeedsPermission: the message says what to do
            return replies, str(e), True
        except Exception as e:  # bad args from the model, etc.
            return replies, f"{step['action']}: {e}", False
    return replies, None, False


def confirmation(step: dict) -> str:
    args = step["args"] + [""] * 3
    return CONFIRM[step["action"]].format(*args)


def describe(steps: list[dict]) -> str:
    said = [SAYING.get(s["action"], s["action"].replace("_", " ") + (" {0}" if s["args"] else "")).format(*s["args"], "")
            for s in steps]
    return said[0] if len(said) == 1 else ", ".join(said[:-1]) + " and " + said[-1]
