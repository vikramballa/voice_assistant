"""What the assistant remembers between commands, in one object (instead of globals spread over modules).

    conversation_on   after "Hey Laya", commands need no wake word until "stop"
    pending           an unfinished command waiting for its missing part: "Open." -> "Open what?" -> "Safari",
                      or steps waiting for a yes: "Should I email Rahul ...?" -> "yes"
    history           the last few commands and their steps, for follow-ups ("what about Delhi?", "in Brave")
    browser           the browser in use: "open YouTube" goes there when the words don't name one
    working_app       the app opened by voice: work goes there until it is closed or something else is
                      opened (in Notes / TextEdit, plain sentences are written into it)
"""

import time
from dataclasses import dataclass, field

from assistant.actions import WEB_ACTIONS
from assistant.actions.web import browser_app
from assistant.language.catalog import BROWSER_KEYS, SITES

MEMORY_TURNS = 3      # how many earlier commands the LLM sees
MEMORY_SECONDS = 120  # ...and how old they may be
# Apps where a plain sentence is text to write, not a request ("I'm a developer" goes into the note).
WRITING_APPS = {"notes", "textedit", "pages", "microsoft word", "stickies", "obsidian"}
TYPED_CHARS = 400  # how much of what was typed is remembered (context for hearing and auto-correction)


@dataclass
class Session:
    conversation_on: bool = False
    pending_text: str | None = None
    pending_steps: list[dict] | None = None
    pending_until: float = 0.0
    history: list[tuple[float, str, list[dict]]] = field(default_factory=list)  # (time, text, steps)
    browser: str | None = None
    last: tuple[float, str, list[dict], bool] | None = None  # (time, text, steps, worked) of the last command
    working_app: str | None = None
    typed: str = ""  # the end of what was typed in the working app (the note so far)

    # --- an unfinished command -------------------------------------------------------------------

    def wait_for(self, text: str, seconds: float) -> None:
        """`text` is unfinished; what is said in the next `seconds` completes it."""
        self.pending_text, self.pending_steps, self.pending_until = text, None, time.time() + seconds

    def wait_for_yes(self, steps: list[dict], seconds: float) -> None:
        """These steps run only if the user says yes within `seconds`."""
        self.pending_text, self.pending_steps, self.pending_until = None, steps, time.time() + seconds

    def is_waiting(self) -> bool:
        return bool(self.pending_text or self.pending_steps) and time.time() < self.pending_until

    def take_steps_to_confirm(self) -> list[dict] | None:
        steps = self.pending_steps if self.is_waiting() else None
        self.pending_steps = None
        return steps

    def take_pending(self) -> str | None:
        """The unfinished command, if it is still waiting (and forget it either way)."""
        text = self.pending_text if self.is_waiting() else None
        self.pending_text = None
        return text

    # --- earlier commands ------------------------------------------------------------------------

    def remember(self, text: str, steps: list[dict]) -> None:
        self.history.append((time.time(), text, steps))
        del self.history[:-MEMORY_TURNS]

    def recent(self) -> list[tuple[str, list[dict]]]:
        """(text, steps) of the last few commands, for the LLM's follow-ups."""
        now = time.time()
        return [(said, did) for t0, said, did in self.history if now - t0 < MEMORY_SECONDS][-MEMORY_TURNS:]

    def redo_steps(self, browser: str) -> list[dict] | None:
        """"in Brave" right after a web command: the same web steps again, in that browser."""
        for t0, _, did in reversed(self.history):
            web = [s for s in did if s["action"] in WEB_ACTIONS]
            if web and time.time() - t0 < MEMORY_SECONDS:
                return [{"action": s["action"], "args": [s["args"][0], browser]} for s in web]
            # "open ChatGPT" opened the app, then "in Brave": its website, in Brave
            sites = [SITES[s["args"][0].lower()] for s in did
                     if s["action"] == "open_app" and s["args"] and s["args"][0].lower() in SITES]
            if sites and time.time() - t0 < MEMORY_SECONDS:
                return [{"action": "open_website", "args": [site, browser]} for site in sites]
        return None

    def note_command(self, text: str, steps: list[dict], worked: bool) -> None:
        """Every handled command, also failed / not understood ones (history only keeps those that worked)."""
        self.last = (time.time(), text, steps, worked)

    def last_command(self, seconds: float) -> tuple[str, list[dict], bool] | None:
        """(text, steps, worked) of the command handled within the last `seconds`, else None."""
        if self.last and time.time() - self.last[0] < seconds:
            return self.last[1:]
        return None

    # --- the app being worked in ------------------------------------------------------------------

    def note_working_app(self, steps: list[dict]) -> None:
        """Opening an app (or a new note) makes it the working app; closing it ends it; the web makes the
        browser the working app. Answers, sums, keys and typing keep it."""
        for step in steps:
            action, args = step["action"], step["args"]
            if action in ("open_app", "write_note"):
                self.working_app = args[0] if action == "open_app" and args else "Notes"
                self.typed = args[0] if action == "write_note" and args else ""
            elif action == "close_app" and args and self.working_app and args[0].lower() == self.working_app.lower():
                self.working_app, self.typed = None, ""
            elif action in WEB_ACTIONS or action == "switch_tab":
                self.working_app, self.typed = browser_app(args[1] if len(args) > 1 else self.browser or ""), ""
            elif action == "type_text" and args:
                self.typed = (self.typed + args[0])[-TYPED_CHARS:]
            elif action == "clear_text":
                self.typed = ""

    def writing_app(self) -> str | None:
        """The working app when it is one to write in (Notes, TextEdit...), else None."""
        return self.working_app if self.working_app and self.working_app.lower() in WRITING_APPS else None

    def browser_in_use(self) -> str | None:
        """"brave" when the working app is a browser, else None."""
        return BROWSER_KEYS.get((self.working_app or "").lower())

    # --- the browser in use ----------------------------------------------------------------------

    def note_browser(self, steps: list[dict]) -> None:
        """The browser just opened, used or closed decides where the next "open youtube" goes."""
        for step in steps:
            action, args = step["action"], step["args"]
            if action in ("open_app", "close_app") and args and args[0].lower() in BROWSER_KEYS:
                name = BROWSER_KEYS[args[0].lower()]
                if action == "open_app":
                    self.browser = name
                elif self.browser == name:
                    self.browser = None
            elif (action in WEB_ACTIONS or action == "switch_tab") and len(args) > 1:
                self.browser = args[1]
