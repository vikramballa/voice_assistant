"""Check the LLM's steps against the user's actual words before anything runs.

A 4B model makes predictable mistakes: made-up app names, placeholder arguments ("query"), steps
nobody asked for (a screenshot after "message mom"), a search written as a results link, numbers the
user never said. repair() fixes what it can and drops the rest. The rule of thumb: the user's words
win over the model (which browser, which search, which app).
"""

import inspect
import re
from urllib.parse import parse_qs, urlparse

from assistant.actions import ACTIONS, WEB_ACTIONS, calculator
from assistant.language import arguments, catalog, in_app
from assistant.language.sentence import clean, split_commands
from assistant.language.spoken_math import math_expression, words_to_numbers
from assistant.language.words import normalize

# An action the LLM adds must match something the user actually said
# ("send a message to mom" must not take a screenshot; "dark mode" must not touch the volume).
GUARDS = {
    "screenshot": r"screen|capture|snap|shot",
    "lock_screen": r"lock|secure",
    "type_text": r"type|write|enter|dictate|fill|put",
    "mute": r"mute|silen|sound|quiet|audio|shut",
    "set_volume": r"volume|sound|loud|quiet|\d",
    "volume_up": r"volume|sound|loud|up|hear",
    "volume_down": r"volume|sound|loud|quiet|down|soft",
    "close_app": r"close|quit|exit|kill|shut",
    "dark_mode": r"dark|light|night|theme|appearance",
    "tell_time": r"time|date|day|clock|today|o'clock",
    "battery_status": r"batter|charg|power|juice",
    "set_timer": r"timer|countdown|minute|second|hour",
    "add_reminder": r"remind|reminder|alarm|wake|don't let me forget|remember",
    "click": r"click|tap|press|hit|select|choose|button|link|check|tick|menu",
    "press_keys": r"press|hit|key|enter|return|escape|shortcut|tab|back|forward|refresh|reload|undo|redo|copy|paste|"
                  r"cut|select|save|zoom|find|delete|space|arrow|up|down|left|right|clear|new|close|next|previous|quit",
    "scroll": r"scroll|page|down|up|top|bottom",
    "web_search": r"search|google|look\s*up|find|browse|web|internet|online|result|brave|chrome|safari|bing|"
                  r"play|watch|video|youtube|song|music",
    "look_at_screen": r"screen|see|look|read|show|display|window|page|this|that|error|say|summar|describ",
    "do_on_screen": r"screen|page|site|tab|window|app|form|fill|for me|take over|find|sign|log ?in|button|menu|setting",
    "look_through_camera": r"camera|webcam|holding|wearing|hand|in front|see me|how do i look|look at me",
    "watch_screen": r"when|watch|until|wait for|eye on|notify|alert|let me know",
    "calculate": r"\d|calculat|comput|math|plus|minus|times|into|multipl|divid|add|subtract|percent|squared|sum|total|"
                 r"twice|double|half|one|two|three|four|five|six|seven|eight|nine|ten|twenty|hundred|thousand",
    "weather": r"weather|rain|temperature|hot|cold|sunny|forecast|degrees|umbrella|humid",
    # music only when music was asked for (2026-10-03: a misheard "remove that" played a song)
    "media_play_pause": r"play|pause|resume|stop|music|song|track|continue|unpause|spotify|tune|hold",
    "next_track": r"next|skip|song|track|music|another|change",
    "previous_track": r"previous|back|last|song|track|before|again|replay",
    "read_mail": r"mail|inbox|repl|message",
    "send_email": r"mail|repl|write|send",
    "send_message": r"message|text|imessage|tell|send|whatsapp",
    "calendar_events": r"calendar|meeting|schedule|event|agenda|busy|free|plan|appointment",
    "add_event": r"calendar|meeting|event|schedule|book|appointment",
}
# The page a search opens anyway: dropped when the plan also searches.
SEARCH_HOMES = re.compile(r"^(?:https?://)?(?:www\.|search\.)?(?:google|bing|duckduckgo|brave|youtube)\.(?:com|in)/?$")
# Arguments a model writes when it has nothing real to put there.
PLACEHOLDERS = {"", "query", "name", "x", "text", "search", "url", "app", "none", "null", "unknown"}
OPEN_ACTIONS = ("open_app", "open_website", "open_folder")


def repair(steps: list[dict], text: str, browser: str | None = None) -> list[dict]:
    """Check the LLM's steps against reality; fix what can be fixed, drop the rest.
    `browser` is the one in use, for when the words don't name one."""
    fixed = []
    text = clean(text)
    browser = catalog.pick_browser(text, default=browser)  # the user's words pick the browser, not the model
    clauses = split_commands(text)
    # "open X" where the words name a real app / site / folder exactly: the words win over the model.
    found = catalog.open_candidates(clauses[0], fuzzy=1) \
        if len(clauses) == 1 and catalog.is_open_command(clauses[0]) else {}
    named = next(((a, found[a]) for a in OPEN_ACTIONS if a in found), None)
    missing = catalog.missing_app(clauses[0]) if len(clauses) == 1 else None
    for step in steps:
        action = step.get("action")
        # a curly quote the model used to "close" the text, and the JSON it typed after it
        args = [re.sub(r"\s*[”“]+[\]}\s\"]*$", "", str(a)).strip() for a in step.get("args") or []]
        if action not in ACTIONS or (action in GUARDS and not re.search(GUARDS[action], text, re.IGNORECASE)):
            continue
        if args and not re.search(r"\w", args[0]) and _arity(action)[0]:
            continue  # "," as an app name: the model lost its way
        if action == "mute" and arguments.volume_direction(text) == "unmute":
            action, args = "unmute", []
        if action == "media_play_pause" and arguments.play_title(text):
            action, args = "youtube_search", [arguments.play_title(text)]
        elif action == "media_play_pause":  # the words decide: "stop" only ever pauses
            args = [arguments.play_or_pause(text)] if arguments.play_or_pause(text) else []
        if action == "open_website":
            args.sort(key=lambda a: not _looks_like_url(a))  # the url first, whatever order the model used
            if args and _search_query(args[0]):  # a search written as a results link
                action, args = "web_search", [_search_query(args[0])]

        if action in OPEN_ACTIONS and missing:
            action, args = "answer", [f"{missing} isn't installed on this Mac"]
        elif action in OPEN_ACTIONS and named:
            action, args = named
            args = args + [browser] if action == "open_website" else args
        elif action == "close_app" and args:
            target = catalog.resolve_name(args[0], fuzzy=0.8)
            if target and target[0] == "open_app":
                args = target[1]
        elif action == "open_app":
            if not args or args[0].lower() in PLACEHOLDERS:
                continue
            target = catalog.resolve_name(args[0])
            if target:  # misheard app -> real app; a web service -> its website
                action, args = target
                if action == "open_website":
                    args = args + [browser]
        elif action in ("web_search", "youtube_search"):
            action = arguments.search_kind(text)  # the user's words pick Google or YouTube
            if not args or args[0].lower() in PLACEHOLDERS:
                if action == "youtube_search":
                    action, args = "open_website", ["youtube.com", browser]
                else:
                    continue
        elif action == "open_website":
            if not args or args[0].lower() in PLACEHOLDERS:
                continue
            args[0] = re.sub(r"^https?://", "", args[0]).rstrip("/")
            if "." not in args[0]:  # "gemini" -> gemini.google.com
                target = catalog.resolve_name(args[0])
                if not target or target[0] != "open_website":
                    continue
                args = target[1] + args[1:]
        elif action == "set_volume":
            if not args or not re.fullmatch(r"\d{1,3}", args[0]):
                continue
        elif action == "answer" and (not args or not args[0]):
            continue
        elif action == "calculate":
            expression = _real_arithmetic(args[0]) if args else None
            if not expression:
                continue
            expression = _said_numbers(expression, text)
            if expression is None:  # numbers the user never said (taken from earlier commands)
                action, args = "answer", ["What should I calculate?"]
            else:
                args = [expression]
        elif action in ("click", "press_keys") and (not args or args[0].lower() in PLACEHOLDERS):
            continue
        elif action == "press_keys":
            args = args[:1] + ([browser] if in_app.is_browser_command(text) else [])
        elif action in ("send_email", "send_message") and not (args[-1:] and args[-1].strip()):
            action, args = "answer", ["What should the message say?"]  # never send an empty message

        if action in WEB_ACTIONS:
            args = [args[0] if action == "open_website" else args[0].lower(), browser]
        required, total = _arity(action)
        step = {"action": action, "args": args[:total]}
        if len(args) < required or fixed[-1:] == [step]:
            continue  # missing arguments, or the same step twice in a row
        fixed.append(step)

    # "open brave and search X": the search opens the browser (and the search engine) already
    if any(s["action"] in ("web_search", "youtube_search") for s in fixed):
        fixed = [s for s in fixed if not (s["action"] == "open_website" and SEARCH_HOMES.match(s["args"][0]))]
    used = {s["args"][1] for s in fixed if s["action"] in WEB_ACTIONS}
    return [s for s in fixed if not (s["action"] == "open_app" and s["args"]
                                     and catalog.BROWSER_KEYS.get(s["args"][0].lower()) in used)]


def _arity(action: str) -> tuple[int, int]:
    """(required, total) number of arguments of an action."""
    params = inspect.signature(ACTIONS[action]).parameters.values()
    return sum(p.default is p.empty for p in params), len(params)


def _looks_like_url(arg: str) -> bool:
    return bool(re.match(r"(https?://)?[\w-]+(\.[\w-]+)+([/?#]|$)", arg))


def _search_query(url: str) -> str:
    """The query of a search-results link ("google.com/search?q=python+jobs" -> "python jobs"), else ""."""
    parsed = urlparse(url if "://" in url else "https://" + url)
    params = parse_qs(parsed.query)
    if re.search(r"(^|\.)(google|bing|duckduckgo)\.", parsed.netloc) and params.get("q"):
        return params["q"][0].strip()
    if "youtube." in parsed.netloc and parsed.path.startswith("/results") and params.get("search_query"):
        return params["search_query"][0].strip()
    return ""


def _real_arithmetic(arg: str) -> str | None:
    """The model's calculate argument as an expression, if it is real arithmetic ("25 into 25", "3 * (4+1)")."""
    expression = math_expression(arg)
    if not expression and re.fullmatch(r"[\d.+\-*/^%() ]+", arg) and re.search(r"\d\s*[-+*/^%]\s*\(?\d", arg):
        expression = arg.replace(" ", "")
    return expression


def _said_numbers(expression: str, text: str) -> str | None:
    """The expression if its numbers were said; "625*4" after 625 was the last answer -> "*4"; else None."""
    said = set(re.findall(r"\d+(?:\.\d+)?", words_to_numbers(normalize(text))))
    if all(n in said for n in re.findall(r"\d+(?:\.\d+)?", expression)):
        return expression
    last = calculator.last_result()
    m = re.match(r"(\d+(?:\.\d+)?)([-+*/^].*)$", expression)
    if m and last is not None and float(m.group(1)) == last and all(n in said for n in re.findall(r"\d+(?:\.\d+)?", m.group(2))):
        return m.group(2)
    return None
