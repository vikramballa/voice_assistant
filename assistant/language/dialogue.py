"""The conversation around commands: is the sentence finished, what to ask back, was it meant for us at all.

    follow_up_question  "Open." -> "Open what?"
    trails_off          "set the volume to..." -> the rest is probably still coming
    complete            ("open", "safari") -> "open safari"
    is_stop / is_noise  end conversation mode / skip Whisper's noise words and room talk
    is_echo / without_echo   the mic heard our own reply
    redo_in_browser     "in Brave" right after a web command -> do it again there
    correction          "no, I meant open notes" -> the user is fixing what we just did
    dictated            in Notes: "I'm a developer" is text to write, "next song" is still a command
    yes_or_no           the answer to "Should I send it?"
    routine             "when I say start my day, open Slack and Mail" -> a routine to remember
"""

import difflib
import re

from . import arguments, catalog
from .in_app import (clear_request, click_target, play_pause_request, screen_question, scroll_args, shortcut,
                     vision_request)
from .sentence import fix_misheard, strip_wake
from .spoken_math import math_expression, words_to_numbers
from .words import BROWSER_NAMES, POLITE, VERBS, normalize

# A command that is only a verb: "open", "please search" -> ask what, instead of guessing.
INCOMPLETE = [
    (re.compile(rf"^{POLITE}(?:open|launch|start|run|open\s+up|go\s+to|visit)(?:\s+the|\s+my|\s+a)?$"), "Open what?"),
    (re.compile(rf"^{POLITE}(?:close|quit|exit|kill)(?:\s+the|\s+my)?$"), "Close which app?"),
    (re.compile(rf"^{POLITE}(?:search|google|look\s+up|find|search\s+for|search\s+youtube(?:\s+for)?"
                rf"|search\s+google(?:\s+for)?|search\s+on\s+youtube)$"), "Search for what?"),
    (re.compile(rf"^{POLITE}(?:type|write|type\s+out|dictate|take\s+dictation|(?:type|write)\s+(?:down\s+)?(?:what(?:ever)?|everything)"
                rf"\s+i\s+(?:say|tell\s+you)|(?:type|write)\s+(?:a\s+)?(?:message|text)"
                rf"|(?:type|write)\s+(?:in|into|on)\s+(?:the|this|my)\s+(?:new\s+)?notes?)$"), "Type what?"),
    (re.compile(rf"^{POLITE}(?:write|make|take|add)\s+(?:a\s+)?(?:new\s+)?note(?:\s+in\s+notes)?\s+(?:saying|that\s+says|about)$"),
     "What should the note say?"),
    (re.compile(rf"^{POLITE}(?:set\s+(?:the\s+)?volume(?:\s+to)?|volume\s+to)$"), "What volume, from 0 to 100?"),
    (re.compile(rf"^{POLITE}(?:send|write)\s+(?:an?\s+)?(?:message|text|imessage|e-?mail|mail)\s+to\s+(?:my\s+)?[\w.@]+$"
                rf"|^{POLITE}(?:text|message|e-?mail)\s+(?:my\s+)?[\w.@]+$"), "What should the message say?"),
]
# A command that ends like this is not finished: "search youtube for", "set the volume to", "open".
UNFINISHED = re.compile(r"\b(?:and|then|or|but|to|for|about|of|in|on|at|with|the|a|an|my|your|this|that|from|into|"
                        r"open|close|quit|launch|search|google|type|write|set|go|find|look|play\s+some|"
                        r"volume\s+to|like|um+|uh+|so)$")

STOP = re.compile(r"^(?:(?:ok(?:ay)?|thanks|thank\s+you|alright|hey\s+laya|laya)\s+)*"
                  r"(?:(?:you\s+can\s+|please\s+)?stop(?:\s+listening)?|that's\s+(?:all|it)|that\s+is\s+all|"
                  r"go\s+to\s+sleep|sleep|good\s*bye|bye(?:\s+bye)?|see\s+you|exit|stop\s+it|we're\s+done|i'm\s+done|"
                  r"done|never\s*mind|dismiss)(?:\s+(?:now|laya|thanks|thank\s+you|for\s+now|please))*$")
# What Whisper writes for noise / silence; never a command.
HALLUCINATIONS = re.compile(r"^(?:you|thank\s+you|thanks\s+for\s+watching|thank\s+you\s+for\s+watching|"
                            r"please\s+subscribe|bye|so|um+|uh+|hmm+|oh|ah|okay|ok|yeah|yes|no|i|the)?$")
# One word said without the wake word is only taken as a command if it's one of these.
ONE_WORD_COMMANDS = {"pause", "play", "resume", "unpause", "next", "skip", "previous", "back", "mute", "unmute", "louder",
                     "quieter", "screenshot", "copy", "paste", "undo", "redo", "refresh", "reload", "save", "enter"}
FILLERS = {"okay", "ok", "now", "so", "um", "umm", "uh", "well", "right", "alright", "and", "then", "yeah", "yes",
           "hmm", "like", "the", "a", "oh", "ah", "wait", "hey", "hi", "hello", "just", "actually", "also"}

# At least this share of the heard words is in our reply: it is our own voice coming back.
ECHO_SHARE = 0.6

REDO_IN = re.compile(rf"^(?:(?:no|not\s+(?:in\s+)?(?:{BROWSER_NAMES}))\s+)*{POLITE}"
                     rf"(?:(?:open|do|search|play|show)\s+(?:it|that|this)(?:\s+again)?\s+)?"
                     rf"(?:(?:in|on|using|with|use)\s+)(?:the\s+)?(?:google\s+)?({BROWSER_NAMES})(?:\s+browser)?"
                     rf"(?:\s+(?:please|instead|now))*$")


# --- is the command finished? ---------------------------------------------------------------------

LEADING_FILLER = re.compile(r"^(?:(?:ok|okay|alright|so|now|well|done|um+|uh+)\s+)+")


def follow_up_question(text: str) -> str | None:
    """"Open" alone -> "Open what?". None when the command is complete."""
    t = LEADING_FILLER.sub("", normalize(fix_misheard(text)))  # "OK, right. What I see?" -> "write what i say"
    for pattern, question in INCOMPLETE:
        if pattern.match(t):
            return question
    return None


def trails_off(command: str) -> bool:
    """"set the volume to..." - the rest is probably coming after a pause."""
    t = normalize(command)
    if catalog.is_open_command(t) and catalog.open_candidates(t, fuzzy=1):
        return False  # "open google": google is the thing to open, not the start of "google ..."
    if shortcut(t) or scroll_args(t) or math_expression(t) or screen_question(t) or clear_request(t) \
            or play_pause_request(t) or vision_request(t):
        return False  # "zoom in", "scroll up a bit", "remove that": whole commands that end in "in" / "a" / "that"
    return bool(t) and bool(UNFINISHED.search(t)) and follow_up_question(t) is None


NAMES_A_BROWSER = re.compile(rf"\b(?:in|on|using|with)\s+(?:the\s+)?(?:google\s+)?(?:{BROWSER_NAMES})\b")
OPEN_ENDED = re.compile(rf"^{POLITE}(?:search|google|look\s+up|find|play|watch|youtube)\b")


def may_go_on(text: str) -> bool:
    """At a short pause: could the next words still belong to this command? Then listening waits longer
    instead of acting on half a sentence ("open ChatGPT" ... "in Brave browser" opened the ChatGPT app).
    True for: half sentences, "open X" when X is also a website, searches / "play X" (a browser or
    "on YouTube" may follow) - unless a browser is already named."""
    command = strip_wake(text) or text
    t = normalize(command)
    if not t or NAMES_A_BROWSER.search(t):
        return False
    if follow_up_question(t) or trails_off(t):
        return True
    if OPEN_ENDED.match(t):  # but "play the next song" is whole
        return not re.search(r"\b(?:next|previous|last|song|music|track|tune)\b|^(?:\w+\s+)?(?:play|pause|resume)$", t)
    return catalog.is_open_command(t) and "open_website" in catalog.open_candidates(t, fuzzy=1)


def looks_complete(text: str, need_wake: bool = True) -> bool:
    """The words form a whole command: (wake word +) something that doesn't trail off."""
    command = strip_wake(text) if need_wake else text
    if not command:
        return False  # no wake word, or just "Hey Laya": wait normally
    t = normalize(command)
    return bool(t) and not UNFINISHED.search(t) and follow_up_question(t) is None


def complete(pending: str, answer: str) -> str:
    """Join an incomplete command with the answer to our question: ("open", "safari") -> "open safari"."""
    if re.match(rf"(?:{POLITE})(?:{VERBS})\b", normalize(answer)) or math_expression(answer):
        return answer  # the user said a whole new command instead
    question = follow_up_question(pending)
    if question == "Type what?":  # "type what I say" + "see you at 5" -> "type see you at 5"
        return f"type {answer}"
    if question == "What should the note say?":
        return f"write a note saying {answer}"
    if question == "What should the message say?":  # "text mom" + "I'm late" -> "text mom saying I'm late"
        return f"{pending.strip(' .')} saying {answer}"
    return f"{pending.strip(' .')} {answer}"


# --- writing in Notes ---------------------------------------------------------------------------

# Questions for Laya herself, never text to write: "what time is it", "set a timer", "Laya, ...".
ASKS_LAYA = re.compile(r"\blaya\b|^(?:what|can|could|would|will|do)\s+you\b|^(?:what|what's|whats|when|how|is|are|can|could|will|would|tell|set|start|remind|turn|"
                       r"check|give)\b.*\b(?:time|date|weather|temperature|rain(?:ing)?|battery|charging|charged|timer|"
                       r"remind(?:er)?|alarm|dark\s+mode|screen)\b")
SHORT_COMMAND_WORDS = 3  # "next song", "volume up", "louder please": short and the fast brain is sure -> a command


def sounds_like_command(text: str) -> bool:
    """Starts with a command word ("open...", "search...", "undo") or is an exact in-app command."""
    t = re.sub(rf"^{POLITE}", "", LEADING_FILLER.sub("", normalize(text)))
    return bool(re.match(rf"(?:{VERBS})\b", t)) or bool(
        shortcut(t) or scroll_args(t) or click_target(t) or clear_request(t) or play_pause_request(t)
        or math_expression(t) or screen_question(t) or vision_request(t))


def asks_laya(text: str) -> bool:
    """"what time is it", "set a timer", "can you...", "Laya, ...": meant for Laya even while writing."""
    return bool(ASKS_LAYA.search(LEADING_FILLER.sub("", normalize(text))))


def as_dictation(text: str) -> str:
    return arguments.spoken_punctuation(text.strip())


def dictated(text: str, fast_sure: bool) -> str | None:
    """In a writing app (Notes): the words to write, or None when it is a command / a question for Laya.
    "continue, I'm a developer" -> "I'm a developer"; "I'm a developer" -> itself; "next song" (the fast
    brain is sure, short) -> None; "what time is it" -> None."""
    if arguments.NEW_LINE.match(text):
        return "\n"  # "next line" in a note is the Return key, not the next song
    added = arguments.dictation(text)
    if added:
        return arguments.spoken_punctuation(added)
    t = normalize(text)
    if not t or is_stop(t) or ASKS_LAYA.search(LEADING_FILLER.sub("", t)) or sounds_like_command(text):
        return None  # "Okay, quit notes." and "OK, what are you seeing?" were typed into the note
    if fast_sure and len(t.split()) <= SHORT_COMMAND_WORDS:
        return None
    return arguments.spoken_punctuation(text.strip())


# --- answers and routines ------------------------------------------------------------------------

YES = re.compile(r"^(?:yes|yeah|yep|yup|sure|ok(?:ay)?|do\s+it|go\s+ahead|send(?:\s+it)?|please\s+do|confirm(?:ed)?"
                 r"|correct|right|of\s+course|absolutely)\b")
NO = re.compile(r"^(?:no|nope|nah|don'?t|do\s+not|cancel|stop|never\s*mind|wait|hold\s+on|not\s+now)\b")
ROUTINE = re.compile(rf"^(?:when(?:ever)?|if)\s+i\s+say\s+(?P<name>.+?)\s*,?\s+(?:then\s+|you\s+should\s+|please\s+)?"
                     rf"(?P<steps>(?:{VERBS})\b.+)$"
                     r"|^(?:create|make|add|save)\s+(?:a\s+)?routine\s+(?:called|named)\s+(?P<name2>.+?)\s*,?\s+"
                     r"(?:that|which|to)\s+(?:does\s+|will\s+)?(?P<steps2>.+)$", re.IGNORECASE)


def yes_or_no(text: str) -> bool | None:
    t = LEADING_FILLER.sub("", normalize(text)) or normalize(text)
    return True if YES.match(t) else False if NO.match(t) else None


def routine(text: str) -> tuple[str, str] | None:
    """"When I say 'start my work day', open VS Code and Brave" -> ("start my work day", "open VS Code and Brave")."""
    m = ROUTINE.match(text.strip().rstrip("."))
    if not m:
        return None
    name, steps = (m.group("name") or m.group("name2")), (m.group("steps") or m.group("steps2"))
    return name.strip(" \"'“”,"), steps.strip()


# --- was it meant for us? -------------------------------------------------------------------------

def is_stop(text: str) -> bool:
    """"stop", "stop listening", "that's all", "bye": end conversation mode (not "stop the music")."""
    return bool(STOP.match(normalize(text)))


def is_noise(text: str) -> bool:
    """Whisper's noise words, or a single word that isn't a command ("6", "right")."""
    t = normalize(text)
    if t and set(t.split()) <= FILLERS:
        return True  # "okay, now...", "so um"
    return bool(HALLUCINATIONS.match(t)) or (len(t.split()) == 1 and t not in ONE_WORD_COMMANDS)


def is_echo(heard: str, reply: str) -> bool:
    """The mic probably heard our own reply: it reads almost the same, or most of its words are (nearly) words
    of the reply. "Nearly", because Whisper mishears our voice too: "I can help you with bear" for "... with
    various tasks" (2026-09-29: all-words-must-match let such pieces through and Laya answered itself in a loop)."""
    h, r = (words_to_numbers(normalize(t)) for t in (heard, reply))
    if not h:
        return True
    if difflib.SequenceMatcher(None, h, r).ratio() >= 0.75:
        return True
    reply_words = set(r.split())
    heard_words = h.split()
    matched = sum(w in reply_words or (len(w) >= 4 and bool(difflib.get_close_matches(w, reply_words, 1, 0.8)))
                  for w in heard_words)
    return matched / len(heard_words) >= ECHO_SHARE


def without_echo(heard: str, reply: str) -> str:
    """Remove our own spoken reply from the start of what the mic heard."""
    h, r = normalize(heard).split(), normalize(reply).split()
    i = 0
    while i < min(len(h), len(r)) and h[i] == r[i]:
        i += 1
    if i >= max(1, len(r) - 1):
        return " ".join(h[i:])
    # Only the end of our reply got in: "625 multiply that by 4" after "... is 625".
    tail = next((k for k in range(min(len(h) - 1, len(r)), 0, -1) if h[:k] == r[-k:]), 0)
    return " ".join(h[tail:]) if tail else heard


# --- the user correcting what we just did ---------------------------------------------------------

# "I meant X", "I said X", "not that, X": clearly a correction.
CORRECTION_STRONG = re.compile(r"^(?:(?:no|nope|wrong|sorry)\s+)*(?:i\s+(?:meant|said|asked(?:\s+for)?|wanted|want)"
                               r"|not\s+that(?:\s+one)?(?:\s+i\s+(?:meant|said))?)\s+(?:to\s+)?(.+)$")
# "no, X", "wrong, X": a correction, or a change of mind (app.py decides with the earlier command).
CORRECTION_WEAK = re.compile(r"^(?:no|nope|wrong|that's\s+wrong|that's\s+not\s+(?:it|right|what\s+i\s+(?:said|meant)))"
                             r"\s+(?:(?:no|wait|sorry|please)\s+)*(.+)$")


def correction(text: str) -> tuple[bool, str] | None:
    """"no, I meant open notes" -> (True, "open notes"); "no, open notes" -> (False, "open notes");
    None when it isn't a correction (or nothing follows the "no")."""
    t = normalize(text)
    for strong, pattern in ((True, CORRECTION_STRONG), (False, CORRECTION_WEAK)):
        m = pattern.match(t)
        if m and m.group(1).strip():
            return strong, m.group(1).strip()
    return None


def similar(a: str, b: str) -> float:
    """How alike two commands read (0-1): "open not" / "open notes" 0.89, "open spotify" / "close safari" 0.5."""
    return difflib.SequenceMatcher(None, normalize(a), normalize(b)).ratio()


# --- follow-ups about the browser -----------------------------------------------------------------

def redo_in_browser(text: str) -> str | None:
    """"in Brave", "no, open it in Brave", "use Safari instead" -> that browser (redo the last web step there)."""
    m = REDO_IN.match(normalize(fix_misheard(text)))
    return m.group(1) if m else None
