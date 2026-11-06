"""The details of one clause: what to search for, which volume, which kind of search or media key.

args_for() is the entry point: given the intent Laya picked, it returns the argument list for that
action, or None when the rules can't find what the action needs (then the LLM plans the command).
"""

import re

from . import catalog
from .spoken_math import words_to_numbers
from .words import BROWSER_NAMES as BROWSER_WORDS, POLITE, VERBS, normalize

FILLER_WORDS = r"in|on|at|for|about|of|to|with|and|the|a|an|from"


def args_for(intent: str, clause: str, sentence: str | None = None, browser: str | None = None) -> list[str] | None:
    """Argument list for the action `intent`, or None if the rules aren't sure.
    `sentence` is the whole command (it may name the browser when the clause doesn't);
    `browser` is the one in use, for when neither names one."""
    b = catalog.pick_browser(clause, sentence, browser)
    if intent in ("web_search", "youtube_search"):
        q = search_query(clause)
        return [q, b] if q else None
    if intent == "open_website":
        url = catalog.website(clause)
        return [url, b] if url else None
    if intent in ("open_app", "close_app"):
        name = catalog.app_name(clause)
        return [name] if name else None
    if intent == "open_folder":
        f = catalog.folder(clause)
        return [f] if f else None
    if intent == "set_volume":
        n = number(clause)
        return [n] if n else None
    if intent == "type_text":
        s = typed_text(clause)
        return [s] if s else None
    if intent == "tell_time":
        return [time_kind(clause)]
    return []  # actions without arguments


def search_query(text: str) -> str | None:
    """"search youtube for a video about black holes" -> "black holes"; None if nothing to search yet."""
    t = catalog.BROWSER_IN.sub("", normalize(text))
    t = re.sub(r"\s+(?:on|in|using)\s+(?:youtube|google)$", "", t)
    m = re.search(r"\b(?:search(?:\s+(?:on\s+)?(?:youtube|google|the\s+web|the\s+internet))?\s+for"
                  r"|search\s+(?:on\s+)?(?:youtube|google)"
                  r"|(?:youtube|google)(?:\s+search)?(?:\s+for)?|look\s+up|search|find|play|watch|show\s+me)\s+(.+)$", t)
    if not m:
        return None
    q = m.group(1)
    # "a video about black holes" -> "black holes"
    q = re.sub(r"^(?:me\s+)?(?:a\s+|some\s+|the\s+)?(?:youtube\s+)?(?:videos?|clips?|tutorials?)\s+(?:about|on|of|for)\s+", "", q)
    q = re.sub(r"^(?:for|about)\s+", "", q)
    # Speech cut off mid-sentence: "neural networks in..." -> "neural networks"
    q = re.sub(rf"(?:\s+(?:{FILLER_WORDS}))+$", "", q).strip(" .'\"")
    if re.fullmatch(rf"(?:{FILLER_WORDS}|me|something|it)?", q):
        return None  # "search youtube for..." - nothing to search yet
    return q or None


def search_kind(text: str) -> str:
    """The words say which search: YouTube / videos -> youtube_search, anything else -> web_search."""
    if re.search(r"\b(?:youtube|videos?|watch|play|clips?)\b", normalize(text)):
        return "youtube_search"
    return "web_search"


def play_title(text: str) -> str | None:
    """"play arijit singh songs" -> "arijit singh songs"; "play the music" -> None (that's play/pause)."""
    m = re.match(rf"^{POLITE}(?:play|put\s+on)\s+(.+)$", normalize(text))
    if not m or re.fullmatch(r"(?:the\s+|my\s+|some\s+)?(?:music|songs?|tracks?|it|again|it\s+again|playback|something|"
                             r"next\s+\w+|previous\s+\w+|last\s+\w+)", m.group(1)):
        return None
    return search_query(text)


def media_kind(text: str) -> str | None:
    """"skip this" -> next_track, "go back a song" -> previous_track, "resume" -> media_play_pause."""
    t = normalize(text)
    if re.search(r"\b(?:days?|weeks?|months?|years?|hours?|minutes?|results?|pages?|accounts?)\b", t):
        return None  # "give me the last 5 days" is a chart range, not a song
    if re.search(r"\b(?:next|skip)\b", t):
        return "next_track"
    if re.search(r"\b(?:previous|back|before)\b|\blast\s+(?:song|track|tune|one)\b", t):
        return "previous_track"
    if re.search(r"\b(?:resume|pause|unpause|play)\b", t) or \
            re.search(r"\b(?:stop|continue)\b", t) and re.search(MUSIC_WORDS, t):
        return "media_play_pause"  # a bare "stop" in a sentence is not about music ("I made it. Stop.")
    return None


MUSIC_WORDS = r"\b(?:music|songs?|tracks?|tunes?|playlist|playing|spotify|album|podcast|audio|player)\b"


def play_or_pause(text: str) -> str:
    """For media_play_pause: "pause" / "stop the music" -> "pause" (never starts music), "play" / "resume" ->
    "play", else "" (toggle)."""
    t = normalize(text)
    if re.search(r"\b(?:pause|stop|hold|quiet)\b", t):
        return "pause"
    if re.search(r"\b(?:play|resume|unpause|continue|start)\b", t):
        return "play"
    return ""


def number(text: str) -> str | None:
    """A volume level, 0-100: digits anywhere ("volume 30"), or words after "to" / "at" / "volume"
    ("set the volume to fifty"). Words elsewhere don't count: "turn it up one more time" is not 1.
    A step ("up by 10") is not a level: see volume_step."""
    if volume_step(text):
        return None
    m = re.search(r"\b(\d{1,3})\b", text) or \
        re.search(r"\b(?:to|at|volume)\s+(\d{1,3})\b", words_to_numbers(normalize(text)))
    return m.group(1) if m and int(m.group(1)) <= 100 else None


def volume_step(text: str) -> str | None:
    """"turn the volume up by 20" -> "20" (a change, not a level)."""
    m = re.search(r"\b(?:up|down|louder|quieter|softer|increase|decrease|raise|lower|reduce)\b.*\bby\s+(\d{1,3})\b"
                  r"(?:\s*(?:%|percent))?$", words_to_numbers(normalize(text)))
    return m.group(1) if m and 0 < int(m.group(1)) <= 100 else None


def volume_direction(text: str) -> str | None:
    """"turn it up" -> volume_up, "too loud" -> volume_down, "mute the volume" -> mute, else None."""
    t = normalize(text)
    if re.search(r"\bunmute\b|\bturn\s+(?:the\s+)?(?:sound|audio)\s+back\s+on\b", t):
        return "unmute"
    if re.search(r"\b(?:mute|silence)\b", t):
        return "mute"
    if re.search(r"\b(?:down|quieter|softer|lower|decrease|reduce|less|too\s+loud)\b", t):
        return "volume_down"
    if re.search(r"\b(?:up|louder|increase|raise|higher|more|can't\s+hear)\b", t):
        return "volume_up"
    return None


def typed_text(text: str) -> str | None:
    """"type hello world" -> "hello world", "type a message saying hi" -> "hi" (original spelling kept)."""
    m = re.search(r"\b(?:type|write|enter)\s+(?:(?:in|into|on)\s+(?:the|this|my)\s+(?:new\s+)?(?:note|notes|document|box)\W*\s+"
                  r"|out\s+|down\s+|in\s+|the\s+(?:text|words)\s+)?"
                  r"(?:(?:a|the|this|my)\s+(?:message|text|reply)\s*(?:saying|that\s+says|which\s+says|to\s+say|:)?\s+)?(.+)$",
                  text.strip(), re.IGNORECASE)
    return m.group(1).strip() if m else None


NOTE_PATTERNS = [
    r"(?:write|make|create|add|take|start)\s+(?:a\s+|me\s+a\s+)?(?:new\s+|quick\s+)?note\s*(?:saying|that\s+says|which\s+says"
    r"|to\s+say|reading|with|about|:)\s*(?P<text>.+)",
    r"(?:write|type|add|put|save|jot)\s+(?:down\s+)?(?P<text>.+?)\s+(?:(?:in|into|to|on)\s+(?:my\s+|the\s+)?"
    r"(?:notes|notes\s+app|note\s+app)|(?:in|into|on)\s+a\s+(?:new\s+)?note)",
    r"(?:in|on|to)\s+(?:the\s+|my\s+)?(?:notes?|notes\s+app)\s*,?\s+(?:write|type|add|put|jot)\s+(?:down\s+)?(?P<text>.+)",
    r"(?:note|jot)\s+down\s+(?P<text>.+)",
    r"(?:create|make|start|open|add|take|write)\s+(?:a\s+)?(?:new\s+)?note(?:\s+in\s+(?:the\s+)?notes(?:\s+app)?)?",
]


# While writing in a note: "continue, I'm a developer" / "also write X" -> the words to add.
DICTATION = re.compile(rf"^(?:(?:ok|okay|so|now|right)\W+)*{POLITE}(?:continue(?:\s+writing|\s+typing)?|keep\s+(?:writing|typing)|also\s+(?:write|type))"
                       r"\W+(?P<text>\w.*)$", re.IGNORECASE)


# Spoken punctuation while dictating: "hi comma how are you question mark" -> "hi, how are you?"
NEW_LINE = re.compile(r"^\W*(?:(?:new|next)\s+(?:line|paragraph)|(?:go\s+to\s+(?:the\s+)?)?next\s+line|line\s+break|enter)\W*$",
                      re.IGNORECASE)
SPOKEN_MARKS = [(r"\s*,?\s*\b(?:new|next)\s+paragraph\b[.,]?\s*", "\n\n"), (r"\s*,?\s*\b(?:new|next)\s+line\b[.,]?\s*", "\n"),
                (r"\s*,?\s*\bcomma\b[.,]?", ","), (r"\s*,?\s*\bfull\s+stop\b[.,]?", "."), (r"\s*,?\s*\bperiod[.,]?$", "."),
                (r"\s*,?\s*\bquestion\s+mark\b[.,]?", "?"), (r"\s*,?\s*\bexclamation\s+(?:mark|point)\b[.,]?", "!"),
                (r"\s*,?\s*\bcolon\b[.,]?", ":")]


def spoken_punctuation(text: str) -> str:
    """"Hi comma how are you question mark." -> "Hi, how are you?"; "new line" alone -> "\n"."""
    if NEW_LINE.match(text):
        return "\n"
    out = text
    for pattern, mark in SPOKEN_MARKS:
        out = re.sub(pattern, mark, out, flags=re.IGNORECASE)
    if out != text:
        out = re.sub(r"([,.?!:])[.,]+(?=\s|$)", r"\1", out)  # "how are you question mark." -> "?" not "?."
        out = re.sub(r"(^|[.?!]\s+|\n)([a-z])", lambda m: m.group(1) + m.group(2).upper(), out)  # capital after a full stop
    return out.strip(" ")


def dictation(text: str) -> str | None:
    """"Continue. I'm a developer." -> "I'm a developer." (original spelling kept); None otherwise."""
    m = DICTATION.match(text.strip())
    if not m or note_text(text) is not None or re.match(rf"(?:{VERBS})\b", m.group("text"), re.IGNORECASE) \
            or re.fullmatch(rf"(?:(?:the|my|this)\s+)?(?:video|{MUSIC_WORDS})\W*", m.group("text"), re.IGNORECASE):
        return None  # "write a note saying X" is a new note, "continue, open safari" a command, "continue the music"
    return m.group("text").strip()


# Mail, messages and the calendar: always planned by the LLM (they may need a look first, and a yes to send).
PERSONAL = re.compile(rf"^{POLITE}(?:(?:send|draft|reply)\b.*\b(?:e-?mail|mail|message|text|imessage|to)\b"
                      r"|(?:text|message|e-?mail)\s+\w+)|\b(?:e-?mails?|inbox|calendar|meetings?|schedule|agenda|appointments?)\b")


def personal_task(text: str) -> bool:
    """"email Rahul that I'm late", "what's on my calendar" (not "open mail": that's just the app)."""
    t = normalize(text)
    return bool(PERSONAL.search(t)) and not re.match(rf"{POLITE}(?:open|close|quit|launch|start)\b", t)


def note_text(text: str) -> str | None:
    """A new note in Apple Notes: "write a note saying buy milk" / "write buy milk in notes" -> "buy milk",
    "create a new note" -> "" (an empty one); None when it isn't about a note."""
    t = re.sub(r"[.!?]+$", "", text.strip())
    for pattern in NOTE_PATTERNS:
        m = re.fullmatch(rf"{POLITE}{pattern}(?:\s+(?:please|for\s+me))?", t, re.IGNORECASE)
        if m:
            return (m.groupdict().get("text") or "").strip()
    return None


# AI chat sites that take the question in the address: it is asked the moment the page opens.
ASK_URLS = {"chatgpt": "https://chatgpt.com/?q={}", "perplexity": "https://www.perplexity.ai/search?q={}",
            "claude": "https://claude.ai/new?q={}"}
ASK = re.compile(rf"^{POLITE}(?:(?:ask|tell)\s+(?:the\s+)?(?P<a>chat\s*gpt|perplexity|claude)\s+(?:to\s+|about\s+|whether\s+|if\s+)?"
                 r"(?P<q1>.+?)|(?:ask|search(?:\s+for)?)\s+(?P<q2>.+?)\s+(?:in|on|with|using)\s+(?P<b>chat\s*gpt|perplexity|claude)"
                 rf"|(?:in|on)\s+(?P<c>chat\s*gpt|perplexity|claude)\s*,?\s+(?:ask|search(?:\s+for)?|type)\s+(?P<q3>.+?))"
                 rf"(?:\s+(?:in|on|using)\s+(?:the\s+)?(?:{BROWSER_WORDS})(?:\s+browser)?)?$", re.I)


def ask_ai(text: str) -> tuple[str, str] | None:
    """"ask chatgpt what is a neural network" -> ("chatgpt", "what is a neural network"); None otherwise."""
    m = ASK.match(re.sub(r"[.!]+$", "", text.strip()))
    if not m:
        return None
    site = re.sub(r"\s+", "", (m.group("a") or m.group("b") or m.group("c")).lower())
    question = (m.group("q1") or m.group("q2") or m.group("q3")).strip(" ,")
    return (site, question) if question else None


def ask_url(site: str, question: str) -> str:
    from urllib.parse import quote
    return ASK_URLS[site].format(quote(question))


def time_kind(text: str) -> str:
    return "date" if re.search(r"\b(?:date|day|today)\b", normalize(text)) else "time"
