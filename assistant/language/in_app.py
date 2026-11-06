"""Commands inside the app in front: keyboard shortcuts, scrolling, clicking, and questions about the screen.

Laya's 17 intents don't cover these, so they are decided by exact rules here, or planned by the LLM.
"""

import re

from .sentence import clean
from .words import BROWSER_NAMES, POLITE, normalize

# Everyday keyboard commands, decided by the words alone (no LLM). First match wins, so order matters.
SHORTCUTS = [
    (r"(?:open\s+|make\s+|start\s+)?(?:a\s+)?new\s+tab", "cmd+t"),
    (r"(?:open\s+|make\s+)?(?:a\s+)?new\s+window", "cmd+n"),
    (r"(?:close|shut)\s+(?:this\s+|the\s+|that\s+|current\s+)*tab", "cmd+w"),
    (r"(?:close|shut)\s+(?:this\s+|the\s+|that\s+|current\s+)*window", "cmd+w"),
    (r"(?:refresh|reload)(?:\s+(?:this|the))?(?:\s+(?:page|tab|website|site))?", "cmd+r"),
    (r"(?:go\s+)?back(?:\s+to)?\s+(?:the\s+)?(?:previous|last)\s+page|(?:go\s+)?back\s+a\s+page|previous\s+page"
     r"|go\s+back\s+(?:in\s+the\s+)?browser", "cmd+["),
    (r"(?:go\s+)?forward(?:\s+a|\s+to\s+the\s+next)?\s+page|next\s+page", "cmd+]"),
    (r"(?:go\s+to\s+|switch\s+to\s+)?(?:the\s+)?next\s+tab", "ctrl+tab"),
    (r"(?:go\s+to\s+|switch\s+to\s+)?(?:the\s+)?(?:previous|last)\s+tab", "ctrl+shift+tab"),
    (r"select\s+all(?:\s+(?:the\s+)?text)?\s+and\s+copy(?:\s+it)?", "cmd+a cmd+c"),
    (r"select\s+all(?:\s+(?:the\s+)?text)?", "cmd+a"),
    (r"copy(?:\s+(?:this|that|it|the\s+text))?", "cmd+c"),
    (r"paste(?:\s+(?:it|this|that|the\s+text))?", "cmd+v"),
    (r"cut(?:\s+(?:it|this|that))?", "cmd+x"),
    (r"undo(?:\s+(?:that|it|this))?", "cmd+z"),
    (r"redo(?:\s+(?:that|it|this))?", "cmd+shift+z"),
    (r"save(?:\s+(?:it|this|the\s+file|the\s+document))?", "cmd+s"),
    (r"zoom\s+in(?:\s+(?:a\s+bit|more|again))?|make\s+(?:it|the\s+(?:text|page))\s+bigger|bigger\s+text", "cmd+="),
    (r"zoom\s+out(?:\s+(?:a\s+bit|more|again))?|make\s+(?:it|the\s+(?:text|page))\s+smaller|smaller\s+text", "cmd+-"),
    (r"(?:reset|re\s+set|resat|we\s+sat|recent|resize)\s+(?:the\s+)?(?:zoom|size)|(?:normal|actual)\s+(?:zoom|size)"
     r"|zoom\s+(?:to\s+)?(?:normal|100(?:\s+percent)?)", "cmd+0"),
    (r"zoom", "cmd+="),
    (r"(?:go\s+|make\s+it\s+|enter\s+|exit\s+|leave\s+)?full\s*screen(?:\s+mode)?", "ctrl+cmd+f"),
    (r"minimi[sz]e(?:\s+(?:this|the|it))?(?:\s+window)?", "cmd+m"),
    (r"hide\s+(?:this|the|it)(?:\s+(?:app|window))?", "cmd+h"),
    (r"(?:switch|go)\s+to\s+(?:the\s+)?(?:next|other)\s+window", "cmd+`"),
    (r"(?:go\s+to\s+|jump\s+to\s+)?(?:the\s+)?(?:top|start|beginning)\s+of\s+(?:the\s+)?page", "cmd+up"),
    (r"(?:go\s+to\s+|jump\s+to\s+)?(?:the\s+)?(?:bottom|end)\s+of\s+(?:the\s+)?page", "cmd+down"),
    (r"(?:open\s+)?(?:a\s+)?(?:new\s+)?(?:private|incognito)\s+window", "cmd+shift+n"),
    (r"reopen\s+(?:the\s+)?(?:closed|last)\s+tab|open\s+(?:the\s+)?(?:closed|last\s+closed)\s+tab", "cmd+shift+t"),
    (r"bookmark\s+(?:this|the)(?:\s+page)?", "cmd+d"),
    (r"(?:go\s+to\s+|click\s+(?:on\s+)?)?(?:the\s+)?(?:address|url)\s+bar", "cmd+l"),
    (r"(?:send|submit)(?:\s+(?:it|this|that|the\s+(?:message|prompt|question|text|chat|query|form)))?", "enter"),
    (r"(?:press|hit)\s+(?:the\s+)?(enter|return|escape|esc|space|tab|delete|backspace|up|down|left|right|"
     r"page\s+up|page\s+down|home|end)(?:\s+(?:key|button))?(?:\s+(?:again|once))?", r"\1"),
]

# Any in-app command. Those without an exact rule above are planned by the LLM.
UI_COMMAND = re.compile(rf"^{POLITE}(?:click|tap|double\s+click|right\s+click|press|hit|scroll|select|choose|"
                        r"check|uncheck|tick|refresh|reload|new\s+tab|open\s+a\s+new\s+tab|close\s+(?:this\s+|the\s+)?tab|"
                        r"undo|redo|copy|paste|cut|save|zoom|calculate|compute|clear|delete|go\s+to\s+the\s+next\s+tab|"
                        r"(?:go|switch)\s+to\s+the\s+(?:next|previous|last)\s+tab)\b")

SCROLL = re.compile(rf"^{POLITE}scroll(?:\s+(?:the\s+page|the\s+window|it|this|the\s+screen))?"
                    r"(?:\s+(?:(up|down|left|right|back\s+up)|to\s+the\s+(top|bottom)))?"
                    r"(?:\s+(?:(a\s+bit|a\s+little|a\s+little\s+bit|slightly)|(a\s+lot|way|much|fast)|more|again|down|"
                    r"(\d+|one|two|three|four|five)\s+(?:pages?|screens?|times)|please|now))*$")
SCREENS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}

SCREEN_QUESTION = re.compile(
    r"\b(?:(?:what(?:'s|\s+is)|whats)\s+(?:on|in)\s+(?:my|the|this)\s+(?:screen|display|window|page|monitor)"
    r"|(?:can\s+you\s+|do\s+you\s+)?(?:see|look\s+at|check|describe|explain|summari[sz]e|read)\s+(?:my|the|this)\s+"
    r"(?:screen|display|window|page|monitor|error|message|article|email|document|code|chart|image|picture|photo)"
    r"|what\s+(?:do|can|are)\s+you\s+(?:see|seeing|looking\s+at)|what\s+am\s+i\s+(?:looking\s+at|seeing|watching)"
    r"|what\s+(?:does|do)\s+(?:this|it|that|the)(?:\s+\w+)?\s+say|read\s+(?:this|it|that)(?:\s+(?:out|to\s+me|aloud))*"
    r"|(?:what(?:'s|\s+is)|whats)\s+(?:this|that)\s+(?:error|message|page|window|app|website|video|picture|image|code)"
    r"|(?:is\s+there|do\s+you\s+see|can\s+you\s+find)\s+.+\s+on\s+(?:my|the)\s+screen"
    r"|why\s+(?:can't|cannot|can\s+not)\s+you\s+see\s+(?:my|the)\s+screen|see\s+my\s+screen"
    r"|(?:open|show|look\s+at)\s+(?:the|my)\s+screen$"
    # "can you see my recent chats", "read my messages", "what are my last emails": read from the screen
    r"|(?:can\s+you\s+|could\s+you\s+)?(?:see|read(?:\s+out)?|show\s+me|tell\s+me|list|check|go\s+through)\s+"
    r"(?:me\s+)?(?:all\s+)?(?:my|the)\s+(?:recent\s+|last\s+|latest\s+|previous\s+|old(?:er)?\s+|new\s+|unread\s+)*"
    r"(?:chats?|charts|messages|conversations|emails|mails|notifications|comments|replies|search\s+results|results)"
    r"|what\s+(?:are|were)\s+(?:my|the)\s+(?:recent\s+|last\s+|latest\s+|previous\s+)*"
    r"(?:chats|conversations|messages|emails|mails|notifications)"
    r"|what\s+(?:did|have)\s+i\s+(?:ask|asked|chat|chatted|talk|talked)\s+(?:about\s+)?(?:before|earlier|last\s+time|recently))\b")


# "remove that", "delete what I typed", "clear the chat bar": the text in the box the user is looking at
BOX = (r"(?:the\s+|this\s+|my\s+)?(?:chat\s*bar|chat\s+box|chat|text\s*box|text\s+field|search\s*bar|search\s+box|"
       r"input(?:\s+box)?|prompt(?:\s+box)?|message\s+box|box|field|typing\s+area)")
WANT = r"(?:(?:i\s+(?:want|need|would\s+like)|i'd\s+like|help\s+me|let's|lets)\s+(?:you\s+)?(?:to\s+)?)?"
CLEAR_TEXT = re.compile(
    rf"^{POLITE}{WANT}(?:(?:remove|delete|clear|erase|wipe|get\s+rid\s+of)\s+"
    r"(?:(?:all\s+(?:of\s+)?)?(?:that|it|this|everything|all|the\s+(?:text|words|message|prompt|query|search|typing|input|draft)"
    r"|what(?:'s|\s+is|\s+i)\s+(?:typed|written|wrote|there|in\s+there)|what\s+i\s+(?:typed|wrote|said)"
    r"|whatever\s+is\s+(?:there|typed|written)|(?:the\s+)?(?:text|words)\s+there)"
    rf"(?:\s+(?:in|from|inside|on)\s+{BOX})?|{BOX})"
    r"|clear(?:\s+(?:it|this|that))?)(?:\s+(?:please|now|out|off))*$")
CLEAR_WORD = re.compile(rf"^{POLITE}(?:remove|delete|erase)\s+(?:the\s+)?last\s+word(?:\s+(?:please|now))*$")


# "pause", "stop the music", "resume the video": play/pause, whatever Laya's guess ("stop the music" was mute)
PLAY_PAUSE = re.compile(rf"^{POLITE}{WANT}(?:(?:pause|resume|unpause|continue|stop|start|play)"
                        r"(?:\s+(?:the|my|this|that))?(?:\s+(?:music|songs?|tracks?|playback|playing|spotify|video|"
                        r"youtube\s+video|podcast|audio|movie|it))?|(?:pause|resume|unpause))(?:\s+(?:please|now|again))*$")


def play_pause_request(text: str) -> bool:
    """A whole play/pause command. Bare "stop" / "play" / "start" are not (stop ends the conversation,
    "play X" is a search); they need what to stop ("stop the music")."""
    t = normalize(text)
    if not PLAY_PAUSE.match(t):
        return False
    return not re.fullmatch(rf"{POLITE}{WANT}(?:stop|play|start|continue)(?:\s+(?:it|please|now|again))*", t)


def clear_request(text: str) -> str | None:
    """"remove that" / "clear the chat box" -> "all", "delete the last word" -> "word", else None."""
    t = normalize(text)
    if CLEAR_WORD.match(t):
        return "word"
    return "all" if CLEAR_TEXT.match(t) else None


def is_ui_command(text: str) -> bool:
    return bool(UI_COMMAND.match(normalize(text)))


def shortcut(text: str) -> str | None:
    """"open a new tab" -> "cmd+t", "press enter" -> "enter"; None if the words aren't a known key command."""
    t = normalize(text)
    for words, keys in SHORTCUTS:
        m = re.fullmatch(rf"{POLITE}{words}(?:\s+(?:please|now|for\s+me|in\s+(?:the\s+)?(?:browser|{BROWSER_NAMES})))*", t)
        if m:
            return m.expand(keys)
    return None


# "go to the ChatGPT tab", "switch to my YouTube tab", "open the Gmail tab": an open tab, by its name
NOT_A_TAB_NAME = r"(?:new|next|previous|last|other|this|that|current|closed|last\s+closed|private|incognito|first|second|third)"
TAB_BY_NAME = re.compile(rf"^{POLITE}(?:go|switch|jump|move|take\s+me|bring\s+me|change)\s+(?:back\s+)?to\s+(?:the\s+|my\s+)?(?P<a>.+?)\s+tab"
                         rf"|^{POLITE}(?:open|show(?:\s+me)?|use)\s+(?:the\s+|my\s+)(?P<b>.+?)\s+tab$")


def tab_by_name(text: str) -> str | None:
    """"go to ChatGPT tab" -> "chatgpt"; None for "next tab" / "new tab" (those are keys)."""
    m = TAB_BY_NAME.match(normalize(text))
    name = m and (m.group("a") or m.group("b"))
    if not name or re.fullmatch(NOT_A_TAB_NAME, name):
        return None
    return name


# "previous step", "next tap" said in a browser: the noun is not music, so it is a tab
NEXT_OR_PREVIOUS = re.compile(r"^(?:go\s+(?:back\s+)?to\s+|switch\s+to\s+)?(?:the\s+)?(next|previous|last)\s+(\w+)$")
NOT_TABS = {"song", "songs", "track", "tracks", "tune", "music", "video", "one", "page", "result", "episode", "podcast",
            "chapter", "slide", "photo", "picture", "image", "day", "days", "week", "month", "year", "line"}


def tab_step(text: str) -> str | None:
    """In a browser: "previous step" (Whisper's "tab") -> "previous tab"; None when it's about music etc."""
    m = NEXT_OR_PREVIOUS.match(normalize(text))
    if not m or m.group(2) in NOT_TABS or m.group(2) in ("tab", "tabs"):
        return None
    return f"{'next' if m.group(1) == 'next' else 'previous'} tab"


def is_browser_command(text: str) -> bool:
    """Keys meant for the browser ("new tab", "reload the page"), even when another app is in front."""
    return bool(re.search(r"\b(?:tabs?|pages?|address\s+bar|url\s+bar|refresh|reload|forward|browser|website|site)\b",
                          normalize(text)))


ORDINAL_WORDS = r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last|top|\d+(?:st|nd|rd|th)"
# "open the second video", "pick the first account", "play the third one": a thing on the page, by position
NTH_ITEM = re.compile(rf"^{POLITE}(?:open|play|watch|pick|select|choose|use|go\s+to|click(?:\s+on)?|tap(?:\s+on)?)\s+"
                      rf"(?:the\s+|my\s+)?((?:{ORDINAL_WORDS})\s+(?:video|result|search\s+result|link|account|one|option|"
                      r"item|email|profile|google\s+account)s?)(?:\s+(?:please|now|for\s+me|on\s+(?:the|this)\s+(?:page|screen|list)))*$")
# "pick vikram's account", "use my work account", "continue with my personal account"
NAMED_ACCOUNT = re.compile(rf"^{POLITE}(?:pick|choose|select|use|sign\s+in\s+with|log\s+in\s+with|continue\s+with|"
                           r"switch\s+to)\s+(?:the\s+|my\s+)?(.+?\s+(?:google\s+)?account)(?:\s+(?:please|now))*$")
# Chart ranges on stock pages ("give me the last 5 days" -> the "5D" button)
CHART_RANGE = re.compile(rf"^{POLITE}(?:(?:give|show|check|see|display|view|switch\s+to|change\s+(?:it\s+)?to|go\s+to|select|"
                         r"pick|open|what\s+about)\s+)?(?:me\s+)?(?:the\s+)?(?:(?:last|past|previous)\s+)?"
                         r"(\d+|a|one|two|three|five|six|ten)\s+(day|week|month|year)s?"
                         r"(?:\s+(?:chart|graph|data|view|history|range|performance|prices?|trend))?(?:\s+(?:please|now))*$")
CHART_WORDS = re.compile(rf"^{POLITE}(?:(?:show|give|switch\s+to|change\s+(?:it\s+)?to|go\s+to|select)\s+)?(?:me\s+)?(?:the\s+)?"
                         r"(max(?:imum)?|all\s+time|year\s+to\s+date|ytd|today|intraday)"
                         r"(?:\s+(?:chart|graph|data|view|history|range|performance|prices?|trend))?(?:\s+(?:please|now))*$")
RANGE_NUMBERS = {"a": 1, "one": 1, "two": 2, "three": 3, "five": 5, "six": 6, "ten": 10}
RANGE_WORDS = {"max": "MAX", "maximum": "MAX", "all time": "MAX", "year to date": "YTD", "ytd": "YTD",
               "today": "1D", "intraday": "1D"}


def chart_range(text: str) -> str | None:
    """"give me the last 5 days" -> "5D", "show 1 year" -> "1Y", "max" -> "MAX": the range button of a
    stock / chart page (Google, Google Finance, Yahoo...). None if the words aren't a chart range."""
    t = normalize(text)
    m = CHART_RANGE.match(t)
    if m:
        n = RANGE_NUMBERS.get(m.group(1)) or int(m.group(1))
        return f"{n}{m.group(2)[0].upper()}"
    m = CHART_WORDS.match(t)
    return RANGE_WORDS[re.sub(r"\s+", " ", m.group(1))] if m else None


def click_target(text: str) -> str | None:
    """"click on the sign in button" -> "sign in", "open the second video" -> "second video",
    "pick my work account" -> "work account"."""
    t = normalize(text)
    m = NTH_ITEM.match(t) or NAMED_ACCOUNT.match(t)
    if m:
        return re.sub(r"\s+", " ", m.group(1))
    m = re.fullmatch(rf"{POLITE}(?:click|tap|press|hit|select|choose|double\s+click)\s+(?:on\s+)?(?:the\s+)?(.+?)"
                     r"(?:\s+(?:button|link|tab|menu|icon|option|item|box|field|bar|checkbox|key))?(?:\s+(?:please|now|for\s+me))*",
                     normalize(text))
    if not m or m.group(1) in ("it", "that", "this", "there", "here", "all", "enter", "play", "pause") \
            or re.search(r"\b(?:song|music|track|tune|playlist|volume)\b", m.group(1)):
        return None  # "hit this song" is Whisper's "skip this song"
    return m.group(1)


def scroll_args(text: str) -> list[str] | None:
    """"scroll up a bit" -> ["up", "bit"], "scroll to the bottom" -> ["bottom"], "scroll" -> ["down"]."""
    t = normalize(clean(text))
    m = SCROLL.match(t)
    if not m:
        m2 = re.fullmatch(rf"{POLITE}(?:page\s+(up|down)|go\s+(?:to\s+the\s+)?(top|bottom)|(?:go\s+)?(up|down)\s+a\s+bit)", t)
        if not m2:
            return None
        return [m2.group(1) or m2.group(2) or m2.group(3)] + (["bit"] if m2.group(3) else [])
    direction = (m.group(1) or m.group(2) or "down").replace("back up", "up")
    screens = m.group(5)
    if screens:
        screens = str(SCREENS.get(screens, screens))
    amount = "bit" if m.group(3) else "lot" if m.group(4) else screens or ""
    return [direction] + ([amount] if amount else [])


def scroll_direction(text: str) -> str | None:
    args = scroll_args(text)
    return args[0] if args else None


def screen_question(text: str) -> str | None:
    """"what's on my screen", "read this", "what does this error say" -> the question to ask about the screen."""
    t = normalize(text)
    if re.search(r"\b(?:screenshot|screen\s+shot|capture|record)\b", t):
        return None  # taking a screenshot, not looking
    if not SCREEN_QUESTION.search(t):
        return None
    if re.search(r"why\s+(?:can't|cannot|can\s+not)\s+you\s+see|^(?:can\s+you\s+)?see\s+my\s+screen$"
                 r"|(?:open|show|look\s+at)\s+(?:the|my)\s+screen$", t):
        return "What is on the screen?"
    return text.strip()


# "what am I holding", "how do I look", "look at me through the camera": one photo from the camera
CAMERA_QUESTION = re.compile(
    r"\b(?:camera|webcam)\b(?!\s+app)"
    r"|what\s+(?:am\s+i|i'm|im)\s+(?:holding|wearing|showing(?:\s+you)?|pointing\s+at|doing)\b"
    r"|\bhow\s+do\s+i\s+look\b|\bdo\s+i\s+look\s+(?:good|okay|ok|fine|tired|alright|nice)\b"
    r"|\bcan\s+you\s+see\s+me\b|\bin\s+my\s+hands?\b|\blook\s+at\s+me\b"
    r"|what(?:'s|\s+is)\s+in\s+front\s+of\s+(?:you|me)\b")
CAMERA_ONLY = re.compile(rf"^{POLITE}(?:(?:use|open|turn\s+on|start|check)\s+(?:the\s+|your\s+|my\s+)?(?:camera|webcam)"
                         r"|(?:look|see)\s+(?:through|with|using)\s+(?:the\s+|your\s+|my\s+)?(?:camera|webcam)"
                         r"|look\s+at\s+me)(?:\s+(?:please|now))*$")


def camera_question(text: str) -> str | None:
    """"what am I holding" -> that question about a camera photo; "look through the camera" -> "What do you see?".
    None for other words, "open the camera app" and taking pictures."""
    t = normalize(text)
    if not CAMERA_QUESTION.search(t) or re.search(r"\b(?:take|snap)\s+(?:a\s+)?(?:photo|picture|selfie)\b", t):
        return None
    if re.fullmatch(rf"{POLITE}open\s+(?:the\s+)?camera(?:\s+app)?|{POLITE}open\s+photo\s+booth", t):
        return None
    return "Describe what you see, in one or two sentences." if CAMERA_ONLY.match(t) else text.strip()


# "on this page, search for shoes and open the first result", "take over and fill in the form":
# a task done on the screen step by step (actions/agent.py)
SCREEN_TASK = re.compile(
    rf"^{POLITE}(?:(?:on|in)\s+(?:this|the|my)\s+(?:screen|page|website|site|window|app|tab)\s+(?:{POLITE})(.+)"
    r"|(?:take\s+(?:over|control)(?:\s+of\s+(?:the|my)\s+(?:screen|mac|computer))?|use\s+the\s+screen)\s+(?:and\s+|to\s+)?(.+)"
    r"|(.+?)\s+for\s+me\s+on\s+(?:this|the|my)\s+(?:screen|page|website|site|window))$")


def screen_task(text: str) -> str | None:
    """The task in "on this page, X" / "take over and X" / "X for me on the screen", else None.
    In the user's own spelling ("type my name Vikram"): the same number of words from the original text."""
    m = SCREEN_TASK.match(normalize(text))
    if not m:
        return None
    words = re.sub(r"[^\w\s.'@/-]", " ", text).split()
    n = len(next(g for g in m.groups() if g).split())
    return " ".join(words[:n] if m.group(3) else words[-n:]).strip(" .")


# "tell me when Build succeeded shows up", "watch the screen for done", "let me know when the page says ready"
QUOTES = "'\"“”‘’"
WATCH = [
    re.compile(rf"^{POLITE}(?:tell|let|notify|alert|ping)\s+me(?:\s+know)?\s+(?:when|once|as\s+soon\s+as)\s+"
               r"(?:the\s+(?:words?|text|message)\s+)?(.+?)\s+(?:appears?|shows?\s+up|pops?\s+up|comes?\s+up|"
               r"is\s+on\s+(?:the|my)\s+screen)(?:\s+on\s+(?:the|my)\s+screen)?(?:\s+please)?$"),
    re.compile(rf"^{POLITE}(?:tell|let|notify|alert|ping)\s+me(?:\s+know)?\s+(?:when|once|as\s+soon\s+as)\s+"
               r"(?:the\s+(?:screen|page|window|app)|it)\s+(?:says|shows|reads)\s+(.+?)(?:\s+please)?$"),
    re.compile(rf"^{POLITE}(?:watch|keep\s+an\s+eye\s+on|monitor)\s+(?:the|my)\s+screen\s+(?:for|until)\s+"
               r"(?:the\s+(?:words?|text|message)\s+)?(.+?)(?:\s+(?:appears?|shows?\s+up|comes?\s+up))?(?:\s+please)?$"),
]
STOP_WATCHING = re.compile(rf"^{POLITE}(?:stop|cancel|quit)\s+(?:watching|monitoring|looking)"
                           r"(?:\s+(?:the|my)\s+screen)?(?:\s+(?:please|now))*$")


def watch_request(text: str) -> str | None:
    """What to wait for on the screen, or None. Not "tell me when it's 5 pm" (a reminder)."""
    t = normalize(text.translate({ord(q): " " for q in QUOTES}))
    m = next((w.match(t) for w in WATCH if w.match(t)), None)
    return re.sub(r"^the\s+|\s+(?:message|text|words?|notification|sign)$", "", m.group(1).strip()) if m else None


def stop_watching_request(text: str) -> bool:
    return bool(STOP_WATCHING.match(normalize(text)))


def vision_request(text: str) -> bool:
    """A whole command for the eyes: camera, a task on screen, or watching the screen."""
    return bool(camera_question(text) or screen_task(text) or watch_request(text) or stop_watching_request(text))
