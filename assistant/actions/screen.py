"""Actions that need to know what is on screen: click something, answer a question about the screen.

click() tries these ways, fastest and surest first:
    0. in a browser, inside the page itself (~0.1 s)     "click sign in", "the second video", "5D"
       (needs the browser's "Allow JavaScript from Apple Events"; Chrome/Brave 154 hide pages from Accessibility)
    1. by name through Accessibility (instant)          "click compose"
    2. by the text on screen, Apple OCR (~0.1 s)         when the app doesn't expose the name
       "the first account": the e-mail addresses on screen, top to bottom
    3. by description, the vision model (~2 s)           "click the video with the red thumbnail"
"""

import re
import subprocess
import time

from assistant.mac import web_page
from assistant.mac.permissions import screen_recording_message

from .errors import Failed, NeedsPermission
from .macos import accessibility

ORDINAL = re.compile(r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last|top|"
                     r"\d+(?:st|nd|rd|th))\b", re.I)
IS_ACCOUNT = re.compile(r"\b(?:account|profile)\b", re.I)
ACCOUNT_WORDS = re.compile(r"(?:^my\s+|'s\b|\s*\b(?:google\s+)?(?:account|profile)\b)", re.I)
EMAIL = re.compile(r"\S+@\S+\.\w+")


def click(name: str) -> str:
    """Click something in the app the user is working in, by its name or a short description."""
    ax = accessibility()
    app = bring_forward(ax)
    name = re.sub(r"^(?:on\s+)?(?:the\s+)?|\s+(?:button|link|tab|menu|icon|option|item|box|field|bar)$", "",
                  name.strip(), flags=re.I)
    described = bool(ORDINAL.search(name)) or len(name.split()) > 4  # "the second video": look, don't match names
    scripts_off = None
    if web_page.is_browser(app):
        try:
            clicked, count = web_page.click(app.localizedName(), name)
            if clicked:
                return f"Clicked {_short(clicked)}"
            if count:  # "the fifth video" when the page shows three: say so, don't guess
                raise Failed(f"I only see {count} on this page")
        except web_page.ScriptsOff as e:
            scripts_off = str(e)
        except (RuntimeError, OSError, subprocess.TimeoutExpired):
            pass  # a page that refuses scripts (the new-tab page, the store): look at the screen instead
    target = None if described else ax.find(name, app)
    if target is not None:
        ax.press(target)
        return f"Clicked {ax.label(target)[:40] or name}"
    from assistant.mac import vision
    image, bounds = window_shot(ax, app)
    try:
        spot = _account_on_screen(image, name) if IS_ACCOUNT.search(name) else None
        if spot is None and not described:
            spot = vision.find_text(image, name)
        description = _description(name) if described else name if " " in name else f"the {name} button or link"
        if spot is None:
            spot = vision.locate_click(image, description)  # GUI-grounding model: surer on ordinals/icons
        if spot is None:
            spot = vision.point(image, description)
    finally:
        vision.discard(image)
    if spot is None:
        raise Failed(f"I couldn't find {name} on the screen" + (f". {scripts_off}" if scripts_off else ""))
    ax.click_at(*on_screen(bounds, spot))
    return f"Clicked {name}"


def _short(label: str) -> str:
    """What to say back: a video title without its channel / duration part, at most ~40 letters."""
    label = re.split(r"\s+[|·•]\s+", label)[0]
    return label if len(label) <= 40 else label[:40].rsplit(" ", 1)[0]


def _description(name: str) -> str:
    """What to ask the vision model to point at: "second video" -> counted in reading order."""
    if ORDINAL.search(name):
        return f"the {name} (counting left to right, then top to bottom; point at its title or thumbnail)"
    return f"the {name}"


def _account_on_screen(image, name: str) -> tuple[float, float] | None:
    """"the second account" on a "Choose an account" page: the e-mail addresses on screen, top to bottom.
    "vikram's account": the line with that name. None if no account is visible."""
    from assistant.mac import vision
    lines = vision.read_text(image)
    emails = sorted((1 - (box.origin.y + box.size.height / 2), box.origin.x + box.size.width / 2)
                    for text, _, box in lines if EMAIL.search(text))
    m = ORDINAL.search(name)
    if m and emails:
        n = web_page.ORDINALS.get(m.group(0).lower(), 0)
        if n >= len(emails):
            return None
        y, x = emails[n]
        return x, y
    who = ACCOUNT_WORDS.sub("", name).strip()
    return vision.find_text(image, who) if who and not m else None


# Questions about how things look go to the vision model; everything else is answered from the screen's text.
VISUAL = re.compile(r"\b(?:colou?rs?|look(?:s|ing)?\s+like|picture|image|photo|drawing|logo|design|layout|icon|"
                    r"thumbnail|face|wearing|graph|chart\s+(?:look|show|trend)|diagram|map)\b", re.I)
MIN_WORDS = 12  # less text than this on screen: a picture, a video... let the vision model look

# answer_from_text(question, screen text, where) -> spoken answer. Set by the app to the LLM (brain.slow):
# actions don't import the brains. Without it, the vision model answers everything.
answer_from_text = None


def look_at_screen(question: str = "What is on the screen?") -> str:
    """Answer a question about what the user is looking at. Text questions ("read my recent chats", "what does
    this error say") are answered by the LLM from the exact text on screen (the page's own text in a browser,
    else Apple OCR); questions about how things look, by the vision model on a screenshot."""
    from assistant.mac import vision
    ax = accessibility()
    app = bring_forward(ax)
    where = app.localizedName() if app is not None else "the screen"
    text = None
    if answer_from_text is not None and not VISUAL.search(question) and web_page.is_browser(app):
        try:
            title, text = web_page.page_text(app.localizedName())
            where = f"{title} in {where}" if title else where
        except (web_page.ScriptsOff, RuntimeError, OSError, subprocess.TimeoutExpired):
            text = None
    if text and len(text.split()) >= MIN_WORDS:
        return list_from_screen(question, text) or answer_from_text(question, text, where)
    image, _ = window_shot(ax, app)
    try:
        if answer_from_text is not None and not VISUAL.search(question):
            text = screen_text(vision.read_text(image))
            if len(text.split()) >= MIN_WORDS:
                return list_from_screen(question, text) or answer_from_text(question, text, where)
        return vision.ask(image, question) or "I couldn't make out the screen"
    finally:
        vision.discard(image)


# "read my recent chats", "what did I ask before", "what are my emails": the titles under the list's heading,
# read by rule (the 4B LLM mixed menu buttons like "New chat, Library" into the list).
LIST_QUESTION = re.compile(r"\b(chats?|charts|conversations?|messages|e?mails|notifications)\b"
                           r"|\bwhat\s+(?:did|have)\s+i\s+(?:ask|asked|chat|chatted|talk|talked)", re.I)
LIST_HEADING = re.compile(r"^(?:chats|your\s+chats|recent|recents|recent\s+chats|today|yesterday|previous\s+\d+\s+days|"
                          r"this\s+week|inbox(?:\s+\d+)?|primary|conversations|messages|notifications)$", re.I)
NOT_A_TITLE = re.compile(r"^(?:new\s+chat|search(?:\s+chats)?|library|sora|gpts|explore\s+gpts|codex|projects?|new\s+project|"
                         r"settings|help|upgrade(?:\s+plan)?|see\s+plans.*|log\s*in|sign\s+in|sign\s+up|compose|starred|"
                         r"snoozed|sent|drafts|more|show\s+more|see\s+all|what\s+can\s+i\s+help\s+with\??|ask\s+anything|"
                         r"promotions|social|updates|chatgpt|images|apps|deep\s+research|my\s+gpts|.{0,2})$", re.I)
LIST_NOUN = {"chat": "chats", "chats": "chats", "charts": "chats", "conversation": "chats", "conversations": "chats",
             "messages": "messages", "emails": "emails", "mails": "emails", "notifications": "notifications"}


def list_from_screen(question: str, text: str, most: int = 5) -> str | None:
    """"Your recent chats are A, B, C" from the screen text, or None (not a list question / no list heading)."""
    m = LIST_QUESTION.search(question)
    if not m:
        return None
    noun = LIST_NOUN.get((m.group(1) or "chats").lower(), "chats")
    lines = [" ".join(line.split()) for line in text.splitlines()]
    start = next((i for i, line in enumerate(lines) if LIST_HEADING.match(line)), None)
    if start is None:
        return None
    titles = []
    for line in lines[start + 1:]:
        if LIST_HEADING.match(line) or NOT_A_TITLE.match(line) or line in titles:
            continue
        titles.append(line)
        if len(titles) == most:
            break
    if not titles:
        return None
    return f"Your recent {noun} are: " + ", ".join(t.rstrip(".") for t in titles)


def screen_text(lines) -> str:
    """OCR lines in reading order: a narrow left side panel (chat list, folders) first, then the rest,
    each top to bottom."""
    def place(line):
        box = line[2]
        side = box.origin.x < 0.22 and box.size.width < 0.25
        return (0 if side else 1, -(box.origin.y + box.size.height), box.origin.x)
    return "\n".join(text for text, _, _ in sorted(lines, key=place))


def bring_forward(ax):
    """The app the user is working in, brought to the front and settled (its window finished sliding in)."""
    app = ax.target_app()
    if ax.activate(app):
        time.sleep(ax.SETTLE_SECONDS - 0.25)
    return app


def window_shot(ax, app):
    """(screenshot of the app's window, its bounds in screen points)."""
    from assistant.mac import vision
    _, bounds = ax.window_of(app)
    number = ax.window_number(app, bounds) if bounds else None
    try:
        return vision.capture(number), bounds
    except vision.NoScreenAccess:
        raise NeedsPermission(screen_recording_message()) from None


def on_screen(bounds, spot) -> tuple[float, float]:
    """A spot given as fractions of a window screenshot -> screen point."""
    if bounds:
        return bounds[0] + spot[0] * bounds[2], bounds[1] + spot[1] * bounds[3]
    from AppKit import NSScreen
    frame = NSScreen.mainScreen().frame()
    return spot[0] * frame.size.width, spot[1] * frame.size.height
