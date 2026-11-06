"""What exists to be opened: installed apps, web services, folders and browsers.

The main question answered here is what "open X" means. Laya only decides that the user wants to
open something; whether X is an installed app, a website or a folder is a lookup, done exactly here.
"""

import difflib
import re
from functools import cache
from pathlib import Path

from .words import BROWSER_NAMES, DEFAULT_BROWSER, OPEN_VERB, POLITE, normalize

APP_DIRS = ["/Applications", "/Applications/Utilities", "/System/Applications",
            "/System/Applications/Utilities", str(Path.home() / "Applications")]
# Spoken name -> real app name. Only used for apps that are installed (see aliases()).
APP_ALIASES = {
    "chrome": "Google Chrome", "google chrome": "Google Chrome", "brave": "Brave Browser",
    "vs code": "Visual Studio Code", "vscode": "Visual Studio Code", "visual studio code": "Visual Studio Code",
    "settings": "System Settings", "system preferences": "System Settings", "preferences": "System Settings",
    "finder": "Finder", "calculator": "Calculator", "app store": "App Store", "teams": "Microsoft Teams",
    "voice memos": "VoiceMemos", "find my": "FindMy", "quicktime": "QuickTime Player", "vnc": "RealVNC Connect Viewer",
}
# Web services by spoken name. Multi-word names are matched as a whole ("google drive").
# "spotify" is left out on purpose: its web player can't be controlled by "next song" (that goes to Music).
SITES = {
    "youtube": "youtube.com", "youtube music": "music.youtube.com", "gmail": "mail.google.com",
    "google mail": "mail.google.com", "github": "github.com", "google": "google.com",
    "gemini": "gemini.google.com", "google gemini": "gemini.google.com", "bard": "gemini.google.com",
    "google drive": "drive.google.com", "drive": "drive.google.com", "google docs": "docs.google.com",
    "google sheets": "sheets.google.com", "google slides": "slides.google.com",
    "google calendar": "calendar.google.com", "google maps": "maps.google.com",
    "google photos": "photos.google.com", "google meet": "meet.google.com", "google translate": "translate.google.com",
    "google classroom": "classroom.google.com", "google keep": "keep.google.com", "google news": "news.google.com",
    "google account": "myaccount.google.com", "my google account": "myaccount.google.com",
    "chatgpt": "chatgpt.com", "chat gpt": "chatgpt.com", "claude": "claude.ai", "perplexity": "perplexity.ai",
    "grok": "grok.com", "deepseek": "chat.deepseek.com", "copilot": "copilot.microsoft.com",
    "reddit": "reddit.com", "instagram": "instagram.com", "linkedin": "linkedin.com", "netflix": "netflix.com",
    "twitter": "x.com", "x": "x.com", "facebook": "facebook.com", "amazon": "amazon.in", "flipkart": "flipkart.com",
    "wikipedia": "wikipedia.org", "stackoverflow": "stackoverflow.com", "stack overflow": "stackoverflow.com",
    "whatsapp web": "web.whatsapp.com", "telegram web": "web.telegram.org", "outlook": "outlook.live.com",
    "notion": "notion.so", "figma": "figma.com", "canva": "canva.com",
    "discord": "discord.com/app", "slack": "app.slack.com", "zoom": "zoom.us", "twitch": "twitch.tv",
    "leetcode": "leetcode.com", "hacker news": "news.ycombinator.com", "kaggle": "kaggle.com",
    "hugging face": "huggingface.co", "huggingface": "huggingface.co", "arxiv": "arxiv.org", "medium": "medium.com",
    "hotstar": "hotstar.com", "prime video": "primevideo.com", "dropbox": "dropbox.com", "icloud": "icloud.com",
    "pinterest": "pinterest.com", "quora": "quora.com", "coursera": "coursera.org", "udemy": "udemy.com",
}
FOLDERS = ["downloads", "documents", "desktop", "pictures", "movies", "music", "home", "applications"]
# App name (lowercase) -> the short browser name the web actions take.
BROWSER_KEYS = {"google chrome": "chrome", "chrome": "chrome", "safari": "safari", "brave browser": "brave", "brave": "brave"}

# "... in chrome", "... using the brave browser"
BROWSER_IN = re.compile(rf"\s+(?:in|on|using|with)\s+(?:the\s+)?(?:google\s+)?({BROWSER_NAMES})(?:\s+browser)?\b")
# "... in my google account", "... for me", "... please": not part of the name being opened.
OPEN_TAIL = re.compile(r"\s+(?:(?:in|on|with|using|from|through)\s+(?:my\s+|the\s+)?(?:google|gmail)(?:\s+account|\s+profile)?"
                       r"|(?:in|on|with|using|from)\s+(?:my\s+|the\s+)?(?:account|profile|browser|laptop|mac|computer)"
                       r"|for\s+me|please|now|right\s+now|quickly|real\s+quick)$")
# "the calculator app", "the downloads folder": the kind of thing is not part of its name.
OPEN_KIND = re.compile(r"\s+(?:app|application|website|web\s+site|site|web\s+page|page|folder|window|in\s+finder)$")
DOMAIN = r"[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|org|net|io|ai|dev|edu|in|co|gov|app|so|us|tv)"


# --- installed apps (read once, on first use) ---------------------------------------------------

def installed_apps() -> dict[str, str]:
    """lowercase name -> real app name, for everything in the Applications folders."""
    found = {}
    for folder in APP_DIRS:
        for path in Path(folder).glob("*.app"):
            found[path.stem.lower()] = path.stem
    return found


@cache
def apps() -> dict[str, str]:
    """installed_apps(), plus Finder (it lives in /System/Library/CoreServices)."""
    found = installed_apps()
    found.setdefault("finder", "Finder")
    return found


@cache
def aliases() -> dict[str, str]:
    """APP_ALIASES, only for apps that are on this Mac."""
    return {k: v for k, v in APP_ALIASES.items() if v.lower() in apps()}


# --- browsers -----------------------------------------------------------------------------------

def pick_browser(clause: str, sentence: str | None = None, default: str | None = None) -> str:
    """The browser named in the clause, else in the whole sentence, else `default` (the one in use), else chrome."""
    for text in (clause, sentence or ""):
        m = re.search(rf"\b({BROWSER_NAMES})\b", text.lower())
        if m:
            return m.group(1)
    return default or DEFAULT_BROWSER


# --- "open X" -----------------------------------------------------------------------------------

def is_open_command(text: str) -> bool:
    """Starts with an unmistakable open verb ("open", "go to"; not "start"/"play")."""
    return bool(re.match(rf"^{POLITE}(?:open|launch|go\s+to|visit|navigate\s+to|take\s+me\s+to|pull\s+up"
                         rf"|fire\s+up|bring\s+up)\b", normalize(text)))


def open_object(text: str) -> str | None:
    """The thing after an "open"-type verb, without filler: "please open Gemini in my Google
    account" -> "gemini", "open the calculator app" -> "calculator"."""
    t = normalize(text)
    m = re.match(rf"^{POLITE}(?:{OPEN_VERB}|close|quit|exit|kill|show\s+me|show)\s+(?:up\s+)?(.+)$", t)
    if not m:
        return None
    obj = BROWSER_IN.sub("", m.group(1))
    while True:  # strip fillers until nothing changes: "the calculator app please" -> "calculator"
        stripped = OPEN_KIND.sub("", OPEN_TAIL.sub("", obj))
        stripped = re.sub(r"^(?:the|my|a|an|me)\s+", "", stripped)
        if stripped == obj:
            break
        obj = stripped
    return obj.strip(" .'") or None


def open_candidates(text: str, fuzzy: float = 0.8) -> dict[str, list[str]]:
    """Every exact reading of "open X": {"open_app": [...], "open_website": [...], "open_folder": [...]}.
    Falls back to the closest-sounding app/site (above `fuzzy` similarity) when nothing matches exactly;
    fuzzy=1 turns that off."""
    obj = open_object(text)
    if not obj:
        return {}
    domain = re.fullmatch(rf"({DOMAIN}(?:/\S*)?)", obj)
    if domain:
        return {"open_website": [_fix_domain(domain.group(1))]}
    found = {}
    if obj in FOLDERS:
        found["open_folder"] = [obj]
    site = SITES.get(obj) or SITES.get(f"{obj} web")
    if site and BROWSER_IN.search(normalize(text)):
        return {"open_website": [site]}  # "open whatsapp in chrome" means the website
    if obj in SITES:
        found["open_website"] = [SITES[obj]]
    app = _app_exact(obj)
    if app:
        found["open_app"] = [app]
    if not found and fuzzy < 1:
        guess = _fuzzy(obj, fuzzy)
        if guess:
            found[guess[0]] = guess[1]
    return found


def app_name(text: str) -> str | None:
    """"open the calculator app" -> "Calculator" (installed apps only)."""
    obj = open_object(text)
    return _app_exact(obj) if obj else None


def missing_app(text: str) -> str | None:
    """"open vs code" when VS Code isn't on this Mac -> "Visual Studio Code" (so we can say so)."""
    obj = open_object(text)
    if obj and obj in APP_ALIASES and obj not in aliases() and obj not in SITES:
        return APP_ALIASES[obj]
    return None


def resolve_name(name: str, fuzzy: float = 0.6) -> tuple[str, list[str]] | None:
    """An app/site/folder name (e.g. from the LLM) -> (action, args), or None if nothing is close.
    Installed apps win, then websites, then folders."""
    found = open_candidates(f"open {name}", fuzzy)
    for action in ("open_app", "open_website", "open_folder"):
        if action in found:
            return action, found[action]
    return None


def website(text: str) -> str | None:
    """A domain said in the text ("go to github.com"), or a known web service's address ("open gmail")."""
    t = normalize(text)
    m = re.search(r"\b([a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|org|net|io|ai|dev|edu|in|co|gov|app))\b", t)
    if m:
        return _fix_domain(m.group(1))
    obj = open_object(t)
    return SITES.get(obj) if obj else None


def folder(text: str) -> str | None:
    m = re.search(r"\b(" + "|".join(FOLDERS) + r")\b", normalize(text))
    return m.group(1) if m else None


def _app_exact(name: str) -> str | None:
    if name in aliases():
        return aliases()[name]
    if name in apps():
        return apps()[name]
    # "word" -> "Microsoft Word", only when exactly one installed app matches
    matches = {v for k, v in apps().items() if re.search(rf"\b{re.escape(name)}\b", k)}
    return matches.pop() if len(matches) == 1 else None


def _fuzzy(name: str, cutoff: float) -> tuple[str, list[str]] | None:
    """Closest app or website to a misheard name ("braille" -> Brave Browser)."""
    names = {}
    for k, v in apps().items():
        names[k] = ("open_app", [v])
        first = k.split()[0]
        if len(first) > 3:
            names.setdefault(first, ("open_app", [v]))
    for k, v in aliases().items():
        names.setdefault(k, ("open_app", [v]))
    for k, v in SITES.items():
        if len(k) > 2:
            names.setdefault(k, ("open_website", [v]))
    if len(name) < 5:  # "mac" is not "Macky": short words need a near-exact match
        cutoff = max(cutoff, 0.85)
    close = difflib.get_close_matches(name, names, n=1, cutoff=cutoff)
    return names[close[0]] if close else None


def _fix_domain(domain: str) -> str:
    """Speech-to-text slips on names: "jithub.com" -> "github.com".
    Only very close matches to well-known sites are corrected."""
    known = set(SITES.values()) | {s.replace(" ", "") + ".com" for s in SITES}
    if domain in known:
        return domain
    close = difflib.get_close_matches(domain, known, n=1, cutoff=0.85)
    return close[0] if close else domain
