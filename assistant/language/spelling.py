"""Auto-correction of misheard / misspelled names before the brains see the words.

Whisper writes names it doesn't know as near-misses: "jithub", "spotifi", "notez", "chat gpt". A word that is
not an English word but is close to a name Laya knows (installed apps, web services, browsers, folders, the
command verbs) is replaced by that name: "search jithub for laya" -> "search GitHub for laya".

English words are never changed (so "crome" -> Chrome stays the job of sentence.MISHEARD), and nothing is
changed in text the user dictates ("type ...", "write a note ...").
"""

import difflib
import re
from functools import cache
from pathlib import Path

from . import catalog
from .words import POLITE, VERBS, normalize

DICTIONARY = Path("/usr/share/dict/words")  # macOS's word list (235k words)
MIN_LETTERS = 4      # shorter words carry too little to correct safely
CUTOFF = 0.8         # difflib similarity needed ("gmial" -> gmail, "notez" -> notes)...
SHORT_CUTOFF = 0.86  # ...for 4-letter words
EXTRA_WORDS = {"laya", "hey", "okay", "ok", "app", "apps", "online", "website", "email", "emails", "wifi", "url",
               "login", "signup", "username", "password", "screenshot", "youtube", "google", "iphone", "ipad",
               "macbook", "bluetooth", "airpods", "playlist", "podcast", "podcasts", "selfie", "lofi"}
# Names people say that may be neither installed nor a known site.
BRANDS = ["Spotify", "WhatsApp", "Instagram", "Facebook", "Netflix", "Telegram", "Discord", "Slack", "Zoom", "Notion",
          "Amazon", "Flipkart", "Twitter", "Reddit", "Wikipedia", "Gemini", "ChatGPT", "Claude", "Perplexity"]
# Dictation: what follows is the user's own text, never "corrected" into app names.
DICTATION = re.compile(rf"^{POLITE}(?:type|write|enter|dictate|note\s+down|jot|(?:make|take|create|add)\s+a\s+note)\b"
                       r"|\b(?:in|to|into)\s+(?:my\s+|the\s+)?notes\b")


@cache
def english() -> frozenset[str]:
    try:
        return frozenset(w.strip().lower() for w in DICTIONARY.read_text().split())
    except OSError:
        return frozenset()


@cache
def names() -> dict[str, str]:
    """{one-word lowercase form: how to write it}: "github" -> "GitHub", "chatgpt" -> "ChatGPT"."""
    found: dict[str, str] = {}
    for key, name in catalog.apps().items():
        found.setdefault(key.replace(" ", ""), name if " " not in name else key)
        first = key.split()[0]
        if len(first) >= MIN_LETTERS:
            found.setdefault(first, name.split()[0])
    for key in catalog.SITES:
        found.setdefault(key.replace(" ", ""), _site_spelling(key))
    for name in BRANDS:
        found.setdefault(name.lower(), name)
    for word in VERBS.split("|") + list(catalog.FOLDERS):
        found.setdefault(word, word)
    return found


def _site_spelling(key: str) -> str:
    known = {"github": "GitHub", "chatgpt": "ChatGPT", "youtube": "YouTube", "gmail": "Gmail", "linkedin": "LinkedIn",
             "whatsapp": "WhatsApp", "netflix": "Netflix", "gemini": "Gemini", "spotify": "Spotify"}
    return known.get(key.replace(" ", ""), key)


def _known_word(word: str) -> bool:
    return word in english() or word in EXTRA_WORDS or word.rstrip("s") in english() or word in names()


def _closest(word: str) -> str | None:
    cutoff = CUTOFF if len(word) >= 5 else SHORT_CUTOFF
    close = difflib.get_close_matches(word, names(), n=1, cutoff=cutoff)
    return names()[close[0]] if close else None


def correct(text: str) -> tuple[str, list[tuple[str, str]]]:
    """(corrected text, [(heard, fixed)...]). Unchanged when nothing needs fixing or it is dictation."""
    if DICTATION.search(normalize(text)):
        return text, []
    tokens = re.split(r"(\s+)", text)  # keeps the spaces, so the rest of the text stays exactly as said
    fixes = []
    i = 0
    while i < len(tokens):
        word = re.sub(r"[^\w]", "", tokens[i]).lower()
        if not word:
            i += 1
            continue
        # two words that are one name: "chat gpt" -> ChatGPT, "what's app" -> WhatsApp
        if i + 2 < len(tokens):
            nxt = re.sub(r"[^\w]", "", tokens[i + 2]).lower()
            joined = word + nxt
            if nxt and joined in names() and not (_known_word(word) and _known_word(nxt) and joined not in catalog.SITES):
                fixes.append((f"{tokens[i]} {tokens[i + 2]}", names()[joined]))
                tokens[i:i + 3] = [_keep_punctuation(tokens[i + 2], names()[joined])]
                i += 1
                continue
        if len(word) >= MIN_LETTERS and word.isalpha() and not _known_word(word):
            fixed = _closest(word)
            if fixed and fixed.lower() != word:
                fixes.append((tokens[i], fixed))
                tokens[i] = _keep_punctuation(tokens[i], fixed)
        i += 1
    return "".join(tokens), fixes


def _keep_punctuation(token: str, word: str) -> str:
    """"jithub." -> "GitHub." (the full stop Whisper put after it stays)."""
    m = re.match(r"^(\W*)\w[\w']*(\W*)$", token)
    return f"{m.group(1)}{word}{m.group(2)}" if m else word
