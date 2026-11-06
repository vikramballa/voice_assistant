"""One sentence as Whisper wrote it -> clean clauses: fix mishearings, find the wake word, split "A and B"."""

import re

from .words import BROWSER_NAMES, OPEN_VERB, POLITE, VERBS, normalize

WAKE = re.compile(r"^\W*(?:(?:(?:hey|hi|okay|ok|a)\W+)?(?:laya|leia|layla|laila|leila|lyla|lyra|lara|laia|liya|leya|lea|leah|lya|laya's)"
                  r"|h[ae]y?l[eaiy]+a?h?)\b\W*",  # fused by speech-to-text: "Helea", "Haleya", "Heleia"
                  re.IGNORECASE)

# Whisper's usual slips on names we care about.
MISHEARD = [
    (re.compile(r"\b(?:braille|brave's|breve|brev)\b", re.IGNORECASE), "Brave"),
    (re.compile(r"\b(?:crome|krome|chrom)\b", re.IGNORECASE), "Chrome"),
    (re.compile(r"\b(?:sapari|safary|saffari)\b", re.IGNORECASE), "Safari"),
    (re.compile(r"\b(?:you\s+tube|utube)\b", re.IGNORECASE), "YouTube"),
    # "Create a new node.", "write it in nodes": Whisper's "note(s)" (only after a/new or in/to, so "node.js" stays)
    (re.compile(r"\b(new|a)\s+node\b", re.IGNORECASE), r"\1 note"),
    (re.compile(r"\b(in|to|my)\s+nodes\b", re.IGNORECASE), r"\1 notes"),
    # "OK, right. What I see?" was "write what I say" (2026-10-04 log: it took a screenshot)
    (re.compile(r"\b(?:right|write|rite)\W+what\s+i\s+(?:say|see|said)\b", re.IGNORECASE), "write what I say"),
    # "Scrone up.", "School down", "Scrawl down": only before a direction, so "school" stays a word
    (re.compile(r"\b(?:scrone|scrawl|scroal|skroll|scrol|scroll's|scrolls|school|scroller|scroe|scrow|screw|crawl|scold)"
                r"(?=\s+(?:up|down|left|right|back|to\s+the|a\s+bit|a\s+little|more|a\s+lot)\b)", re.IGNORECASE), "scroll"),
]
# "In Brave, open YouTube" -> "open YouTube in brave"
LEADING_BROWSER = re.compile(rf"^\s*(?:in|on|using|with)\s+(?:the\s+)?(?:google\s+)?({BROWSER_NAMES})(?:\s+browser)?\s*,?\s+(.+)$",
                             re.IGNORECASE)
# Speech cut off after a joining word: "open safari and it"
DANGLING = re.compile(r"(?:\s*,?\s+(?:and|then|and\s+then|and\s+it|and\s+the|and\s+also|so|um+|uh+))+[\s.,!?]*$",
                      re.IGNORECASE)

SPLIT = re.compile(rf"\s*,?\s+(?:and\s+then|and|then|after\s+that)\s+(?=(?:{VERBS})\b)", re.IGNORECASE)
BROWSER_OPEN = re.compile(rf"^{POLITE}(?:open|launch|start)\s+(?:google\s+)?({BROWSER_NAMES})(?:\s+browser)?$")
# What can follow "open chrome and ..." in that browser ("open safari and open notes" is two apps)
WEB_CLAUSE = re.compile(r"^(?:search|google|look\s+up|find|play|watch|youtube|go\s+to|visit|ask|"
                        r"open\s+(?:the\s+)?(?:youtube|google|\S+\.[a-z]{2,}\b|.*\b(?:website|site|page)\b))")
SITE_OPEN = re.compile(rf"^{POLITE}{OPEN_VERB}\s+(?:the\s+)?(youtube|google)(?:\s+website|\s+site|\.com)?$")
# "Okay, quit notes.", "Done. Go to previous tab.": words before the command that aren't part of it
LEADING_FILLER = re.compile(r"^\s*(?:(?:(?:ok(?:ay)?|alright|all\s+right|um+|uh+|hmm+|so|well)\s*[,.!]?|done\s*[,.!])\s+)+(?=\w)",
                            re.IGNORECASE)
CONTINUATION = re.compile(r"^\W*(?:and|then|and\s+then|also|after\s+that)\b\W*(.+)$", re.IGNORECASE)


def fix_misheard(text: str) -> str:
    for pattern, fix in MISHEARD:
        text = pattern.sub(fix, text)
    return text


def clean(text: str) -> str:
    """Fix what speech-to-text gets wrong before deciding: misheard names, "In Brave, open
    YouTube" -> "open YouTube in brave", a cut-off "... and it"."""
    t = LEADING_FILLER.sub("", fix_misheard(text.strip()))
    m = LEADING_BROWSER.match(t)
    if m:
        t = f"{m.group(2).rstrip(' .!?')} in {m.group(1).lower()}"
    return DANGLING.sub("", t).strip()


def strip_wake(text: str) -> str | None:
    """"Hey Laya, open Safari" -> "open Safari".  None if the wake word isn't there."""
    m = WAKE.match(text)
    return text[m.end():].strip() if m else None


def split_commands(text: str) -> list[str]:
    """Split into one clause per command, dropping a leading "open chrome" / "open youtube"
    that only sets where the following search happens."""
    clauses = [c.strip(" ,.") for c in SPLIT.split(text.strip()) if c.strip(" ,.")]
    if len(clauses) > 1 and BROWSER_OPEN.match(normalize(clauses[0])) and WEB_CLAUSE.match(normalize(clauses[1])):
        clauses = clauses[1:]  # "open chrome and search youtube for x" -> one clause, browser from the sentence
    elif len(clauses) > 1 and re.match(r"(?:search|play|find|look|watch|show)\b", normalize(clauses[1])):
        site = SITE_OPEN.match(normalize(clauses[0]))
        if site:  # "open youtube and search cats" -> "search cats on youtube"
            clauses = [clauses[1] + (" on youtube" if site.group(1) == "youtube" else "")] + clauses[2:]
    return clauses or [text]


def continuation(text: str) -> str | None:
    """"and type hello" -> "type hello" (the rest of a command we stopped listening to too early)."""
    m = CONTINUATION.match(text.strip())
    return m.group(1).strip() if m else None
