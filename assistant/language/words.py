"""The shared vocabulary of the text rules: normalizing text, and word patterns used by several modules."""

import re

# "please", "can you", "just now and"...: allowed in front of any command.
POLITE = r"(?:(?:please|can\s+you|could\s+you|would\s+you|will\s+you|just|now|and)\s+)*"

# Words a command starts with. A new clause starts at "and"/"then" only before one of these:
# "take a screenshot and lock my mac" splits; "search for salt and pepper" doesn't.
VERBS = (r"open|close|quit|exit|launch|start|take|lock|set|turn|mute|unmute|play|pause|resume|stop|type|write|"
         r"google|search|find|look|go|visit|show|increase|decrease|raise|lower|skip|next|previous|capture|"
         r"screenshot|volume|make|youtube|tell|what|how|calculate|compute|click|tap|press|hit|scroll|select|choose|multiply|"
         r"divide|check|uncheck|tick|refresh|reload|copy|paste|undo|redo|zoom|save|create|pick|watch|jot|send|submit|ask|remove|erase|text|message|email|reply")

OPEN_VERB = (r"(?:open|launch|start|run|fire\s+up|bring\s+up|pull\s+up|go\s+to|visit|load|navigate\s+to"
             r"|take\s+me\s+to|switch\s+to)")

BROWSER_NAMES = r"chrome|safari|brave"
DEFAULT_BROWSER = "chrome"


def normalize(text: str) -> str:
    """Lowercase, drop punctuation except dots in domains, collapse spaces."""
    t = re.sub(r"[^\w\s.'/-]", " ", text.lower())
    t = re.sub(r"\s+dot\s+(com|org|net|io|ai|dev|in|co)\b", r".\1", t)  # "github dot com"
    return " ".join(t.split()).strip(" .")
