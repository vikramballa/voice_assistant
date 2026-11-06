"""How words sound, for judging speech-to-text mistakes: Whisper swaps words that sound alike
("hacker" -> "hackle", "tab" -> "tap"), so a fix is believable only when the new word sounds like the old one.

    code         a Metaphone-style sound code: "hacker" -> "HKR", "hackle" -> "HKL", "phone" -> "FN"
    sound_alike  0..1: half spelling similarity, half sound-code similarity
"""

import difflib
import re

VOWELS = set("AEIOU")


def code(word: str) -> str:
    """Simplified Metaphone (Lawrence Philips, 1990): consonant sounds, the first vowel kept."""
    w = re.sub(r"[^A-Z]", "", word.upper())
    if not w:
        return ""
    for start, repl in (("KN", "N"), ("GN", "N"), ("PN", "N"), ("WR", "R"), ("AE", "E"), ("WH", "W"), ("X", "S")):
        if w.startswith(start):
            w = repl + w[len(start):]
            break
    out = []
    for i, c in enumerate(w):
        prev, nxt = w[i - 1] if i else "_", w[i + 1] if i + 1 < len(w) else "_"  # "_": never in a letter set
        if c == prev and c != "C":
            continue
        if c in VOWELS:
            out.append(c if i == 0 else "")
        elif c == "B":
            out.append("" if prev == "M" and nxt == "_" else "B")
        elif c == "C":
            out.append("K" if nxt == "H" and (w[i + 2:i + 3] in ("R", "L") or prev == "S") else "X" if nxt == "H" else "S" if nxt in set("IEY") else "K")
        elif c == "D":
            out.append("J" if nxt == "G" and w[i + 2:i + 3] in ("I", "E", "Y") else "T")
        elif c == "G":
            if nxt == "H" and w[i + 2:i + 3] not in VOWELS | {""} or (nxt == "H" and i + 2 >= len(w)):
                out.append("")  # "night", "though"
            elif nxt == "N" and i + 2 >= len(w):
                out.append("")  # "sign"
            else:
                out.append("J" if nxt in set("IEY") and prev != "G" else "K")
        elif c == "H":
            out.append("H" if nxt in VOWELS and prev not in set("CSPTG") else "")
        elif c == "K":
            out.append("" if prev == "C" else "K")
        elif c == "P":
            out.append("F" if nxt == "H" else "P")
        elif c == "Q":
            out.append("K")
        elif c == "S":
            out.append("X" if nxt == "H" or (nxt == "I" and w[i + 2:i + 3] in ("O", "A")) else "S")
        elif c == "T":
            out.append("0" if nxt == "H" else "X" if nxt == "I" and w[i + 2:i + 3] in ("O", "A") else "T")
        elif c == "V":
            out.append("F")
        elif c in "WY":
            out.append(c if nxt in VOWELS else "")
        elif c == "X":
            out.append("KS")
        elif c == "Z":
            out.append("S")
        else:
            out.append(c)
    return "".join(out)


def phrase_code(text: str) -> str:
    return " ".join(code(w) for w in text.split())


def sound_alike(a: str, b: str) -> float:
    """0..1, how alike two words / phrases sound: 'hackle'/'hacker' 0.75, 'tab'/'tap' 0.83, 'milk'/'chrome' 0.1."""
    a, b = a.lower().strip(), b.lower().strip()
    if a == b:
        return 1.0
    spelling = difflib.SequenceMatcher(None, a, b).ratio()
    sound = difflib.SequenceMatcher(None, phrase_code(a), phrase_code(b)).ratio()
    return (spelling + sound) / 2
