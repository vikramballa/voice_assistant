"""Spoken arithmetic -> an expression: "what is 25 into 25" -> "25*25", "multiply that by 3" -> "*3".

Arithmetic is exact, so no model is needed for it: the rules here decide, and actions.calculator computes.
"""

import re

from .words import normalize

_UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                     "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if i > 1}
_SCALES = {"hundred": 100, "thousand": 1000, "lakh": 100000, "million": 10**6, "crore": 10**7, "billion": 10**9}

MATH_PREFIX = re.compile(r"^(?:(?:in|on|using|with)\s+(?:the\s+|my\s+)?calculator\s+)?"
                         r"(?:(?:please|can\s+you|could\s+you|now|ok(?:ay)?|so)\s+)*"
                         r"(?:(?:what\s+is|what's|whats|what\s+are|how\s+much\s+is|calculate|compute|solve|"
                         r"work\s+out|find|tell\s+me|do|perform|type|enter)\s+)*"
                         r"(?:(?:the\s+)?(?:value|result|answer|sum|product)\s+of\s+)?")
MATH_SUFFIX = re.compile(r"\s+(?:(?:in|on|using|with)\s+(?:the\s+|my\s+)?calculator|please|for\s+me|equals|is)$")
NUM = r"-?\d+(?:\.\d+)?"
_IT = r"(?:(?:it|that|this|the\s+(?:result|answer|number))\s+)?"

# Whole phrasings with a verb, tried first. A result starting with an operator ("+5") goes on from the last answer.
VERB_FORMS = [
    (rf"^multiply\s+({NUM})\s+(?:by|with|and|times)\s+({NUM})$", r"\1*\2"),
    (rf"^divide\s+({NUM})\s+by\s+({NUM})$", r"\1/\2"),
    (rf"^add\s+({NUM})\s+(?:and|to|plus)\s+({NUM})$", r"\1+\2"),
    (rf"^subtract\s+({NUM})\s+from\s+({NUM})$", r"\2-\1"),
    (rf"^({NUM})\s+percent\s+of\s+({NUM})$", r"\1*\2/100"),
    (rf"^({NUM})\s+squared$", r"\1^2"),
    (rf"^(?:the\s+)?square\s+root\s+of\s+({NUM})$", r"\1^0.5"),
    (rf"^half\s+of\s+({NUM})$", r"\1/2"),
    (rf"^(?:double|twice)\s+({NUM})$", r"\1*2"),
    (rf"^({NUM})\s+cubed$", r"\1^3"),
    (rf"^(?:add|plus)\s+({NUM})(?:\s+to\s+(?:it|that|this))?$", r"+\1"),
    (rf"^(?:subtract|minus|take\s+away|remove)\s+({NUM})(?:\s+from\s+(?:it|that|this))?$", r"-\1"),
    (rf"^(?:multiply\s+{_IT}(?:by\s+)?|multiplied\s+{_IT}by\s+|times\s+|into\s+)({NUM})$", r"*\1"),
    (rf"^(?:divide\s+{_IT}by\s+|divided\s+by\s+)({NUM})$", r"/\1"),
]
# Operator words between numbers -> symbols ("25 into 25" is how many Indian English speakers say "times").
OPERATORS = [(r"multiplied\s+by|times|into|x|\*", "*"), (r"divided\s+by|over|/", "/"), (r"plus|\+", "+"),
             (r"minus|-(?=\s)", "-"), (r"to\s+the\s+power\s+of|raised\s+to|\^", "^"), (r"mod(?:ulo)?", "%")]


def math_expression(text: str) -> str | None:
    """Spoken arithmetic -> "25*25"; "+5" / "*3" continue from the last answer ("add 5", "times 3").
    None when the words are anything but a calculation."""
    t = re.sub(r"(?<=\d),(?=\d{3})", "", text)  # "1,000" -> "1000"
    for symbol, words in (("×", "times"), ("*", "times"), ("÷", "divided by"), ("+", "plus"), ("=", " ")):
        t = t.replace(symbol, f" {words} ")
    t = normalize(t)
    t = re.sub(r"(\d)\s*x\s*(?=\d)", r"\1 times ", t)
    t = words_to_numbers(t)
    t = MATH_SUFFIX.sub("", MATH_PREFIX.sub("", t)).strip(" ?.")
    for _ in range(2):
        t = MATH_SUFFIX.sub("", t)
    for pattern, expr in VERB_FORMS:
        m = re.match(pattern, t)
        if m:
            return m.expand(expr)
    expr = t
    for words, symbol in OPERATORS:
        expr = re.sub(rf"(?<=[\d)\s])\s*(?:{words})\s*(?=[\d(\s-])", f" {symbol} ", f" {expr} ").strip()
    expr = re.sub(r"\s+", "", expr)
    if re.fullmatch(rf"\(?{NUM}\)?(?:[-+*/^%]\(?{NUM}\)?)+", expr):
        return expr
    return None


def words_to_numbers(text: str) -> str:
    """"twenty five times three point five" -> "25 times 3.5"."""
    out, total, current, number, decimals = [], 0, 0, False, None

    def flush():
        nonlocal total, current, number, decimals
        if number:
            out.append(str(total + current) + (f".{decimals}" if decimals else ""))
        total, current, number, decimals = 0, 0, False, None

    for word in text.split():
        parts = word.replace("-", " ").split() if "-" in word and word.replace("-", "").isalpha() else [word]
        for part in parts:
            if decimals is not None and part in _UNITS and _UNITS[part] < 10:
                decimals += str(_UNITS[part])
            elif part in _UNITS and decimals is None:
                current += _UNITS[part]
                number = True
            elif part in _TENS and decimals is None:
                current += _TENS[part]
                number = True
            elif part in _SCALES and (number or part == "hundred"):
                scale = _SCALES[part]
                if scale == 100:
                    current = max(current, 1) * 100
                else:
                    total += max(current, 1) * scale
                    current = 0
                number = True
            elif part == "point" and number and decimals is None:
                decimals = ""
            elif part in ("a", "and") and not number:
                out.append(part)
            elif part == "and" and number:
                continue  # "one hundred and five"
            else:
                flush()
                out.append(part)
    flush()
    return " ".join(out)


def mentions_calculator(text: str) -> bool:
    return bool(re.search(r"\bcalculator\b", text, re.IGNORECASE))


def calculator_redo(text: str) -> bool:
    """"do it in the calculator", "show that in calculator": the last calculation again, in the Calculator
    app (no numbers said, and not "open calculator")."""
    t = normalize(text)
    return bool(re.search(r"\bcalculator\b", t)) and not re.search(r"\d", words_to_numbers(t)) and bool(
        re.search(r"\b(?:do|perform|show|put|enter|type|display|see|proceed|it|that|this|again|same)\b", t)) \
        and not re.search(r"\b(?:open|launch|start|close|quit)\b", t)
