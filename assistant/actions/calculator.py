"""Arithmetic: work it out exactly, say it, and press the real buttons if Calculator.app is open.

It remembers the last result, so "multiply that by 4" (expression "*4") goes on from it,
and the last calculation, so "do it in the calculator" (expression "again") repeats it.
"""

import ast
import operator
import re
import time

from assistant.mac.permissions import accessibility_message

from .errors import Failed, NeedsPermission

# Calculator.app buttons by their accessibility identifier.
CALC_KEYS = {**{str(i): n for i, n in enumerate(("Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven",
                                                   "Eight", "Nine"))},
             ".": "Decimal", "+": "Add", "-": "Subtract", "*": "Multiply", "/": "Divide", "^": "Power",
             "(": "OpenParenthesis", ")": "CloseParenthesis", "=": "Equals", "clear": "AllClear"}
OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
             ast.Pow: operator.pow, ast.Mod: operator.mod, ast.USub: operator.neg, ast.UAdd: operator.pos}
SPOKEN = {"+": " plus ", "-": " minus ", "*": " times ", "/": " divided by ", "^": " to the power of ",
          "%": " modulo ", "(": "", ")": ""}

_last_result: float | None = None
_last_expression: str | None = None


def last_result() -> float | None:
    """The answer of the last calculation (None before the first one)."""
    return _last_result


def calculate(expression: str) -> str:
    """Work out plain arithmetic ("25*25"; "+5" / "-4" go on from the last result; "again" repeats the last
    calculation, e.g. to show it in the Calculator). If the Calculator app is open, its buttons are pressed too."""
    global _last_result, _last_expression
    if expression.strip().lower() == "again":
        if _last_expression is None:
            raise Failed("What should I calculate?")
        expression = _last_expression
    expr = re.sub(r"\s+", "", expression.replace("×", "*").replace("÷", "/").replace("x", "*"))
    # "+5", "*3", and "-4" when there is an answer to go on from ("minus 4")
    follow_on = bool(re.match(r"[+*/^]", expr)) or (expr.startswith("-") and _last_result is not None)
    if follow_on and _last_result is None:
        raise Failed("Calculate what?")
    full = f"{_as_operand(_last_result)}{expr}" if follow_on else expr
    try:
        value = _evaluate(full)
    except ZeroDivisionError:
        raise Failed("You can't divide by zero") from None
    except (ValueError, SyntaxError, TypeError, OverflowError):
        raise Failed(f"I couldn't work out {expression}") from None
    _last_result, _last_expression = value, full
    said = "That makes" if follow_on else _spoken(expr) + " is"
    answer = f"{said} {_number_words(value)}"
    try:
        shown = _calculator_press(expr, fresh=not follow_on)
    except NeedsPermission:
        raise NeedsPermission(f"{answer}. To press the Calculator buttons, {accessibility_message()}") from None
    if shown and shown.replace(",", "").replace(".", "", 1).lstrip("-").isdigit():
        answer = f"{said} {shown}"  # what the Calculator shows, in case it rounds differently
    return answer


def _evaluate(expression: str) -> float:
    """Plain arithmetic only (+ - * / ^ % and brackets); anything else raises. Never eval()."""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("too big")
            return OPERATORS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in OPERATORS:
            return OPERATORS[type(node.op)](ev(node.operand))
        raise ValueError("not arithmetic")
    return ev(ast.parse(expression.replace("^", "**"), mode="eval").body)


def _number_words(value: float) -> str:
    value = round(value, 10)  # 0.1+0.2 -> 0.3, (1/3)*3 -> 1
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.6g}"


def _as_operand(value: float) -> str:
    """The last result to go on from, exactly: "625" (as the Calculator buttons can type it), or "(-5)" /
    "(0.3333333333333333)" when rounding or a leading minus would change the answer."""
    return _number_words(value) if value >= 0 and value == int(value) and value < 1e15 else f"({value!r})"


def _spoken(expression: str) -> str:
    root = re.fullmatch(r"(\d+(?:\.\d+)?)\^0\.5", expression)
    if root:
        return f"the square root of {root.group(1)}"
    return "".join(SPOKEN.get(c, c) for c in expression).strip()


def _calculator_press(expression: str, fresh: bool) -> str | None:
    """Press the keys in Calculator.app if it's open; its display afterwards, or None.
    Raises NeedsPermission when it's open but macOS won't let us press its buttons."""
    from assistant.mac import accessibility as ax
    app = ax.running_app("Calculator")
    if app is None:
        return None
    if not ax.trusted():
        raise NeedsPermission(accessibility_message())
    buttons = {}
    display = None
    for element, role, text in ax.elements([ax.app_element(app)]):
        ident = ax.attr(element, "AXIdentifier") if role == "AXButton" else None
        if ident:
            buttons[ident] = element
        elif role == "AXStaticText" and text and ax.attr(element, "AXDescription") == "Edit field":
            display = element
    keys = (["clear"] if fresh else []) + [c for c in expression if c != " "] + ["="]
    if any(CALC_KEYS.get(k) not in buttons for k in keys):
        return None  # e.g. brackets in the basic mode: just say the answer
    for k in keys:
        ax.press(buttons[CALC_KEYS[k]])
    if display is None:
        return None
    time.sleep(0.05)
    value = ax.attr(display, "AXValue")  # sometimes words ("two point five"): then our own result is said
    return value if isinstance(value, str) and value.strip() else None
