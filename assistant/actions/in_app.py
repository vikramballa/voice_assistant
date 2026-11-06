"""Working inside the app the user is using: type text, press keys and shortcuts, scroll.

"The app the user is using" is never the terminal Laya runs in (see mac.accessibility.target_app).
Needs the Accessibility permission.
"""

import plistlib
import re
import subprocess
import time

from .errors import Failed
from .macos import accessibility, osascript, run
from .web import BROWSERS, browser_app

KEY_CODES = {
    "enter": 36, "return": 36, "tab": 48, "space": 49, "delete": 51, "backspace": 51, "escape": 53, "esc": 53,
    "left": 123, "right": 124, "down": 125, "up": 126, "pagedown": 121, "page down": 121, "pageup": 116,
    "page up": 116, "home": 115, "end": 119, "forwarddelete": 117,
    # typed as characters, "=" and "-" don't reach apps as the zoom shortcuts; the physical keys do
    "=": 24, "+": 24, "-": 27, "`": 50, "[": 33, "]": 30,
    **{f"f{i}": code for i, code in enumerate((122, 120, 99, 118, 96, 97, 98, 100, 101, 109, 103, 111), 1)},
}
MODIFIERS = {"cmd": "command", "command": "command", "ctrl": "control", "control": "control",
             "option": "option", "alt": "option", "opt": "option", "shift": "shift", "fn": "function"}
SCROLL_SCREENS = {"bit": 0.3, "little": 0.3, "": 0.8, "more": 0.8, "page": 0.8, "lot": 2.5}  # amount -> screens


TEXT_ROLES = {"AXTextField", "AXTextArea", "AXComboBox", "AXSearchField"}


def type_text(text: str, app_name: str = "") -> str:
    """Type where the user works; with `app_name` ("Notes"), into that app even if another one is in front."""
    ax = accessibility()
    app = (ax.running_app(app_name) if app_name else None) or ax.target_app()
    ax.activate(app)  # the user's app, not the terminal Laya runs in (no-op if already in front)
    _find_text_box(ax, app)
    _keystroke(text)
    return "Typed"


def _keystroke(text: str) -> None:
    """Types the text; a line break is the Return key ("new line" while dictating)."""
    if "\n" not in text:
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        osascript(f'tell application "System Events" to keystroke "{escaped}"')
        return
    lines = []
    for i, part in enumerate(text.split("\n")):
        if i:
            lines.append("key code 36")  # Return
        if part:
            escaped = part.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'keystroke "{escaped}"')
    if lines:
        osascript('tell application "System Events"\n' + "\n".join(lines) + "\nend tell")


def _find_text_box(ax, app) -> None:
    """In a web page with no text box selected, typing goes nowhere: put the cursor in the page's search
    box (or first text box). The browser's own address bar counts as selected. Raises Failed when the page
    has no box at all; does nothing when it can't tell (other apps, scripts turned off)."""
    from assistant.mac import web_page
    if not web_page.is_browser(app):
        return
    focused = ax.attr(ax.app_element(app), "AXFocusedUIElement")
    if focused is not None and ax.attr(focused, "AXRole") in TEXT_ROLES:
        return  # the address bar, or a page box the browser does show
    try:
        if web_page.focus_text_box(app.localizedName()) is None:
            raise Failed("There's no text box on this page to type into")
    except (web_page.ScriptsOff, RuntimeError, OSError, subprocess.TimeoutExpired):
        pass


def clear_text(what: str = "all") -> str:
    """"remove that", "clear the chat box": empty the text box the user is working in ("word": only the
    last word). In a web page, the box being typed in or the one with text in it (ChatGPT's message box).
    Never touches anything but text: with no text box, it says so."""
    from assistant.mac import web_page
    ax = accessibility()
    app = ax.target_app()
    ax.activate(app)
    keys = "alt+delete" if what == "word" else "cmd+a delete"
    if web_page.is_browser(app):
        try:
            if what == "word":
                if web_page.focus_text_box(app.localizedName()) is None:
                    raise Failed("There's no text box on this page")
            else:
                had = web_page.clear_text_box(app.localizedName())
                if had is None:
                    raise Failed("There's nothing typed here to clear")
                return "Cleared. Say undo to bring it back"
        except (web_page.ScriptsOff, RuntimeError, OSError, subprocess.TimeoutExpired):
            pass  # can't look inside the page: keys into whatever box has the cursor (harmless outside a box)
    else:
        focused = ax.attr(ax.app_element(app), "AXFocusedUIElement") if app is not None else None
        if focused is None or ax.attr(focused, "AXRole") not in TEXT_ROLES:
            raise Failed("There's no text box selected to clear. Click in it first")
    press_keys(keys)
    return "Cleared. Say undo to bring it back" if what != "word" else "Deleted the last word"


def write_note(text: str = "") -> str:
    """A new note in Apple Notes with this text ("" = an empty new note, ready for type_text)."""
    from .apps import wait_for_app
    run(["open", "-a", "Notes"])
    wait_for_app("Notes")
    press_keys("cmd+n")
    time.sleep(0.4)  # the new note's body takes the cursor
    if text:
        _keystroke(text)
        return "Written in a new note"
    return "New note"


# Menu commands a key press must never reach. People remap keys in System Settings > Keyboard > Keyboard
# Shortcuts > App Shortcuts: on the user's Mac cmd+T was "Shut Down…" (2026-10-03), so "new tab" opened the
# shut-down dialog.
DANGEROUS_MENU = re.compile(r"shut\s*down|restart|log\s*out|sleep|power\s*off|lock\s+screen|erase|empty\s+(?:the\s+)?trash"
                            r"|delete|quit", re.I)
MENU_MODIFIERS = {"@": "command", "^": "control", "~": "option", "$": "shift"}


def press_keys(keys: str, browser: str = "") -> str:
    """"cmd+t", "enter", "cmd+shift+n", "escape"; several combos separated by spaces or commas. They go to the
    app the user is working in; with a browser ("new tab"), to that browser unless a browser is in front.
    A new tab in a browser is made through the browser itself, not cmd+T (that key can be remapped)."""
    lines = _key_script(keys)
    ax = accessibility()
    app = ax.target_app()
    if browser and (app is None or app.localizedName() not in BROWSERS.values()):
        name = browser_app(browser)
        app = ax.running_app(name)
        if app is None:
            run(["open", "-a", name])
            app = ax.running_app(name)
    if _combos(keys) == [("command", "t")] and app is not None and app.localizedName() in BROWSERS.values():
        _new_tab(app.localizedName())
        ax.activate(app)
        return "New tab"
    _refuse_remapped(keys, app)
    ax.activate(app)
    osascript('tell application "System Events"\n' + "\ndelay 0.05\n".join(lines) + "\nend tell")
    return "Done"


def _new_tab(app_name: str) -> None:
    """A new tab in the browser's front window (a new window if it has none), with the address bar ready."""
    if app_name == "Safari":
        osascript('tell application "Safari"\nif (count of windows) = 0 then make new document\n'
                  'tell front window to set current tab to (make new tab)\nend tell')
    else:
        osascript(f'tell application "{app_name}"\nif (count of windows) = 0 then\nmake new window\nelse\n'
                  'make new tab at end of tabs of front window\nend if\nend tell')


def _combos(keys: str) -> list[tuple[str, ...]]:
    """"cmd+shift+t enter" -> [("command", "shift", "t"), ("enter",)]: modifiers sorted, then the key."""
    keys = re.sub(r"\s*\+\s*", "+", keys.strip().lower())
    out = []
    for combo in re.split(r"[\s,]+", keys):
        parts = [p for p in combo.split("+") if p]
        if parts:
            out.append(tuple(sorted(MODIFIERS.get(m, m) for m in parts[:-1])) + (parts[-1],))
    return out


def custom_shortcuts(bundle_id: str | None = None) -> dict[tuple[str, ...], str]:
    """The user's own key remaps (all apps + this app): {("command", "t"): "Shut Down…"}."""
    found = {}
    for domain in ["-g"] + ([bundle_id] if bundle_id else []):
        out = subprocess.run(["defaults", "export", domain, "-"], capture_output=True)
        if out.returncode:
            continue
        try:
            remaps = plistlib.loads(out.stdout).get("NSUserKeyEquivalents") or {}
        except Exception:
            continue
        for title, key in remaps.items():
            mods = tuple(sorted(MENU_MODIFIERS[c] for c in key[:-1] if c in MENU_MODIFIERS))
            if key:
                found[mods + (key[-1].lower(),)] = title
    return found


def _refuse_remapped(keys: str, app) -> None:
    """Don't press a key the user has set to shut down / restart / sleep... (Failed says which)."""
    if not any(len(c) > 1 for c in _combos(keys)):
        return  # plain keys ("enter") can't be menu shortcuts
    remaps = custom_shortcuts(app.bundleIdentifier() if app is not None else None)
    for combo in _combos(keys):
        title = remaps.get(combo)
        if title and DANGEROUS_MENU.search(title):
            raise Failed(f"On this Mac {'+'.join(combo)} is set to {title.rstrip('…. ')}, so I didn't press it")


def _key_script(keys: str) -> list[str]:
    """"cmd + shift + n, enter" -> one AppleScript line per key combo."""
    lines = []
    keys = re.sub(r"\s*\+\s*", "+", keys.strip().lower()).replace("page down", "pagedown").replace("page up", "pageup")
    for combo in re.split(r"[\s,]+", keys):
        parts = [p.strip() for p in combo.split("+") if p.strip()]
        if not parts:
            continue
        *mods, key = parts
        if any(m not in MODIFIERS for m in mods) or (key not in KEY_CODES and len(key) != 1):
            raise Failed(f"I don't know the key {combo}")
        using = f" using {{{', '.join(MODIFIERS[m] + ' down' for m in mods)}}}" if mods else ""
        press = f"key code {KEY_CODES[key]}" if key in KEY_CODES else f'keystroke "{key.replace(chr(34), "")}"'
        lines.append(f"{press}{using}")
    if not lines:
        raise Failed("Which key?")
    return lines


def scroll(direction: str = "down", amount: str = "") -> str:
    """Scroll the app the user is working in: up / down / left / right by a screen ("bit", "lot" or a number
    of screens), or "top" / "bottom"."""
    ax = accessibility()
    direction = direction.lower().strip()
    if direction in ("top", "bottom"):
        press_keys("cmd+up" if direction == "top" else "cmd+down")
        return f"Scrolled to the {direction}"
    app = ax.target_app()
    if app is None:
        raise Failed("I can't tell which app to scroll")
    if ax.activate(app):
        time.sleep(ax.SETTLE_SECONDS - 0.25)  # scroll events during the slide-in animation are lost
    _, bounds = ax.window_of(app)
    if bounds is None:
        raise Failed(f"{app.localizedName()} has no window to scroll")
    amount = str(amount).lower().strip()
    screens = float(amount) if re.fullmatch(r"\d+(?:\.\d+)?", amount) else SCROLL_SCREENS.get(amount, 0.8)
    horizontal = direction in ("left", "right")
    pixels = screens * (bounds[2] if horizontal else bounds[3]) * (-1 if direction in ("up", "left") else 1)
    center = (bounds[0] + bounds[2] / 2, bounds[1] + bounds[3] / 2)  # scroll events go to the window under them
    if horizontal:
        ax.scroll(center, right=pixels)
    else:
        ax.scroll(center, down=pixels)
    return f"Scrolled {direction}"
