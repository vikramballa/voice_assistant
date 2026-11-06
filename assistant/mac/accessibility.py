"""Working inside other apps through the macOS Accessibility API: find a button / link / menu item
by its name and press it, click at a point, scroll. Needs the Accessibility permission for the app
Laya runs in (Terminal, an IDE...), which this module calls the "host".

Test from the terminal:
    python -m assistant.mac.accessibility                  # list the named elements of the frontmost app
    python -m assistant.mac.accessibility "Show Sidebar"   # press it
"""

import difflib
import os
import re
import subprocess
import sys
import time

import ApplicationServices as AX
import Quartz
from AppKit import NSRunningApplication, NSWorkspace

MAX_ELEMENTS = 4000  # big web pages have more; the search stops here
MAX_SECONDS = 2.0
SETTLE_SECONDS = 0.8  # after bringing an app forward, its window slides in for this long (Stage Manager)
CHROMIUM = ("Google Chrome", "Brave Browser", "Microsoft Edge", "Arc", "Chromium", "Vivaldi", "Opera")
# Roles a person would click on; plain text is pressed through its clickable parent.
CLICKABLE = {"AXButton", "AXLink", "AXMenuItem", "AXMenuBarItem", "AXCheckBox", "AXRadioButton", "AXPopUpButton",
             "AXMenuButton", "AXTab", "AXCell", "AXRow", "AXDisclosureTriangle", "AXTextField", "AXSearchField",
             "AXComboBox", "AXTextArea", "AXImage", "AXSwitch", "AXToggle"}
_chromium_ready: set[int] = set()


def trusted() -> bool:
    return bool(AX.AXIsProcessTrusted())


def front_app():
    """The app the user is looking at (NSRunningApplication): the owner of the topmost normal window.
    (NSWorkspace's frontmostApplication only updates inside an app's event loop, and Accessibility's focused
    app was once seen a step behind right after a switch; the window list was always right.)"""
    wins = Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionOnScreenOnly |
                                             Quartz.kCGWindowListExcludeDesktopElements, Quartz.kCGNullWindowID)
    for w in wins or []:
        if w.get("kCGWindowLayer") != 0:
            continue
        app = NSRunningApplication.runningApplicationWithProcessIdentifier_(w.get("kCGWindowOwnerPID", 0))
        if app is not None and app.activationPolicy() == 0:  # skips Stage Manager's strip, our island, etc.
            return app
    focused = attr(AX.AXUIElementCreateSystemWide(), "AXFocusedApplication")
    if focused is not None:
        err, pid = AX.AXUIElementGetPid(focused, None)
        app = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid) if err == 0 else None
        if app is not None:
            return app
    return NSWorkspace.sharedWorkspace().frontmostApplication()


_host = None          # the app Laya runs in (Terminal, an IDE...): never the target of clicks and keys
_last_target = None   # the last other app the user was working in


def host_app():
    """The app this process belongs to: first regular app up the parent-process chain."""
    global _host
    if _host is None:
        pid = os.getppid()
        while pid > 1:
            app = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
            if app is not None and app.activationPolicy() == 0:  # a regular app with a Dock icon
                _host = app
                break
            out = subprocess.run(["ps", "-o", "ppid=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
            pid = int(out) if out.isdigit() else 1
    return _host


def host_name() -> str:
    app = host_app()
    return app.localizedName() if app is not None else "your terminal app"


def is_host(app) -> bool:
    host = host_app()
    return app is not None and host is not None and app.processIdentifier() == host.processIdentifier()


def note_front_app() -> None:
    """Remember the app the user is working in (called when a command arrives and after acting)."""
    global _last_target
    app = front_app()
    if app is not None and not is_host(app) and app.activationPolicy() == 0:
        _last_target = app


def target_app():
    """Where clicks, keys and scrolling go: the app in front, but never the terminal running Laya
    (then the app used before it)."""
    app = front_app()
    if is_host(app) and _last_target is not None and not _last_target.isTerminated():
        return _last_target
    return app


def activate(app, timeout: float = 1.5) -> bool:
    """Bring the app to the front (with Stage Manager, apps in the side strip aren't even on screen).
    True if it had to switch. Keys work 0.25 s after the switch; scrolling needs the window to finish
    sliding in (SETTLE_SECONDS, measured 0.8 s)."""
    if app is None or _pid(front_app()) == app.processIdentifier():
        return False
    # activateWithOptions_ is refused for a background process on recent macOS; Launch Services ("open") is allowed.
    bundle = app.bundleIdentifier()
    subprocess.run(["open", "-b", bundle] if bundle else ["open", "-a", app.bundleURL().path()], capture_output=True)
    deadline = time.time() + timeout
    while time.time() < deadline and _pid(front_app()) != app.processIdentifier():
        time.sleep(0.05)
    time.sleep(0.25)  # window animation
    return True


def _pid(app) -> int | None:
    return app.processIdentifier() if app is not None else None


def window_of(app):
    """(AX window element, (x, y, width, height) in screen points) of the app's main window, or (None, None)."""
    root = app_element(app)
    win = attr(root, "AXFocusedWindow") or (attr(root, "AXWindows") or [None])[0]
    position, size = (attr(win, "AXPosition"), attr(win, "AXSize")) if win is not None else (None, None)
    if position is None or size is None:
        return None, None
    _, p = AX.AXValueGetValue(position, AX.kAXValueCGPointType, None)
    _, s = AX.AXValueGetValue(size, AX.kAXValueCGSizeType, None)
    return win, (p.x, p.y, s.width, s.height)


def window_number(app, bounds) -> int | None:
    """The window server's number for the app's window with these bounds (for screencapture -l)."""
    wins = Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionOnScreenOnly |
                                             Quartz.kCGWindowListExcludeDesktopElements, Quartz.kCGNullWindowID)
    best = None
    for w in wins:
        if w.get("kCGWindowOwnerPID") != app.processIdentifier() or w.get("kCGWindowLayer") != 0:
            continue
        b = w["kCGWindowBounds"]
        gap = abs(b["X"] - bounds[0]) + abs(b["Y"] - bounds[1]) + abs(b["Width"] - bounds[2])
        if best is None or gap < best[0]:
            best = (gap, w["kCGWindowNumber"])
    return best[1] if best else None


def running_app(name: str):
    for app in NSWorkspace.sharedWorkspace().runningApplications():
        if (app.localizedName() or "").lower() == name.lower():
            return app
    return None


def attr(element, name: str):
    err, value = AX.AXUIElementCopyAttributeValue(element, name, None)
    return value if err == 0 else None


def app_element(app):
    element = AX.AXUIElementCreateApplication(app.processIdentifier())
    if app.localizedName() in CHROMIUM and app.processIdentifier() not in _chromium_ready:
        # Chrome/Brave only build the web page's accessibility tree when asked to.
        AX.AXUIElementSetAttributeValue(element, "AXManualAccessibility", True)
        _chromium_ready.add(app.processIdentifier())
        time.sleep(0.3)
    return element


def label(element) -> str:
    """What a person would call the element: its title, description or text."""
    for name in ("AXTitle", "AXDescription", "AXValue", "AXPlaceholderValue", "AXHelp"):
        value = attr(element, name)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    return ""


def elements(roots):
    """(element, role, label) for everything under each root in turn (breadth first: the visible,
    top-level things come early). Stops after MAX_ELEMENTS / MAX_SECONDS."""
    start = time.perf_counter()
    seen = 0
    for root in roots:
        queue = [root]
        while queue and seen < MAX_ELEMENTS and time.perf_counter() - start < MAX_SECONDS:
            element = queue.pop(0)
            seen += 1
            role = attr(element, "AXRole") or ""
            yield element, role, label(element)
            queue.extend(attr(element, "AXChildren") or [])


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def _score(want: str, have: str) -> float:
    if not have:
        return 0.0
    if have == want:
        return 1.0
    if re.search(rf"\b{re.escape(want)}\b", have):
        return 0.9 - min(0.3, (len(have) - len(want)) / 200)  # "sign in" inside "sign in to your account"
    return difflib.SequenceMatcher(None, want, have).ratio() * 0.85


def find(name: str, app=None):
    """The best matching element of the app (default: the frontmost one), or None."""
    app = app or front_app()
    root = app_element(app)
    windows = attr(root, "AXWindows") or []
    focused = attr(root, "AXFocusedWindow")
    if focused is not None:
        windows = [focused] + [w for w in windows if w != focused]
    want = _norm(name)
    best, best_score = None, 0.0
    menu_bar = attr(root, "AXMenuBar")
    for element, role, text in elements(windows + ([menu_bar] if menu_bar is not None else [])):
        if role in ("AXWindow", "AXApplication") or not text:
            continue
        score = _score(want, _norm(text)) + (0.05 if role in CLICKABLE else 0)
        if score > best_score:
            best, best_score = element, score
            if score >= 1.05:
                break  # exact name on a clickable thing: stop looking
    return best if best_score >= 0.75 else None


def _in_chromium(element) -> bool:
    err, pid = AX.AXUIElementGetPid(element, None)
    app = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid) if err == 0 else None
    return app is not None and app.localizedName() in CHROMIUM


def press(element) -> None:
    """Click it: its own "press" action, else its clickable parent's, else a real mouse click.
    In Chrome/Brave pages "press" is accepted but does nothing on links, so those get a real click."""
    if _in_chromium(element) and attr(element, "AXRole") not in ("AXMenuItem", "AXMenuBarItem"):
        _, size = AX.AXValueGetValue(attr(element, "AXSize"), AX.kAXValueCGSizeType, None)
        if size.height < 1:  # off screen (Chrome reports it squashed at the edge): bring it into view first
            AX.AXUIElementPerformAction(element, "AXScrollToVisible")
            time.sleep(0.3)
        click_at(*center(element))
        return
    node = element
    for _ in range(4):
        err, actions = AX.AXUIElementCopyActionNames(node, None)
        if err == 0 and actions and "AXPress" in actions:
            if AX.AXUIElementPerformAction(node, "AXPress") == 0:
                return
        node = attr(node, "AXParent")
        if node is None:
            break
    click_at(*center(element))


def center(element) -> tuple[float, float]:
    pos, size = attr(element, "AXPosition"), attr(element, "AXSize")
    _, p = AX.AXValueGetValue(pos, AX.kAXValueCGPointType, None)
    _, s = AX.AXValueGetValue(size, AX.kAXValueCGSizeType, None)
    return p.x + s.width / 2, p.y + s.height / 2


def pointer() -> tuple[float, float]:
    """Where the mouse pointer is (screen points)."""
    at = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
    return at.x, at.y


def click_at(x: float, y: float, clicks: int = 1) -> None:
    """A real left click (clicks=2: a double click) at this screen point; the pointer is put back afterwards."""
    before = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
    for count in range(1, clicks + 1):
        for kind in (Quartz.kCGEventLeftMouseDown, Quartz.kCGEventLeftMouseUp):
            event = Quartz.CGEventCreateMouseEvent(None, kind, (x, y), Quartz.kCGMouseButtonLeft)
            Quartz.CGEventSetIntegerValueField(event, Quartz.kCGMouseEventClickState, count)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
            time.sleep(0.05)
    time.sleep(0.1)
    Quartz.CGWarpMouseCursorPosition(before)


def scroll(at: tuple[float, float], down: float = 0, right: float = 0) -> None:
    """Scroll the window at this screen point by pixels (positive = down / right). Scroll events go to the
    window under their location, so they are aimed at the window's centre; the pointer is put back after."""
    before = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
    steps = max(1, int(max(abs(down), abs(right)) // 400))  # big jumps in a few pieces: pages load lazily
    for _ in range(steps):
        if right:  # two-axis events are ignored by Chrome/Brave for plain vertical scrolling: only use them sideways
            event = Quartz.CGEventCreateScrollWheelEvent(None, Quartz.kCGScrollEventUnitPixel, 2, 0, int(-right / steps))
        else:
            event = Quartz.CGEventCreateScrollWheelEvent(None, Quartz.kCGScrollEventUnitPixel, 1, int(-down / steps))
        Quartz.CGEventSetLocation(event, at)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
        time.sleep(0.03)
    Quartz.CGWarpMouseCursorPosition(before)


if __name__ == "__main__":
    time.sleep(1)  # time to switch to the app to look at
    if len(sys.argv) > 1:
        target = find(" ".join(sys.argv[1:]))
        print("not found" if target is None else f"pressing {label(target)!r}")
        if target is not None:
            press(target)
    else:
        root = app_element(front_app())
        for _, role, text in elements([root]):
            if text:
                print(f"{role:18} {text[:80]}")
