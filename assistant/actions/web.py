"""Browsers: open a website, search Google, search YouTube. Searches jump straight to the results page."""

import difflib
import re
import subprocess
from urllib.parse import quote_plus, urlparse

from .errors import Failed
from .macos import osascript, run

BROWSERS = {"chrome": "Google Chrome", "safari": "Safari", "brave": "Brave Browser"}  # short name -> app
DEFAULT_BROWSER = "chrome"


def browser_app(browser: str) -> str:
    """"brave", "Brave Browser", "google chrome" -> the app name; unknown -> Chrome."""
    key = browser.lower().replace(" browser", "").replace("google ", "")
    return BROWSERS.get(key, BROWSERS[DEFAULT_BROWSER])


def open_url(url: str, browser: str = DEFAULT_BROWSER) -> None:
    run(["open", "-a", browser_app(browser), url])


def web_search(query: str, browser: str = DEFAULT_BROWSER) -> str:
    open_url(f"https://www.google.com/search?q={quote_plus(query)}", browser)
    return f"Searching Google for {query}"


def youtube_search(query: str, browser: str = DEFAULT_BROWSER) -> str:
    open_url(f"https://www.youtube.com/results?search_query={quote_plus(query)}", browser)
    return f"Searching YouTube for {query}"


def open_website(url: str, browser: str = DEFAULT_BROWSER) -> str:
    if "://" not in url:
        url = "https://" + url
    open_url(url, browser)
    return "Opening website"


# --- tabs that are already open ----------------------------------------------------------------------
# AppleScript, no page JavaScript needed. Only asked when the browser is running ("tell" would launch it).
TABS_SCRIPT = '''if application "{app}" is running then
  set out to ""
  tell application "{app}"
    repeat with w from 1 to (count windows)
      repeat with i from 1 to (count tabs of window w)
        set t to tab i of window w
        set out to out & w & (ASCII character 9) & i & (ASCII character 9) & ({title} of t) & (ASCII character 9) & (URL of t) & linefeed
      end repeat
    end repeat
  end tell
  return out
end if'''
TAB_MATCH = 0.75  # how well the spoken name must match a tab's title or address


def browser_tabs(browser: str) -> list[tuple[int, int, str, str]]:
    """(window, tab, title, url) of every open tab, front window first; [] if the browser isn't running."""
    app = browser_app(browser)
    script = TABS_SCRIPT.format(app=app, title="name" if app == "Safari" else "title")
    try:
        out = osascript(script, timeout=5)
    except (Failed, subprocess.CalledProcessError):
        return []
    tabs = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) == 4 and parts[0].isdigit() and parts[1].isdigit():
            tabs.append((int(parts[0]), int(parts[1]), parts[2], parts[3]))
    return tabs


def tab_score(name: str, title: str, url: str) -> float:
    """How well "chatgpt" / "my youtube" names a tab: 1.0 when the words are in its title or address."""
    words = [w for w in re.findall(r"[a-z0-9]+", name.lower()) if w not in ("my", "the", "a", "tab", "page", "website")]
    if not words:
        return 0.0
    host = re.sub(r"^www\.", "", urlparse(url).netloc.lower())
    title_words = re.findall(r"[a-z0-9]+", title.lower())
    joined = "".join(words)
    if joined in host.replace("-", "") or all(w in title_words or w in host for w in words) \
            or joined in "".join(title_words):
        return 1.0
    site = host.split(".")[-2] if host.count(".") >= 1 else host
    candidates = [site] + title_words + [" ".join(title_words[i:i + len(words)]) for i in range(len(title_words))]
    return max(difflib.SequenceMatcher(None, " ".join(words), c).ratio() for c in candidates if c)


def find_tab(name: str, browser: str) -> tuple[int, int, str] | None:
    """(window, tab, title) of the open tab that `name` means, or None."""
    best, found = 0.0, None
    for w, i, title, url in browser_tabs(browser):
        score = tab_score(name, title, url)
        if score > best:
            best, found = score, (w, i, title)
    return found if best >= TAB_MATCH else None


def switch_tab(name: str, browser: str = DEFAULT_BROWSER) -> str:
    """"go to the ChatGPT tab": bring that open tab (and its window) to the front."""
    found = find_tab(name, browser)
    if not found:
        raise Failed(f"I don't see a {name} tab in {browser_app(browser)}")
    w, i, title = found
    app = browser_app(browser)
    select = (f"set current tab of window {w} to tab {i} of window {w}" if app == "Safari"
              else f"set active tab index of window {w} to {i}")
    osascript(f'tell application "{app}"\n{select}\nset index of window {w} to 1\nactivate\nend tell', timeout=5)
    return f"Switched to {title.split(' | ')[0].split(' - ')[0][:40]}"
