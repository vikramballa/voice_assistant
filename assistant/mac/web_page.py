"""Clicking and typing inside the web page in front, through the browser's own JavaScript.

Chrome and Brave (version 154 here) no longer show the page's links and buttons to macOS Accessibility,
so a click by name could only guess from a screenshot. With the browser setting
    View > Developer > Allow JavaScript from Apple Events      (Safari: Develop > Allow JavaScript from Apple Events)
turned on once, by hand (a script can't switch it), Laya finds the exact element in the page: by name
("sign in", "5D") or by position ("the second video", "the first account"), in ~0.1 s.

Test from the terminal (the browser in front):
    python -m assistant.mac.web_page "second video"
"""

import json
import re
import subprocess
import sys

CHROMIUM = ("Google Chrome", "Brave Browser", "Microsoft Edge")
BROWSERS = CHROMIUM + ("Safari",)
ORDINALS = {"first": 0, "top": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2, "fourth": 3, "4th": 3,
            "fifth": 4, "5th": 4, "sixth": 5, "6th": 5, "seventh": 6, "eighth": 7, "ninth": 8, "tenth": 9, "last": -1}
KINDS = {"video": "video", "videos": "video", "account": "account", "result": "result", "results": "result",
         "link": "link", "search result": "result", "email": "account", "profile": "account"}
NTH = re.compile(rf"^(?:the\s+)?({'|'.join(ORDINALS)})\s+(?:one\s+)?(\w+(?:\s+result)?)?$", re.I)
RANGE = re.compile(r"^(\d+)\s*([dwmy])$", re.I)  # chart ranges "5D", "1Y": also try "5 days", "1 year"
UNITS = {"d": "day", "w": "week", "m": "month", "y": "year"}


class ScriptsOff(Exception):
    """The browser refuses JavaScript from Apple Events (the setting is off)."""

    def __init__(self, app_name: str):
        menu = "Develop" if app_name == "Safari" else "View, Developer"
        super().__init__(f"To click on web pages, turn on {menu}, Allow JavaScript from Apple Events, "
                         f"in {app_name}'s menu bar")


# The page side. `a` is {mode: "nth", n, kind} | {mode: "name", names: [...]} | {mode: "focus"} | {mode: "clear"}
# | {mode: "text", limit} | {mode: "media", what: "pause" | "play" | ""}.
# Answers JSON: {ok, label} / {ok: false, count}. With a.dry nothing is clicked (tests).
PAGE_SCRIPT = r"""
(function (a) {
  var vis = function (e) {
    var r = e.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) return false;
    var s = getComputedStyle(e);
    return s.visibility != 'hidden' && s.display != 'none' && s.opacity != '0';
  };
  var onScreen = function (e) {
    var r = e.getBoundingClientRect();
    return r.bottom > 0 && r.top < innerHeight && r.right > 0 && r.left < innerWidth;
  };
  var txt = function (e) {
    return (e.getAttribute('aria-label') || e.innerText || e.value || e.title || e.getAttribute('placeholder') ||
            e.getAttribute('alt') || '').replace(/\s+/g, ' ').trim();
  };
  var norm = function (s) { return s.toLowerCase().replace(/[^\w\s@.]/g, ' ').replace(/\s+/g, ' ').trim(); };
  var all = function (sel) { return Array.prototype.slice.call(document.querySelectorAll(sel)); };
  var reading = function (list) {
    return list.sort(function (p, q) {
      var r = p.getBoundingClientRect(), s = q.getBoundingClientRect();
      return Math.abs(r.top - s.top) > 24 ? r.top - s.top : r.left - s.left;
    });
  };
  var byHref = function (list) {
    var seen = {};
    return list.filter(function (e) { var h = e.href || txt(e); if (seen[h]) return false; seen[h] = 1; return true; });
  };
  var outermostOnly = function (list) {
    return list.filter(function (e) { return !list.some(function (o) { return o !== e && o.contains(e); }); });
  };
  var editable = function (e) {
    if (!e) return false;
    if (e.isContentEditable) return true;
    if (e.tagName == 'TEXTAREA') return true;
    return e.tagName == 'INPUT' && /^(text|search|email|url|tel|password|number|)$/.test(e.type || '');
  };
  var hit = function (e) {
    if (a.dry) return JSON.stringify({ok: true, label: txt(e).slice(0, 80)});
    e.scrollIntoView({block: 'center', inline: 'nearest'});
    if (editable(e)) e.focus();
    e.click();
    return JSON.stringify({ok: true, label: txt(e).slice(0, 80)});
  };

  var content = function (e) { return ((e.isContentEditable ? e.innerText : e.value) || '').trim(); };
  if (a.mode == 'clear') {
    /* the box being typed in, else the one with text in it (ChatGPT's message box), else nothing to clear */
    var box = editable(document.activeElement) && content(document.activeElement) ? document.activeElement : null;
    if (!box) box = all('textarea,[contenteditable=true],[contenteditable=""],input').filter(function (e) {
      return editable(e) && vis(e) && !e.disabled && !e.readOnly && content(e) &&
             !(e.parentElement && e.parentElement.closest('[contenteditable=true],[contenteditable=""]'));
    })[0];
    if (!box) return JSON.stringify({ok: false, count: 0});
    var had = content(box);
    box.focus();
    if (box.isContentEditable) {
      var range = document.createRange(); range.selectNodeContents(box);
      var sel = getSelection(); sel.removeAllRanges(); sel.addRange(range);
    } else { box.select(); }
    document.execCommand('delete');
    if (content(box)) {  /* a box that ignores execCommand */
      if (box.isContentEditable) box.innerText = ''; else box.value = '';
      box.dispatchEvent(new Event('input', {bubbles: true}));
    }
    return JSON.stringify({ok: true, label: had.slice(0, 60)});
  }

  if (a.mode == 'media') {  /* the biggest video on the page: {ok, label: "paused" | "playing", was: ...} */
    var videos = all('video').filter(vis).sort(function (p, q) {
      var r = p.getBoundingClientRect(), s = q.getBoundingClientRect();
      return s.width * s.height - r.width * r.height;
    });
    var v = videos[0];
    if (!v || !(v.currentTime > 0 || !v.paused)) return JSON.stringify({ok: false, count: videos.length});
    var was = v.paused ? 'paused' : 'playing';
    var want = a.what == 'pause' ? 'paused' : a.what == 'play' ? 'playing' : (v.paused ? 'playing' : 'paused');
    if (want == 'paused' && !v.paused) v.pause();
    if (want == 'playing' && v.paused) v.play();
    return JSON.stringify({ok: true, label: want, was: was});
  }

  if (a.mode == 'text') {  /* the words on the page, for questions about it ("read my recent chats") */
    return JSON.stringify({ok: true, label: document.title,
                           text: (document.body.innerText || '').replace(/\n\s*\n+/g, '\n').slice(0, a.limit || 6000)});
  }

  if (a.mode == 'focus') {
    var active = document.activeElement;
    if (editable(active) || (active && active.tagName == 'IFRAME')) return JSON.stringify({ok: true, label: 'already'});
    var boxes = all('input[type=search],input[name=q],input[name=search_query],[role=searchbox],[role=combobox]')
      .concat(all('textarea,[contenteditable=true],[contenteditable=""],input'))
      .filter(function (e) { return editable(e) && vis(e) && !e.disabled && !e.readOnly; });
    if (!boxes.length) return JSON.stringify({ok: false, count: 0});
    boxes[0].scrollIntoView({block: 'center'});
    boxes[0].focus();
    boxes[0].click();
    return JSON.stringify({ok: true, label: txt(boxes[0]).slice(0, 80) || 'text box'});
  }

  if (a.mode == 'nth') {
    var items = [];
    if (a.kind == 'video') {
      /* YouTube: the video cards (search results, home grid, side list), clicked on their title */
      items = outermostOnly(all('ytd-video-renderer,ytd-rich-item-renderer,ytd-compact-video-renderer,' +
                                'ytd-grid-video-renderer,yt-lockup-view-model').filter(vis))
        .map(function (card) {
          return card.querySelector('a#video-title') || card.querySelector('a#video-title-link') ||
                 card.querySelector('h3 a') || card.querySelector('a[href*="watch?v="]');
        }).filter(function (e) { return e && vis(e) && txt(e).length >= 12; });  /* not "Watch" on an ad */
    }
    if (a.kind == 'video' && !items.length) {
      var titles = {};  /* one link per video: its longest text is the title (thumbnails and durations are shorter) */
      all('a[href*="watch?v="],a[href*="/shorts/"],a[href*="/video/"]').forEach(function (e) {
        var m = (e.getAttribute('href') || '').match(/[?&]v=([\w-]+)|\/shorts\/([\w-]+)|\/video\/([\w-]+)/);
        if (!m || !vis(e)) return;
        var id = m[1] || m[2] || m[3], t = txt(e);
        if (t.length >= 8 && (!titles[id] || t.length > txt(titles[id]).length)) titles[id] = e;
      });
      items = Object.keys(titles).map(function (k) { return titles[k]; });
    } else if (a.kind == 'account') {
      items = all('[data-identifier],[data-email]').filter(vis);
      if (!items.length)
        items = outermostOnly(all('a,button,li,[role=link],[role=button],[role=option],[role=menuitem]')
          .filter(function (e) { return vis(e) && /\S+@\S+\.\w+/.test(e.innerText || ''); }));
    } else if (a.kind == 'result') {
      items = all('a h3').map(function (h) { return h.closest('a'); }).filter(vis);
    }
    if (!items.length && a.kind != 'account')
      items = all('a[href]').filter(function (e) { return vis(e) && txt(e).length >= 15; });
    items = byHref(items);
    var seen = items.filter(onScreen);
    items = reading(seen.length ? seen : items);
    var i = a.n < 0 ? items.length + a.n : a.n;
    if (i < 0 || i >= items.length) return JSON.stringify({ok: false, count: items.length});
    return hit(items[i]);
  }

  var names = a.names.map(norm);
  var clickable = 'a,button,input,select,textarea,summary,label,[role=button],[role=link],[role=tab],' +
                  '[role=menuitem],[role=option],[role=checkbox],[role=radio],[role=switch],[onclick],[tabindex]';
  var best = null, bestScore = 0;
  var score = function (have) {
    var top = 0;
    names.forEach(function (want) {
      if (!have || !want) return;
      var s = 0;
      if (have == want) s = 1;
      else if ((' ' + have + ' ').indexOf(' ' + want + ' ') >= 0) s = 0.9 - Math.min(0.3, (have.length - want.length) / 200);
      if (s > top) top = s;
    });
    return top;
  };
  var consider = function (e, bonus) {
    if (!vis(e)) return;
    var s = score(norm(txt(e))) + bonus + (onScreen(e) ? 0.05 : 0);
    if (s > bestScore) { best = e; bestScore = s; }
  };
  all(clickable).forEach(function (e) { consider(e, 0.05); });
  if (bestScore < 1) {
    all('span,div,p,li,td,h1,h2,h3,h4,img').forEach(function (e) {
      if (e.children.length == 0 || e.tagName == 'IMG') consider(e, 0);
    });
  }
  if (!best || bestScore < 0.75) return JSON.stringify({ok: false, count: 0});
  return hit(best);
})
"""


def is_browser(app) -> bool:
    return app is not None and app.localizedName() in BROWSERS


def run(app_name: str, script: str, arg: dict, timeout: float = 5.0) -> dict:
    """Run PAGE_SCRIPT-style JavaScript (a function of one argument) in the front tab; its JSON answer."""
    js = f"{script.strip()}({json.dumps(arg)})"
    js = js.replace("\\", "\\\\").replace('"', '\\"')
    if app_name == "Safari":
        apple = f'tell application "Safari" to do JavaScript "{js}" in front document'
    else:
        apple = f'tell application "{app_name}" to execute front window\'s active tab javascript "{js}"'
    out = subprocess.run(["osascript", "-e", apple], capture_output=True, text=True, timeout=timeout)
    if out.returncode:
        if "Allow JavaScript from Apple Events" in out.stderr or "turned off" in out.stderr:
            raise ScriptsOff(app_name)
        raise RuntimeError(out.stderr.strip()[:200])
    try:
        return json.loads(out.stdout.strip() or "{}")
    except json.JSONDecodeError:
        return {}


def request_for(name: str) -> dict:
    """"second video" -> nth video; "5D" -> by name, also "5 days"; "sign in" -> by name."""
    m = NTH.match(name.strip())
    if m:
        return {"mode": "nth", "n": ORDINALS[m.group(1).lower()], "kind": KINDS.get((m.group(2) or "").lower(), "")}
    names = [name]
    account = re.match(r"^(?:my\s+)?(.+?)(?:'s)?\s+(?:account|profile|email)$", name.strip(), re.I)
    if account:  # "vikram account": the account showing "vikram"
        names.append(account.group(1))
    r = RANGE.match(name.strip())
    if r:
        n, unit = int(r.group(1)), UNITS[r.group(2).lower()]
        names += [f"{n} {unit}" + ("s" if n > 1 else ""), f"{n}{r.group(2).lower()}"]
    return {"mode": "name", "names": names}


def click(app_name: str, name: str) -> tuple[str | None, int]:
    """Click `name` in the page: (label of what was clicked or None, how many of that kind the page has:
    "the fifth video" when there are 3 -> (None, 3)). Raises ScriptsOff when the browser doesn't allow it."""
    result = run(app_name, PAGE_SCRIPT, request_for(name))
    return ((result.get("label") or name) if result.get("ok") else None), result.get("count", 0)


def focus_text_box(app_name: str) -> str | None:
    """Make sure typing lands in a text box of the page: "already" if one has the focus, the label of the
    box it put the focus in, or None when the page has none. Raises ScriptsOff."""
    result = run(app_name, PAGE_SCRIPT, {"mode": "focus"})
    return result.get("label") or "text box" if result.get("ok") else None


def clear_text_box(app_name: str) -> str | None:
    """Empty the text box being typed in (or the one with text in it): the text that was there, or None
    when no box has text. Raises ScriptsOff."""
    result = run(app_name, PAGE_SCRIPT, {"mode": "clear"})
    return (result.get("label") or "") if result.get("ok") else None


def page_video(app_name: str, what: str = "") -> tuple[str, str] | None:
    """Pause / play / toggle the video on the page: (now "paused"/"playing", before), or None when the page
    has no started video. Raises ScriptsOff."""
    result = run(app_name, PAGE_SCRIPT, {"mode": "media", "what": what})
    return (result["label"], result.get("was", "")) if result.get("ok") else None


def page_text(app_name: str, limit: int = 6000) -> tuple[str, str]:
    """(page title, the page's text) of the front tab. Raises ScriptsOff."""
    result = run(app_name, PAGE_SCRIPT, {"mode": "text", "limit": limit})
    return result.get("label", ""), result.get("text", "")


if __name__ == "__main__":
    from assistant.mac.accessibility import front_app
    print(click(front_app().localizedName(), " ".join(sys.argv[1:])))
