"""Action helpers that can be checked without doing anything on the Mac."""

from datetime import datetime, timedelta

import subprocess

import pytest

from assistant.actions import Failed, NeedsPermission, calculator, in_app, reminders
from assistant.mac import accessibility


@pytest.mark.parametrize("amount, seconds", [
    ("5 minutes", 300), ("90 seconds", 90), ("1 hour 30 minutes", 5400), ("10", 600),
    ("half an hour", 1800), ("an hour", 3600), ("2 mins", 120), ("soon", None),
])
def test_timer_amounts(amount, seconds):
    assert reminders._seconds(amount) == seconds


def test_duration_words():
    assert reminders._duration_words(5400) == "1 hour and 30 minutes"
    assert reminders._duration_words(61) == "1 minute and 1 second"


def test_reminder_times():
    now = datetime.now()
    in20 = reminders.parse_when("in 20 minutes")
    assert timedelta(minutes=19) < in20 - now < timedelta(minutes=21)
    six = reminders.parse_when("at 6 pm")
    assert (six.hour, six.minute) == (18, 0) and six > now
    tomorrow = reminders.parse_when("tomorrow at 9 am")
    assert (tomorrow.date() - now.date()).days == 1 and tomorrow.hour == 9
    assert reminders.parse_when("18:30").hour == 18
    assert reminders.parse_when("whenever") is None


def test_timer_without_a_duration_asks():
    with pytest.raises(Failed):
        reminders.set_timer("a while")


def test_calculate_without_the_calculator_app(monkeypatch):
    monkeypatch.setattr(calculator, "_calculator_press", lambda expression, fresh: None)
    monkeypatch.setattr(calculator, "_last_result", None)
    assert calculator.calculate("25*25") == "25 times 25 is 625"
    assert calculator.calculate("+5") == "That makes 630"
    assert calculator.calculate("10/4") == "10 divided by 4 is 2.5"
    with pytest.raises(Failed):
        calculator.calculate("1/0")
    with pytest.raises(Failed):
        calculator.calculate("__import__('os')")


def test_press_keys_builds_applescript(monkeypatch):
    scripts = []
    monkeypatch.setattr(in_app, "osascript", scripts.append)
    in_app.press_keys("cmd + shift + n, enter")
    assert 'keystroke "n" using {command down, shift down}' in scripts[0]
    assert "key code 36" in scripts[0]
    with pytest.raises(Failed):
        in_app.press_keys("hyper+x")


def test_calculate_minus_goes_on_and_again_repeats(monkeypatch):
    pressed = []
    monkeypatch.setattr(calculator, "_calculator_press", lambda expression, fresh: pressed.append((expression, fresh)))
    monkeypatch.setattr(calculator, "_last_result", None)
    assert calculator.calculate("5*35") == "5 times 35 is 175"
    assert calculator.calculate("-4") == "That makes 171"
    assert calculator.calculate("again") == "175 minus 4 is 171"
    assert pressed == [("5*35", True), ("-4", False), ("175-4", True)]


def test_calculator_open_without_permission_says_so(monkeypatch):
    monkeypatch.setattr(accessibility, "running_app", lambda name: object())
    monkeypatch.setattr(accessibility, "trusted", lambda: False)
    monkeypatch.setattr(calculator, "_last_result", None)
    with pytest.raises(NeedsPermission, match=r"^5 times 35 is 175\. To press the Calculator buttons, I need Accessibility"):
        calculator.calculate("5*35")


def test_zoom_keys_are_sent_as_physical_keys(monkeypatch):
    scripts = []
    monkeypatch.setattr(in_app, "osascript", scripts.append)
    monkeypatch.setattr(accessibility, "trusted", lambda: True)
    monkeypatch.setattr(accessibility, "target_app", lambda: None)
    monkeypatch.setattr(accessibility, "activate", lambda app: False)
    in_app.press_keys("cmd+=")
    in_app.press_keys("cmd+-")
    assert "key code 24 using {command down}" in scripts[0] and "key code 27 using {command down}" in scripts[1]


@pytest.mark.parametrize("amount, seconds", [
    ("five minutes", 300), ("twenty five seconds", 25), ("an hour and a half", 5400), ("a minute", 60),
])
def test_timer_amounts_said_in_words(amount, seconds):
    assert reminders._seconds(amount) == seconds


def test_reminder_tomorrow_without_a_time_is_the_morning():
    due = reminders.parse_when("tomorrow")
    assert (due.date() - datetime.now().date()).days == 1 and (due.hour, due.minute) == (9, 0)
    assert reminders.parse_when("at noon").hour == 12


def test_calculate_goes_on_from_the_exact_last_result(monkeypatch):
    monkeypatch.setattr(calculator, "_calculator_press", lambda expr, fresh: None)
    calculator.calculate("1/3")
    assert calculator.calculate("*3") == "That makes 1"  # was 0.999999 (rounded result reused)
    calculator.calculate("2-7")
    assert calculator.calculate("^2") == "That makes 25"  # was -25 (-5^2)
    assert calculator.calculate("0.1+0.2") == "0.1 plus 0.2 is 0.3"
    assert calculator.calculate("16^0.5") == "the square root of 16 is 4"


def test_volume_step_given_as_text(monkeypatch):
    from assistant.actions import media
    monkeypatch.setattr(media, "_current_volume", lambda: 40)
    monkeypatch.setattr(media, "osascript", lambda script: "")
    assert media.volume_up("20") == "Volume 60"
    assert media.volume_down("a lot") == "Volume 30"
    assert media.unmute() == "Unmuted"


# --- 2026-10-03: clicking inside web pages, typing into a page, notes --------------------------

from assistant.actions import screen  # noqa: E402
from assistant.mac import web_page  # noqa: E402


class FakeApp:
    def __init__(self, name):
        self.name = name

    def localizedName(self):
        return self.name

    def bundleIdentifier(self):
        return None


def test_web_page_requests():
    assert web_page.request_for("second video") == {"mode": "nth", "n": 1, "kind": "video"}
    assert web_page.request_for("last result") == {"mode": "nth", "n": -1, "kind": "result"}
    assert web_page.request_for("first account") == {"mode": "nth", "n": 0, "kind": "account"}
    assert web_page.request_for("5D")["names"] == ["5D", "5 days", "5d"]
    assert web_page.request_for("vikram account")["names"] == ["vikram account", "vikram"]


def test_page_script_is_escaped_for_applescript(monkeypatch):
    sent = []

    class Out:
        returncode, stdout, stderr = 0, '{"ok": true, "label": "5D"}', ""
    monkeypatch.setattr(web_page.subprocess, "run", lambda cmd, **kw: sent.append(cmd) or Out())
    assert web_page.click("Brave Browser", 'say "hi"') == ("5D", 0)
    script = sent[0][2]
    assert script.startswith('tell application "Brave Browser" to execute front window\'s active tab javascript "')
    assert '[\\"say \\\\\\"hi\\\\\\"\\"]' in script  # JSON ["say \"hi\""], escaped once more for AppleScript


def test_page_script_turned_off_is_explained(monkeypatch):
    class Out:
        returncode, stdout = 1, ""
        stderr = "Executing JavaScript through AppleScript is turned off."
    monkeypatch.setattr(web_page.subprocess, "run", lambda cmd, **kw: Out())
    with pytest.raises(web_page.ScriptsOff, match="View, Developer, Allow JavaScript from Apple Events"):
        web_page.click("Google Chrome", "sign in")


def _fake_screen(monkeypatch, app):
    monkeypatch.setattr(accessibility, "trusted", lambda: True)
    monkeypatch.setattr(accessibility, "target_app", lambda: app)
    monkeypatch.setattr(accessibility, "activate", lambda app: False)


def test_click_in_a_browser_uses_the_page(monkeypatch):
    _fake_screen(monkeypatch, FakeApp("Brave Browser"))
    monkeypatch.setattr(web_page, "click", lambda app, name: ("But what is a neural network? | Deep learning", 0))
    monkeypatch.setattr(accessibility, "find", lambda *a: pytest.fail("Accessibility not needed"))
    assert screen.click("second video") == "Clicked But what is a neural network?"


def test_click_past_the_last_item_says_how_many(monkeypatch):
    _fake_screen(monkeypatch, FakeApp("Brave Browser"))
    monkeypatch.setattr(web_page, "click", lambda app, name: (None, 3))
    with pytest.raises(Failed, match="only see 3"):
        screen.click("fifth video")


def test_click_falls_back_to_the_screen_and_explains_the_setting(monkeypatch):
    from assistant.mac import vision
    _fake_screen(monkeypatch, FakeApp("Brave Browser"))

    def off(app, name):
        raise web_page.ScriptsOff(app)
    monkeypatch.setattr(web_page, "click", off)
    monkeypatch.setattr(screen, "window_shot", lambda ax, app: (None, (0, 0, 100, 100)))
    monkeypatch.setattr(vision, "discard", lambda image: None)
    monkeypatch.setattr(vision, "read_text", lambda image: [])
    monkeypatch.setattr(vision, "point", lambda image, description: None)
    with pytest.raises(Failed, match="Allow JavaScript from Apple Events"):
        screen.click("first account")


class Box:
    def __init__(self, x, y, w=0.2, h=0.02):
        self.origin = type("P", (), {"x": x, "y": y})()
        self.size = type("S", (), {"width": w, "height": h})()


def test_account_by_position_from_the_emails_on_screen(monkeypatch):
    from assistant.mac import vision
    lines = [("Choose an account", None, Box(0.4, 0.9)), ("Alex Lee", None, Box(0.4, 0.62)),
             ("alex.lee@example.com", None, Box(0.4, 0.6)), ("Work", None, Box(0.4, 0.52)),
             ("alex.work@example.com", None, Box(0.4, 0.5)), ("Use another account", None, Box(0.4, 0.4))]
    monkeypatch.setattr(vision, "read_text", lambda image: lines)
    x, y = screen._account_on_screen(None, "second account")
    assert round(y, 2) == 0.49  # the lower e-mail (screen y grows downwards)
    x, y = screen._account_on_screen(None, "first account")
    assert round(y, 2) == 0.39
    assert screen._account_on_screen(None, "third account") is None


def test_typing_in_a_page_puts_the_cursor_in_a_text_box(monkeypatch):
    typed, focused = [], []
    _fake_screen(monkeypatch, FakeApp("Google Chrome"))
    monkeypatch.setattr(accessibility, "app_element", lambda app: None)
    monkeypatch.setattr(accessibility, "attr", lambda element, name: None)
    monkeypatch.setattr(web_page, "focus_text_box", lambda app: focused.append(app) or "Search")
    monkeypatch.setattr(in_app, "osascript", typed.append)
    assert in_app.type_text("lofi music") == "Typed"
    assert focused == ["Google Chrome"] and 'keystroke "lofi music"' in typed[0]

    monkeypatch.setattr(web_page, "focus_text_box", lambda app: None)
    with pytest.raises(Failed, match="no text box"):
        in_app.type_text("hello")


def test_typing_outside_a_browser_just_types(monkeypatch):
    typed = []
    _fake_screen(monkeypatch, FakeApp("Notes"))
    monkeypatch.setattr(web_page, "focus_text_box", lambda app: pytest.fail("not a browser"))
    monkeypatch.setattr(in_app, "osascript", typed.append)
    in_app.type_text('say "hi"')
    assert typed == ['tell application "System Events" to keystroke "say \\"hi\\""']


def test_write_note_opens_a_new_note_and_types(monkeypatch):
    from assistant.actions import apps
    calls = []
    monkeypatch.setattr(in_app, "run", lambda cmd: calls.append(cmd))
    monkeypatch.setattr(apps, "wait_for_app", lambda name: calls.append(("wait", name)))
    monkeypatch.setattr(in_app, "press_keys", lambda keys: calls.append(("keys", keys)))
    monkeypatch.setattr(in_app, "osascript", lambda script: calls.append(script))
    monkeypatch.setattr(in_app.time, "sleep", lambda s: None)
    assert in_app.write_note("buy milk") == "Written in a new note"
    assert calls == [["open", "-a", "Notes"], ("wait", "Notes"), ("keys", "cmd+n"),
                     'tell application "System Events" to keystroke "buy milk"']


def test_a_key_set_to_shut_down_is_never_pressed(monkeypatch):
    # 2026-10-03: on the user's Mac cmd+T was set to "Shut Down…" in App Shortcuts; "new tab" opened that dialog.
    scripts = []
    monkeypatch.setattr(in_app, "osascript", scripts.append)
    _fake_screen(monkeypatch, FakeApp("Notes"))
    monkeypatch.setattr(in_app, "custom_shortcuts", lambda bundle: {("command", "t"): "Shut Down…"})
    with pytest.raises(Failed, match="Shut Down"):
        in_app.press_keys("cmd+t")
    assert scripts == []
    in_app.press_keys("cmd+w")  # other keys still work
    assert len(scripts) == 1


def test_new_tab_is_made_by_the_browser_not_cmd_t(monkeypatch):
    scripts = []
    monkeypatch.setattr(in_app, "osascript", scripts.append)
    _fake_screen(monkeypatch, FakeApp("Brave Browser"))
    assert in_app.press_keys("cmd+t", "brave") == "New tab"
    assert scripts and 'tell application "Brave Browser"' in scripts[0] and "make new tab" in scripts[0]
    assert "keystroke" not in "".join(scripts)


def test_custom_shortcuts_are_read_from_the_settings(monkeypatch):
    import plistlib

    class Out:
        returncode = 0
        stdout = plistlib.dumps({"NSUserKeyEquivalents": {"Shut Down…": "@t", "Sleep …": "@$p"}})
    monkeypatch.setattr(in_app.subprocess, "run", lambda cmd, **kw: Out())
    found = in_app.custom_shortcuts()
    assert found[("command", "t")] == "Shut Down…" and found[("command", "shift", "p")] == "Sleep …"


def test_clear_text_in_a_page_empties_the_box(monkeypatch):
    _fake_screen(monkeypatch, FakeApp("Brave Browser"))
    monkeypatch.setattr(web_page, "clear_text_box", lambda app: "some words")
    assert in_app.clear_text() == "Cleared. Say undo to bring it back"
    monkeypatch.setattr(web_page, "clear_text_box", lambda app: None)
    with pytest.raises(Failed, match="nothing typed"):
        in_app.clear_text()


def test_clear_text_never_acts_without_a_text_box(monkeypatch):
    # "delete it" in Finder with a file selected must not press delete keys there
    keys = []
    _fake_screen(monkeypatch, FakeApp("Finder"))
    monkeypatch.setattr(accessibility, "app_element", lambda app: None)
    monkeypatch.setattr(accessibility, "attr", lambda element, name: None)
    monkeypatch.setattr(in_app, "press_keys", keys.append)
    with pytest.raises(Failed, match="no text box"):
        in_app.clear_text()
    assert keys == []


def test_pause_never_starts_music(monkeypatch):
    from assistant.actions import media
    scripts = []
    monkeypatch.setattr(media, "_video_in_front", lambda what: None)
    monkeypatch.setattr(media, "osascript", lambda s: scripts.append(s) or ("false" if "is running" in s else ""))
    assert media.media_play_pause("pause") == "Nothing is playing"
    assert not any("playpause" in s or " to play" in s for s in scripts)


def test_pause_goes_to_the_video_in_front(monkeypatch):
    from assistant.actions import media
    _fake_screen(monkeypatch, FakeApp("Brave Browser"))
    monkeypatch.setattr(web_page, "page_video", lambda app, what: ("paused", "playing"))
    monkeypatch.setattr(media, "osascript", lambda s: pytest.fail("the music app must not be touched"))
    assert media.media_play_pause("pause") == "Paused the video"


CHATS = ("New chat\nSearch chats\nLibrary\nSora\nGPTs\nChats\nPython list comprehension help\nTrip plan for Goa\n"
         "Resume review for data analyst\nTCS share price analysis\nNeural networks explained\nWeekend workout plan\n"
         "Upgrade plan\nWhat can I help with?\nAsk anything")


def test_recent_chats_are_read_from_the_list_not_the_menu():
    assert screen.list_from_screen("Can you see my recent charts?", CHATS) == (
        "Your recent chats are: Python list comprehension help, Trip plan for Goa, Resume review for data analyst, "
        "TCS share price analysis, Neural networks explained")
    days = "New chat\nToday\nTCS share price analysis\nYesterday\nTrip plan for Goa\nUpgrade plan"
    assert screen.list_from_screen("read my chats", days) == "Your recent chats are: TCS share price analysis, Trip plan for Goa"
    assert screen.list_from_screen("did Rahul reply", CHATS) is None


def test_screen_text_questions_use_the_page_text(monkeypatch):
    _fake_screen(monkeypatch, FakeApp("Brave Browser"))
    monkeypatch.setattr(web_page, "page_text", lambda app, limit=6000: ("ChatGPT", CHATS))
    monkeypatch.setattr(screen, "answer_from_text", lambda q, text, where: f"LLM: {where}")
    assert screen.look_at_screen("read my recent chats").startswith("Your recent chats are: Python")
    assert screen.look_at_screen("what is this page about") == "LLM: ChatGPT in Brave Browser"


def test_new_line_is_the_return_key(monkeypatch):
    calls = []
    monkeypatch.setattr(in_app, "osascript", lambda script: calls.append(script))
    in_app._keystroke("Hi.\nSee you")
    assert calls == ['tell application "System Events"\nkeystroke "Hi."\nkey code 36\nkeystroke "See you"\nend tell']


def test_tab_names_match_title_or_address():
    from assistant.actions import web
    assert web.tab_score("chatgpt", "Mental Health Product Ideas", "https://chatgpt.com/c/123") == 1.0
    assert web.tab_score("gmail", "Inbox (3) - me@gmail.com - Gmail", "https://mail.google.com/") == 1.0
    assert web.tab_score("my youtube", "YouTube", "https://www.youtube.com/") == 1.0
    assert web.tab_score("youtube", "ChatGPT", "https://chatgpt.com/") < web.TAB_MATCH


def test_switch_tab_picks_the_best_tab(monkeypatch):
    from assistant.actions import web
    tabs = [(1, 1, "YouTube", "https://www.youtube.com/"), (1, 2, "New chat", "https://chatgpt.com/"),
            (2, 1, "New Tab", "chrome://newtab/")]
    scripts = []
    monkeypatch.setattr(web, "browser_tabs", lambda browser: tabs)
    monkeypatch.setattr(web, "osascript", lambda script, timeout=20: scripts.append(script) or "")
    assert web.switch_tab("chat gpt", "brave") == "Switched to New chat"
    assert "set active tab index of window 1 to 2" in scripts[0] and 'tell application "Brave Browser"' in scripts[0]
    with pytest.raises(web.Failed):
        web.switch_tab("gmail", "brave")


def test_read_mail_lists_unread_or_from_a_sender(monkeypatch):
    from assistant.actions import messaging
    sep = messaging.SEP
    inbox = (sep * 2).join([f"Rahul <r@x.com>{sep}Sync{sep}false{sep}Can you join\nat 5?",
                           f"Shop <s@y.com>{sep}Sale{sep}true{sep}50% off"]) + sep * 2
    monkeypatch.setattr(messaging, "tell_app", lambda app, script: inbox)
    assert messaging.read_mail() == "From Rahul <r@x.com>: Sync - Can you join at 5?"
    assert messaging.read_mail("shop") == "From Shop <s@y.com>: Sale - 50% off"
    assert messaging.read_mail("bob") == "No emails from bob"


def test_contact_lookup(monkeypatch):
    from assistant.actions import messaging
    monkeypatch.setattr(messaging, "tell_app", lambda app, script: "")
    assert messaging.contact("rahul@x.com", "email") == "rahul@x.com"
    assert messaging.contact("+91 98765 43210", "phone") == "+91 98765 43210"
    with pytest.raises(messaging.Failed, match="phone number for Mom"):
        messaging.contact("Mom", "phone")


def test_calendar_events_in_time_order(monkeypatch):
    from assistant.actions import calendar
    monkeypatch.setattr(calendar, "tell_app", lambda app, script: "840\tDesign review\n600\tStandup\n")
    assert calendar.calendar_events("today") == "10:00 AM Standup\n2:00 PM Design review"
    monkeypatch.setattr(calendar, "tell_app", lambda app, script: "")
    assert calendar.calendar_events("tomorrow") == "Nothing on your calendar tomorrow"


def test_automation_refused_says_how_to_allow(monkeypatch):
    from assistant.actions import macos
    def refused(script, timeout=20):
        raise subprocess.CalledProcessError(1, "osascript", "", "Not authorized (-1743)")
    monkeypatch.setattr(macos, "osascript", refused)
    with pytest.raises(macos.Failed, match="permission to use Mail"):
        macos.tell_app("Mail", "x")
