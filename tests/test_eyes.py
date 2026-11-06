"""The eyes: the camera, tasks done on the screen step by step, and watching the screen.
Models and the Mac are faked here; tests/test_vision.py runs the real vision models."""

import time
from pathlib import Path

import pytest

from assistant.actions import Failed, NeedsPermission, agent, camera, watch
from assistant.brain import fast, repair
from assistant.language import dialogue, in_app
from assistant.mac import vision


# --- words ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("text, question", [
    ("What am I holding?", "What am I holding?"),
    ("How do I look?", "How do I look?"),
    ("Can you see me?", "Can you see me?"),
    ("Look through the camera", "Describe what you see, in one or two sentences."),
    ("look at me", "Describe what you see, in one or two sentences."),
    ("Use the webcam and tell me what color my shirt is", "Use the webcam and tell me what color my shirt is"),
    ("Open the camera app", None),
    ("Open Photo Booth", None),
    ("take a photo with the camera", None),
    ("What do you see on my screen?", None),  # the screen, not the camera
])
def test_camera_question(text, question):
    assert in_app.camera_question(text) == question


@pytest.mark.parametrize("text, task", [
    ("On this page, search for running shoes and open the first result.",
     "search for running shoes and open the first result"),
    ("In this app, can you turn on dark theme?", "turn on dark theme"),
    ("Take over and fill in the form with my name Vikram.", "fill in the form with my name Vikram"),
    ("Find the Pricing page for me on this website", "Find the Pricing page"),
    ("open notes on the screen", None),
    ("on the other hand", None),
])
def test_screen_task(text, task):
    assert in_app.screen_task(text) == task


@pytest.mark.parametrize("text, wanted", [
    ('Tell me when "Build succeeded" shows up', "build succeeded"),
    ("let me know when the page says ready", "ready"),
    ("Watch the screen for done", "done"),
    ("notify me when the download complete message appears on the screen", "download complete"),
    ("tell me when it's 5 pm", None),  # a reminder, not the screen
    ("tell me a joke", None),
])
def test_watch_request(text, wanted):
    assert in_app.watch_request(text) == wanted


def test_stop_watching_is_not_the_end_of_the_conversation():
    assert in_app.stop_watching_request("Stop watching the screen")
    assert not dialogue.is_stop("stop watching")


def test_eyes_commands_never_wait_for_more_and_are_never_typed():
    for text in ("On this page, search for", "What am I holding", "Tell me when done shows up"):
        assert not dialogue.trails_off(text)
        assert dialogue.sounds_like_command(text)


@pytest.mark.parametrize("text, steps", [
    ("On this page, search for running shoes and open the first result",  # one task, not split at "and"
     [{"action": "do_on_screen", "args": ["search for running shoes and open the first result"]}]),
    ("On this page, click sign in", [{"action": "click", "args": ["sign in"]}]),  # a plain click stays one
    ("What am I holding?", [{"action": "look_through_camera", "args": ["What am I holding?"]}]),
    ("Tell me when Build succeeded shows up", [{"action": "watch_screen", "args": ["build succeeded"]}]),
    ("Stop watching", [{"action": "stop_watching", "args": []}]),
])
def test_fast_brain_rules(monkeypatch, text, steps):
    monkeypatch.setattr(fast, "classify_many", lambda clauses: [{i: 1 / len(fast.INTENTS) for i in fast.INTENTS}
                                                                for _ in clauses])
    assert fast.decide(text)[0] == steps


def test_llm_cannot_invent_eye_actions():
    assert repair.repair([{"action": "look_through_camera", "args": ["what is it"]}], "tell me a joke") == []
    assert repair.repair([{"action": "do_on_screen", "args": ["x"]}], "what time is it") == []
    kept = repair.repair([{"action": "do_on_screen", "args": ["sign up with my email"]}], "sign me up on this page")
    assert kept == [{"action": "do_on_screen", "args": ["sign up with my email"]}]


# --- the model's answers ---------------------------------------------------------------------------

def test_next_actions_reads_the_model_answer(monkeypatch):
    seen = {}

    def model(image, prompt, max_tokens):
        seen["prompt"] = prompt
        return "<action>click(start_box='<|box_start|>(500,38)<|box_end|>')</action>"
    monkeypatch.setattr(vision, "_agent_run", model)
    actions = vision.next_actions(Path("x.png"), "search for shoes", ["Click Menu"])
    assert actions == [{"name": "click", "spot": (0.5, 0.038), "args": {"start_box": "<|box_start|>(500,38)<|box_end|>"}}]
    assert "search for shoes" in seen["prompt"] and "1. Click Menu" in seen["prompt"]
    # what the real model answered on the shop page: two steps in one answer, parentheses inside quotes
    monkeypatch.setattr(vision, "_agent_run", lambda *a: "<action>click(start_box='<|box_start|>(105,218)<|box_end|>')"
                                                         "\n\ntype(content='it\\'s (fine)')</action>")
    click, typing = vision.next_actions(Path("x.png"), "t", [])
    assert click["spot"] == (0.105, 0.218) and typing == {"name": "type", "spot": None, "args": {"content": "it's (fine)"}}
    monkeypatch.setattr(vision, "_agent_run", lambda *a: "I am not sure")
    assert vision.next_actions(Path("x.png"), "t", []) == []


def test_locate_click_reads_a_box_with_or_without_tags(monkeypatch):
    monkeypatch.setattr(vision, "_agent_run", lambda *a: "click(start_box='<|box_start|>(250,500)<|box_end|>')")
    assert vision.locate_click(Path("x.png"), "the menu") == (0.25, 0.5)

    def broken(*a):
        raise MemoryError
    monkeypatch.setattr(vision, "_agent_run", broken)
    assert vision.locate_click(Path("x.png"), "the menu") is None


# --- doing a task on the screen ----------------------------------------------------------------

class FakeAx:
    def __init__(self):
        self.clicks, self.scrolls, self.at = [], [], (5.0, 5.0)

    def pointer(self):
        return self.at

    def click_at(self, x, y, clicks=1):
        self.clicks.append((round(x), round(y), clicks))

    def scroll(self, at, down=0, right=0):
        self.scrolls.append((round(at[0]), round(at[1]), round(down), round(right)))

    def attr(self, element, name):
        return None

    def app_element(self, app):
        return None


def step(name, spot=None, near="", **args):
    return {"name": name, "spot": spot, "args": args, "near": near}


@pytest.fixture
def screen_agent(monkeypatch):
    """Runs agent.do_on_screen against a script of model answers; returns (run, ax, typed, keys, seen)."""
    ax, typed, keys, seen = FakeAx(), [], [], []
    monkeypatch.setattr(agent, "accessibility", lambda: ax)
    monkeypatch.setattr(agent, "bring_forward", lambda ax: "app")
    monkeypatch.setattr(agent, "window_shot", lambda ax, app: (Path("shot.png"), (0, 0, 1000, 500)))
    monkeypatch.setattr(agent, "type_text", typed.append)
    monkeypatch.setattr(agent, "press_keys", keys.append)
    monkeypatch.setattr(agent, "STEP_SECONDS", 0)
    monkeypatch.setattr(vision, "discard", lambda image: None)

    def run(task, script, during=None):
        """Each item of the script is one answer of the model: a step, or a list of steps."""
        answers = iter(script)
        nears = []

        def next_actions(image, t, done):
            seen.append(list(done))
            if during:
                during(len(done))
            answer = next(answers)
            answer = answer if isinstance(answer, list) else [answer]
            nears[:] = [a["near"] for a in answer]
            return [dict(a) for a in answer]
        monkeypatch.setattr(vision, "next_actions", next_actions)
        monkeypatch.setattr(vision, "text_at", lambda image, spot: nears.pop(0) if nears else "")
        return agent.do_on_screen(task)
    return run, ax, typed, keys, seen


def test_agent_does_the_steps_until_finished(screen_agent):
    run, ax, typed, keys, seen = screen_agent
    reply = run("search for shoes", [[step("click", (0.5, 0.1), near="Search products"), step("type", content="shoes")],
                                      step("click", (0.6, 0.1), near="Search"),
                                      step("finish")])
    assert reply == "Done: click Search products, type shoes and click Search"
    assert ax.clicks == [(500, 50, 1), (600, 50, 1)] and typed == ["shoes"]
    assert seen == [[], ["Click Search products", "Type shoes"], ["Click Search products", "Type shoes", "Click Search"]]


def test_wanting_the_same_click_again_means_done(screen_agent):
    run, ax, *_ = screen_agent  # the real model kept clicking Pricing on the Pricing page
    assert run("find the pricing page", [step("click", (0.2, 0.03), near="Pricing")] * 3) == "Done: click Pricing"
    assert len(ax.clicks) == 1


def test_agent_never_pays(screen_agent):
    run, ax, *_ = screen_agent
    reply = run("buy these shoes", [step("click", (0.8, 0.9), near="Place your order")])
    assert reply.startswith("I stopped before clicking Place your order") and ax.clicks == []


def test_agent_deletes_or_sends_only_when_asked(screen_agent):
    run, ax, *_ = screen_agent
    reply = run("open the draft", [step("click", (0.1, 0.1), near="Delete")])
    assert reply == 'I stopped before clicking Delete. Say "click Delete" if you want me to' and ax.clicks == []
    assert run("delete the draft", [step("click", (0.1, 0.1), near="Delete"), step("finish")]) == "Done: click Delete"
    assert ax.clicks == [(100, 50, 1)]


def test_agent_never_types_passwords(screen_agent):
    run, ax, typed, *_ = screen_agent
    reply = run("log in as me@x.com with password hunter2", [
        step("click", (0.5, 0.4), near="Email address"), step("type", content="me@x.com"),
        step("click", (0.5, 0.5), near="Password"), step("type", content="hunter2")])
    assert reply.startswith("I don't type passwords") and typed == ["me@x.com"]  # the e-mail, not the password


def test_agent_never_types_a_password_it_was_told(screen_agent):
    run, ax, typed, *_ = screen_agent  # the shop test: the password box showed no text where the model clicked
    reply = run("sign in with email me@x.com and password hunter2.", [
        [step("click", (0.2, 0.3), near="Email address"), step("type", content="me@x.com")],
        [step("click", (0.2, 0.4)), step("type", content="hunter2")]])
    assert reply.startswith("I don't type passwords") and typed == ["me@x.com"]


def test_agent_adds_to_the_cart_only_when_asked(screen_agent):
    run, ax, *_ = screen_agent
    assert run("open the second result", [step("click", (0.1, 0.5), near="Add to cart")]).startswith(
        "I stopped before clicking Add to cart")
    assert run("add it to my cart", [step("click", (0.1, 0.5), near="Add to cart"), step("finish")]) == \
        "Done: click Add to cart"


def test_moving_the_mouse_stops_the_agent(screen_agent):
    run, ax, *_ = screen_agent

    def user_moves_mouse(steps_done):  # while the model looks at the second screenshot
        if steps_done == 1:
            ax.at = (300.0, 300.0)
    reply = run("open settings", [step("click", (0.5, 0.5), near="Menu"), step("click", (0.6, 0.6), near="Settings")],
                during=user_moves_mouse)
    assert reply == "You moved the mouse, so I stopped, after 1 step" and len(ax.clicks) == 1


def test_agent_presses_only_safe_keys(screen_agent):
    run, ax, typed, keys, _ = screen_agent
    assert run("copy it", [step("hotkey", key="Command + C"), step("finish")]) == "Done: press Command + C"
    assert keys == ["cmd+c"]
    reply = run("close it", [step("hotkey", key="cmd+q")])
    assert reply.startswith("I stopped: it wanted the keys cmd+q") and keys == ["cmd+c"]


def test_agent_stops_when_stuck_or_out_of_steps(screen_agent, monkeypatch):
    run, ax, *_ = screen_agent
    down = step("scroll", (0.5, 0.5), direction="down")
    assert run("find the footer", [down] * 4) == "I seem to be stuck, after 3 steps"
    monkeypatch.setattr(agent, "MAX_STEPS", 2)
    scrolls = [step("scroll", (0.5, 0.5), direction="down"), step("scroll", (0.2, 0.5), direction="up")]
    assert run("go on", scrolls) == "I stopped after 2 steps, after 2 steps"
    assert ax.scrolls[-2:] == [(500, 250, 300, 0), (200, 250, -300, 0)]  # aimed where the model looked


def test_agent_says_when_the_model_gives_up_or_cannot_load(screen_agent, monkeypatch):
    run, *_ = screen_agent
    assert run("find the pricing", [step("stop", reason="There is no pricing page here")]) == \
        "There is no pricing page here"

    def broken(*a):
        raise MemoryError
    monkeypatch.setattr(vision, "next_actions", broken)
    with pytest.raises(Failed, match="screen agent model"):
        agent.do_on_screen("anything")


# --- camera ---------------------------------------------------------------------------------------

def test_camera_answers_and_deletes_the_photo(monkeypatch):
    deleted = []
    monkeypatch.setattr(vision, "photo", lambda: Path("photo.png"))
    monkeypatch.setattr(vision, "ask", lambda image, question, what: f"You are holding a red mug ({what[:7]})")
    monkeypatch.setattr(vision, "discard", deleted.append)
    assert camera.look_through_camera("What am I holding?") == "You are holding a red mug (a photo)"
    assert deleted == [Path("photo.png")]


def test_camera_without_permission_says_how(monkeypatch):
    def refused():
        raise vision.NoCameraAccess()
    monkeypatch.setattr(vision, "photo", refused)
    with pytest.raises(NeedsPermission, match="Privacy and Security, Camera"):
        camera.look_through_camera()


# --- watching the screen ----------------------------------------------------------------------------

class WatchedApp:
    def localizedName(self):
        return "Xcode"

    def isTerminated(self):
        return False


@pytest.fixture
def watcher(monkeypatch):
    said = []
    fake_ax = type("Ax", (), {"target_app": staticmethod(lambda: WatchedApp())})()
    monkeypatch.setattr(watch, "accessibility", lambda: fake_ax)
    monkeypatch.setattr(watch, "window_shot", lambda ax, app: (Path("shot.png"), (0, 0, 10, 10)))
    monkeypatch.setattr(watch, "announce", said.append)
    monkeypatch.setattr(watch, "WATCH_SECONDS", 0.01)
    monkeypatch.setattr(vision, "discard", lambda image: None)
    yield said
    watch.stop_watching()


def _wait_for(said, seconds=2.0):
    end = time.time() + seconds
    while not said and time.time() < end:
        time.sleep(0.01)


def test_watch_tells_when_the_text_shows_up(watcher, monkeypatch):
    looks = iter([None, None, (0.5, 0.5)])
    monkeypatch.setattr(vision, "find_text", lambda image, text: next(looks, (0.5, 0.5)))
    assert watch.watch_screen("Build succeeded") == "Okay, I'll tell you when Build succeeded shows up in Xcode"
    _wait_for(watcher)
    assert watcher == ["Build succeeded is on the screen now"]
    assert watch.stop_watching() == "I wasn't watching anything"


def test_stop_watching_cancels_quietly(watcher, monkeypatch):
    monkeypatch.setattr(vision, "find_text", lambda image, text: None)
    watch.watch_screen("done")
    assert watch.stop_watching() == "Okay, I stopped watching for done"
    time.sleep(0.1)
    assert watcher == []


def test_watch_needs_something_to_watch_for():
    with pytest.raises(Failed):
        watch.watch_screen("  ")
