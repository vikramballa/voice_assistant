"""The whole pipeline with the real brains but pretend actions (nothing happens on the Mac)."""

import threading
import time

import numpy as np
import pytest

from assistant import app, logbook, modes
from assistant.audio import listen, speaker, voice_id
from assistant.brain import fast, slow
from assistant.config import Options


@pytest.fixture(scope="module", autouse=True)
def warm():
    fast.warm_up()  # as at startup, so timings exclude model loading


@pytest.fixture(autouse=True)
def no_voiceprint(monkeypatch, tmp_path):
    """The owner's real voiceprint (models/voiceprint.npz) must not decide these tests."""
    monkeypatch.setattr(voice_id, "VOICEPRINT", tmp_path / "voiceprint.npz")
    monkeypatch.setattr(voice_id, "_print", None)


@pytest.fixture
def fake_mac(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(app.task, "run_action", lambda name, *args: calls.append((name, args)) or "ok")
    monkeypatch.setattr(app.task, "wait_for_app", lambda name: calls.append(("wait_for_app", (name,))))
    monkeypatch.setattr(app, "speak", lambda text: calls.append(("speak", (text,))))
    monkeypatch.setattr(app, "_is_running", lambda name: True)
    monkeypatch.setattr(app.web, "find_tab", lambda name, browser: None)  # never the real browser's tabs
    monkeypatch.setattr(app.proofread, "correct", lambda text, before="": (text, []))  # no LLM unless a test wants it
    monkeypatch.setattr(logbook, "DB", tmp_path / "commands.db")
    return calls


@pytest.fixture
def assistant():
    return app.Assistant()  # a fresh session: no conversation, history, pending question or browser


def actions(calls):
    return [c for c in calls if c[0] != "speak"]


def spoken(calls):
    return [c[1][0] for c in calls if c[0] == "speak"]


def test_fast_path_runs_and_logs(fake_mac, assistant):
    assistant.handle("open chrome and search youtube for neural networks")
    assert actions(fake_mac) == [("youtube_search", ("neural networks", "chrome"))]
    entry = logbook.entries()[-1]
    assert entry["path"] == "laya" and entry["laya_ms"] < 500


def test_waits_for_app_before_typing(fake_mac, assistant):
    assistant.handle("open notes and type hello")
    assert actions(fake_mac) == [("open_app", ("Notes",)), ("wait_for_app", ("Notes",)), ("type_text", ("hello",))]


def test_unsure_goes_to_llm(fake_mac, assistant, monkeypatch):
    monkeypatch.setattr(slow, "plan", lambda text, history=None, browser=None:
                        [{"action": "web_search", "args": ["pizza", "chrome"]}])
    assistant.handle("order me a pizza")
    assert actions(fake_mac) == [("web_search", ("pizza", "chrome"))]
    assert logbook.entries()[-1]["path"] == "llm"


def test_failed_action_stops_remaining_steps(fake_mac, assistant, monkeypatch):
    def boom(name, *args):
        raise RuntimeError("app not found")
    monkeypatch.setattr(app.task, "run_action", boom)
    assistant.handle("take a screenshot and lock my mac")
    assert logbook.entries()[-1]["error"].startswith("screenshot")


def test_gemini_in_google_account_opens_the_website(fake_mac, assistant):
    assistant.handle("Please open Gemini in my Google account.")
    assert actions(fake_mac) == [("open_website", ("gemini.google.com", "chrome"))]
    assert logbook.entries()[-1]["path"] == "laya"


def test_incomplete_command_asks_then_uses_the_answer(fake_mac, assistant, monkeypatch):
    monkeypatch.setattr(app, "ASK_AFTER_SECONDS", 0.05)
    assistant.handle("Open.")
    time.sleep(0.2)  # nothing else was said: now it asks
    assert spoken(fake_mac) == ["Open what?"] and actions(fake_mac) == []
    assistant.handle("Safari")
    assert actions(fake_mac) == [("open_app", ("Safari",))]


def test_failure_says_why(fake_mac, assistant, monkeypatch):
    def not_found(name, *args):
        raise app.task.Failed("I couldn't find an app called Foo")
    monkeypatch.setattr(app.task, "run_action", not_found)
    assistant.handle("open safari")
    assert spoken(fake_mac) == ["I couldn't find an app called Foo"]


def test_answers_are_spoken(fake_mac, assistant, monkeypatch):
    monkeypatch.setattr(app.task, "run_action", lambda name, *args: fake_mac.append((name, args)) or "It's 3:21 PM")
    assistant.handle("what time is it")
    assert actions(fake_mac) == [("tell_time", ("time",))] and spoken(fake_mac) == ["It's 3:21 PM"]


def test_trailing_off_waits_silently_for_the_rest(fake_mac, assistant):
    assistant.handle("set the volume to...")
    assistant.handle("30.")
    assert actions(fake_mac) == [("set_volume", ("30",))]
    assert "What volume, from 0 to 100?" not in spoken(fake_mac)  # the rest came in time: no question


def test_gpu_wake_up_runs_beside_a_real_decision(fake_mac):
    fast._last_wake = 0.0
    fast.wake_gpu()  # background pass, as when speech starts
    fast.wake_gpu()  # throttled: no second pass within a second
    steps, _ = fast.decide("open safari")
    assert steps == [{"action": "open_app", "args": ["Safari"]}]


def test_browser_in_use_is_remembered(fake_mac, assistant):
    assistant.handle("open brave")
    assistant.handle("open youtube")
    assert actions(fake_mac)[-1] == ("open_website", ("youtube.com", "brave"))


def test_in_brave_redoes_the_last_web_step(fake_mac, assistant):
    assistant.handle("open youtube")
    assistant.handle("in Brave")
    assert actions(fake_mac) == [("open_website", ("youtube.com", "chrome")), ("open_website", ("youtube.com", "brave"))]


def test_calculator_question_uses_the_calculator(fake_mac, assistant):
    assistant.handle("In calculator, what is the value of 25 into 25?")
    assert actions(fake_mac) == [("open_app", ("Calculator",)), ("wait_for_app", ("Calculator",)),
                                 ("calculate", ("25*25",))]


def test_in_app_part_alone_goes_to_the_llm(fake_mac, assistant, monkeypatch):
    asked = []
    monkeypatch.setattr(slow, "plan", lambda text, history=None, browser=None: asked.append(text) or
                        [{"action": "click", "args": ["subscribe"]}])
    assistant.handle("open youtube in brave and check subscribe")
    assert asked == ["check subscribe"]
    assert actions(fake_mac) == [("open_website", ("youtube.com", "brave")), ("click", ("subscribe",))]


class FakeTranscriber:
    """Hands out the given sentences as if Whisper had heard them."""

    def __init__(self, said):
        self.said = iter(said)

    def on_pause(self, audio):
        return False

    def transcribe(self, audio):
        return next(self.said), {"stt_ms": 1}


def test_conversation_mode_until_stop(fake_mac, assistant, monkeypatch):
    """ "Hey Laya" once, then commands without it, until "stop"."""
    said = ["open safari",                  # no wake word yet: ignored
            "Hey Laya, open notes",         # starts conversation mode
            "open safari",                  # no wake word needed now
            "thank you",                    # Whisper noise: skipped
            "we should get lunch later",    # people talking: not a request, skipped quietly
            "stop",                         # ends it
            "open chrome"]                  # ignored again
    monkeypatch.setattr(listen, "utterances", lambda **kw: (np.zeros(16000) for _ in said))
    monkeypatch.setattr(modes, "spoke_during", lambda t0, t1: False)
    monkeypatch.setattr(modes, "chime", lambda: None)
    monkeypatch.setattr(slow, "plan", lambda text, history=None, browser=None: [
        {"action": "answer", "args": ["Sorry, I didn't catch that."]}])
    loop = modes.WakeWordLoop(assistant)
    loop.transcriber = FakeTranscriber(said)
    loop.run()
    assert actions(fake_mac) == [("open_app", ("Notes",)), ("open_app", ("Safari",))]
    assert "Okay" in spoken(fake_mac) and not any("catch" in s for s in spoken(fake_mac))
    assert not assistant.session.conversation_on


def test_echo_check_uses_when_the_audio_was_recorded(monkeypatch):
    """A reply that started after the user finished talking is not an overlap, even if we got to the
    audio late; one that played during it is."""
    monkeypatch.setattr(speaker, "_spans", [[100.0, 102.0, "Opening Safari"]])
    monkeypatch.setattr(speaker, "_queued", False)
    assert speaker.spoke_during(101.0, 101.5)        # said while the reply played
    assert not speaker.spoke_during(95.0, 99.0)      # said before the reply (processed late): not echo
    assert not speaker.spoke_during(103.0, 104.0)    # after it (and after the short echo tail)
    assert speaker.said_during(101.0, 101.5) == ["Opening Safari"]


def test_unclear_question_is_not_waited_on(fake_mac, assistant, monkeypatch):
    """2026-09-28 crash: "I'm not sure I understand. Could you clarify ...?" waited for an answer, background talk
    was glued onto it and asked about again every few seconds."""
    reply = "I'm not sure I understand. Could you clarify what you mean by 'put a drop of light'?"
    monkeypatch.setattr(slow, "plan", lambda text, history=None, browser=None: [{"action": "answer", "args": [reply]}])
    assert assistant.handle("put a drop of light on it", quiet_unclear=True) == []  # people talking: skipped
    assert spoken(fake_mac) == []
    assistant.handle("put a drop of light on it")  # said with the wake word: answered, but not waited on
    assert len(spoken(fake_mac)) == 1 and assistant.session.pending_text is None


class FakeStream:
    """Records which thread calls what; `fail_writes` writes raise a device error."""
    log: list = []
    fail_writes = 0

    def __init__(self, **kw):
        self._record("open")

    def _record(self, what):
        FakeStream.log.append((what, threading.current_thread()))

    def start(self):
        self._record("start")

    def write(self, block):
        if FakeStream.fail_writes:
            FakeStream.fail_writes -= 1
            import sounddevice
            raise sounddevice.PortAudioError("device changed")
        self._record("write")
        time.sleep(0.002)

    def abort(self):
        self._record("abort")

    def close(self):
        self._record("close")


@pytest.fixture
def fake_sound(monkeypatch):
    import sounddevice
    FakeStream.log, FakeStream.fail_writes = [], 0
    monkeypatch.setattr(sounddevice, "OutputStream", FakeStream)
    monkeypatch.setattr(speaker, "_model", object())
    monkeypatch.setattr(speaker, "_synthesize", lambda text: np.zeros(speaker.BLOCK * 40, dtype=np.float32))
    yield FakeStream
    speaker.shutdown()


def test_only_the_speaker_thread_touches_the_audio_stream(fake_sound):
    """sd.play in the worker + sd.stop in the caller freed PortAudio's stream twice and crashed Python."""
    for i in range(20):  # every new reply interrupts the one playing
        speaker.speak(f"reply {i}")
        time.sleep(0.01)
    speaker.speak("last", wait=True)
    worker = speaker._worker
    speaker.shutdown()
    assert fake_sound.log and all(t is worker for _, t in fake_sound.log)


def test_one_stream_for_many_replies_and_shutdown_closes_it(fake_sound):
    """Opening/closing a stream per reply collided with macOS rebuilding the device; exiting while the worker
    was still writing crashed Python (2026-09-29). One stream, closed by the worker itself on shutdown."""
    for i in range(3):
        speaker.speak(f"reply {i}", wait=True)
    worker = speaker._worker
    speaker.speak("a long reply that is still playing")
    time.sleep(0.02)
    speaker.shutdown()
    opened = [w for w, _ in fake_sound.log if w == "open"]
    assert len(opened) == 1 and fake_sound.log[-1][0] == "close"
    assert not worker.is_alive() and speaker._worker is None


def test_device_change_reopens_the_stream(fake_sound):
    fake_sound.fail_writes = 1  # e.g. headphones plugged in while speaking
    speaker.speak("hello", wait=True)
    events = [w for w, _ in fake_sound.log]
    assert events.count("open") == 2 and "write" in events


def test_exit_stops_the_voice_before_python_ends():
    """Exiting while a reply plays: the old normal Python exit crashed 5 of 6 times (segfault/abort)."""
    import subprocess
    import sys
    code = """
import sys, time
import numpy as np
from assistant.audio import speaker
speaker._model = object()
speaker._synthesize = lambda text: np.zeros(speaker.RATE * 20, dtype=np.float32)
speaker.speak("long reply")
time.sleep(0.5)
from assistant.__main__ import _exit
_exit(0)
"""
    for _ in range(3):
        assert subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=30).returncode == 0


def test_startup_is_offline_unless_a_model_is_missing(monkeypatch, tmp_path):
    """A dead network froze "loading models..." for minutes (Hugging Face update checks): models on disk -> offline."""
    import os
    import sys
    import assistant.__main__ as start
    from huggingface_hub import constants
    assert os.environ["HF_HUB_OFFLINE"] == "1"  # set when assistant.__main__ was imported
    assert app.missing_models() == []           # everything is downloaded on this Mac
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))  # an empty cache: all missing
    assert fast.BASE in app.missing_models()

    monkeypatch.setattr(app, "missing_models", lambda: ["some/model"])
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    started = []
    monkeypatch.setattr(os, "execv", lambda exe, argv: started.append((exe, argv, os.environ["HF_HUB_OFFLINE"])))
    monkeypatch.setattr(app, "warm_up", lambda **kw: (_ for _ in ()).throw(SystemExit))  # stop right after
    with pytest.raises(SystemExit):
        start.run(Options(ui=False))
    assert started == [(sys.executable, sys.orig_argv, "0")]  # the same command again, online


def test_ctrl_c_while_loading_quits_at_once(monkeypatch):
    """Leaving a `with ThreadPoolExecutor` waited for every model, so Ctrl+C during loading did nothing for a minute."""
    stop = threading.Event()
    monkeypatch.setattr(fast, "warm_up", lambda: stop.wait(30))  # a model that takes long

    def ctrl_c():
        raise KeyboardInterrupt
    monkeypatch.setattr(slow, "warm_up", ctrl_c)
    monkeypatch.setattr(speaker, "warm_up", lambda use_say: None)
    monkeypatch.setattr(app, "_warm_up_vision", lambda: None)
    t = time.perf_counter()
    with pytest.raises(KeyboardInterrupt):
        app.warm_up(use_mic=False)
    stop.set()
    assert time.perf_counter() - t < 2


def test_command_line_options():
    from assistant.config import parse_args
    assert parse_args([]) == Options()
    assert parse_args(["--ptt", "--fast", "--no-ui"]) == Options(mode="ptt", fast=True, ui=False)
    assert parse_args(["--anyone"]) == Options(anyone=True)


def test_only_the_owner_gives_commands_but_anyone_can_say_stop(fake_mac, assistant, monkeypatch):
    said = [("stranger", "Hey Laya, open notes"),  # someone else (TV, people in the room): ignored
            ("owner", "Hey Laya, open safari"),    # the owner starts conversation mode
            ("stranger", "open chrome"),           # still ignored inside the conversation
            ("owner", "open notes"),
            ("stranger", "stop")]                   # anyone may stop it
    monkeypatch.setattr(listen, "utterances", lambda **kw: (np.full(16000, i, dtype=np.float32) for i in range(len(said))))
    monkeypatch.setattr(modes, "spoke_during", lambda t0, t1: False)
    monkeypatch.setattr(voice_id, "check", lambda audio: (said[int(audio[0])][0] == "owner", 0.5))
    loop = modes.WakeWordLoop(assistant)
    loop.owner_only = True  # as when a voiceprint exists
    loop.transcriber = FakeTranscriber([text for _, text in said])
    loop.run()
    assert actions(fake_mac) == [("open_app", ("Safari",)), ("open_app", ("Notes",))]
    assert not assistant.session.conversation_on and spoken(fake_mac)[-1] == "Okay"


def test_voice_check_is_off_until_someone_enrolls(assistant):
    assert not modes.WakeWordLoop(assistant).owner_only
    assert voice_id.check(np.zeros(16000, dtype=np.float32)) == (True, 1.0)


REPLY = ("I am a voice assistant controlling your Mac. I can help you with various tasks like opening applications, "
         "searching the web, controlling media, and more. How can I assist you today?")


def run_loop_over_reply(assistant, monkeypatch, said, reply=REPLY):
    """The wake loop in conversation mode, every utterance heard while `reply` was playing."""
    monkeypatch.setattr(listen, "utterances", lambda **kw: (np.zeros(16000) for _ in said))
    monkeypatch.setattr(modes, "spoke_during", lambda t0, t1: True)
    monkeypatch.setattr(modes, "said_during", lambda t0, t1: [reply])
    monkeypatch.setattr(modes, "chime", lambda: None)
    asked = []
    monkeypatch.setattr(slow, "plan", lambda text, history=None, browser=None: asked.append(text) or
                        [{"action": "answer", "args": [REPLY]}])
    assistant.session.conversation_on = True
    loop = modes.WakeWordLoop(assistant)
    loop.transcriber = FakeTranscriber(said)
    loop.run()
    return asked


def test_own_reply_heard_back_never_starts_a_loop(fake_mac, assistant, monkeypatch):
    """2026-09-29 log: pieces of our long reply, misheard by Whisper, were answered again and again."""
    asked = run_loop_over_reply(assistant, monkeypatch, [
        "I have a voice assist.", "Controlling your Mac.", "I can help you with bear.",
        "Control it closely.", "I'm asking you."])
    assert asked == [] and actions(fake_mac) == [] and spoken(fake_mac) == []


def test_clear_command_over_our_reply_still_works(fake_mac, assistant, monkeypatch):
    asked = run_loop_over_reply(assistant, monkeypatch, ["625 multiply that by 4", "open safari"],
                                reply="25 times 25 is 625")
    assert asked == []
    assert actions(fake_mac) == [("calculate", ("*4",)), ("open_app", ("Safari",))]


def test_stop_over_our_reply_ends_conversation(fake_mac, assistant, monkeypatch):
    run_loop_over_reply(assistant, monkeypatch, ["stop"])
    assert not assistant.session.conversation_on and spoken(fake_mac) == ["Okay"]


@pytest.mark.parametrize("text, steps", [
    ("unmute", [("unmute", ())]),  # Laya alone picked "mute": the opposite
    ("turn the volume up by 20", [("volume_up", ("20",))]),  # was set_volume 20
    ("set the volume to fifty", [("set_volume", ("50",))]),  # was a hand-off to the LLM
])
def test_volume_words_on_the_fast_path(fake_mac, assistant, text, steps):
    assistant.handle(text)
    assert actions(fake_mac) == steps
    assert logbook.entries()[-1]["path"] == "laya"


# --- learning from corrections (phrasebook) ---------------------------------------------------------

def test_correction_is_learned_and_used_next_time(fake_mac, assistant):
    from assistant import phrasebook
    assistant.handle("open safari")
    assistant.handle("no, I meant open notes")
    assert actions(fake_mac)[-1] == ("open_app", ("Notes",))
    assert phrasebook.lookup("open safari") == [{"action": "open_app", "args": ["Notes"]}]
    entry = logbook.entries()[-1]
    assert entry["corrects"] == "open safari" and entry["learned"] is True
    fake_mac.clear()
    assistant.handle("Open Safari.")  # same words: the lesson, no model
    assert actions(fake_mac) == [("open_app", ("Notes",))]
    assert logbook.entries()[-1]["path"] == "memory"


@pytest.mark.parametrize("first, second", [
    ("open safari", "no, take a screenshot"),  # a plain "no" with a different command: a change of mind
    ("search youtube for cats", "no, search youtube for cats in brave"),  # only the browser: not a lesson
])
def test_change_of_mind_is_not_learned(fake_mac, assistant, first, second):
    from assistant import phrasebook
    assistant.handle(first)
    assistant.handle(second)
    assert phrasebook.entries() == {}


def test_correction_too_late_is_not_learned(fake_mac, assistant):
    from assistant import phrasebook
    assistant.handle("open safari")
    t, text, steps, worked = assistant.session.last
    assistant.session.last = (t - app.CORRECTION_SECONDS - 1, text, steps, worked)
    assistant.handle("no, I meant open notes")
    assert actions(fake_mac)[-1] == ("open_app", ("Notes",))  # still done...
    assert phrasebook.entries() == {}                         # ...but not learned


def test_in_brave_after_an_app_opens_its_website(fake_mac, assistant):
    # "open ChatGPT" (the app) ... "in Brave browser": the user wanted chatgpt.com in Brave
    assistant.handle("open ChatGPT")
    assistant.handle("in Brave browser")
    assert actions(fake_mac)[-1] == ("open_website", ("chatgpt.com", "brave"))


def test_misspelled_names_are_corrected_before_deciding(fake_mac, assistant):
    assistant.handle("search jithub for laya")
    assert actions(fake_mac)[-1][0] == "web_search" and "github" in actions(fake_mac)[-1][1][0].lower()


def test_new_note_from_a_misheard_node(fake_mac, assistant):
    assistant.handle("Create a new node.")
    assert actions(fake_mac) == [("write_note", ())]


def test_misheard_remove_that_never_plays_music(fake_mac, assistant):
    # 2026-10-03: "remove that" was heard as "I made it. Stop." and a song started
    assistant.handle("I made it. Stop.")
    assert all(a != ("media_play_pause", ()) and a != ("media_play_pause", ("play",)) for a in actions(fake_mac))


def test_remove_that_clears_the_text_box(fake_mac, assistant):
    assistant.handle("remove that")
    assert actions(fake_mac) == [("clear_text", ("all",))]


def test_stop_the_music_pauses_not_mutes(fake_mac, assistant):
    assistant.handle("stop the music")
    assert actions(fake_mac) == [("media_play_pause", ("pause",))]


def test_writing_in_a_note_like_the_users_log(fake_mac, assistant, monkeypatch):
    # 2026-10-04 log: "OK, right. What I see?" (= "write what I say") took a screenshot, the dictated
    # "Hello, how are you?" got a chat answer, "Write in the note. X" typed "the note. X", "Continue. I'm a
    # developer." got a chat answer. The user: once Notes is open, work there until it is closed.
    monkeypatch.setattr(app, "ASK_AFTER_SECONDS", 0.05)
    monkeypatch.setattr(slow, "plan", lambda *a, **k: pytest.fail("the LLM must not be asked"))
    assistant.handle("Open nodes.")
    assistant.handle("Create a new node.")
    assistant.handle("OK, right. What I see?")
    time.sleep(0.2)
    assistant.handle("Hello, how are you?")
    assistant.handle("Write in the note. Hello, how are you?")
    assistant.handle("Continue. I'm a developer.")
    assistant.handle("I'm a developer.")
    assert actions(fake_mac) == [
        ("open_app", ("Notes",)), ("write_note", ()), ("type_text", ("Hello, how are you?", "Notes")),
        ("type_text", (" Hello, how are you?", "Notes")), ("type_text", (" I'm a developer.", "Notes")),
        ("type_text", (" I'm a developer.", "Notes"))]
    assert "Type what?" in spoken(fake_mac)


def test_notes_stays_the_working_app_until_something_else(fake_mac, assistant, monkeypatch):
    monkeypatch.setattr(slow, "plan", lambda text, history=None, browser=None:
                        [{"action": "answer", "args": ["It's 5 pm"]}])
    assistant.handle("open notes")
    assistant.handle("Hello, how are you?")               # written into the note
    assistant.handle("what time is it")                   # a question for Laya: answered, still in Notes
    assistant.handle("next song")                         # a command: done, still in Notes
    assistant.handle("see you at five")                   # written
    assistant.handle("close notes")                       # Notes is done
    assistant.handle("I'm a developer")                   # not written anywhere: goes to the brains
    did = actions(fake_mac)
    assert did[:2] == [("open_app", ("Notes",)), ("type_text", ("Hello, how are you?", "Notes"))]
    assert ("next_track", ()) in did and ("type_text", (" see you at five", "Notes")) in did
    assert ("close_app", ("Notes",)) in did
    assert not any(a == "type_text" and "developer" in args[0] for a, args in did)
    assert ("answer", ("It's 5 pm",)) in did


def test_opening_something_else_ends_writing(fake_mac, assistant, monkeypatch):
    monkeypatch.setattr(slow, "plan", lambda text, history=None, browser=None:
                        [{"action": "answer", "args": ["Nice!"]}])
    assistant.handle("Continue. I'm a developer.")  # nothing open by voice: not typed
    assistant.handle("open notes")
    assistant.handle("search youtube for lofi")      # went to the web: Notes is no longer the working app
    assistant.handle("continue, I'm a developer")
    assert not any(a == "type_text" for a, _ in actions(fake_mac))


def test_notes_quit_by_hand_ends_writing(fake_mac, assistant, monkeypatch):
    monkeypatch.setattr(slow, "plan", lambda text, history=None, browser=None: [{"action": "answer", "args": ["Ok"]}])
    assistant.handle("open notes")
    monkeypatch.setattr(app, "_is_running", lambda name: False)
    assistant.handle("I'm a developer")
    assert not any(a == "type_text" for a, _ in actions(fake_mac))


def test_second_notes_session_from_the_log(fake_mac, assistant, monkeypatch):
    # 2026-10-04 01:40 log: "OK, what are you seeing?" and "Okay, quit notes." were typed into the note,
    # "Next line." played the next song, "And a hackle." stayed misheard.
    monkeypatch.setattr(app.proofread, "correct", lambda text, before="":
                        ("And a hacker.", [("hackle", "hacker")]) if text == "And a hackle." else (text, []))
    monkeypatch.setattr(slow, "plan", lambda text, history=None, browser=None:
                        [{"action": "look_at_screen", "args": [text]}] if "seeing" in text else pytest.fail(text))
    for said in ["Open nodes.", "OK, what are you seeing?", "Hi, how are you?", "Next line.", "I am Vikram.",
                 "I'm a developer.", "And a hackle.", "Hi comma how are you question mark", "Okay, quit notes."]:
        assistant.handle(said)
    typed = [args[0] for a, args in actions(fake_mac) if a == "type_text"]
    assert "".join(typed) == "Hi, how are you?\nI am Vikram. I'm a developer. And a hacker. Hi, how are you?"
    assert ("look_at_screen", ("OK, what are you seeing?",)) in actions(fake_mac) or \
        any(a == "look_at_screen" for a, _ in actions(fake_mac))
    assert actions(fake_mac)[-1] == ("close_app", ("Notes",))
    assert not any(a == "next_track" for a, _ in actions(fake_mac))


def test_go_to_chatgpt_in_brave_switches_to_its_tab(fake_mac, assistant, monkeypatch):
    # 2026-10-04 01:43 log: in Brave, "Go to ChatGPT." opened the ChatGPT app, "Go to ChatGPT tab." pressed cmd+T
    monkeypatch.setattr(app.web, "find_tab", lambda name, browser:
                        (1, 2, "ChatGPT") if "chat" in name.lower() else None)
    assistant.handle("Open Brave.")
    assistant.handle("Go to ChatGPT.")
    assistant.handle("Go to ChatGPT tab.")
    assistant.handle("Done. Previous step.")
    assistant.handle("open gmail")                     # no Gmail tab: the website, in Brave
    assistant.handle("open the chatgpt app")           # the app when the user says so
    assert actions(fake_mac) == [
        ("open_app", ("Brave Browser",)), ("switch_tab", ("ChatGPT", "brave")), ("switch_tab", ("chatgpt", "brave")),
        ("press_keys", ("ctrl+shift+tab", "brave")), ("open_website", ("mail.google.com", "brave")),
        ("open_app", ("ChatGPT",))]


def test_chat_reply_in_notes_is_written_instead(fake_mac, assistant, monkeypatch):
    monkeypatch.setattr(slow, "plan", lambda text, history=None, browser=None: [{"action": "answer", "args": ["Sure!"]}])
    assistant.handle("open notes")
    assistant.handle("Make sure to call mom tonight.")
    assert actions(fake_mac)[-1] == ("type_text", ("Make sure to call mom tonight.", "Notes"))


def test_reply_to_an_email_asks_before_sending(fake_mac, assistant, monkeypatch):
    read, send = {"action": "read_mail", "args": ["Rahul"]}, {"action": "send_email", "args": ["Rahul", "Re: Sync", "I'll join at 5"]}
    monkeypatch.setattr(slow, "plan", lambda text, history=None, browser=None, seen=None: [send] if seen else [read])
    assistant.handle("reply to Rahul's email saying I'll join at 5")
    assert [a for a, _ in actions(fake_mac)] == ["read_mail"]
    assert spoken(fake_mac)[-1] == 'Should I email Rahul: "I\'ll join at 5"?'
    assistant.handle("Yes.")
    assert actions(fake_mac)[-1] == ("send_email", ("Rahul", "Re: Sync", "I'll join at 5"))


def test_no_means_nothing_is_sent(fake_mac, assistant, monkeypatch):
    monkeypatch.setattr(slow, "plan", lambda *a, **k: [{"action": "send_message", "args": ["Mom", "I'm late"]}])
    assistant.handle("text mom that I'm late")
    assistant.handle("No, don't.")
    assert not any(a == "send_message" for a, _ in actions(fake_mac))
    assert spoken(fake_mac)[-1] == "Okay, I won't send it"


def test_something_else_instead_of_yes_drops_the_send(fake_mac, assistant, monkeypatch):
    monkeypatch.setattr(slow, "plan", lambda *a, **k: [{"action": "send_message", "args": ["Mom", "I'm late"]}])
    assistant.handle("text mom that I'm late")
    assistant.handle("open safari")
    assistant.handle("yes")
    assert ("open_app", ("Safari",)) in actions(fake_mac)
    assert not any(a == "send_message" for a, _ in actions(fake_mac))


def test_a_routine_is_taught_then_used(fake_mac, assistant):
    assistant.handle("When I say start my work day, open Safari and open Notes")
    assert actions(fake_mac) == [] and spoken(fake_mac)[-1] == "Okay. When you say start my work day, I'll open Safari and open Notes"
    assistant.handle("Start my work day.")
    assert actions(fake_mac) == [("open_app", ("Safari",)), ("wait_for_app", ("Safari",)), ("open_app", ("Notes",))]
