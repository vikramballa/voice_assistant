"""The pipeline for one command, whatever it came from (voice or typed):

    text -> a yes / no to "Should I send it?"   |   "when I say X, do Y": a routine to remember
         -> unfinished? wait for the rest
         -> decide:  taught phrase  |  "in Brave" redo  |  fast brain (Laya)  |  slow brain (LLM) if Laya isn't sure
         -> act:     task.run: the steps; after a lookup (mail, calendar) the LLM plans on; sending waits for a yes
         -> reply:   speak, show on the island, log one row
         -> learn:   "no, I meant X" right after a command: X's steps are remembered for those words (phrasebook.py)

Everything the assistant remembers between commands is in self.session (session.py).
"""

import re
import threading
import time
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait

from assistant import logbook, phrasebook, task
from assistant.actions import WEB_ACTIONS, web
from assistant.actions import screen
from assistant.audio.speaker import speak
from assistant.brain import fast, proofread, slow
from assistant.language import dialogue, spelling
from assistant.language.catalog import SITES
from assistant.mac import island
from assistant.session import Session

screen.answer_from_text = slow.answer_about_screen  # "read my chats": the LLM answers from the screen's text

FOLLOW_UP_SECONDS = 5    # after a bare "Hey Laya" or a question back, how long to wait for the answer
TRAIL_SECONDS = 4        # "search youtube for..." waits this long for the rest
ASK_AFTER_SECONDS = 1.2  # ...and asks "Search for what?" out loud if nothing came by then
LOADING_NOTE_SECONDS = 15  # startup prints which models are still loading this often, so it never looks frozen
ISLAND_TEXT = 90         # longest reply shown on the island
CORRECTION_SECONDS = 20  # "no, I meant ..." this soon after a command corrects it
CONFIRM_SECONDS = 15     # "Should I send it?" waits this long for a yes
# A plain "no, X" may be a change of mind, not a correction: it only teaches when X reads like the earlier
# command ("open not" -> "open notes" 0.89), or a bit like it when the earlier one failed.
SIMILAR_ENOUGH = 0.75
SIMILAR_AFTER_FAILURE = 0.5
# Their reply is the whole point: always said in full (other actions just say "Done").
INFO_ACTIONS = ("answer", "tell_time", "battery_status", "weather", "calculate", "look_at_screen", "read_mail",
                "calendar_events", "do_on_screen", "look_through_camera", "watch_screen", "stop_watching")
# The LLM didn't understand. Such a question is never waited on: background talk would be glued onto it and
# asked about again every few seconds (2026-09-28: 6 replies in 25 s, each cutting off the last).
UNCLEAR = re.compile(r"sorry, i didn'?t (?:catch|get) that|\bi'?m (?:still )?not sure (?:i|what)\b"
                     r"|\bi (?:don'?t|didn'?t|do not|did not) (?:understand|follow)\b"
                     r"|\bcould you (?:please )?(?:clarify|repeat|rephrase)", re.I)


class Assistant:
    def __init__(self, session: Session | None = None):
        self.session = session or Session()

    def handle(self, text: str, timings: dict | None = None, quiet_unclear: bool = False,
               fast_only: bool = False) -> list[dict]:
        """Do what `text` asks; returns the steps that were planned.
        quiet_unclear (conversation mode, no wake word): something that isn't a request is skipped silently,
        because it was probably not meant for us.
        fast_only (said while our reply was playing): only a command the fast brain is sure about is done;
        nothing goes to the LLM, so a piece of our own reply heard back can never start an answer loop."""
        timings = dict(timings or {})
        to_confirm = self.session.take_steps_to_confirm()
        if to_confirm and dialogue.yes_or_no(text) is not None:
            return self._confirmed(text, to_confirm, dialogue.yes_or_no(text))
        routine = dialogue.routine(text)
        if routine:
            return self._teach(*routine, timings)
        pending = self.session.take_pending()
        if pending:
            text = dialogue.complete(pending, text)
        fixes = []
        if not self._writing_app() or dialogue.sounds_like_command(text):  # in Notes, plain words are the user's own
            text, fixes = spelling.correct(text)  # "open spotifi" -> "open Spotify" (names only, never dictation)
        if fixes:
            print("  [spelling: " + ", ".join(f"{a!r} -> {b!r}" for a, b in fixes) + "]")
        corrects, corrected_steps = (None, []) if pending else self._corrects(text)
        if corrects:
            text = dialogue.correction(text)[1]  # "no, I meant open notes" -> "open notes"
        question = dialogue.follow_up_question(text)
        if fast_only and (question or dialogue.trails_off(text)):
            print(f"  (said over our reply, not a whole command: {text!r})")
            return []
        if question or dialogue.trails_off(text):  # "set the volume to..." - the rest may still be coming
            print(f"  [waiting for the rest of {text!r}]")
            self._wait_for_rest(text, question)
            return []
        steps, decision, path = self._decide(text, timings, fast_only)
        if steps is None:  # fast_only and the fast brain wasn't sure
            print(f"  (said over our reply, not a clear command: {text!r})")
            return []
        unclear = not steps or (len(steps) == 1 and steps[0]["action"] == "answer"
                                and bool(UNCLEAR.search(steps[0]["args"][0])))
        if quiet_unclear and unclear:
            print(f"  (not a command: {text!r})")
            island.show("hide")
            return []

        t = time.perf_counter()
        outcome = task.run(steps, self._planner(text, path), self.say)
        timings["action_ms"] = _ms_since(t)
        self.session.note_browser(outcome.ran)
        self.session.note_working_app(outcome.ran)
        _hear_as_dictation(self.session.typed if self._writing_app() else None)
        _note_front_app()

        error = outcome.error
        if outcome.ran and not error:
            self.session.remember(text, outcome.ran)
        worked = bool(outcome.ran) and not error and not unclear and any(s["action"] != "answer" for s in outcome.ran)
        self.session.note_command(text, outcome.ran, worked)
        learned = bool(corrects) and worked and path != "redo" \
            and not _only_browser_changed(corrected_steps, outcome.ran) and phrasebook.learn(corrects, outcome.ran)
        if learned:
            print(f"  [learned: {phrasebook.key(corrects)!r} now means this]")
        reply = reply_for(outcome.last, outcome.replies, error, outcome.error_is_message)
        if outcome.waiting and not error:  # "Should I email Rahul ...?": runs on a yes
            reply = task.confirmation(outcome.waiting[0])
            self.session.wait_for_yes(outcome.waiting, CONFIRM_SECONDS)
        elif steps and reply.endswith("?") and not error and not unclear:  # the LLM asked back ("Open what?")
            self.session.wait_for(text, FOLLOW_UP_SECONDS + 3)
        self._finish(text, path, outcome.ran + outcome.waiting, reply, error, timings,
                     laya=decision, **({"corrects": corrects, "learned": learned} if corrects else {}),
                     **({"spelling": fixes} if fixes else {}))
        return outcome.ran + outcome.waiting

    def _finish(self, text: str, path: str, steps: list[dict], reply: str, error: str | None, timings: dict,
                **log) -> None:
        """Say the reply, show it on the island, print and log one row."""
        self.say(reply)
        if self.session.is_waiting():
            island.show("listening", reply)
        else:
            island.show("error" if error or not steps else "reply",
                        reply if len(reply) < ISLAND_TEXT else reply[:ISLAND_TEXT - 3] + "...")
        shown = ", ".join(f"{s['action']}{tuple(s['args'])}" for s in steps) or "nothing"
        print(f"  [{path}] {shown}  {timings}" + (f"  ERROR {error}" if error else ""))
        print(f"  said: {reply}")
        logbook.add({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "text": text, "path": path, "steps": steps,
                     "error": error, **timings, **log})

    def _planner(self, text: str, path: str) -> task.PlanNext | None:
        """After a lookup in an LLM plan, the LLM plans on from what was found."""
        if path not in ("llm", "laya+llm"):
            return None
        return lambda seen: slow.plan(text, self.session.recent(), self.session.browser, seen)

    def _confirmed(self, text: str, steps: list[dict], yes: bool) -> list[dict]:
        """The user's answer to "Should I send it?"."""
        if not yes:
            self._finish(text, "confirm", [], "Okay, I won't send it", None, {})
            return []
        outcome = task.run(steps, confirmed=True)
        self._finish(text, "confirm", outcome.ran, reply_for(outcome.last, outcome.replies, outcome.error,
                                                              outcome.error_is_message), outcome.error, {})
        return outcome.ran

    def _teach(self, name: str, meaning: str, timings: dict) -> list[dict]:
        """"When I say start my work day, open VS Code and Brave": remembered, not done now."""
        steps, _, path = self._decide(meaning, timings)
        if steps and phrasebook.learn(name, steps):
            reply = f"Okay. When you say {name}, I'll {task.describe(steps)}"
        else:
            reply, steps = "Sorry, I can't make a routine of that", []
        self._finish(name, "routine", steps, reply, None, timings, meaning=meaning)
        return []

    def _decide(self, text: str, timings: dict, fast_only: bool = False) -> tuple[list[dict] | None, dict, str]:
        """(steps, the decision for the log, path = which brain decided). Adds laya_ms / llm_ms to timings.
        With fast_only, steps is None when the LLM would be needed."""
        island.show("heard", text)
        _note_front_app()
        t = time.perf_counter()
        taught = phrasebook.lookup(text)
        redo = None if taught else dialogue.redo_in_browser(text)
        steps = taught or (self.session.redo_steps(redo) if redo else None)
        if taught:
            decision, path = {"taught": phrasebook.key(text)}, "memory"
        elif steps:
            decision, path = {"redo_in": redo}, "redo"
        else:
            steps, decision = fast.decide(text, self.session.browser)
            path = "laya"
            timings["laya_ms"] = _ms_since(t)
            writing = None if fast_only else self._writing_app()
            words = writing and dialogue.dictated(text, steps is not None)
            if words:  # in Notes: "I'm a developer" / "continue, I'm a developer" is written into the note
                steps, decision, path = self._dictate(words, timings), {"dictation": words}, "dictation"
            elif steps:
                steps = self._in_browser(text, steps)
        if steps:
            steps = self._type_in_working_app(steps)
        timings.setdefault("laya_ms", _ms_since(t))
        if steps is not None or fast_only:
            return steps, decision, path

        island.show("thinking")
        t = time.perf_counter()
        recent, browser = self.session.recent(), self.session.browser
        if "parts" in decision:  # "open youtube in brave and scroll down": the LLM only plans the in-app part
            path, steps = "laya+llm", []
            for part in decision["parts"]:
                steps += part["steps"] if "steps" in part else slow.plan(part["llm"], recent, browser)
        else:
            path, steps = "llm", slow.plan(text, recent, browser)
        timings["llm_ms"] = _ms_since(t)
        if self._writing_app() and steps and all(s["action"] == "answer" for s in steps) and not dialogue.asks_laya(text):
            # in Notes, "Make sure to call mom" is not a request (the LLM only chatted back): write it
            steps, path = self._type_in_working_app(self._dictate(dialogue.as_dictation(text), timings)), "dictation"
        return steps, decision, path

    def _dictate(self, words: str, timings: dict) -> list[dict]:
        """type_text for dictated words, with misheard words fixed ("And a hackle." -> "And a hacker.")."""
        if words.strip():
            t = time.perf_counter()
            words, fixes = proofread.correct(words, self.session.typed)
            timings["proofread_ms"] = _ms_since(t)
            if fixes:
                timings["autocorrect"] = fixes
                print("  [auto-correct: " + ", ".join(f"{a!r} -> {b!r}" for a, b in fixes) + "]")
        return [{"action": "type_text", "args": [words]}]

    def _in_browser(self, text: str, steps: list[dict]) -> list[dict]:
        """While working in a browser, a site goes there: "go to ChatGPT" switches to its open tab (it opened
        the ChatGPT app, 2026-10-04), or opens chatgpt.com in that browser. "the ChatGPT app" is still the app."""
        browser = self._browser_in_use()
        if not browser or len(steps) != 1:
            return steps
        action, args = steps[0]["action"], steps[0]["args"]
        if action == "open_app" and args and args[0].lower() in SITES and not re.search(r"\bapp\b", text, re.I):
            name, site, where = args[0], SITES[args[0].lower()], browser
        elif action == "open_website" and args and not re.search(r"[/?]", args[0]):
            name, site, where = args[0], args[0], args[1] if len(args) > 1 else browser
        else:
            return steps
        if web.find_tab(name, where):
            return [{"action": "switch_tab", "args": [name, where]}]
        return [{"action": "open_website", "args": [site, where]}] if action == "open_app" else steps

    def _browser_in_use(self) -> str | None:
        browser = self.session.browser_in_use()
        if browser and not _is_running(self.session.working_app):
            self.session.working_app = None
            return None
        return browser

    def _writing_app(self) -> str | None:
        """Notes / TextEdit... when the user is writing in it (forgotten once the user quit it by hand)."""
        app = self.session.writing_app()
        if app and not _is_running(app):
            self.session.working_app = None
            return None
        return app

    def _type_in_working_app(self, steps: list[dict]) -> list[dict]:
        """Typing goes into the working app (even when another window is in front), after a space when
        something was typed there before."""
        app = self._writing_app()
        if not app:
            return steps
        out, typed = [], self.session.typed
        for step in steps:
            if step["action"] == "type_text" and len(step["args"]) == 1:
                text = step["args"][0]
                space = typed and not typed[-1].isspace() and not re.match(r"[\s,.?!:;]", text)
                step = {"action": "type_text", "args": [(" " if space else "") + text, app]}
                typed += step["args"][0]
            elif step["action"] in ("open_app", "write_note") or step["action"] in WEB_ACTIONS:
                return out + steps[len(out):]  # the steps go elsewhere from here
            out.append(step)
        return out

    def _corrects(self, text: str) -> tuple[str | None, list[dict]]:
        """(the earlier command that `text` corrects, its steps): "no, I meant open notes" right after "open not".
        (None, []) when it isn't a correction."""
        found = dialogue.correction(text)
        last = self.session.last_command(CORRECTION_SECONDS)
        if not found or not last or dialogue.is_stop(found[1]):
            return None, []
        strong, meant = found
        before, steps, worked = last
        alike = dialogue.similar(before, meant)
        if strong or alike >= SIMILAR_ENOUGH or (not worked and alike >= SIMILAR_AFTER_FAILURE):
            return before, steps
        return None, []

    def ask(self, text: str, question: str) -> None:
        """Ask for the missing part; the next thing the user says completes `text`."""
        self.session.wait_for(text, FOLLOW_UP_SECONDS + 3)
        self.say(question)
        island.show("listening", question)

    def say(self, text: str) -> None:
        speak(text)

    def _wait_for_rest(self, text: str, question: str | None) -> None:
        """Listen for the rest without the wake word; if nothing comes soon, ask `question` out loud."""
        self.session.wait_for(text, TRAIL_SECONDS)
        island.show("listening", text.rstrip(" .") + "…")
        if question:
            token = self.session.pending_until

            def ask_if_still_waiting() -> None:
                if self.session.pending_text == text and self.session.pending_until == token:
                    self.ask(text, question)
            threading.Timer(ASK_AFTER_SECONDS, ask_if_still_waiting).start()


def _only_browser_changed(before: list[dict], after: list[dict]) -> bool:
    """"open youtube" in Chrome, then "no, in Brave": a choice for this time, not a lesson for those words."""
    def without_browser(steps):
        return [(s["action"], s["args"][:1]) if s["action"] in WEB_ACTIONS else (s["action"], s["args"]) for s in steps]
    return bool(before) and without_browser(before) == without_browser(after)


def reply_for(steps: list[dict], replies: list[str], error: str | None, error_is_message: bool) -> str:
    """The one sentence to say after running the steps."""
    if error:
        return error if error_is_message else "Sorry, that didn't work"
    if not steps:
        return "Sorry, I didn't get that. Can you say it again?"
    info = [r for s, r in zip(steps, replies) if s["action"] in INFO_ACTIONS]
    if info:
        return ". ".join(r.rstrip(".") for r in info)
    return replies[-1] if len(replies) == 1 else "Done"


def warm_up(use_mic: bool, fast_end: bool = False, use_say: bool = False) -> None:
    """Load all models at the same time instead of one after another (~6 s instead of ~15)."""
    from assistant.audio import speaker
    t = time.perf_counter()
    # transformers loads its parts lazily; importing it from several threads at once half-loads it
    # ("cannot import name 'AutoTokenizer'"), so it is imported here first.
    from transformers import AutoProcessor, AutoTokenizer  # noqa: F401
    jobs = {"Laya": fast.warm_up, "LLM": slow.warm_up, "voice": lambda: speaker.warm_up(use_say),
            "screen vision": _warm_up_vision}
    if use_mic:
        from assistant.audio import listen, transcribe, voice_id
        jobs |= {"speech detection": listen.warm_up, "Whisper": transcribe.warm_up, "voice check": voice_id.warm_up}
        if fast_end:
            from assistant.audio import turn
            jobs["end of turn"] = turn.warm_up
    # No `with`: leaving a `with` block waits for every model, so Ctrl+C during loading did nothing for a minute.
    pool = ThreadPoolExecutor()
    try:
        running = {pool.submit(job): name for name, job in jobs.items()}
        while True:
            done, pending = wait(running, timeout=LOADING_NOTE_SECONDS, return_when=FIRST_EXCEPTION)
            for job in done:
                job.result()  # a failed model stops startup with its own error
            if not pending:
                break
            print(f"  still loading: {', '.join(running[job] for job in pending)}...")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    print(f"ready in {time.perf_counter() - t:.1f}s")


def missing_models() -> list[str]:
    """The Hugging Face models that are not downloaded yet. Startup runs offline when there are none (__main__)."""
    from pathlib import Path
    from huggingface_hub.constants import HF_HUB_CACHE
    from assistant.audio import speaker, transcribe
    from assistant.brain import mlx_llm
    from assistant.mac import vision

    def downloaded(repo: str) -> bool:  # the libraries fetch only the files they use, so any snapshot counts
        folder = Path(HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}"
        try:
            return any((folder / "snapshots" / (folder / "refs" / "main").read_text().strip()).iterdir())
        except OSError:
            return False
    return [repo for repo in [fast.BASE, mlx_llm.MODEL, transcribe.WHISPER_MODEL, speaker.KOKORO_MODEL, vision.MODEL,
                              vision.AGENT_MODEL] if not downloaded(repo)]


def _warm_up_vision() -> None:
    """The screen model (1.8 GB) loads with the others at startup, so the first look isn't slow."""
    try:
        from assistant.mac import vision
        vision.warm_up()
    except Exception as e:
        print(f"  (screen vision unavailable: {str(e).splitlines()[0][:100]})")


def _hear_as_dictation(text_so_far: str | None) -> None:
    """Whisper's hint: the note so far while writing (app names pulled "I'm from" to "I'm the Chrome")."""
    import sys
    transcribe = sys.modules.get("assistant.audio.transcribe")  # only loaded when listening to the mic
    if transcribe:
        transcribe.set_dictation(text_so_far)


def _is_running(app: str) -> bool:
    try:
        from assistant.mac import accessibility
        return accessibility.running_app(app) is not None
    except Exception:
        return True


def _note_front_app() -> None:
    """Remember the app the user works in, so clicks / keys / scrolling go there and not to our terminal."""
    try:
        from assistant.mac import accessibility
        accessibility.note_front_app()
    except Exception:
        pass


def _ms_since(t: float) -> int:
    return round((time.perf_counter() - t) * 1000)
