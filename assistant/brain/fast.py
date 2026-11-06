"""The fast brain: exact rules first, then Laya picks the intent of each clause, all clauses in one forward pass.

decide() splits the sentence into clauses ("take a screenshot and lock my mac" -> 2 clauses) and returns
the steps, or None to hand the whole command to the LLM when any clause is:
  - below CONFIDENCE, or "other"
  - missing details the rules in language/ can't find
When the only unsure clauses are in-app ones ("... and scroll down"), info["parts"] says which clauses
the LLM should plan, so the rest still takes the fast path.

Test from the terminal:
    python -m assistant.brain.fast "open chrome and search youtube for neural networks"
"""

import sys
import threading
import time

from assistant.config import TRAIN_DIR
from assistant.language import arguments, catalog, in_app, spoken_math
from assistant.language.sentence import clean, split_commands

FINETUNED = TRAIN_DIR / "laya-mac"  # made by train/finetune.py; used when it exists
BASE = "convaiinnovations/laya"     # the base model is near-random on these intents: always fine-tune
CONFIDENCE = 0.6

# Laya's options. The descriptions are its "criteria": Laya reads them, so wording matters.
# Changing this set means retraining (the old checkpoint gives garbage for a new option set).
INTENTS = {
    "youtube_search": "search for or play videos on YouTube",
    "web_search": "search Google or the web for information",
    "open_website": "go to a website address like github.com or gmail",
    "open_app": "open, launch or start an application",
    "close_app": "close, quit or exit an application",
    "open_folder": "open a folder like downloads, documents or desktop",
    "media_play_pause": "play, pause or resume music",
    "next_track": "skip to the next song",
    "previous_track": "go back to the previous song",
    "set_volume": "set the volume to a specific number",
    "volume_up": "make the sound louder",
    "volume_down": "make the sound quieter",
    "mute": "mute or silence the sound",
    "type_text": "type or write some text on the keyboard",
    "screenshot": "take a screenshot or capture the screen",
    "lock_screen": "lock the computer or screen",
    "other": "anything else, questions, or tasks not listed",
}
QUESTIONS = {"intent": {"type": "choice", "instructions": "What does the user want the Mac to do?", "criteria": INTENTS}}

# Families: intents that share a phrasing and differ only in details the words decide exactly.
# Laya's confidence for the family is the sum over its members; the rules then pick the member.
OPEN_FAMILY = ("open_app", "open_website", "open_folder")        # "open X": X is an app, a site or a folder
OPEN_LIKE = OPEN_FAMILY + ("youtube_search", "web_search")       # "open youtube" can look like a search to Laya
VOLUME_FAMILY = ("set_volume", "volume_up", "volume_down", "mute")
SEARCH_FAMILY = ("web_search", "youtube_search")                 # "search google for X" is web, whatever Laya leans to
MEDIA_FAMILY = ("media_play_pause", "next_track", "previous_track")

_agent = None
_lock = threading.Lock()  # one forward pass at a time (wake_gpu runs in the background)
_last_wake = 0.0


def decide(text: str, browser: str | None = None) -> tuple[list[dict] | None, dict]:
    """(steps or None, info about the decision for the log). `browser` is the one in use (None = chrome)."""
    text = clean(text)
    # "on this page, search for X and open the first result" is one task for the screen agent, not two commands
    clauses = [text] if in_app.screen_task(text) or in_app.watch_request(text) else split_commands(text)
    info = {"clauses": []}
    steps, parts = [], []
    for clause, probs in zip(clauses, classify_many(clauses)):
        intent, conf, args = _resolve(clause, text, probs, browser)
        info["clauses"].append({"text": clause, "intent": intent, "confidence": round(conf, 3)})
        if intent == "other":
            info["handoff"] = "other"
        elif conf < CONFIDENCE:
            info["handoff"] = "low confidence"
        elif args is None:
            info["handoff"] = "missing details"
        else:
            decided = [{"action": intent, "args": args}]
            if intent == "calculate" and spoken_math.mentions_calculator(clause) and \
                    not any(s["args"] == ["Calculator"] for s in steps):
                decided.insert(0, {"action": "open_app", "args": ["Calculator"]})  # "in calculator, 25 into 25"
            steps += decided
            parts.append({"steps": decided})
            continue
        if not in_app.is_ui_command(clause) or len(clauses) == 1:
            return None, info
        parts.append({"llm": clause})  # an unsure in-app clause: only this one goes to the LLM
    if any("llm" in p for p in parts):
        info["parts"] = parts
        return None, info
    return steps, info


def _resolve(clause: str, sentence: str, probs: dict[str, float],
             browser: str | None) -> tuple[str, float, list[str] | None]:
    """(intent, confidence, args) for one clause. Exact rules first (confidence 1.0), then Laya's pick,
    corrected by the words within its family."""
    def pick_browser() -> str:
        return catalog.pick_browser(clause, sentence, browser)

    # 1. Exact rules: no model needed.
    question = in_app.camera_question(clause)
    if question:  # "what am I holding", "how do I look": one photo from the camera
        return "look_through_camera", 1.0, [question]
    task = in_app.screen_task(clause)
    if task:  # "on this page, find the pricing and open it": step by step on the screen
        if not (in_app.click_target(task) or in_app.scroll_args(task) or in_app.shortcut(task)):
            return "do_on_screen", 1.0, [task]
        clause = task  # "on this page, click sign in" is one plain click
    watched = in_app.watch_request(clause)
    if watched:  # "tell me when Build succeeded shows up"
        return "watch_screen", 1.0, [watched]
    if in_app.stop_watching_request(clause):
        return "stop_watching", 1.0, []
    question = in_app.screen_question(clause)
    if question:  # "what's on my screen", "read this": the vision model looks
        return "look_at_screen", 1.0, [question]
    note = arguments.note_text(clause)
    if note is not None:  # "write a note saying buy milk", "write buy milk in notes", "create a new note"
        return "write_note", 1.0, [note] if note else []
    if in_app.play_pause_request(clause):  # "stop the music" (Laya: mute), "resume", "pause the video"
        what = arguments.play_or_pause(clause)
        return "media_play_pause", 1.0, [what] if what else []
    asked = arguments.ask_ai(clause)
    if asked:  # "ask chatgpt what is X": the site with the question already asked
        return "open_website", 1.0, [arguments.ask_url(*asked), pick_browser()]
    clear = in_app.clear_request(clause)
    if clear:  # "remove that", "clear the chat box": the text in the box on screen, nothing else
        return "clear_text", 1.0, [clear]
    span = in_app.chart_range(clause)
    if span:  # "give me the last 5 days" on a stock page: its "5D" button (Laya read it as "previous song")
        return "click", 1.0, [span]
    if spoken_math.calculator_redo(clause):  # "do it in the calculator": the last sum again, in Calculator.app
        return "calculate", 1.0, ["again"]
    expression = spoken_math.math_expression(clause)
    if expression:  # "what is 25 into 25"
        return "calculate", 1.0, [expression]
    if arguments.personal_task(clause):  # mail, messages, calendar: the LLM plans (and sending waits for a yes)
        return "other", 1.0, None
    tab = in_app.tab_by_name(clause)
    if tab:  # "go to the ChatGPT tab": that open tab (the LLM pressed cmd+T here, 2026-10-04)
        return "switch_tab", 1.0, [tab, pick_browser()]
    step = in_app.tab_step(clause) if browser else None
    if step:  # in a browser, "previous step" (Whisper's "tab") was previous_track
        clause = step
    keys = in_app.shortcut(clause)  # "open a new tab", "press enter"
    if keys:
        return "press_keys", 1.0, [keys] + ([pick_browser()] if in_app.is_browser_command(clause) else [])
    scroll = in_app.scroll_args(clause)  # "scroll up a bit", "scroll to the bottom"
    if scroll:
        return "scroll", 1.0, scroll
    target = in_app.click_target(clause)  # "click on the sign in button"
    if target:
        return "click", 1.0, [target]
    if in_app.is_ui_command(clause):  # other in-app commands: the LLM plans those
        return "other", 1.0, None
    if arguments.volume_direction(clause) == "unmute":  # not one of Laya's options: it would pick "mute"
        return "unmute", 1.0, []

    # 2. Laya's pick, with the words deciding inside a family.
    intent = max(probs, key=probs.get)
    explicit = catalog.is_open_command(clause)
    if intent in OPEN_FAMILY or explicit:
        # "open X" where X is a real app/site/folder is decided by the words, whatever Laya thinks.
        found = catalog.open_candidates(clause)
        if found:
            conf = sum(probs[i] for i in OPEN_LIKE)
            conf = max(conf, CONFIDENCE) if explicit else conf
            if "open_app" in found:  # an installed app beats its website ("open claude")
                found.pop("open_website", None)
            best = max(found, key=probs.get)
            return best, conf, found[best] + ([pick_browser()] if best == "open_website" else [])
        missing = catalog.missing_app(clause)
        if missing and (intent in ("open_app", "close_app") or explicit):
            return "answer", 1.0, [f"{missing} isn't installed on this Mac"]
        if intent in OPEN_FAMILY:
            return intent, sum(probs[i] for i in OPEN_FAMILY), None
    if intent in VOLUME_FAMILY:
        conf = sum(probs[i] for i in VOLUME_FAMILY)
        if arguments.number(clause):
            return "set_volume", conf, [arguments.number(clause)]
        direction = arguments.volume_direction(clause)
        if direction:  # "turn it up by 20": a bigger step than the usual 10
            step = arguments.volume_step(clause) if direction in ("volume_up", "volume_down") else None
            return direction, conf, [step] if step else []
        return intent, conf, arguments.args_for(intent, clause, sentence, browser)
    if intent in MEDIA_FAMILY:
        title = arguments.play_title(clause)
        if title:  # "play lofi on youtube" is a search
            return "youtube_search", probs[intent] + probs["youtube_search"], [title, pick_browser()]
        kind = arguments.media_kind(clause)
        # No music word at all ("okay, now...", "perform the top"): not a media command, whatever Laya says.
        if kind == "media_play_pause":  # "pause" never starts music, "play" never stops it
            what = arguments.play_or_pause(clause)
            return kind, sum(probs[i] for i in MEDIA_FAMILY), [what] if what else []
        return (kind, sum(probs[i] for i in MEDIA_FAMILY), []) if kind else (intent, probs[intent], None)
    if intent in SEARCH_FAMILY:
        kind = arguments.search_kind(clause) or intent
        conf = sum(probs[i] for i in SEARCH_FAMILY)
        if arguments.play_title(clause):  # "play <title>": play/pause and YouTube are the same reading
            conf += probs["media_play_pause"]
        return kind, conf, arguments.args_for(kind, clause, sentence, browser)
    return intent, probs[intent], arguments.args_for(intent, clause, sentence, browser)


# --- the model ------------------------------------------------------------------------------------

def classify_many(clauses: list[str]) -> list[dict[str, float]]:
    """Probability of every intent, for each clause, in one forward pass."""
    with _lock:
        results = _get_agent().predict_batch(clauses, QUESTIONS)
    return [r["answers"]["intent"]["probabilities"] for r in results]


def _get_agent():
    # laya.agent.Agent directly instead of laya.Router: only the English checkpoint is used,
    # so the Router's language detection would be wasted time on every command.
    global _agent
    if _agent is None:
        from laya.agent import Agent
        _agent = Agent(str(FINETUNED) if FINETUNED.exists() else BASE, device="mps")
    return _agent


def warm_up() -> None:
    """Load the model once at startup (~10 s) so commands are fast."""
    classify_many(["open safari", "take a screenshot"])


def wake_gpu() -> None:
    """Call when the user starts speaking. After a few idle seconds the GPU clocks down and
    Laya takes 150-320 ms instead of ~80 ms; one throwaway pass now wakes it before the
    real command arrives. Runs in the background, at most once a second."""
    global _last_wake
    if _agent is None or time.monotonic() - _last_wake < 1.0:
        return
    _last_wake = time.monotonic()
    threading.Thread(target=classify_many, args=(["open safari"],), daemon=True).start()


if __name__ == "__main__":
    warm_up()
    t = time.perf_counter()
    steps, info = decide(" ".join(sys.argv[1:]))
    print(steps, info, f"({(time.perf_counter() - t) * 1000:.0f} ms)")
