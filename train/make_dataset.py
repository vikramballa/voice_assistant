"""Build the Laya training set: train/dataset.jsonl with {"text", "intent"} rows.

Three sources:
  1. Templates  - many ways of phrasing each command, filled with apps,
                  search topics, numbers, etc. Labels are always correct.
  2. Local LLM  - qwen3 (Ollama) writes extra casual phrasings per intent,
                  so Laya also sees wording the templates don't cover.
  3. learned.jsonl - your real commands the LLM handled (train/learn_from_logs.py).
Sentences that appear in test_set.jsonl are removed so the test stays fair.

Run from the project folder:
    python train/make_dataset.py            # templates + LLM (~2 min)
    python train/make_dataset.py --no-llm   # templates only
"""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from assistant.brain.fast import INTENTS  # noqa: E402
from assistant.language.catalog import SITES as KNOWN_SITES, installed_apps  # noqa: E402
from assistant.language.words import normalize  # noqa: E402

random.seed(0)

APPS = ["safari", "chrome", "google chrome", "spotify", "notes", "calendar", "mail", "messages", "terminal",
        "vs code", "slack", "zoom", "photos", "music", "preview", "xcode", "whatsapp", "discord",
        "the calculator", "facetime", "reminders", "maps", "the app store", "pages", "keynote",
        "finder", "system settings", "settings", "activity monitor", "textedit", "photo booth"]
APPS += sorted(installed_apps())  # every app actually on this Mac
TOPICS = ["neural networks", "machine learning", "lofi beats", "how to cook biryani", "world news",
          "the latest iphone", "gradient descent", "football highlights", "python decorators",
          "stock market today", "cheap flights to goa", "best laptops 2026", "quantum computing",
          "how to tie a tie", "taylor swift songs", "react hooks tutorial", "the weather tomorrow",
          "chess openings", "yoga for beginners", "mars rover", "linear algebra", "funny dog videos",
          "restaurants near me", "movie reviews", "the history of rome", "docker basics"]
SITES = ["github.com", "gmail", "youtube", "reddit.com", "stackoverflow.com", "amazon.in", "chatgpt",
         "linkedin", "netflix", "claude.ai", "instagram", "twitter.com", "google.com", "python.org",
         "wikipedia", "arxiv.org", "kaggle.com", "huggingface.co", "medium.com", "reddit", "wikipedia.org"]
SITES += sorted(KNOWN_SITES)  # gemini, google drive, notion... every web service the catalog knows
SITE_TAIL = ["", "", "", "", " in my google account", " on my google account", " for me", " please"]
MUSIC = ["arijit singh songs", "some lofi", "despacito", "old hindi songs", "coldplay", "relaxing piano music",
         "the new taylor swift song", "punjabi songs", "rain sounds", "ar rahman hits"]
FOLDERS = ["downloads", "documents", "desktop", "pictures", "movies", "home", "applications"]
TEXTS = ["hello world", "good morning", "i will be late today", "thanks a lot", "see you soon",
         "meeting at five", "call me back", "lorem ipsum", "my name is vikram", "ok sounds good",
         "buy milk", "see you tomorrow", "dear sir", "happy birthday", "the quick brown fox", "on my way"]
BROWSER = ["", "", "", " in chrome", " on safari", " in safari", " using chrome"]
OPEN_B = ["", "", "", "open chrome and ", "open safari and ", "launch chrome and "]

TEMPLATES = {
    "youtube_search": ["search youtube for {t}", "youtube {t}", "{o}search {t} on youtube", "play {t} on youtube",
                       "find videos about {t}", "show me a video on {t}", "{o}look up {t} on youtube",
                       "i want to watch {t}", "play a video about {t}{b}", "youtube search {t}",
                       "find a {t} tutorial on youtube", "watch {t} videos", "play {m} on youtube"],
    "web_search": ["google {t}", "search for {t}{b}", "{o}google {t}", "search the web for {t}",
                   "look up {t}", "can you google {t}", "search about {t}", "google {t} near me", "search google for {t}", "find information about {t}",
                   "what does the internet say about {t}", "do a web search for {t}", "{o}search {t}"],
    "open_website": ["open {s}", "go to {s}{b}", "visit {s}", "take me to {s}", "{o}go to {s}",
                     "open the {s} website", "load {s}", "navigate to {s}", "open {s}{b}", "visit {s}{b}",
                     "open the {s} site", "pull up {s}", "please open {s}{g}", "open my {s}{g}", "open {s}{g}",
                     "can you open {s}{g}"],
    "open_folder": ["open {f}", "open my {f} folder", "show me the {f} folder", "open the {f} folder",
                    "go to my {f}", "show my {f}", "open {f} in finder", "take me to the {f} folder"],
    "next_track": ["next song", "skip", "skip this song", "next track", "play the next song", "skip it",
                   "go to the next track", "change the song", "i don't like this song skip", "next"],
    "previous_track": ["previous song", "go back a song", "previous track", "play the last song again",
                       "back to the previous song", "play the previous track", "last song", "go back one track"],
    "open_app": ["open {a}", "launch {a}", "start {a}", "open up {a}", "can you open {a}", "fire up {a}",
                 "run {a}", "bring up {a}", "please open {a}", "{a}", "open {a} app", "open the {a} application"],
    "close_app": ["close {a}", "quit {a}", "exit {a}", "shut {a}", "kill {a}", "close the {a} app",
                  "please quit {a}", "get rid of {a} window", "force quit {a}"],
    "media_play_pause": ["pause", "play", "pause music", "resume music", "stop the song", "play the song",
                         "pause the song", "continue playing", "resume playback", "pause playback", "unpause",
                         "resume the music", "continue the song", "resume it", "pause it", "play it",
                         "hit play", "unpause the music", "stop music", "start the music again"],
    "set_volume": ["set volume to {n}", "volume {n}", "volume to {n} percent", "make the volume {n}",
                   "change volume to {n}", "put volume at {n}", "set the sound to {n}", "volume at {n}"],
    "volume_up": ["volume up", "louder", "turn up the volume", "turn it up", "increase volume",
                  "raise the volume", "make it louder", "a bit louder", "crank it up", "i can't hear it",
                  "sound up", "turn the sound up"],
    "volume_down": ["volume down", "quieter", "turn down the volume", "turn it down", "decrease volume",
                    "lower the volume", "make it softer", "a bit quieter", "too loud", "reduce the sound",
                    "sound down", "turn the sound down"],
    "mute": ["mute", "mute it", "mute audio", "mute the volume", "silence", "turn off the sound",
             "shut the sound off", "no sound please", "mute everything", "kill the audio"],
    "type_text": ["type {x}", "write {x}", "write {x}", "type out {x}", "write out {x}", "please type {x}", "enter the text {x}",
                  "write down {x}", "type the words {x}", "can you type {x}"],
    "screenshot": ["screenshot", "take a screen shot", "grab a screenshot", "capture my screen",
                   "screen capture", "snap the screen", "save a picture of my screen", "screenshot this",
                   "take a picture of the screen", "capture the display"],
    "lock_screen": ["lock", "lock computer", "lock the mac", "lock it", "lock my computer",
                    "lock the display", "secure my mac", "lock my screen now", "lock up the laptop"],
    # time/date/battery are "other" for Laya: the LLM picks tell_time / battery_status (a 19-intent
    # model was tried 2026-09-26 and made 4 wrong decisions, so the 17-intent model stays)
    "other": ["what time is it", "what's the date", "how much battery do i have", "is my mac charging",
              "tell me a joke", "what's the capital of japan",
              "read the containers what are there", "turn off dark mode", "enable night mode", "how far is the moon",
              "explain neural networks", "what's 12 plus 30", "good morning", "you are awesome", "stop listening", "who is the president of america", "how are you",
              "send a message to mom", "set an alarm for 7", "what's the weather", "remind me to call john",
              "book a cab", "read my emails", "how much is 25 times 4", "delete all my files",
              "shut down the computer", "turn on bluetooth", "connect to wifi", "summarize this page",
              "translate hello to hindi", "create a new folder", "empty the trash", "reply to that email",
              "what is machine learning", "call dad", "make a presentation", "rename this file",
              "who won the world cup", "move this window left", "hello", "never mind", "thank you",
              "what can you do", "schedule a meeting tomorrow", "order food", "write me a poem about rain",
              "turn on dark mode", "restart my mac", "how do i fix my code", "copy that", "undo",
              "order me a pizza", "buy a new phone", "email my boss", "what's on my calendar today",
              "write an essay about climate change", "open the pod bay doors", "how tall is mount everest",
              "delete this file", "is it going to rain",
              "send a whatsapp to riya", "make me a sandwich", "who are you", "book a flight to delhi"]
}


def fill(template: str) -> str:
    return template.format(
        t=random.choice(TOPICS), a=random.choice(APPS), s=random.choice(SITES), n=random.randint(0, 100),
        x=random.choice(TEXTS), b=random.choice(BROWSER), o=random.choice(OPEN_B), f=random.choice(FOLDERS),
        m=random.choice(MUSIC), g=random.choice(SITE_TAIL),
    ).strip()


def from_templates(per_intent: int = 80) -> list[dict]:
    rows = []
    for intent, templates in TEMPLATES.items():
        seen = set()
        for _ in range(per_intent * 5):
            if len(seen) >= per_intent:
                break
            seen.add(fill(random.choice(templates)))
        rows += [{"text": s, "intent": intent} for s in seen]
    return rows


def from_llm(per_intent: int = 25) -> list[dict]:
    import ollama

    rows = []
    schema = {"type": "object", "properties": {"sentences": {"type": "array", "items": {"type": "string"}}},
              "required": ["sentences"]}
    for intent, desc in INTENTS.items():
        prompt = (f"Write {per_intent} different short things a person might SAY to a voice assistant on "
                  f"their Mac when they want to: {desc}. Casual spoken English, lowercase, varied wording, "
                  f"real app names / topics / numbers where needed. No numbering.")
        if intent == "other":
            prompt = (f"Write {per_intent} different short things a person might say to a voice assistant that "
                      f"are NOT about: searching the web or YouTube, opening/closing apps or websites, music "
                      f"play/pause, volume, typing text, screenshots, locking the screen, the time or date, or "
                      f"the battery. Mix questions, "
                      f"chit-chat and other computer tasks. Lowercase.")
        resp = ollama.chat(model="qwen3:4b", messages=[{"role": "user", "content": prompt}],
                           format=schema, think=False, options={"temperature": 0.9})
        sentences = json.loads(resp.message.content)["sentences"]
        rows += [{"text": s.strip().lower().rstrip("."), "intent": intent} for s in sentences if s.strip()]
        print(f"  llm {intent}: {len(sentences)}")
    return rows


def main() -> None:
    rows = from_templates()
    if "--no-llm" not in sys.argv:
        rows += from_llm()
    learned = ROOT / "train" / "learned.jsonl"  # real commands, from learn_from_logs.py
    if learned.exists():
        rows += [json.loads(l) for l in learned.read_text().splitlines() if l.strip()]
    held_out = [ROOT / "train" / "test_set.jsonl", ROOT / "train" / "dev_set.jsonl"]  # never train on these
    test = {normalize(json.loads(l)["text"]) for f in held_out if f.exists()
            for l in f.read_text().splitlines() if l.strip()}
    unique = {}
    for r in rows:
        r["text"] = r["text"].strip().rstrip("?.!")
        if normalize(r["text"]) not in test:
            unique.setdefault(r["text"], r)
    rows = list(unique.values())
    random.shuffle(rows)
    out = ROOT / "train" / "dataset.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    counts = {i: sum(r["intent"] == i for r in rows) for i in INTENTS}
    print(f"wrote {len(rows)} rows to {out}")
    print(counts)


if __name__ == "__main__":
    main()
