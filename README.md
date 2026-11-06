# Laya — a fast, private voice assistant for macOS

![Python](https://img.shields.io/badge/python-3.12-blue)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Apple%20Silicon-lightgrey)
![Runs locally](https://img.shields.io/badge/runs-100%25%20local-brightgreen)
![Tests](https://img.shields.io/badge/tests-pytest-informational)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)

Say **"Hey Laya, open Chrome and search YouTube for neural networks"** and it is done about **0.9 seconds**
after you stop talking.

Laya listens, understands and controls your Mac: apps, browser, music, volume, typing, Mail, Messages,
Calendar, and even the screen itself. **Everything runs on your Mac.** No cloud, no API keys, no
subscription, and your voice never leaves the computer.

---

## Table of contents

- [Why Laya](#why-laya)
- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [What you can say](#what-you-can-say)
- [Safety and privacy](#safety-and-privacy)
- [Configuration](#configuration)
- [Project structure](#project-structure)
- [Architecture](#architecture)
- [Testing and results](#testing-and-results)
- [Making it smarter (training)](#making-it-smarter-training)
- [Extending Laya](#extending-laya)
- [Optional: vision model as a server](#optional-vision-model-as-a-server)
- [Troubleshooting](#troubleshooting)
- [Known limitations](#known-limitations)
- [Acknowledgements and licenses](#acknowledgements-and-licenses)
- [License](#license)

---

## Why Laya

Most voice assistants send every word to a big model in the cloud. That is slow (2–5 s) and not private.
Laya uses **two brains**:

| | Fast brain | Slow brain |
|---|---|---|
| What | [Laya](https://pypi.org/project/laya/), a 421M decision model, fine-tuned on Mac commands | Qwen3 4B, a local LLM (Apple MLX) |
| Speed | ~80 ms | ~0.8 s |
| Used for | common commands ("open Safari", "next song", "volume 30") | everything else (questions, mail, multi-step tasks) |
| Rule | acts **only when it is sure**, otherwise hands over | plans the steps; every plan is checked by plain rules |

About 80% of everyday commands never need the LLM, so most commands feel instant.

**Highlights**

- 🎙️ **Hands-free**: say "Hey Laya" once, then keep talking until you say "stop".
- ⚡ **Fast**: ~0.9 s from end of speech to action for common commands.
- 🔒 **Private**: speech, models and logs all stay on the Mac. Works offline after the first download.
- 🗣️ **Only your voice**: after a 30-second enrollment, other people and the TV are ignored.
- 👀 **Eyes**: reads the screen, clicks buttons, does small tasks on web pages, answers about the camera.
- ✉️ **Real tasks**: reads mail, writes replies, texts, checks your calendar — and **never sends without your "yes"**.
- 🧠 **Learns from you**: say "no, I meant …" and it remembers next time.

---

## How it works

```
 🎤 mic
  │
  ▼
 Speech detection ─────── Silero VAD + loudness gate (works in noisy rooms)
  │
  ▼
 Speech → text ────────── Whisper small.en on Apple MLX (~230 ms, started early at short pauses)
  │                       + voice check: is this the owner? (TitaNet, ~10 ms, runs at the same time)
  ▼
 Wake word / conversation mode / echo filter
  │
  ▼
 Text rules ───────────── clean the sentence, split "A and B", fix misheard names
  │
  ▼
 Fast brain (Laya) ────── one pass over all parts of the sentence (~80 ms)
  │            │
  sure ✓      unsure ✗
  │            ▼
  │       Slow brain (Qwen3 4B) plans the steps → rules check the plan (~0.8 s)
  ▼            │
 Actions ◄─────┘ ──────── open apps, search, type, click, mail, calendar …
  │
  ▼
 Spoken reply (Kokoro voice) + on-screen "island" + one row in the local log
```

**Example.** "Hey Laya, open Chrome and search YouTube for neural networks"

1. Whisper turns the audio into text.
2. The text rules split it into two parts and drop the extra "open Chrome" (the search already opens Chrome).
3. Laya is sure: `youtube_search`. The rules find the details: `"neural networks"`, browser `chrome`.
4. The action runs: `{"action": "youtube_search", "args": ["neural networks", "chrome"]}`.
5. Laya replies out loud. Total ≈ 0.9 s after you stopped speaking.

---

## Requirements

| | |
|---|---|
| Computer | Mac with Apple Silicon (M1 or newer). Developed on an M4 with 16 GB RAM |
| Memory | 16 GB RAM recommended |
| Disk | ~13 GB for the models (downloaded once) + ~1.6 GB for the trained Laya model |
| Python | 3.12 (Laya needs 3.10 or newer; the macOS system Python is too old) |
| Tools | [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip` |
| Optional | [Ollama](https://ollama.com) with `qwen3:4b` (backup LLM, and for creating extra training sentences) |

Models downloaded automatically on the first start (all from Hugging Face):

| Model | Job | Size |
|---|---|---|
| `convaiinnovations/laya` | fast decisions (base for fine-tuning) | ~1.7 GB |
| `mlx-community/Qwen3-4B-4bit` | slow brain (LLM) | 2.3 GB |
| `mlx-community/whisper-small.en-mlx` | speech → text | ~0.5 GB |
| `mlx-community/Kokoro-82M-bf16` | Laya's voice | ~0.3 GB |
| `mlx-community/Qwen3-VL-2B-Instruct-4bit` | reading the screen / camera | 1.8 GB |
| `Mininglamp-2718/Mano-CUA-2.0-4B-MLX-8bit` | doing tasks on the screen | ~5 GB |
| TitaNet-small, Smart Turn v3.2 (ONNX) | voice check, end-of-sentence | 48 MB |

---

## Installation

### 1. Get the code and install

```bash
git clone https://github.com/vikramballa/voice_assistant.git
cd voice_assistant

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

<details>
<summary>Without uv</summary>

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```
</details>

### 2. Train the fast brain (one time, ~15 minutes)

The trained Laya model is too big for Git (1.6 GB), so you create it once on your Mac.
The training sentences are already in the repo (`train/dataset.jsonl`).

```bash
.venv/bin/python train/finetune.py    # downloads base Laya, trains, saves train/laya-mac/  (~15 min on M4)
.venv/bin/python train/evaluate.py    # check: "WRONG" must be 0
```

> **Important:** do not skip this step. Base Laya without training is close to random on these commands,
> and Laya uses the trained model automatically as soon as `train/laya-mac/` exists.
> Keep the Mac plugged in and the lid open while training.

### 3. Give macOS permissions

Open **System Settings → Privacy & Security** and allow the app you run Laya from
(Terminal, iTerm, VS Code …):

| Permission | Needed for |
|---|---|
| **Microphone** | hearing you |
| **Accessibility** | typing, pressing keys, clicking, Calculator |
| **Screen & System Audio Recording** | screenshots, reading the screen, screen tasks |
| **Camera** (asked on first use) | "what am I holding?" |
| **Automation** (asked on first use) | Mail, Messages, Calendar, Reminders |

Laya checks these at startup and tells you what is missing. Restart the terminal after changing them.

### 4. Start it

```bash
.venv/bin/python main.py
```

The first start downloads the models (~13 GB, once). After that Laya starts **offline** in about 6–9 s
and says **"Ready"**.

### 5. (Recommended) Teach it your voice

```bash
.venv/bin/python -m assistant.audio.voice_id enroll   # say 10 short phrases (~30 s)
```

From now on only your voice gives commands. See [Only your voice](#only-your-voice).

---

## Usage

```bash
.venv/bin/python main.py            # wake word: say "Hey Laya" once, then just talk; "stop" ends
.venv/bin/python main.py --ptt      # push-to-talk: press Enter, then speak
.venv/bin/python main.py --text     # type commands instead of speaking (no mic)
```

| Option | What it does |
|---|---|
| `--ptt` | push-to-talk mode: press Enter, then speak |
| `--text` | type commands (good for testing without a mic) |
| `--debug` | print every sound the mic picks up |
| `--no-ui` | hide the Siri-style "island" at the top of the screen |
| `--say` | use the classic macOS voice instead of Kokoro |
| `--fast` | reply ~0.25 s sooner at short pauses (may cut a long sentence in two) |
| `--anyone` | accept every voice, not only the enrolled owner |

`python -m assistant` does the same as `python main.py`.

### Conversation mode

Say **"Hey Laya"** once (alone or with a command). After that, every command works **without** the wake word:

> "open YouTube in Brave" → "scroll down" → "open calculator" → "what is 25 into 25" → "multiply that by 4"

Say **"stop"** (or "stop listening", "that's all", "bye") to end it. Talk that isn't a request (people in the
room) is skipped quietly, and you can talk over Laya's replies.

### Only your voice

```bash
.venv/bin/python -m assistant.audio.voice_id enroll   # record your voiceprint (models/voiceprint.npz)
.venv/bin/python -m assistant.audio.voice_id test     # talk and see your match score (Ctrl+C to stop)
.venv/bin/python -m assistant.audio.voice_id add      # 5 more phrases, e.g. with AirPods
```

- After enrolling, other voices are shown as `(ignored '...' - not your voice, similarity 0.21)`.
- **Anyone** can still say "stop".
- Laya's own voice coming back through the speakers is rejected too.
- Without a voiceprint (or with `--anyone`) everyone is accepted. Push-to-talk and typed commands are not checked.
- This is a convenience filter, not security: a recording of your voice would pass.

### Try one part on its own

```bash
.venv/bin/python -m assistant.brain.fast "open chrome and search youtube for cats"   # the fast brain's decision
.venv/bin/python -m assistant.brain.slow "what's the capital of france"             # the LLM's plan
.venv/bin/python -m assistant.actions open_app Safari                              # run one action
.venv/bin/python -m assistant.audio.listen                                         # mic → text
.venv/bin/python assistant/mac/island.py --demo                                    # watch the island animate
```

---

## What you can say

### Everyday

| Area | Examples |
|---|---|
| Apps | "open Safari", "quit Chrome", "open VS Code", "open my Downloads folder" |
| Web | "search YouTube for lofi music", "google the weather in Hyderabad", "open GitHub", "open Gemini in my Google account" |
| Music | "play", "pause", "next song", "previous track" (Spotify if running, else Music) |
| Volume | "volume 30", "turn it up by 20", "mute", "unmute" |
| System | "take a screenshot", "lock my Mac", "turn on dark mode", "how much battery do I have?" |
| Info | "what time is it?", "will it rain in Mumbai?", "what's the capital of France?" |
| Timers | "set a timer for 5 minutes", "remind me to call Mom tomorrow at 9 am" |
| Several at once | "take a screenshot and lock my Mac", "open Brave and search for Python jobs" |

### Working inside apps

| Area | Examples |
|---|---|
| Calculator | "what is 25 into 25", then "multiply that by 4" or "add 5" (presses the real buttons) |
| Click | "click compose", "click the sign in button", "open the second video", "pick the first account" |
| Keys | "new tab", "close this tab", "refresh the page", "go back", "copy", "paste", "undo", "press enter" |
| Scroll | "scroll down", "scroll up a bit", "scroll to the top" |
| Typing | "type hello world", "type what I say" → "Type what?" → your next sentence is typed |
| Fix typing | "remove that", "delete the last word", "send it" |
| Charts | on a stock page: "give me the last 5 days" → presses **5D** ("show 1 year", "max") |
| Tabs | "go to the ChatGPT tab", "next tab", "previous tab" |
| AI chats | "ask ChatGPT what is a neural network" (opens ChatGPT with the question already asked) |

**Dictation.** Open Notes, TextEdit, Pages or Word and just talk: what you say is written into the document.
Say "comma", "full stop", "question mark", "new line". Commands ("close notes", "undo") and questions for
Laya ("what time is it?") still work. Misheard words are auto-corrected, but only when the fix *sounds like*
what was heard, so your meaning is never rewritten.

**Your app and browser are remembered.** After "open Brave", "open YouTube" and "search cats" use Brave.
Opened in the wrong browser? Just say **"in Brave"** and it is redone there.

### Mail, Messages and Calendar

| Area | Examples |
|---|---|
| Mail | "read my emails", "any emails from Rahul?", "reply to Rahul's email saying I'll join at 5" |
| Messages | "text Mom that I'm running late" (names come from Contacts) |
| Calendar | "what's on my calendar today?", "add a meeting with the design team tomorrow at 3 pm" |

Laya **looks first, then plans**: for a reply it reads the mail before writing. Before anything is sent it asks
*"Should I text Mom: 'I'm running late'?"* — only **"yes"** sends it.

### Eyes (screen and camera)

| Feature | Examples |
|---|---|
| Read the screen | "what does this error say?", "can you see my recent chats?" |
| Do it for me | "on this page, search for running shoes and open the second result", "find the pricing page for me on this website" |
| Watch the screen | "tell me when *Build succeeded* shows up" (checks every 3 s, up to 30 min; "stop watching" cancels) |
| Camera | "what am I holding?", "how do I look?" (one photo, answered in ~1 s, then deleted) |

> **Tip for web pages:** in your browser, turn on **View → Developer → Allow JavaScript from Apple Events**
> (Safari: **Develop → Allow JavaScript from Apple Events**). Then clicks like "open the second video" are
> exact and fast. Without it Laya falls back to reading the screen, which is slower.

### Routines and corrections

- **Routines**: "When I say *start my work day*, open VS Code, open Brave and play music."
  Afterwards "start my work day" does all of it at once (no AI needed).
- **Corrections**: if Laya gets something wrong, say **"no, I meant …"** within 20 s. It does the right thing
  and remembers it for those exact words next time.
- Manage them: `python -m assistant.phrasebook list`, `forget "start my work day"`, `clear`.

---

## Safety and privacy

**Privacy**

- All models run on the Mac. Nothing is sent to a cloud service (only the weather uses `wttr.in`, and
  "search …" opens your browser).
- Your command log (`logs/commands.db`), voiceprint and routines (`models/`) stay on your Mac and are
  excluded from Git by `.gitignore`.
- Screenshots and camera photos are deleted right after use.

**Safety rules (plain code, not the AI's choice)**

| Rule | Why |
|---|---|
| Mail and messages are only sent after you say "yes" | no surprise messages |
| The screen agent **never pays** (stops at buy / pay / order / checkout) | money is always your click |
| It clicks delete / send / submit / sign out **only if your own words asked for it** | no risky clicks by guessing |
| It **never types passwords** or codes | secrets stay with you |
| **Move the mouse and the screen agent stops at once**; max 8 steps | you are always in charge |
| Keys mapped to Shut Down / Restart / Log Out / Sleep in your shortcuts are never pressed | some Macs map ⌘T or ⌘P to these |
| Startup never touches the network once models are downloaded | no hangs on a bad connection |

---

## Configuration

Command-line options are listed under [Usage](#usage). Environment variables:

| Variable | Default | What it does |
|---|---|---|
| `LAYA_BACKEND` | `mlx` | `ollama` uses Ollama for the slow brain instead of MLX (also the automatic fallback) |
| `LAYA_MLX_MODEL` | `mlx-community/Qwen3-4B-4bit` | a different MLX LLM |
| `LAYA_LLM` | `qwen3:4b` | the Ollama model (when `LAYA_BACKEND=ollama`) |

Example: `LAYA_BACKEND=ollama .venv/bin/python main.py`

Tuning numbers (silence length, confidence, timeouts …) are constants at the top of the module they tune,
each with a comment, e.g. `SILENCE_SECONDS` in `assistant/audio/listen.py`, `CONFIDENCE` in `assistant/brain/fast.py`.

---

## Project structure

```
voice_assistant/
├── main.py                  start here (same as `python -m assistant`)
├── requirements.txt         everything the assistant needs
├── requirements-vision.txt  only the optional vision server (Linux / no Mac needed)
├── assistant/
│   ├── __main__.py          startup and safe exit
│   ├── config.py            paths and command-line options
│   ├── app.py               one command: decide → act → speak → log  (class Assistant)
│   ├── modes.py             input loops: wake word / push-to-talk / typed text
│   ├── session.py           memory between commands (conversation, browser, working app …)
│   ├── task.py              carrying out a plan (look first, ask before sending)
│   ├── phrasebook.py        routines and learned corrections
│   ├── logbook.py           every command → one row in logs/commands.db (SQLite)
│   ├── language/            pure text rules, no models (cleaning, splitting, names, numbers, dialogue)
│   ├── brain/               fast.py (Laya), slow.py (LLM), prompts, repair (checks plans), mlx_llm (engine)
│   ├── actions/             what Laya can do: apps, web, media, system, mail, calendar, screen, agent, camera …
│   ├── audio/               listen (mic + speech detection), transcribe (Whisper), voice_id, speaker (voice)
│   ├── mac/                 accessibility (click/press), vision (see the screen), web_page, island UI, permissions
│   └── vision_service.py    optional HTTP server for the vision model
├── tests/                   pytest: fast unit tests + the whole pipeline with pretend actions
├── evals/                   slower measurements (full pipeline, hard cases, voice, voice ID)
└── train/                   dataset, fine-tuning, evaluation, learning from your logs
```

---

## Architecture

### Layers

Each layer only imports from the layers above it, so every part can be read and tested on its own.

| Layer | Package | Job |
|---|---|---|
| 0 | `config`, `gpu` | paths, options, the lock that keeps two models off the GPU at once |
| 1 | `language/` | pure text rules (no models, no Mac) — hundreds of tests run in a fraction of a second |
| 2 | `mac/`, `audio/` | macOS APIs; microphone, Whisper, voice |
| 3 | `actions/` | what Laya can do, one function per action |
| 4 | `brain/` | deciding: fast (Laya) and slow (LLM) |
| 5 | `session`, `logbook`, `task` | memory, the command log, running a plan |
| 6 | `app` | the pipeline for one command |
| 7 | `modes`, `__main__` | input loops and startup |

### Design principles

1. **Right or silent.** The fast brain must never be *wrong*; when unsure it returns `None` and the LLM decides.
   The key test (`train/evaluate.py`) requires **0 wrong** decisions.
2. **The user's words beat the model.** The model picks the kind of action; the words decide the details
   (app vs. website, which browser, what to search). The same check is applied to every LLM plan (`brain/repair.py`).
3. **One job per module, state in one place.** Everything remembered between commands lives in one `Session` object.
4. **Heavy things load lazily, in parallel.** Importing never loads a model; startup loads all models at once in threads.
5. **Failures are spoken, not hidden.** Actions raise `Failed("I couldn't find an app called Foo")`; Laya says it.
6. **Isolation where a crash would hurt.** The island UI is a separate process; only one thread touches the speaker;
   every exit stops that thread first.

### Key decisions (with the measurement behind them)

| Decision | Why |
|---|---|
| Laya + LLM instead of only an LLM | ~80 ms vs ~0.8 s; most commands never need the LLM |
| Always the fine-tuned Laya | base Laya scored 52% here and is overconfident; fine-tuned: 99%, 0 wrong |
| LLM on Apple MLX with a JSON grammar | about 2× faster than Ollama with the same accuracy; Ollama kept as fallback |
| Whisper small.en | large-v3-turbo was 4× slower with no accuracy gain on commands |
| Start Whisper at 0.29 s pauses | the text is ready when you finish: ~250 ms saved per command |
| TitaNet-small for the voice check | lowest error of 5 speaker models tested, and the fastest (~10 ms) |
| Learn from "no, I meant …", not reinforcement learning | a correction is a clear label; self-training would learn its own mistakes |
| Mail tasks as "look, then plan again" | the 4B LLM guessed what it hadn't read; now each lookup feeds the next plan |
| Screen agent checked by plain rules every step | the model is good at clicking, but safety must not depend on it |
| Loudness gate on top of Silero VAD | in a noisy room, sentences otherwise never ended |

### Suggested reading order

1. `assistant/app.py` → `Assistant.handle`: one command end to end, then `assistant/task.py`.
2. `assistant/brain/fast.py` → `decide()` and `_resolve()`: rules first, then Laya.
3. `assistant/language/` → `words.py`, `sentence.py`, then `catalog.py`.
4. `assistant/actions/__init__.py` → the action table, then any action module.
5. `assistant/brain/slow.py`, `repair.py`, `mlx_llm.py` (the grammar is the most technical part).
6. `assistant/modes.py` (wake word, echo, conversation mode) and `assistant/audio/listen.py`.
7. `tests/test_app.py` → the behaviour, written down as examples.

---

## Testing and results

```bash
.venv/bin/python -m pytest -q -p no:warnings   # unit + pipeline tests (~2 min)
.venv/bin/python train/evaluate.py             # fast brain on 82 held-out commands
.venv/bin/python evals/full_eval.py            # 119 realistic commands, whole pipeline
.venv/bin/python evals/hard_cases.py           # 29 tricky commands (real past failures)
.venv/bin/python evals/voice_benchmark.py      # 51 spoken commands, 3 voices (several minutes)
.venv/bin/python evals/voice_id_eval.py        # voice check: 14 voices + Laya's own, in room noise
```

Measured on a MacBook (M4, 16 GB):

| Test | Result |
|---|---|
| pytest | 589 tests |
| Fast brain, held-out commands | **99%** correct, **0 wrong**, ~80 ms |
| Full pipeline, 119 realistic commands | **119 / 119** (95 by the fast brain, 0 wrong) |
| Hard cases | **29 / 29** |
| Spoken commands (Whisper → action) | 43 / 51 (misses are mostly Whisper homophones like "buy" → "by") |
| Voice check | owner accepted 70/70 sentences; Laya's own voice let in 0/70 |
| End of speech → action | ~0.9 s (fast brain), ~1.3–2 s (LLM) |
| Startup | ~6–9 s (offline) |

Some tests need the models downloaded (Whisper, the vision model); they are skipped or fail without them.

---

## Making it smarter (training)

Commands the fast brain wasn't sure about were answered by the LLM and logged on your Mac. Teach them to Laya:

```bash
ollama stop qwen3:4b                         # free memory for training (if you use Ollama)
.venv/bin/python train/learn_from_logs.py    # adds your logged commands to train/learned.jsonl (review it!)
.venv/bin/python train/make_dataset.py       # rebuild train/dataset.jsonl (--no-llm: without Ollama)
.venv/bin/python train/finetune.py           # ~15 min on an M4
.venv/bin/python train/evaluate.py           # check: WRONG must stay 0
```

The test sets (`train/test_set.jsonl`, `train/dev_set.jsonl`) are never used for training.

---

## Extending Laya

**A new action** (the LLM can use it right away):

1. Write the function in the right `assistant/actions/` module. It takes simple string arguments, returns
   a sentence to speak, and raises `Failed("...")` for errors the user should hear.
2. Add it to `ACTIONS` in `assistant/actions/__init__.py`.
3. Describe it in `assistant/brain/prompts.py`.
4. If the LLM may only use it when certain words were said, add a `GUARDS` entry in `assistant/brain/repair.py`.

**A new fast-brain intent** (so it runs in ~80 ms): add it to `INTENTS` in `assistant/brain/fast.py`, an
argument rule in `assistant/language/arguments.py`, templates in `train/make_dataset.py`, test rows in
`train/test_set.jsonl`; then rebuild the dataset, retrain and evaluate. Changing `INTENTS` always needs a retrain.

**A new exact rule** (a phrase that should never need a model): add it to the matching `assistant/language/`
module, call it from the "exact rules" block in `brain/fast._resolve`, and add examples to `tests/test_language.py`.

Before sending a change, run `pytest` and `train/evaluate.py` (0 wrong).

---

## Optional: vision model as a server

The screen-reading model can also run on another machine (e.g. a Linux server) as a small HTTP API:

```bash
pip install -r requirements-vision.txt
python -m assistant.vision_service          # port 8000 (or $PORT); $VISION_HOST sets the address
```

Endpoints: `GET /health`, `POST /ask`, `POST /point`, `POST /locate_click` (image as a file upload).

> It has **no login of its own**. Keep it on a private network or behind a reverse proxy / VPN.
> On non-Mac machines it uses plain PyTorch (`transformers`); this path is implemented but tested only on a Mac so far.

---

## Troubleshooting

| What you see | Fix |
|---|---|
| `(ignored '...' - start with "Hey Laya")` | Conversation mode is off (you said "stop"). Say "Hey Laya" again. |
| `(ignored '...' - not your voice ...)` but it was you | Run `voice_id test` to see your score; run `enroll` again (or `add`) with the mic you use now. |
| `!! The microphone ... is sending pure silence` | Open the MacBook lid (the built-in mic is off when closed), or allow Microphone for your terminal. |
| Nothing happens at all | Run with `--debug` to see every sound it hears. In a loud room, speak closer to the Mac. |
| "I need … permission" | Turn on that switch for your terminal in System Settings → Privacy & Security, then restart the terminal. |
| Stuck at `loading models...` | It prints `still loading: ...` every 15 s; Ctrl+C quits at once. Make sure only one Laya is running (two copies fill 16 GB). |
| Wrong or random decisions | Check that `train/laya-mac/` exists (see [Installation step 2](#2-train-the-fast-brain-one-time-15-minutes)). |
| Clicks on web pages miss | Turn on **Allow JavaScript from Apple Events** in your browser's View → Developer menu. |
| Opened in the wrong browser | Say "in Brave" (or "in Safari") right after. |

---

## Known limitations

- **macOS only**, Apple Silicon only (MLX, Accessibility, AppleScript).
- English only (Whisper `small.en`).
- Whisper mixes up words that sound the same ("buy milk" → "by milk"); dictation auto-correct fixes some.
- The screen agent takes ~5 s per step and is new; it has been tested on test pages, not on many real sites.
- Mail, Messages and Calendar are tested with fakes; first real use asks for Automation permission. WhatsApp is not supported.
- "next song" controls the Spotify app or Music, not a web player.
- The voice check can be fooled by similar voices (~4% of sentences) and by recordings.

---

## Acknowledgements and licenses

Laya is built on these open models and libraries (each under its own license):

| Project | Use | License |
|---|---|---|
| [Laya](https://pypi.org/project/laya/) (ConvAI Innovations) | fast decision model | Apache 2.0 |
| [Qwen3](https://huggingface.co/Qwen) / Qwen3-VL | LLM, vision | Apache 2.0 |
| [Mano-CUA](https://huggingface.co/Mininglamp-2718) | screen agent | Apache 2.0 |
| [Whisper](https://github.com/openai/whisper) via [mlx-whisper](https://github.com/ml-explore/mlx-examples) | speech → text | MIT |
| [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) via mlx-audio | voice | Apache 2.0 |
| [Silero VAD](https://github.com/snakers4/silero-vad) | speech detection | MIT |
| [Smart Turn](https://github.com/pipecat-ai/smart-turn) (pipecat-ai) | end-of-sentence detection | BSD-2 |
| TitaNet-small via [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | voice check | Apache 2.0 |
| [MLX](https://github.com/ml-explore/mlx), mlx-lm, mlx-vlm | running models on Apple Silicon | MIT |

## License

Copyright 2026 Vikram Balla.

Licensed under the **Apache License, Version 2.0** — see [LICENSE](LICENSE).
The models listed above are separate projects under their own licenses.
