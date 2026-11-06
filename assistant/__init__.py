"""Laya: a local voice assistant that controls the Mac.

    mic ─► audio.listen ─► audio.transcribe ─► app.Assistant.handle ─► actions ─► audio.speaker
                                                  │
                                    brain.fast (Laya, ~80 ms) ─ unsure? ─► brain.slow (local LLM, ~0.8 s)

Packages, lowest layer first (a layer only imports from the layers above it in this list):

    config, gpu   paths on disk, command-line options, the MLX lock
    language/     pure text rules: cleaning, splitting, names of apps and sites, arguments, math, dialogue
    mac/          macOS itself: Accessibility (click, press), screen vision, the "island" window, permissions
    audio/        microphone, speech detection, Whisper, end-of-turn model, the voice
    actions/      what Laya can do (open_app, web_search, calculate...): one function per action
    brain/        deciding what to do: fast (Laya classifier) and slow (LLM planner + repair)
    logbook       every handled command, one row in SQLite
    session       what the assistant remembers between commands
    app           the pipeline for one command: decide -> act -> speak -> log
    modes         the input loops: wake word, push-to-talk, typed text
    __main__      startup (python -m assistant, or python main.py)

See README.md ("Architecture") for the design and a suggested reading order.
"""
