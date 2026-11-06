"""Start Laya, the local voice assistant for your Mac. The code is in the assistant/ package.

    python main.py            # "Hey Laya, open Safari", then more commands without "Hey Laya" until "stop"
    python main.py --ptt      # push-to-talk: press Enter, then speak
    python main.py --text     # type commands instead (no mic)
    python main.py --debug    # also show every sound the mic picked up
    python main.py --no-ui    # no Siri-style island at the top of the screen
    python main.py --say      # classic macOS voice instead of Kokoro
    python main.py --fast     # reply ~0.25 s sooner on short pauses (may cut a sentence at a pause)
"""

from assistant.__main__ import main

if __name__ == "__main__":
    main()
