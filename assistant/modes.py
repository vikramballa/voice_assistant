"""The three ways to give commands. Each one turns input into text and hands it to Assistant.handle().

    run_text           type commands (no microphone)
    run_push_to_talk   press Enter, then speak
    WakeWordLoop       always listening: "Hey Laya" starts a conversation; then no wake word is needed
                       until "stop". Also handles our own voice coming back through the mic.
"""

import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

from assistant.app import FOLLOW_UP_SECONDS, Assistant
from assistant.audio import listen
from assistant.audio import speaker, voice_id
from assistant.audio.speaker import said_during, spoke_during
from assistant.audio.transcribe import EndEarly, SpeculativeTranscriber
from assistant.brain import fast
from assistant.language import dialogue
from assistant.language.sentence import continuation, strip_wake
from assistant.mac import island

CONTINUE_SECONDS = 2.5  # "... and type hello" this soon after a command continues it (no wake word)
TURN_THRESHOLD = 0.8    # --fast: Smart Turn "finished" probability needed to stop listening at a short pause
ECHO_MARGIN = 0.3       # seconds before the utterance in which our own reply still counts as overlapping


def chime() -> None:
    subprocess.Popen(["afplay", "/System/Library/Sounds/Tink.aiff"])


def run_text(assistant: Assistant) -> None:
    while True:
        try:
            text = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            return
        if text:
            assistant.handle(text)


def run_push_to_talk(assistant: Assistant, fast_end: bool = False) -> None:
    transcriber = SpeculativeTranscriber(fast_end_policy(lambda: False) if fast_end else None, dialogue.may_go_on)
    while True:
        try:
            input("\n[Enter] to talk (Ctrl+C to quit) ")
        except (KeyboardInterrupt, EOFError):
            return
        chime()
        island.show("listening")
        audio = listen.listen_once(early_end=transcriber.on_pause, on_start=fast.wake_gpu)
        if audio is None:
            print("  (didn't hear anything)")
            island.show("hide")
            continue
        text, timings = transcriber.transcribe(audio)
        print(f"  heard: {text!r}")
        if text:
            assistant.handle(text, timings)


def fast_end_policy(need_wake) -> EndEarly:
    """--fast: end listening at a short pause when the voice sounds finished (Smart Turn) AND the words are a
    whole command. ~0.26 s quicker on half the commands, but can cut "google the weather | in hyderabad" in two.
    need_wake() says whether the words must start with "Hey Laya" right now."""
    def end_early(audio, text) -> bool:
        from assistant.audio import turn
        if turn.finished_probability(audio) < TURN_THRESHOLD:
            return False
        return dialogue.looks_complete(text(), need_wake())
    return end_early


class WakeWordLoop:
    """Wake-word mode. One microphone stream; every utterance is transcribed, then:

      1. our own reply heard back through the mic is dropped (unless the user talked over it)
      2. the wake word is looked for; without it, the utterance still counts in conversation mode,
         as the answer to our question, or as "and ..." right after a command
      3. only the enrolled owner's voice counts (voice_id; anyone may say "stop")
      4. "stop" ends conversation mode; noise and room talk are skipped
      5. what is left is a command for Assistant.handle()
    """

    def __init__(self, assistant: Assistant, debug: bool = False, fast_end: bool = False, owner_only: bool = True):
        self.assistant = assistant
        self.session = assistant.session
        self.debug = debug
        self.owner_only = owner_only and voice_id.enrolled()
        self._voice_pool = ThreadPoolExecutor(1)  # the voice check (CPU) runs while Whisper (GPU) transcribes
        need_wake = lambda: not self.session.conversation_on  # noqa: E731
        self.transcriber = SpeculativeTranscriber(fast_end_policy(need_wake) if fast_end else None, dialogue.may_go_on)
        self.awaiting_until = 0.0  # after a bare "Hey Laya", the next utterance is the command
        self.continue_until = 0.0  # right after a command, "and ..." / "then ..." continues it

    def run(self) -> None:
        if self.owner_only:
            print("voice check on: only your voice gives commands (--anyone turns it off)")
        elif not voice_id.enrolled():
            print("voice check off: anyone can give commands (enroll: python -m assistant.audio.voice_id enroll)")
        print('\nlistening... say "Hey Laya", then give commands without it until you say "stop"  (Ctrl+C to quit)')
        try:
            for audio in listen.utterances(early_end=self.transcriber.on_pause, on_start=fast.wake_gpu):
                self.on_utterance(audio)
        except KeyboardInterrupt:
            return

    def on_utterance(self, audio) -> None:
        voice = self._voice_pool.submit(voice_id.check, audio) if self.owner_only else None
        heard = self._transcribe_unless_echo(audio)
        if heard is None:
            return
        text, timings, over_reply = heard
        command, woke, waiting = self._command_in(text)
        if command is None:
            return
        if voice is not None and not self._from_owner(voice, text, command):
            return
        if self.session.conversation_on and dialogue.is_stop(command):
            self.end_conversation()
            return
        if woke and dialogue.is_stop(command):
            self.assistant.say("Okay")
            return
        if not woke and not waiting and dialogue.is_noise(command):
            self._debug(f"skipped {command!r} - not a command")
            return
        print(f"\n  heard: {text!r}")
        if woke and not self.session.conversation_on:
            self.session.conversation_on = True
            print('  [conversation mode: no need to say "Hey Laya" until you say "stop"]')
        if not command:  # a bare "Hey Laya": the next utterance is the command
            chime()
            island.show("listening", 'Listening… say "stop" when done')
            self.awaiting_until = time.time() + FOLLOW_UP_SECONDS
            return
        self.awaiting_until = 0.0
        # Without the wake word, talk that isn't a request (people in the room) is skipped quietly.
        # Said over our own reply: only a clear fast-path command counts (never the LLM: no echo loops).
        self.assistant.handle(command, timings, quiet_unclear=not woke and not waiting,
                              fast_only=over_reply and not woke)
        self.continue_until = time.time() + CONTINUE_SECONDS  # pending answers need no wake word either

    def end_conversation(self) -> None:
        self.session.conversation_on = False
        self.session.pending_text = None
        print('  [conversation ended - say "Hey Laya" to start again]')
        self.assistant.say("Okay")  # also cuts off a reply that is still playing
        island.show("reply", 'Stopped listening. Say "Hey Laya" to start')

    def _transcribe_unless_echo(self, audio) -> tuple[str, dict, bool] | None:
        """(text, timings, said over our reply), or None when the utterance is only our own reply coming
        back through the mic."""
        seconds = len(audio) / listen.SAMPLE_RATE
        # When it was *recorded* (it may have waited while we were busy), not when we got to it.
        recorded_end = listen.captured_at or time.time()
        start = recorded_end - seconds - ECHO_MARGIN
        overlapped = spoke_during(start, recorded_end)
        if overlapped and time.time() > self.continue_until and not self.session.conversation_on:
            self._debug(f"skipped {seconds:.1f}s of audio - overlapped our own reply")
            return None
        text, timings = self.transcriber.transcribe(audio)
        if overlapped:  # over our own voice only "stop", "... and <rest>" or clearly-not-us speech is kept
            reply = " ".join(said_during(start, recorded_end)) or speaker.last_text
            text = dialogue.without_echo(text, reply)
            if self.session.conversation_on and dialogue.is_stop(text):
                self.end_conversation()
                return None
            if dialogue.is_echo(text, reply):
                self._debug(f"skipped {text!r} - our own reply heard back")
                return None
            talked_over = self.session.conversation_on and len(text.split()) >= 2
            if not continuation(text) and not talked_over:
                self._debug(f"skipped {text!r} - overlapped our own reply")
                return None
        self._debug(f"{seconds:.1f}s of audio -> {text!r}, {timings['stt_ms']} ms")
        return text, timings, overlapped

    def _command_in(self, text: str) -> tuple[str | None, bool, bool]:
        """(the command, whether the wake word was said, whether we were waiting for an answer).
        The command is None when the utterance should be ignored."""
        command = strip_wake(text)
        woke = command is not None
        if command is None and time.time() < self.continue_until and continuation(text):
            command = continuation(text)  # we stopped listening too early; this is the rest
        waiting = time.time() < self.awaiting_until or self.session.is_waiting()
        if command is None:
            if not waiting and not self.session.conversation_on:
                # Show it anyway, so it's clear the mic works but the wake word was missing.
                print(f'  (ignored {text!r} - start with "Hey Laya")')
                return None, woke, waiting
            command = text
        return command, woke, waiting

    def _from_owner(self, voice, text: str, command: str) -> bool:
        """Only the owner gives commands; anyone may say "stop" (to end a conversation, or cut off a reply)."""
        is_owner, similarity = voice.result()
        self._debug(f"voice similarity {similarity:.2f}")
        if is_owner or dialogue.is_stop(command):
            return True
        print(f"  (ignored {text!r} - not your voice, similarity {similarity:.2f})")
        return False

    def _debug(self, message: str) -> None:
        if self.debug:
            print(f"  (debug: {message})")
