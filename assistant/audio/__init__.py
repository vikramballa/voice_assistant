"""Sound in and out.

    listen      microphone -> speech detection (Silero VAD + a loudness gate) -> one audio array per utterance
    transcribe  audio -> text with Whisper (MLX), plus speculative transcription during short pauses
    turn        Smart Turn: "does this voice sound finished?" (only with --fast)
    speaker     the voice: Kokoro (MLX), or macOS `say`; a new reply interrupts the one playing
"""
