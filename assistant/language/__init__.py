"""Text rules that turn what Whisper wrote into something we can act on. No models, no side effects.

    words        the shared vocabulary: normalize(), polite fillers, command verbs, browser names
    catalog      what exists: installed apps, web services, folders, browsers; "open X" -> app / site / folder
    sentence     one sentence: fix mishearings, the wake word, split "A and B" into clauses
    arguments    the details of one clause: search words, volume number, which search, which media key
    spoken_math  spoken arithmetic -> "25*25"
    in_app       commands inside the app in front: keyboard shortcuts, scrolling, clicking, questions about the screen
    dialogue     the conversation around commands: "Open what?", unfinished sentences, "stop", noise, our own echo

Every rule that extracts something returns None when it isn't sure. None means "let the LLM decide",
so the rules only have to cover the common phrasings, and never have to guess.
"""
