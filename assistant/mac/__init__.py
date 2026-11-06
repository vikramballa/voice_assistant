"""macOS itself, wrapped in plain Python functions.

    accessibility  find / press buttons, links and menu items in other apps; click at a point; scroll
    vision         screenshot of a window; Apple's text recognition; a small local vision-language model
    island         the Siri-style pill at the top of the screen (a separate process)
    permissions    check Accessibility and Screen Recording at startup and explain what is missing

The pyobjc frameworks these need are imported only when a module is used, so the rest of the
assistant (and most tests) never pays for them.
"""
