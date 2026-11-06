"""The two macOS permissions Laya needs for the app it runs in (Terminal, an IDE...), and what to say
when one is missing. Both are granted per host app in System Settings > Privacy & Security.
(The camera is asked for only when the user first asks Laya to look through it.)"""

import subprocess


def host_name() -> str:
    """The app Laya runs in, by name ("Terminal"), for messages that tell the user which switch to turn on."""
    try:
        from . import accessibility
        return accessibility.host_name()
    except Exception:
        return "your terminal app"


def accessibility_message() -> str:
    return (f"I need Accessibility permission. Open System Settings, Privacy and Security, Accessibility, "
            f"turn on {host_name()}, then restart me")


def screen_recording_message() -> str:
    return (f"I need Screen Recording permission to see the screen. Open System Settings, Privacy and Security, "
            f"Screen and System Audio Recording, turn on {host_name()}, then restart me")


def camera_message() -> str:
    return (f"I can't use the camera. Open System Settings, Privacy and Security, Camera, turn on {host_name()}, "
            f"then restart me. If it's on, the camera may be busy in another app or the lid may be closed")


def check() -> str | None:
    """Accessibility (clicks, keys, Calculator) and Screen Recording (seeing the screen) for the host app.
    Missing ones are printed, macOS's own prompt is opened, and a sentence to say is returned."""
    try:
        import ApplicationServices as AX
        import Quartz

        from . import accessibility
    except Exception:
        return None
    host = accessibility.host_name()
    missing = []
    if not accessibility.trusted():
        missing.append("Accessibility")
        AX.AXIsProcessTrustedWithOptions({AX.kAXTrustedCheckOptionPrompt: True})  # adds the app to the list
    if not Quartz.CGPreflightScreenCaptureAccess():
        missing.append("Screen Recording")
        Quartz.CGRequestScreenCaptureAccess()
    if not missing:
        print(f"permissions ok for {host}: Accessibility, Screen Recording")
        return None
    what = " and ".join(missing)
    lost = []
    if "Accessibility" in missing:
        lost.append("no clicks, keys, scrolling or Calculator buttons")
    if "Screen Recording" in missing:
        lost.append("no seeing the screen")
    print(f"\n!! {host} needs {what} permission (System Settings > Privacy & Security).")
    print(f"   Without it: {'; '.join(lost)}. Turn {host} on there, then restart Laya.")
    pane = "Privacy_Accessibility" if "Accessibility" in missing else "Privacy_ScreenCapture"
    subprocess.Popen(["open", f"x-apple.systempreferences:com.apple.preference.security?{pane}"])
    return f"I need {what} permission for {host}. I opened the settings. Turn it on, then restart me"
