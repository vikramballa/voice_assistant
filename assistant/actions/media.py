"""Music and sound: play/pause, next, previous (Spotify if running, else Apple Music), volume, mute."""

from .macos import osascript

VOLUME_STEP = 10


def _player() -> str:
    """Spotify if it's running, otherwise Apple Music."""
    running = osascript('tell application "System Events" to (name of processes) contains "Spotify"')
    return "Spotify" if running == "true" else "Music"


def media_play_pause(what: str = "") -> str:
    """what: "pause" (also "stop") only ever pauses: nothing starts playing when the user said stop
    (2026-10-03: "I made it. Stop." started a song). "play" only plays. "" toggles."""
    what = what.lower().strip()
    video = _video_in_front(what)
    if video:
        return video
    player = _player()
    if what in ("pause", "stop"):
        if not _running(player) or osascript(f'tell application "{player}" to player state as string') != "playing":
            return "Nothing is playing"
        osascript(f'tell application "{player}" to pause')
        return "Paused"
    osascript(f'tell application "{player}" to {"play" if what == "play" else "playpause"}')
    return "Done"


def _video_in_front(what: str) -> str | None:
    """Watching a video in the browser in front (YouTube...): "pause" means that video, not the music app.
    None when there's no started video there (or the browser doesn't allow page scripts)."""
    import subprocess
    from assistant.mac import accessibility, web_page
    try:
        app = accessibility.target_app()
        if not web_page.is_browser(app):
            return None
        state = web_page.page_video(app.localizedName(), "pause" if what == "stop" else what)
    except (web_page.ScriptsOff, RuntimeError, OSError, subprocess.TimeoutExpired):
        return None
    if state is None:
        return None
    now, before = state
    if now == before:
        return None  # the video already is that way: "pause" may mean the music playing in the background
    return "Paused the video" if now == "paused" else "Playing the video"


def _running(app: str) -> bool:
    """Asking a player that isn't running anything would launch it."""
    return osascript(f'application "{app}" is running') == "true"


def next_track() -> str:
    osascript(f'tell application "{_player()}" to next track')
    return "Next"


def previous_track() -> str:
    osascript(f'tell application "{_player()}" to previous track')
    return "Previous"


def set_volume(level: int) -> str:
    level = max(0, min(100, int(level)))
    osascript(f"set volume without output muted\nset volume output volume {level}")  # also unmutes
    return f"Volume {level}"


def volume_up(step: int | str = VOLUME_STEP) -> str:
    return set_volume(_current_volume() + _step(step))


def volume_down(step: int | str = VOLUME_STEP) -> str:
    return set_volume(_current_volume() - _step(step))


def _step(step: int | str) -> int:
    """Plans pass text ("20"); anything that isn't a number ("a lot") is the usual step."""
    return int(step) if str(step).strip().isdigit() else VOLUME_STEP


def mute() -> str:
    osascript("set volume with output muted")
    return "Muted"


def unmute() -> str:
    osascript("set volume without output muted")
    return "Unmuted"


def _current_volume() -> int:
    return int(osascript("output volume of (get volume settings)"))
