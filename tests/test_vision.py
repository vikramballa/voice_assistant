"""Seeing the screen, on a real YouTube screenshot (tests/data/youtube_results.png, 1440x900) instead of a live
capture, with the real text recognizer and vision model. Clicks are recorded, not made."""

import shutil
from pathlib import Path

import pytest

from assistant.actions import NeedsPermission, screen
from assistant.mac import accessibility as ui
from assistant.mac import vision

SHOT = Path(__file__).parent / "data" / "youtube_results.png"
BOUNDS = (100.0, 50.0, 1440.0, 900.0)  # pretend the window is here on screen, same size as the screenshot


@pytest.fixture
def fake_screen(monkeypatch, tmp_path):
    clicks = []
    monkeypatch.setattr(ui, "trusted", lambda: True)
    monkeypatch.setattr(ui, "target_app", lambda: None)
    monkeypatch.setattr(ui, "activate", lambda app: False)
    monkeypatch.setattr(ui, "find", lambda name, app=None: None)  # nothing by Accessibility: use the screen
    monkeypatch.setattr(ui, "window_of", lambda app: (None, BOUNDS))
    monkeypatch.setattr(ui, "window_number", lambda app, bounds: 1)
    monkeypatch.setattr(ui, "click_at", lambda x, y: clicks.append((x - BOUNDS[0], y - BOUNDS[1])))

    def capture(window_id=None):
        copy = tmp_path / f"shot{len(list(tmp_path.iterdir()))}.png"
        shutil.copy(SHOT, copy)
        return copy
    monkeypatch.setattr(vision, "capture", capture)
    return clicks


def inside(point, box):
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def test_text_on_screen_is_found_and_clicked(fake_screen):
    assert screen.click("Recently uploaded") == "Clicked Recently uploaded"
    assert inside(fake_screen[0], (650, 67, 790, 99))


def test_description_is_found_by_the_vision_model(fake_screen):
    screen.click("second video")
    assert inside(fake_screen[0], (264, 410, 1416, 690))


def test_questions_about_the_screen(fake_screen):
    answer = screen.look_at_screen("What is on my screen?")
    assert "youtube" in answer.lower() and "neural" in answer.lower()


def test_no_screen_recording_permission_is_explained(monkeypatch, fake_screen):
    def refused(window_id=None):
        raise vision.NoScreenAccess()
    monkeypatch.setattr(vision, "capture", refused)
    with pytest.raises(NeedsPermission, match="Screen Recording"):
        screen.look_at_screen("what is this")
