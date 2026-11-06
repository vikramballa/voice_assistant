"""Shared test setup."""

import pytest

from assistant import phrasebook


@pytest.fixture(autouse=True)
def empty_phrasebook(monkeypatch, tmp_path):
    """The user's taught phrases (models/phrasebook.json) must never decide, or be written by, a test."""
    monkeypatch.setattr(phrasebook, "PATH", tmp_path / "phrasebook.json")
