"""What Laya may learn from a correction, and forgetting it (conftest gives each test an empty phrasebook)."""

from assistant import phrasebook

NOTES = [{"action": "open_app", "args": ["Notes"]}]


def test_learn_lookup_forget():
    assert phrasebook.learn("Hey Laya, Open Not.", NOTES)
    assert phrasebook.lookup("open not") == NOTES
    assert phrasebook.lookup("open notes app") is None  # exact words only
    assert phrasebook.forget("open not") and phrasebook.lookup("open not") is None


def test_a_new_correction_replaces_the_old_one():
    phrasebook.learn("open not", NOTES)
    phrasebook.learn("open not", [{"action": "open_app", "args": ["Notion"]}])
    assert phrasebook.lookup("open not") == [{"action": "open_app", "args": ["Notion"]}]


def test_stale_or_pointing_back_is_never_learned():
    assert not phrasebook.learn("what is the capital", [{"action": "answer", "args": ["Delhi"]}])
    assert not phrasebook.learn("add five", [{"action": "calculate", "args": ["+5"]}])
    assert not phrasebook.learn("close it", [{"action": "close_app", "args": ["Safari"]}])
    assert not phrasebook.learn("open not", [])
    assert phrasebook.entries() == {}


def test_lookup_returns_a_copy():
    phrasebook.learn("open not", NOTES)
    phrasebook.lookup("open not")[0]["args"].append("x")
    assert phrasebook.lookup("open not") == NOTES
