"""task.run: lookups feed the next plan, sending waits for a yes."""

from assistant import task

READ = {"action": "read_mail", "args": ["Rahul"]}
SEND = {"action": "send_email", "args": ["Rahul", "Re: Meeting", "I'll join at 5"]}


def fake_actions(monkeypatch, results=None):
    ran = []
    monkeypatch.setattr(task, "run_action", lambda name, *args: ran.append(name) or (results or {}).get(name, "ok"))
    return ran


def test_a_lookup_is_read_then_the_next_steps_wait_for_a_yes(monkeypatch):
    ran = fake_actions(monkeypatch, {"read_mail": "From Rahul: Meeting - can you join at 5?"})
    seen_by_planner, said = [], []
    out = task.run([READ], lambda seen: seen_by_planner.extend(seen) or [SEND], said.append)
    assert ran == ["read_mail"] and out.waiting == [SEND] and said == ["Checking your mail"]
    assert seen_by_planner == [([READ], ["From Rahul: Meeting - can you join at 5?"])]
    assert task.confirmation(out.waiting[0]) == 'Should I email Rahul: "I\'ll join at 5"?'


def test_confirmed_steps_run(monkeypatch):
    ran = fake_actions(monkeypatch)
    out = task.run([SEND], confirmed=True)
    assert ran == ["send_email"] and out.ran == [SEND] and not out.waiting


def test_steps_before_a_send_still_run(monkeypatch):
    ran = fake_actions(monkeypatch)
    out = task.run([{"action": "open_app", "args": ["Mail"]}, SEND])
    assert ran == ["open_app"] and out.waiting == [SEND]


def test_the_planner_can_answer_from_what_was_found(monkeypatch):
    fake_actions(monkeypatch, {"calendar_events": "10:00 AM Standup", "answer": "You have standup at 10"})
    answer = {"action": "answer", "args": ["You have standup at 10"]}
    out = task.run([{"action": "calendar_events", "args": ["today"]}], lambda seen: [answer])
    assert out.last == [answer] and out.replies == ["You have standup at 10"]


def test_no_loop_when_the_planner_repeats_itself(monkeypatch):
    ran = fake_actions(monkeypatch)
    task.run([READ], lambda seen: [READ])
    assert ran == ["read_mail"]


def test_rounds_are_limited(monkeypatch):
    ran = fake_actions(monkeypatch)
    rounds = iter(range(100))
    task.run([READ], lambda seen: [{"action": "read_mail", "args": [str(next(rounds))]}])
    assert len(ran) == task.MAX_ROUNDS


def test_describe():
    assert task.describe([{"action": "open_app", "args": ["Visual Studio Code"]},
                          {"action": "open_website", "args": ["chatgpt.com", "brave"]},
                          {"action": "media_play_pause", "args": ["play"]}]) == \
        "open Visual Studio Code, open chatgpt.com and play music"


def test_steps_guessed_after_a_lookup_are_planned_again(monkeypatch):
    ran = fake_actions(monkeypatch, {"read_mail": "From Rahul: Sync today - join at 5?"})
    guessed = {"action": "send_email", "args": ["Rahul", "Re: Joining", "I'll join"]}
    out = task.run([READ, guessed], lambda seen: [SEND])
    assert ran == ["read_mail"] and out.waiting == [SEND]
