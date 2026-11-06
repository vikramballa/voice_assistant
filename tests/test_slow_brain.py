"""The slow brain without the LLM: repair() checks the LLM's steps against the words, the grammar
only allows compact plans with real actions."""

from assistant.actions import ACTIONS, calculator
from assistant.brain import mlx_llm, slow
from assistant.brain.repair import repair


def fix(steps, text):
    return repair(steps, text)


def test_web_service_becomes_website_in_the_users_browser():
    assert fix([{"action": "open_app", "args": ["Gemini"]}], "open gemini in my google account") == \
        [{"action": "open_website", "args": ["gemini.google.com", "chrome"]}]
    assert fix([{"action": "open_website", "args": ["gemini.google.com", "safari"]}], "open gemini") == \
        [{"action": "open_website", "args": ["gemini.google.com", "chrome"]}]


def test_misheard_app_is_corrected():
    assert fix([{"action": "open_app", "args": ["Braille"]}], "open braille") == \
        [{"action": "open_app", "args": ["Brave Browser"]}]


def test_placeholders_are_dropped_or_fixed():
    assert fix([{"action": "open_app", "args": ["name"]}], "please open") == []
    assert fix([{"action": "youtube_search", "args": ["query", "chrome"]}], "open youtube") == \
        [{"action": "open_website", "args": ["youtube.com", "chrome"]}]


def test_steps_the_user_did_not_ask_for_are_dropped():
    steps = [{"action": "open_app", "args": ["Messages"]}, {"action": "type_text", "args": ["hi mom"]},
             {"action": "screenshot", "args": []}]
    assert fix(steps, "send a message to mom") == [{"action": "open_app", "args": ["Messages"]}]
    assert fix([{"action": "set_volume", "args": ["100"]}], "turn on dark mode") == []
    assert fix([{"action": "tell_time", "args": ["capital"]}], "what's the capital of france") == []


def test_argument_count_is_fixed():
    assert fix([{"action": "screenshot", "args": ["now"]}], "take a screenshot") == [{"action": "screenshot", "args": []}]
    assert fix([{"action": "set_volume", "args": ["loud"]}], "volume loud") == []


def test_play_means_youtube():
    assert fix([{"action": "web_search", "args": ["arijit singh songs", "chrome"]}], "play arijit singh songs") == \
        [{"action": "youtube_search", "args": ["arijit singh songs", "chrome"]}]


def test_play_with_a_title_is_youtube_not_pause():
    assert fix([{"action": "media_play_pause", "args": []}], "play arijit singh songs") == \
        [{"action": "youtube_search", "args": ["arijit singh songs", "chrome"]}]
    assert fix([{"action": "media_play_pause", "args": []}], "resume") == [{"action": "media_play_pause", "args": ["play"]}]


def test_weather_needs_weather_words():
    assert fix([{"action": "weather", "args": ["Mumbai"]}], "will it rain in mumbai") == \
        [{"action": "weather", "args": ["Mumbai"]}]
    assert fix([{"action": "weather", "args": [""]}], "tell me a joke") == []


def test_the_words_win_over_the_model_for_open():
    assert fix([{"action": "open_website", "args": ["drive.google.com", "chrome"]}], "open google docs") == \
        [{"action": "open_website", "args": ["docs.google.com", "chrome"]}]
    assert fix([{"action": "open_app", "args": ["WhatsApp"]}], "open whatsapp in chrome") == \
        [{"action": "open_website", "args": ["web.whatsapp.com", "chrome"]}]
    assert fix([{"action": "open_app", "args": ["Xcode"]}], "open vs code") == \
        [{"action": "answer", "args": ["Visual Studio Code isn't installed on this Mac"]}]
    assert fix([{"action": "close_app", "args": ["whatsapp"]}], "close whatsapp") == \
        [{"action": "close_app", "args": ["WhatsApp"]}]


def test_search_results_link_becomes_a_search():
    steps = [{"action": "open_website", "args": ["brave browser", "https://www.google.com/search?q=Python+jobs"]}]
    assert fix(steps, "Open Brave and search for Python jobs.") == [
        {"action": "web_search", "args": ["python jobs", "brave"]}]


def test_youtube_results_link_becomes_a_youtube_search():
    steps = [{"action": "open_website", "args": ["youtube.com/results?search_query=lofi+music", "chrome"]}]
    assert fix(steps, "play lofi music videos on youtube") == [
        {"action": "youtube_search", "args": ["lofi music", "chrome"]}]


def test_website_args_in_wrong_order():
    steps = [{"action": "open_website", "args": ["safari", "github.com"]}]
    assert fix(steps, "go to github in safari") == [{"action": "open_website", "args": ["github.com", "safari"]}]


def _accepts(kind, text):
    g = mlx_llm.Grammar(kind, ACTIONS)
    return g.feed(text) and g.state == "done"


def test_grammar_allows_only_compact_steps_with_real_actions():
    assert _accepts("steps", '{"steps":[{"action":"open_app","args":["Safari"]},{"action":"mute","args":[]}]}')
    assert _accepts("steps", '{"steps":[{"action":"answer","args":["Say \\"hi\\", ok"]}]}')
    assert not _accepts("steps", '{"steps":[{"action":"media_play_pause()","args":[]}]}')  # not an action
    assert not _accepts("steps", '{"steps":[{"action":"close_app","arg":"Safari"}]}')      # wrong key
    assert not _accepts("steps", '{"steps": [{"action":"mute","args":[]}]}')              # spaces cost time


def test_grammar_reply():
    assert _accepts("reply", '{"reply":"Paris."}')
    assert not _accepts("reply", '{"reply":"line\nbreak"}')


def test_steps_from_accepts_dict_or_single_args():
    assert slow._steps_from('{"steps":[{"action":"weather","args":{"place":"Delhi"}}]}') == [
        {"action": "weather", "args": ["Delhi"]}]
    assert slow._steps_from('not json') == []


def test_calculate_needs_real_arithmetic():
    assert fix([{"action": "calculate", "args": ["25 into 25"]}], "what is 25 into 25 on the calculator") == \
        [{"action": "calculate", "args": ["25*25"]}]
    assert fix([{"action": "calculate", "args": ["0"]}], "open calculator and perform operations") == []


def test_browser_shortcuts_go_to_the_browser():
    assert fix([{"action": "press_keys", "args": ["cmd+t", "x"]}], "open a new tab in brave") == \
        [{"action": "press_keys", "args": ["cmd+t", "brave"]}]
    assert fix([{"action": "press_keys", "args": ["cmd+s"]}], "save the file") == \
        [{"action": "press_keys", "args": ["cmd+s"]}]


def test_opening_the_browser_before_a_search_in_it_is_dropped():
    steps = [{"action": "open_app", "args": ["Brave Browser"]}, {"action": "web_search", "args": ["python jobs"]}]
    assert fix(steps, "open brave and search for python jobs") == \
        [{"action": "web_search", "args": ["python jobs", "brave"]}]


def test_junk_args_are_dropped():
    assert fix([{"action": "close_app", "args": [","]}], "close safari") == []


def test_numbers_the_user_never_said_are_not_calculated(monkeypatch):
    monkeypatch.setattr(calculator, "_last_result", 625.0)
    assert fix([{"action": "calculate", "args": ["25*25"]}], "perform a multiplication operation") == \
        [{"action": "answer", "args": ["What should I calculate?"]}]
    assert fix([{"action": "calculate", "args": ["625*4"]}], "and multiply it with four") == \
        [{"action": "calculate", "args": ["*4"]}]


def test_curly_quote_ending_is_cleaned():
    assert fix([{"action": "answer", "args": ["I can't send messages.”]}]}"]}], "send a message to mom") == \
        [{"action": "answer", "args": ["I can't send messages."]}]


def test_grammar_can_close_early():
    g = mlx_llm.Grammar("steps", ACTIONS)
    g.feed('{"steps":[{"action":"answer","args":["hi')
    assert g.feed(g.closing()) and g.state == "done"


def test_questions_are_not_turned_into_searches():
    assert fix([{"action": "web_search", "args": ["prime minister of india"]}], "who is the prime minister of india") == []
    assert fix([{"action": "open_website", "args": ["www.google.com"]}, {"action": "web_search", "args": ["python jobs"]}],
               "open brave and search for python jobs") == [{"action": "web_search", "args": ["python jobs", "brave"]}]


def test_unmute_is_never_mute():
    assert fix([{"action": "mute", "args": []}], "unmute the sound") == [{"action": "unmute", "args": []}]
    assert fix([{"action": "mute", "args": []}], "mute the sound") == [{"action": "mute", "args": []}]


def test_opening_the_calculator_is_not_a_request_to_calculate():
    assert not slow.CALC_WITHOUT_NUMBERS.search("open the calculator app")
    assert slow.CALC_WITHOUT_NUMBERS.search("open calculator and do some math")
    assert slow.CALC_WITHOUT_NUMBERS.search("calculate something")


def test_plan_json_with_a_tab_inside_a_string_is_read():
    assert slow._steps_from('{"steps":[{"action":"type_text","args":["a\tb"]}]}') == \
        [{"action": "type_text", "args": ["a\tb"]}]


def test_repair_never_sends_an_empty_message_or_repeats_a_step():
    assert repair([{"action": "send_message", "args": ["mom", ""]}], "send a message to mom") == \
        [{"action": "answer", "args": ["What should the message say?"]}]
    twice = [{"action": "web_search", "args": ["python jobs", "brave"]}] * 2
    assert repair(twice, "open brave and search for python jobs") == twice[:1]
