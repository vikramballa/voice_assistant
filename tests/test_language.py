"""The text rules in assistant/language/ (pure functions, no models)."""

import pytest

from assistant.language import arguments, catalog, dialogue, in_app, sentence, spoken_math


@pytest.mark.parametrize("text, rest", [
    ("Hey Laya, open Safari.", "open Safari."),
    ("hey leia take a screenshot", "take a screenshot"),
    ("Laya lock my mac", "lock my mac"),
    ("Okay Layla, volume up", "volume up"),
    ("Hey Laya.", ""),
    ("open safari", None),
    ("I was talking to Laura yesterday", None),
])
def test_strip_wake(text, rest):
    assert sentence.strip_wake(text) == rest


@pytest.mark.parametrize("text, clauses", [
    ("open chrome and search youtube for neural networks", ["search youtube for neural networks"]),
    ("take a screenshot and lock my mac", ["take a screenshot", "lock my mac"]),
    ("open notes and then type buy milk", ["open notes", "type buy milk"]),
    ("search for salt and pepper", ["search for salt and pepper"]),
    ("play rock and roll on youtube", ["play rock and roll on youtube"]),
    ("open chrome", ["open chrome"]),
    ("mute, then lock the screen", ["mute", "lock the screen"]),
    ("open safari and open github.com", ["open github.com"]),
])
def test_split_commands(text, clauses):
    assert sentence.split_commands(text) == clauses


@pytest.mark.parametrize("intent, clause, sentence, args", [
    ("youtube_search", "search youtube for neural networks", "open chrome and search youtube for neural networks",
     ["neural networks", "chrome"]),
    ("youtube_search", "play a video about black holes", None, ["black holes", "chrome"]),
    ("youtube_search", "youtube search for how to make pasta in safari", None, ["how to make pasta", "safari"]),
    ("web_search", "google cheap flights to dubai", "open safari and google cheap flights to dubai",
     ["cheap flights to dubai", "safari"]),
    ("web_search", "search the web for transformer architecture", None, ["transformer architecture", "chrome"]),
    ("open_website", "open youtube in safari", None, ["youtube.com", "safari"]),
    ("open_website", "visit wikipedia.org", None, ["wikipedia.org", "chrome"]),
    ("open_app", "open finder", None, ["Finder"]),
    ("open_app", "open system settings", None, ["System Settings"]),
    ("open_app", "launch chrome", None, ["Google Chrome"]),
    ("open_app", "open some app that does not exist", None, None),
    ("close_app", "quit the calculator app", None, ["Calculator"]),
    ("open_folder", "open my downloads folder", None, ["downloads"]),
    ("set_volume", "set volume to 30", None, ["30"]),
    ("set_volume", "set volume to 300", None, None),
    ("set_volume", "set the volume higher", None, None),
    ("type_text", "Type Hello World!", None, ["Hello World!"]),
    ("type_text", "write see you tomorrow", None, ["see you tomorrow"]),
    ("screenshot", "take a screenshot", None, []),
])
def test_args_for(intent, clause, sentence, args):
    assert arguments.args_for(intent, clause, sentence) == args


@pytest.mark.parametrize("clause, url", [
    ("go to jithub.com", "github.com"),
    ("open redit.com", "reddit.com"),
    ("visit python.org", "python.org"),
    ("open my-own-site.dev", "my-own-site.dev"),
])
def test_website_fixes_near_misses_only(clause, url):
    assert catalog.website(clause) == url


@pytest.mark.parametrize("clause, found", [
    ("Please open Gemini in my Google account", {"open_website": ["gemini.google.com"]}),
    ("open youtube", {"open_website": ["youtube.com"]}),
    ("open google drive for me", {"open_website": ["drive.google.com"]}),
    ("open my gmail", {"open_website": ["mail.google.com"]}),
    ("open github dot com", {"open_website": ["github.com"]}),
    ("open the calculator app", {"open_app": ["Calculator"]}),
    ("open calender", {"open_app": ["Calendar"]}),  # misheard, close enough
    ("open my downloads folder", {"open_folder": ["downloads"]}),
    ("open the pod bay doors", {}),
    ("open", {}),
])
def test_open_candidates(clause, found):
    assert catalog.open_candidates(clause) == found


def test_resolve_name_fixes_mishearing_and_web_services():
    assert catalog.resolve_name("Braille") == ("open_app", ["Brave Browser"])
    assert catalog.resolve_name("Gemini") == ("open_website", ["gemini.google.com"])
    assert catalog.resolve_name("mac") is None
    assert catalog.resolve_name("host") is None


@pytest.mark.parametrize("text, question", [
    ("Open.", "Open what?"), ("please open", "Open what?"), ("search", "Search for what?"),
    ("close", "Close which app?"), ("type", "Type what?"), ("set the volume to", "What volume, from 0 to 100?"),
    ("open safari", None), ("play", None), ("mute", None),
])
def test_follow_up_question(text, question):
    assert dialogue.follow_up_question(text) == question


def test_complete_joins_answer_unless_it_is_a_new_command():
    assert dialogue.complete("Open.", "Safari") == "Open Safari"
    assert dialogue.complete("open", "take a screenshot") == "take a screenshot"


@pytest.mark.parametrize("text, clauses", [
    ("open youtube and search cats", ["search cats on youtube"]),
    ("open google and search for pizza", ["search for pizza"]),
    ("open brave and search for pizza", ["search for pizza"]),
    ("open youtube and take a screenshot", ["open youtube", "take a screenshot"]),
])
def test_split_merges_site_then_search(text, clauses):
    assert sentence.split_commands(text) == clauses


def test_brave_and_cut_off_queries():
    assert arguments.args_for("web_search", "search for pizza", "open brave and search for pizza") == ["pizza", "brave"]
    assert arguments.args_for("web_search", "Search neural networks in...", None) == ["neural networks", "chrome"]


@pytest.mark.parametrize("text, direction", [
    ("volume up", "volume_up"), ("turn it up", "volume_up"), ("too loud", "volume_down"),
    ("make it softer", "volume_down"), ("set the volume", None),
])
def test_volume_direction(text, direction):
    assert arguments.volume_direction(text) == direction


def test_known_app_that_is_not_installed():
    assert catalog.missing_app("open vs code") == "Visual Studio Code"
    assert catalog.missing_app("open safari") is None


def test_browser_named_means_the_website():
    assert catalog.open_candidates("open whatsapp in chrome") == {"open_website": ["web.whatsapp.com"]}
    assert catalog.open_candidates("open whatsapp") == {"open_app": ["WhatsApp"]}


@pytest.mark.parametrize("text, cleaned", [
    ("In Braille browser, open YouTube.", "open YouTube in brave"),
    ("In Brave, search for cheap laptops.", "search for cheap laptops in brave"),
    ("Open Brave and it.", "Open Brave"),
    ("open braille", "open Brave"),
    ("search for salt and pepper", "search for salt and pepper"),
])
def test_clean_fixes_speech_slips(text, cleaned):
    assert sentence.clean(text) == cleaned


@pytest.mark.parametrize("text, kind", [
    ("search google for iphone price", "web_search"), ("search about transformers", "web_search"),
    ("find videos about cats", "youtube_search"), ("play lofi on youtube", "youtube_search"),
])
def test_search_kind(text, kind):
    assert arguments.search_kind(text) == kind


@pytest.mark.parametrize("text, title", [
    ("play arijit singh songs", "arijit singh songs"), ("play the music", None), ("play", None),
    ("play it again", None), ("play the next song", None),
])
def test_play_title(text, title):
    assert arguments.play_title(text) == title


@pytest.mark.parametrize("text, expression", [
    ("What is 25 into 25?", "25*25"),
    ("In calculator, what is the value of 25 into 25?", "25*25"),
    ("calculate 12 times 7", "12*7"),
    ("25x25", "25*25"),
    ("30 × 4", "30*4"),
    ("What is twenty five plus seventeen?", "25+17"),
    ("one hundred and five minus five", "105-5"),
    ("What is 1,000 times 3?", "1000*3"),
    ("10 percent of 250", "10*250/100"),
    ("subtract 3 from 10", "10-3"),
    ("multiply that by 3", "*3"),
    ("add 5", "+5"),
    ("what is the time", None),
    ("add milk to my list", None),
    ("open calculator", None),
    ("search 5 plus 5 on google", None),
])
def test_math_expression(text, expression):
    assert spoken_math.math_expression(text) == expression


@pytest.mark.parametrize("text, stop", [
    ("stop", True), ("Stop listening.", True), ("Okay, stop.", True), ("that's all, thanks", True), ("bye", True),
    ("stop the music", False), ("stop playing", False), ("open safari", False),
])
def test_is_stop(text, stop):
    assert dialogue.is_stop(text) == stop


@pytest.mark.parametrize("text, browser", [
    ("in Brave", "brave"), ("No, open it in Brave.", "brave"), ("use brave instead", "brave"),
    ("In Brave browser.", "brave"), ("in braille", "brave"), ("open brave", None), ("search cats in brave", None),
])
def test_redo_in_browser(text, browser):
    assert dialogue.redo_in_browser(text) == browser


@pytest.mark.parametrize("text, keys", [
    ("Open a new tab.", "cmd+t"), ("close this tab", "cmd+w"), ("refresh the page", "cmd+r"),
    ("go back to the previous page", "cmd+["), ("Press the Enter key.", "enter"), ("select all and copy", "cmd+a cmd+c"),
    ("next tab", "ctrl+tab"), ("undo that", "cmd+z"), ("go back", None), ("next song", None), ("open youtube", None),
])
def test_shortcut(text, keys):
    assert in_app.shortcut(text) == keys


def test_click_and_scroll_targets():
    assert in_app.click_target("click the sign in button") == "sign in"
    assert in_app.click_target("Click on the search bar.") == "search"
    assert in_app.click_target("click it") is None
    assert in_app.click_target("hit this song") is None
    assert in_app.scroll_direction("scroll up a bit") == "up"
    assert in_app.scroll_direction("scroll to the comments") is None


def test_browser_in_use_is_the_default():
    assert arguments.args_for("open_website", "open youtube", browser="brave") == ["youtube.com", "brave"]
    assert arguments.args_for("open_website", "open youtube in safari", browser="brave") == ["youtube.com", "safari"]
    assert arguments.args_for("open_website", "open youtube") == ["youtube.com", "chrome"]


def test_open_google_does_not_wait_for_more():
    assert not dialogue.trails_off("open google")


def test_echo_and_noise():
    reply = "25 times 25 is 625"
    assert dialogue.is_echo("25 times 25 is 625.", reply) and dialogue.is_echo("twenty five times twenty five is 625", reply)
    assert not dialogue.is_echo("Multiply that by four.", reply)
    assert dialogue.is_noise("6") and dialogue.is_noise("Thank you.") and not dialogue.is_noise("pause")


def test_reply_tail_is_removed_from_what_the_mic_heard():
    heard = dialogue.without_echo("625 multiplied that by four.", "25 times 25 is 625")
    assert heard == "multiplied that by four" and spoken_math.math_expression(heard) == "*4"


@pytest.mark.parametrize("text, args", [
    ("Scrone up.", ["up"]), ("School down.", ["down"]), ("scroll up a bit", ["up", "bit"]),
    ("scroll down a lot", ["down", "lot"]), ("scroll to the bottom", ["bottom"]), ("scroll", ["down"]),
    ("Scroll down 2 pages", ["down", "2"]), ("page down", ["down"]), ("go to the top", ["top"]),
    ("scroll right", ["right"]), ("scroll to the comments", None), ("go to school", None),
])
def test_scroll_args(text, args):
    assert in_app.scroll_args(text) == args


@pytest.mark.parametrize("text, keys", [
    ("zoom in", "cmd+="), ("zoom out", "cmd+-"), ("reset zoom", "cmd+0"), ("make the text bigger", "cmd+="),
    ("full screen", "ctrl+cmd+f"), ("minimize this window", "cmd+m"), ("go to the top of the page", "cmd+up"),
    ("reopen the closed tab", "cmd+shift+t"),
])
def test_more_shortcuts(text, keys):
    assert in_app.shortcut(text) == keys


@pytest.mark.parametrize("text, question", [
    ("What is on my screen?", "What is on my screen?"), ("See my screen.", "What is on the screen?"),
    ("Why can't you see my Screen.", "What is on the screen?"), ("Now can you open the screen?", "What is on the screen?"),
    ("what does this error say", "what does this error say"), ("summarize this page", "summarize this page"),
    ("take a screenshot", None), ("open safari", None), ("open screen sharing", None),
])
def test_screen_question(text, question):
    assert in_app.screen_question(text) == question


def test_calculator_redo_and_fillers():
    assert spoken_math.calculator_redo("Perform in Calculator.")
    assert spoken_math.calculator_redo("I want to see in Calculator. Please proceed.")
    assert not spoken_math.calculator_redo("open calculator")
    assert not spoken_math.calculator_redo("In calculator, what is 5 times 3")
    assert dialogue.is_noise("Okay, now...") and not dialogue.is_noise("next") and not dialogue.is_noise("open safari")


LONG_REPLY = ("I am a voice assistant controlling your Mac. I can help you with various tasks like opening applications, "
              "searching the web, controlling media, and more. How can I assist you today?")


@pytest.mark.parametrize("heard, echo", [
    ("I have a voice assist.", True),          # misheard pieces of our reply (2026-09-29 log)
    ("Controlling your Mac.", True),
    ("I can help you with bear.", True),
    ("multiply that by 4", False),             # real commands said over it
    ("close chrome", False),
    ("open safari", False),
])
def test_echo_of_a_long_reply_is_fuzzy(heard, echo):
    assert dialogue.is_echo(heard, LONG_REPLY) is echo


@pytest.mark.parametrize("text, level", [
    ("set the volume to fifty", "50"), ("volume to twenty five", "25"), ("set volume at 30 percent", "30"),
    ("turn the volume up by 20", None), ("turn it up one more time", None),
])
def test_volume_level_in_words_but_not_a_step(text, level):
    assert arguments.number(text) == level


def test_volume_step_and_unmute():
    assert arguments.volume_step("turn the volume up by twenty") == "20"
    assert arguments.volume_step("set the volume to 20") is None
    assert arguments.volume_direction("unmute") == "unmute"
    assert arguments.volume_direction("unmute the sound") == "unmute"
    assert arguments.volume_direction("mute the sound") == "mute"


def test_type_in():
    assert arguments.typed_text("type in hello world") == "hello world"


@pytest.mark.parametrize("text, expression", [
    ("what is the square root of 16", "16^0.5"), ("half of 50", "50/2"), ("double 21", "21*2"),
])
def test_more_spoken_math(text, expression):
    assert spoken_math.math_expression(text) == expression


@pytest.mark.parametrize("text, found", [
    ("No, I meant open Notes", (True, "open notes")), ("I said open Brave", (True, "open brave")),
    ("no, open notes", (False, "open notes")), ("nope, wait, open safari", (False, "open safari")),
    ("no", None), ("open notes", None), ("Notes app is open now", None),
])
def test_correction(text, found):
    assert dialogue.correction(text) == found


# --- 2026-10-03: chart ranges, things by position, accounts, notes ------------------------------

@pytest.mark.parametrize("text, button", [
    ("give me last 5 days", "5D"), ("Show the last five days.", "5D"), ("show me 1 month", "1M"),
    ("last one year", "1Y"), ("check the last 5 days", "5D"), ("give me the 1 year chart", "1Y"),
    ("show me the max", "MAX"), ("year to date", "YTD"), ("past 6 months", "6M"),
])
def test_chart_range(text, button):
    assert in_app.chart_range(text) == button


@pytest.mark.parametrize("text", ["set a timer for 5 minutes", "remind me in 2 days to pay rent",
                                  "play the last song", "search for the last 5 days of news in india"])
def test_not_a_chart_range(text):
    assert in_app.chart_range(text) is None


def test_last_days_is_not_a_song():
    assert arguments.media_kind("give me last 5 days") is None
    assert arguments.media_kind("play the last song") == "previous_track"
    assert arguments.media_kind("go back") == "previous_track"


@pytest.mark.parametrize("text, target", [
    ("pick the first account", "first account"), ("use the second account", "second account"),
    ("open the second video", "second video"), ("play the third video", "third video"),
    ("open the first result", "first result"), ("pick my work account", "work account"),
    ("choose vikram account", "vikram account"), ("click the last link", "last link"),
])
def test_click_by_position_or_account(text, target):
    assert in_app.click_target(text) == target


@pytest.mark.parametrize("text, note", [
    ("write a note saying buy milk", "buy milk"), ("Write buy milk in notes.", "buy milk"),
    ("in notes write call mom at 5", "call mom at 5"), ("add eggs to my notes", "eggs"),
    ("note down the wifi password is laya123", "the wifi password is laya123"),
    ("take a note", ""), ("create a new note", ""), ("make a note: Meeting at 3", "Meeting at 3"),
])
def test_note_text(text, note):
    assert arguments.note_text(text) == note


@pytest.mark.parametrize("text", ["type hello in the note", "take a screenshot", "write hello world", "open notes"])
def test_not_a_new_note(text):
    assert arguments.note_text(text) is None


def test_typed_message_text():
    assert arguments.typed_text("type a message saying I will be late") == "I will be late"
    assert arguments.typed_text("write a message hi there") == "hi there"


def test_dictation_asks_then_types_the_answer():
    assert dialogue.follow_up_question("type what I say") == "Type what?"
    assert dialogue.complete("type what I say", "see you at five") == "type see you at five"
    assert dialogue.follow_up_question("write a note saying") == "What should the note say?"
    assert dialogue.complete("write a note saying", "buy milk") == "write a note saying buy milk"


def test_misheard_note():
    assert sentence.fix_misheard("Create a new node.") == "Create a new note."
    assert sentence.fix_misheard("write buy milk in nodes") == "write buy milk in notes"
    assert sentence.fix_misheard("open node js docs") == "open node js docs"
    assert arguments.note_text(sentence.clean("Create a new node.")) == ""


@pytest.mark.parametrize("text, waits", [
    ("Hey Laya, open ChatGPT.", True), ("open youtube", True), ("search for cats", True), ("play despacito", True),
    ("open chatgpt in brave", False), ("open notes", False), ("scroll down", False), ("new tab", False),
    ("play the next song", False), ("what is 2 plus 2", False), ("Hey Laya.", False), ("set the volume to", True),
])
def test_may_go_on(text, waits):
    assert dialogue.may_go_on(text) is waits


@pytest.mark.parametrize("heard, fixed", [
    ("search jithub for laya", "search GitHub for laya"), ("open spotifi", "open Spotify"),
    ("Open chat gpt in brave", "Open ChatGPT in brave"), ("open you tube", "open YouTube"),
    ("open gmial", "open Gmail"), ("close safary.", "close Safari."), ("scrol down", "scroll down"),
])
def test_spelling_fixes_names(heard, fixed):
    from assistant.language import spelling
    assert spelling.correct(heard)[0] == fixed


@pytest.mark.parametrize("text", [
    "search for tata tcs stock", "give me last 5 days", "what is the capital of india", "type jithub is great",
    "write a note saying spotifi rocks", "search for best biryani in hyderabad", "take a screenshot",
])
def test_spelling_leaves_words_and_dictation(text):
    from assistant.language import spelling
    assert spelling.correct(text) == (text, [])


# --- 2026-10-03 (night): what the user means by "remove that", "stop", "read my chats" -----------

@pytest.mark.parametrize("text, what", [
    ("remove that", "all"), ("Remove that.", "all"), ("delete it", "all"), ("clear the chat bar", "all"),
    ("remove the text in the chat box", "all"), ("delete what I typed", "all"), ("I want to remove that", "all"),
    ("can you remove that", "all"), ("clear the search bar", "all"), ("delete the last word", "word"),
])
def test_clear_request(text, what):
    assert in_app.clear_request(text) == what


@pytest.mark.parametrize("text", ["delete this email", "remove the app", "delete the file", "clear history",
                                  "delete my account", "remove spotify", "remove that tab", "delete it from my downloads"])
def test_not_a_clear_request(text):
    assert in_app.clear_request(text) is None


@pytest.mark.parametrize("text, is_it", [
    ("stop the music", True), ("resume", True), ("pause", True), ("pause the video", True), ("start the music", True),
    ("stop", False), ("play", False), ("play despacito", False), ("stop listening", False), ("I made it. Stop.", False),
])
def test_play_pause_request(text, is_it):
    assert in_app.play_pause_request(text) is is_it


def test_stop_in_a_sentence_is_not_music():
    assert arguments.media_kind("I made it. Stop.") is None  # was play/pause: started a song
    assert arguments.media_kind("stop the music") == "media_play_pause"
    assert arguments.play_or_pause("stop the music") == "pause"
    assert arguments.play_or_pause("resume") == "play"


@pytest.mark.parametrize("text", ["Can you see my recent charts?", "read my previous chats", "what are my recent chats",
                                  "what did I ask before", "show me my emails"])
def test_reading_my_chats_is_a_screen_question(text):
    assert in_app.screen_question(text) == text  # the user's own question, not "what's on my screen"


@pytest.mark.parametrize("text", ["show me my downloads", "open my messages", "check my battery", "read me a story"])
def test_not_a_screen_question(text):
    assert in_app.screen_question(text) is None


@pytest.mark.parametrize("text, site, question", [
    ("ask chatgpt what is a neural network", "chatgpt", "what is a neural network"),
    ("Ask ChatGPT to write a poem about rain in Brave", "chatgpt", "write a poem about rain"),
    ("ask what is TCS share price in chatgpt", "chatgpt", "what is TCS share price"),
    ("in chatgpt, ask how to cook rice", "chatgpt", "how to cook rice"),
])
def test_ask_ai(text, site, question):
    assert arguments.ask_ai(text) == (site, question)


@pytest.mark.parametrize("text", ["ask mom to call me", "ask me later", "search for cats in chrome"])
def test_not_ask_ai(text):
    assert arguments.ask_ai(text) is None


def test_send_it_presses_enter():
    assert in_app.shortcut("send it") == "enter"
    assert in_app.shortcut("send the message") == "enter"
    assert in_app.shortcut("send an email to mom") is None


@pytest.mark.parametrize("text", ["OK, right. What I see?", "right what I said", "write in the note", "Write in the note."])
def test_misheard_write_what_i_say_asks_what(text):
    assert dialogue.follow_up_question(text) == "Type what?"


@pytest.mark.parametrize("text, added", [
    ("Continue. I'm a developer.", "I'm a developer."), ("OK, continue. I work at home.", "I work at home."),
    ("also write buy milk", "buy milk"), ("keep writing see you soon", "see you soon"),
    ("continue, open safari", None), ("continue the music", None), ("continue playing", None), ("Continue.", None),
    ("write a note saying hi", None)])
def test_dictation_while_writing(text, added):
    assert arguments.dictation(text) == added


def test_write_in_the_note_types_only_the_text():
    assert arguments.typed_text("Write in the note. Hello, how are you?") == "Hello, how are you?"


@pytest.mark.parametrize("text, sure, words", [
    ("I'm a developer.", False, "I'm a developer."), ("Hello, how are you?", False, "Hello, how are you?"),
    ("continue, see you soon", True, "see you soon"), ("next song", True, None), ("open safari", True, None),
    ("what time is it", False, None), ("set a timer for 5 minutes", False, None), ("Laya, what is this", False, None),
    ("stop", False, None), ("undo", True, None), ("we met at the park yesterday evening", True,
                                                "we met at the park yesterday evening")])
def test_dictated_in_notes(text, sure, words):
    assert dialogue.dictated(text, sure) == words


@pytest.mark.parametrize("text, out", [
    ("Hi comma how are you question mark.", "Hi, how are you?"), ("Next line.", "\n"), ("new paragraph", "\n"),
    ("I am here full stop new line see you", "I am here.\nSee you"), ("Hello.", "Hello."),
    ("a period of time", "a period of time")])
def test_spoken_punctuation(text, out):
    assert arguments.spoken_punctuation(text) == out


@pytest.mark.parametrize("text", ["OK, what are you seeing?", "Okay, quit notes.", "can you read this", "Make sure to call mom"])
def test_commands_with_fillers_are_not_dictated(text):
    assert dialogue.dictated(text, False) is None


def test_next_line_in_notes_is_a_new_line():
    assert dialogue.dictated("Next line.", True) == "\n"


@pytest.mark.parametrize("text, name", [("Go to ChatGPT tab.", "chatgpt"), ("switch to my youtube tab", "youtube"),
                                        ("open the gmail tab", "gmail"), ("go to the next tab", None),
                                        ("open a new tab", None), ("Go to previous tab.", None)])
def test_tab_by_name(text, name):
    assert in_app.tab_by_name(text) == name


@pytest.mark.parametrize("text, tab", [("Previous step.", "previous tab"), ("next tap", "next tab"), ("next song", None),
                                       ("previous page", None), ("previous tab", None), ("next line", None)])
def test_tab_step_in_a_browser(text, tab):
    assert in_app.tab_step(text) == tab


@pytest.mark.parametrize("text, out", [("Done. Go to previous tab.", "Go to previous tab."), ("Okay, quit notes.", "quit notes."),
                                       ("Done.", "Done."), ("so what is 2 plus 2", "what is 2 plus 2")])
def test_leading_fillers_dropped(text, out):
    assert sentence.clean(text) == out


def test_what_are_you_seeing_is_a_screen_question():
    assert in_app.screen_question("OK, what are you seeing?")


def test_previous_step_in_brave_is_previous_tab():
    from assistant.brain import fast
    steps, _ = fast.decide("Done. Previous step.", "brave")
    assert steps == [{"action": "press_keys", "args": ["ctrl+shift+tab", "brave"]}]
    steps, _ = fast.decide("Go to ChatGPT tab.", "brave")
    assert steps == [{"action": "switch_tab", "args": ["chatgpt", "brave"]}]


@pytest.mark.parametrize("a, b, alike", [("hackle", "hacker", True), ("by", "buy", True), ("nodes", "notes", True),
                                         ("milk", "chrome", False), ("developer", "designer", False)])
def test_sound_alike(a, b, alike):
    from assistant.language.phonetic import sound_alike
    from assistant.brain.proofread import SOUND_ALIKE
    assert (sound_alike(a, b) >= SOUND_ALIKE) == alike


@pytest.mark.parametrize("heard, fixed, ok", [
    ("And a hackle.", "And a hacker.", True), ("I need to by milk", "I need to buy milk", True),
    ("I am Vikram.", "I am a developer.", False),          # a different meaning
    ("I like tea.", "I like tea and coffee.", False),      # added words
    ("Hi, how are you?", "Hello, how are you doing?", False)])
def test_auto_correction_only_keeps_sound_alike_fixes(heard, fixed, ok):
    from assistant.brain.proofread import believable
    assert bool(believable(heard, fixed)) == ok


@pytest.mark.parametrize("text, answer", [("Yes.", True), ("Yeah, send it", True), ("okay go ahead", True),
                                          ("No, don't.", False), ("cancel", False), ("open safari", None)])
def test_yes_or_no(text, answer):
    assert dialogue.yes_or_no(text) is answer


@pytest.mark.parametrize("text, found", [
    ("When I say start my work day, open VS Code and Brave", ("start my work day", "open VS Code and Brave")),
    ('When I say "focus mode" then mute and open Notes.', ("focus mode", "mute and open Notes")),
    ("create a routine called good night that locks the screen", ("good night", "locks the screen")),
    ("when I was young I played cricket", None)])
def test_routine(text, found):
    assert dialogue.routine(text) == found


@pytest.mark.parametrize("text, personal", [
    ("email Rahul that I'm late", True), ("send a message to mom saying hi", True), ("text mom I'm on my way", True),
    ("what's on my calendar today", True), ("read my emails", True), ("reply to the last email", True),
    ("open mail", False), ("open calendar", False), ("type a message saying hi", False), ("send it", False)])
def test_personal_task(text, personal):
    assert arguments.personal_task(text) is personal


def test_a_message_without_words_asks_what_to_say():
    assert dialogue.follow_up_question("send a message to mom") == "What should the message say?"
    assert dialogue.follow_up_question("text mom") == "What should the message say?"
    assert dialogue.follow_up_question("text mom I'm late") is None
    assert dialogue.complete("text mom", "I'm late") == "text mom saying I'm late"
