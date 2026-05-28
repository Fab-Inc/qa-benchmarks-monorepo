from primerpro import Word

from .fixtures import (
    create_english_settings,
    create_overlapping_grapheme_settings,
    create_spanish_like_settings,
    grapheme_symbols,
    taught,
    text_data_from_paragraphs,
)


def test_word_handles_empty_input_and_punctuation_only_input():
    settings = create_english_settings()

    assert Word("", settings).grapheme_count() == 0
    assert Word("?!", settings).display_word == ""
    assert Word("?!", settings).grapheme_count() == 0


def test_word_handles_a_single_grapheme_input():
    settings = create_english_settings()
    word = Word("a", settings)

    assert word.grapheme_count() == 1
    assert grapheme_symbols(word) == ["a"]


def test_word_parser_honors_max_size_grapheme_boundary():
    settings = create_english_settings()

    assert grapheme_symbols(Word("night", settings)) == ["n", "igh", "t"]
    assert grapheme_symbols(Word("through", settings)) == [
        "th",
        "r",
        "o",
        "u",
        "g",
        "h",
    ]


def test_word_parser_uses_longest_match_for_overlapping_multigraphs():
    settings = create_overlapping_grapheme_settings()

    assert grapheme_symbols(Word("nga", settings)) == ["nga"]
    assert grapheme_symbols(Word("aaa", settings)) == ["aa", "a"]


def test_text_data_pins_sentence_boundary_edge_cases():
    settings = create_english_settings()
    no_terminator = text_data_from_paragraphs(settings, "ship queen")
    repeated_terminators = text_data_from_paragraphs(settings, "ship?! queen... night")

    assert no_terminator.sentence_count() == 1
    assert no_terminator.word_count() == 2
    assert repeated_terminators.sentence_count() == 3
    assert repeated_terminators.word_count() == 3


def test_spanish_style_fixture_pins_unicode_graphemes_with_accents_and_enye():
    settings = create_spanish_like_settings()

    assert grapheme_symbols(Word("niño", settings)) == ["n", "i", "ñ", "o"]
    assert grapheme_symbols(Word("café", settings)) == ["c", "a", "f", "é"]


def test_english_fixture_exercises_all_configured_extra_graphemes():
    settings = create_english_settings()

    assert grapheme_symbols(Word("day", settings)) == ["d", "ay"]
    assert grapheme_symbols(Word("goat", settings)) == ["g", "oa", "t"]
    assert grapheme_symbols(Word("car", settings)) == ["c", "ar"]
    assert grapheme_symbols(Word("her", settings)) == ["h", "er"]
    assert grapheme_symbols(Word("bird", settings)) == ["b", "ir", "d"]
    assert grapheme_symbols(Word("fork", settings)) == ["f", "or", "k"]
    assert grapheme_symbols(Word("fur", settings)) == ["f", "ur"]
    assert grapheme_symbols(Word("jam", settings)) == ["j", "a", "m"]
    assert grapheme_symbols(Word("van", settings)) == ["v", "a", "n"]
    assert grapheme_symbols(Word("x", settings)) == ["x"]
    assert grapheme_symbols(Word("yes", settings)) == ["y", "e", "s"]
    assert grapheme_symbols(Word("zip", settings)) == ["z", "i", "p"]


def test_spanish_style_fixture_exercises_standalone_q_separately_from_qu():
    settings = create_spanish_like_settings()

    assert grapheme_symbols(Word("qa", settings)) == ["q", "a"]
    assert Word("qa", settings).is_buildable_word(taught("q", "a"))
    assert not Word("queso", settings).is_buildable_word(
        taught("q", "u", "e", "s", "o")
    )
