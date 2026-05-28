from primerpro import Word

from .fixtures import (
    create_bantu_like_settings,
    grapheme_symbols,
    syllable_symbols,
)


def test_word_uses_longest_match_grapheme_decomposition_for_multigraph_consonants():
    settings = create_bantu_like_settings()
    word = Word("mbata", settings)

    assert word.display_word == "mbata"
    assert grapheme_symbols(word) == ["mb", "a", "t", "a"]


def test_word_uses_longest_match_grapheme_decomposition_for_long_vowels():
    settings = create_bantu_like_settings()
    word = Word("kaana", settings)

    assert grapheme_symbols(word) == ["k", "aa", "n", "a"]


def test_word_canonicalizes_uppercase_inventory_matches_to_inventory_symbols():
    settings = create_bantu_like_settings()
    word = Word("MBata", settings)

    assert word.display_word == "mbata"
    assert grapheme_symbols(word) == ["mb", "a", "t", "a"]


def test_explicit_syllable_breaks_are_preserved_while_display_word_removes_dots():
    settings = create_bantu_like_settings()
    word = Word("mba.ta", settings)

    assert word.display_word == "mbata"
    assert word.syllable_count() == 2
    assert syllable_symbols(word) == ["mba", "ta"]
    assert grapheme_symbols(word) == ["mb", "a", "t", "a"]


def test_punctuation_configured_as_general_or_ending_punctuation_is_removed():
    settings = create_bantu_like_settings()
    word = Word('"ngoma,"', settings)

    assert word.display_word == "ngoma"
    assert grapheme_symbols(word) == ["ng", "o", "m", "a"]
