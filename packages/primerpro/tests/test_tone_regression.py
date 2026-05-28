from primerpro import Word

from .fixtures import create_tone_settings, grapheme_symbols


def test_tone_inventory_entries_expose_level_marker_and_tone_bearing_unit():
    settings = create_tone_settings()
    high = settings.grapheme_inventory.get_tone(
        settings.grapheme_inventory.find_tone_index("á")
    )

    assert high.is_tone
    assert high.level == "H"
    assert high.tone_bearing_unit.symbol == "a"
    assert high.get_marker() == "V"
    assert high.matches_level("H")
    # This pins PrimerPro's current behavior, but it may be incorrect: the method
    # name suggests this should match high.tone_bearing_unit instead.
    assert high.matches_tone_bearing_unit(high)
    assert not high.matches_tone_bearing_unit(high.tone_bearing_unit)


def test_word_tone_helpers_strip_tones_to_tone_bearing_units():
    settings = create_tone_settings()
    word = Word("má", settings)

    assert grapheme_symbols(word) == ["m", "á"]
    assert word.get_word_without_tone() == "ma"
    assert word.is_tone_contain_in_word("á")
    assert word.get_cv_shape_of_word() == "CV"
    assert word.cv_pattern == "CV"


def test_minimal_pair_helpers_can_ignore_tone_bearing_differences():
    settings = create_tone_settings()
    inventory = settings.grapheme_inventory
    plain = Word("ma", settings)
    high = Word("má", settings)

    assert plain.is_minimal_pair(high, False) == 1
    assert plain.is_minimal_pair(high, True) == -1
    assert plain.is_minimal_pair_for_graphemes(
        high,
        False,
        inventory.get_vowel(inventory.find_vowel_index("a")),
        inventory.get_tone(inventory.find_tone_index("á")),
    )
