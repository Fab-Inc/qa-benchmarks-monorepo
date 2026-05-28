from primerpro import GraphemeTaughtOrder, SightWords, Syllable, Word

from .fixtures import (
    assert_contains_highlighted,
    create_bantu_like_settings,
    create_english_settings,
    syllable_symbols,
    taught,
    word_list_from_words,
)


def test_word_auto_syllabification_for_ata_is_pinned_explicitly():
    settings = create_bantu_like_settings()
    word = Word("ata", settings)

    assert word.syllable_count() == 2
    assert syllable_symbols(word) == ["a", "ta"]


def test_word_exposes_cv_shape_and_stored_cv_pattern_from_decomposed_graphemes():
    settings = create_english_settings()
    word = Word("ship", settings)

    assert word.get_cv_shape_of_word() == "CVC"
    assert word.cv_pattern == "CVC"


def test_word_syllable_position_queries_report_expected_positions():
    settings = create_bantu_like_settings()
    word = Word("at.tam", settings)

    assert word.is_closed_syllable(0)
    assert word.is_closed_syllable(1)
    assert not word.is_open_syllable(0)
    assert word.is_syllable_onset("t")
    assert word.is_syllable_coda("m")
    assert word.is_syllable_init("t")
    assert word.is_syllable_medial("a")
    assert word.is_syllable_final("m")


def test_syllable_medial_query_matches_inside_multigraphs():
    settings = create_english_settings()
    syllable = Syllable("asha", settings)

    assert syllable.is_syllable_medial("sh")
    # This pins PrimerPro's current behavior, but it may be incorrect: h is not a
    # standalone medial grapheme here, only a substring of the sh grapheme.
    assert syllable.is_syllable_medial("h")


def test_word_highlight_helpers_wrap_whole_words_when_target_graphemes_are_present():
    settings = create_english_settings()
    word = Word("ship", settings)
    graphemes = taught("th", "sh")

    assert_contains_highlighted("ship", word.get_word_with_highlight_grapheme("sh"))
    assert_contains_highlighted(
        "ship", word.get_word_with_highlight_grapheme(graphemes)
    )
    assert word.get_word_with_highlight_grapheme("th") == "ship"


def test_word_sight_word_lookup_is_exact_over_display_words():
    settings = create_english_settings()
    settings.sight_words.add_word("the")

    assert Word("the", settings).is_sight_word()
    assert Word("The", settings).is_sight_word()
    assert not Word("then", settings).is_sight_word()


def test_word_equality_compares_decomposed_grapheme_sequences():
    settings = create_english_settings()

    assert Word("ship", settings).is_same(Word("ship", settings))
    assert not Word("ship", settings).is_same(Word("thin", settings))
    assert not Word("ship", settings).is_same(Word("Ship", settings))


def test_word_minimal_pair_helpers_identify_single_and_harmony_differences():
    settings = create_english_settings()
    ship = Word("ship", settings)
    shop = Word("shop", settings)
    papa = Word("papa", settings)
    pepe = Word("pepe", settings)
    inventory = settings.grapheme_inventory
    i = inventory.get_grapheme("i")
    o = inventory.get_grapheme("o")
    a = inventory.get_grapheme("a")
    e = inventory.get_grapheme("e")

    assert ship.is_minimal_pair(shop, False) == 1
    assert ship.is_minimal_pair_for_graphemes(shop, False, i, o)
    assert not papa.is_minimal_pair_for_graphemes(pepe, False, a, e)
    assert papa.is_minimal_pair_harmony(pepe, False, a, e)


def test_grapheme_taught_order_preserves_order_and_supports_removal():
    settings = create_english_settings()
    graphemes_taught = GraphemeTaughtOrder(settings)

    graphemes_taught.add_grapheme("a")
    graphemes_taught.add_grapheme("sh")
    graphemes_taught.add_grapheme("ee")
    graphemes_taught.del_grapheme(1)

    assert graphemes_taught.count() == 2
    assert graphemes_taught.get_grapheme(0) == "a"
    assert graphemes_taught.get_grapheme(1) == "ee"
    assert graphemes_taught.is_taught_grapheme("ee")
    assert not graphemes_taught.is_taught_grapheme("sh")


def test_sight_words_collection_supports_add_find_and_remove_semantics():
    settings = create_english_settings()
    sight_words = SightWords(settings)

    sight_words.add_word("the")
    sight_words.add_word("said")
    sight_words.del_word(0)

    assert sight_words.count() == 1
    assert sight_words.get_word(0) == "said"
    assert sight_words.is_sight_word("said")
    assert not sight_words.is_sight_word("the")


def test_word_list_preserves_duplicate_display_words_with_distinct_keys():
    settings = create_english_settings()
    word_list = word_list_from_words(settings, "ship", "ship", "shop")

    assert word_list.word_count() == 3
    assert word_list.is_word_in_list("ship")
    assert word_list.get_word_key(0) != word_list.get_word_key(1)
