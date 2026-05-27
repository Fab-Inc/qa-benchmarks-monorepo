from primerpro import Syllable, Word

from .fixtures import (
    create_english_settings,
    grapheme_symbols,
    taught,
    text_data_from_paragraphs,
)


def test_english_inventory_decomposes_common_consonant_digraphs_and_vowel_teams():
    settings = create_english_settings()

    assert_word(settings, "ship", ["sh", "i", "p"])
    assert_word(settings, "thin", ["th", "i", "n"])
    assert_word(settings, "chain", ["ch", "ai", "n"])
    assert_word(settings, "queen", ["qu", "ee", "n"])
    assert_word(settings, "night", ["n", "igh", "t"])
    assert_word(settings, "beach", ["b", "ea", "ch"])
    assert_word(settings, "book", ["b", "oo", "k"])
    assert_word(settings, "sing", ["s", "i", "ng"])


def test_english_inventory_prefers_longest_graphemes():
    settings = create_english_settings()

    assert_word(settings, "check", ["ch", "e", "ck"])
    assert_word(settings, "three", ["th", "r", "ee"])
    assert_word(settings, "whale", ["wh", "a", "l", "e"])


def test_english_uppercase_multigraphs_are_exact():
    settings = create_english_settings()

    assert grapheme_symbols(Word("SHip", settings)) == ["sh", "i", "p"]
    assert grapheme_symbols(Word("Ship", settings)) == ["s", "h", "i", "p"]


def test_english_punctuation_and_apostrophes_are_removed_before_decomposition():
    settings = create_english_settings()
    word = Word('"Sam\'s!"', settings)

    assert word.display_word == "sams"
    assert grapheme_symbols(word) == ["s", "a", "m", "s"]


def test_english_decodability_requires_taught_multigraphs_rather_than_components():
    settings = create_english_settings()
    ship = Word("ship", settings)
    queen = Word("queen", settings)

    assert not ship.is_buildable_word(taught("s", "h", "i", "p"))
    assert not queen.is_buildable_word(taught("q", "u", "ee", "n"))
    assert ship.is_buildable_word(taught("sh", "i", "p"))
    assert queen.is_buildable_word(taught("qu", "ee", "n"))


def test_english_staged_decodability_handles_vowel_teams_and_consonant_digraphs():
    settings = create_english_settings()
    short_vowels = taught("s", "a", "t", "m", "p", "i", "n")
    digraphs = taught("s", "a", "t", "m", "p", "i", "n", "sh", "ch", "th", "ck")
    vowel_teams = taught(
        "s",
        "a",
        "t",
        "m",
        "p",
        "i",
        "n",
        "sh",
        "ch",
        "th",
        "ck",
        "ee",
        "ai",
        "igh",
    )

    assert Word("sat", settings).is_buildable_word(short_vowels)
    assert not Word("ship", settings).is_buildable_word(short_vowels)
    assert Word("ship", settings).is_buildable_word(digraphs)
    assert not Word("chain", settings).is_buildable_word(digraphs)
    assert Word("chain", settings).is_buildable_word(vowel_teams)
    assert Word("night", settings).is_buildable_word(vowel_teams)


def test_english_final_position_marker_can_model_final_ck():
    settings = create_english_settings()
    back = Word("back", settings)
    initial = Syllable("ckab", settings)
    graphemes = taught("b", "a", "_ck")

    assert back.is_buildable_word(graphemes)
    assert not initial.is_buildable(graphemes)


def test_english_text_data_counts_paragraphs_sentences_words_and_unique_display_words():
    settings = create_english_settings()
    text_data = text_data_from_paragraphs(
        settings,
        "Ship, ship! Queen sings.",
        "Night check?",
    )
    unique = text_data.build_word_list_with_no_duplicates()

    assert text_data.paragraph_count() == 2
    assert text_data.sentence_count() == 3
    assert text_data.word_count() == 6
    assert unique.word_count() == 5


def test_english_frequency_counts_use_decomposed_multigraph_graphemes():
    settings = create_english_settings()
    text_data = text_data_from_paragraphs(
        settings,
        "ship shop chain queen night beach book moon sing check.",
    )
    counted = text_data.update_grapheme_counts(
        settings.grapheme_inventory, False, False
    )

    assert consonant_count(counted, "sh") == 2
    assert consonant_count(counted, "ch") == 3
    assert consonant_count(counted, "qu") == 1
    assert consonant_count(counted, "ng") == 1
    assert vowel_count(counted, "igh") == 1
    assert vowel_count(counted, "oo") == 2
    assert vowel_count(counted, "ee") == 1
    assert vowel_count(counted, "ea") == 1


def assert_word(settings, text, expected):
    assert grapheme_symbols(Word(text, settings)) == expected


def consonant_count(inventory, symbol):
    return inventory.get_consonant(
        inventory.find_consonant_index(symbol)
    ).count_in_text_data


def vowel_count(inventory, symbol):
    return inventory.get_vowel(inventory.find_vowel_index(symbol)).count_in_text_data
