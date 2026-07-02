from .fixtures import (
    create_bantu_like_settings,
    create_english_settings,
    create_settings,
    create_tone_settings,
    text_data_from_paragraphs,
)


def test_text_data_parses_non_english_story_text_into_paragraphs_sentences_and_words():
    settings = create_bantu_like_settings()
    text_data = text_data_from_paragraphs(
        settings,
        "Mbata ngoma. Kaana ta?",
        "Ngoma mbata!",
    )

    assert text_data.paragraph_count() == 2
    assert text_data.sentence_count() == 3
    assert text_data.word_count() == 6


def test_tokenization_does_not_split_on_punctuation_inside_a_grapheme():
    """
    GIVEN text with a grapheme that contains an apostrophe ("ng'")
    WHEN the text is parsed into words
    THEN the punctuation inside the grapheme does not split the token
    """
    settings = create_settings(["n", "ng", "ng'"], ["a", "i", "o"])
    assert "'" in settings.option_settings.general_punct

    text_data = text_data_from_paragraphs(settings, "ng'ania ngoma.")
    word_list = text_data.build_word_list()

    # Two words, not three: the internal apostrophe stays inside "ng'", while the
    # space between words still separates tokens.
    assert word_list.word_count() == 2
    assert word_list.get_word(0).display_word == "ng'ania"
    assert word_list.get_word(1).display_word == "ngoma"


def test_text_data_average_words_in_sentences_averages_paragraph_averages():
    settings = create_bantu_like_settings()
    text_data = text_data_from_paragraphs(
        settings,
        "b. b b b.",
        "b b b b b b b b b b.",
    )

    # This pins PrimerPro's current behavior, but it may be incorrect: the text
    # average is computed from paragraph averages, not total words / sentences.
    assert text_data.avg_number_of_words_in_sentences() == 6
    assert text_data.word_count() // text_data.sentence_count() == 4


def test_text_data_can_build_a_word_list_preserving_duplicate_tokens():
    settings = create_bantu_like_settings()
    text_data = text_data_from_paragraphs(settings, "mbata ngoma mbata.")
    word_list = text_data.build_word_list()

    assert word_list.word_count() == 3


def test_text_data_can_build_a_word_list_with_duplicate_display_words_removed():
    settings = create_bantu_like_settings()
    text_data = text_data_from_paragraphs(settings, "mbata ngoma mbata.")
    word_list = text_data.build_word_list_with_no_duplicates()

    assert word_list.word_count() == 2


def test_frequency_counting_respects_longest_match_grapheme_decomposition():
    settings = create_bantu_like_settings()
    text_data = text_data_from_paragraphs(settings, "mbata ngoma.")
    counted = text_data.update_grapheme_counts(
        settings.grapheme_inventory, False, False
    )

    assert (
        counted.get_consonant(counted.find_consonant_index("mb")).count_in_text_data
        == 1
    )
    assert (
        counted.get_consonant(counted.find_consonant_index("m")).count_in_text_data == 1
    )
    assert (
        counted.get_consonant(counted.find_consonant_index("ng")).count_in_text_data
        == 1
    )
    assert (
        counted.get_consonant(counted.find_consonant_index("n")).count_in_text_data == 0
    )


def test_frequency_counting_can_ignore_configured_sight_words():
    settings = create_english_settings()
    settings.sight_words.add_word("ship")
    text_data = text_data_from_paragraphs(settings, "ship shop.")

    counted = text_data.update_grapheme_counts(settings.grapheme_inventory, True, False)

    assert (
        counted.get_consonant(counted.find_consonant_index("sh")).count_in_text_data
        == 1
    )
    assert (
        counted.get_consonant(counted.find_consonant_index("p")).count_in_text_data == 1
    )


def test_frequency_counting_can_fold_tones_into_tone_bearing_units():
    settings = create_tone_settings()
    text_data = text_data_from_paragraphs(settings, "ma má.")

    counted = text_data.update_grapheme_counts(settings.grapheme_inventory, False, True)

    assert counted.get_vowel(counted.find_vowel_index("a")).count_in_text_data == 2
    assert counted.get_tone(counted.find_tone_index("á")).count_in_text_data == 0
