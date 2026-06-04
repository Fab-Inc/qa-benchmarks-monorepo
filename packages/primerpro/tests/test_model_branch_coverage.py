from primerpro import (
    Consonant,
    Grapheme,
    GraphemeInventory,
    GraphemeTaughtOrder,
    Paragraph,
    Settings,
    SightWords,
    Syllable,
    TextData,
    Tone,
    Vowel,
    Word,
    WordList,
)
from primerpro.model import Sentence, Syllograph

from .fixtures import (
    create_english_settings,
    create_settings,
    create_tone_settings,
    text_data_from_paragraphs,
)


def test_inventory_accessors_and_marker_fallbacks_cover_grapheme_types() -> None:
    settings = Settings()
    inventory = GraphemeInventory(settings)
    settings.grapheme_inventory = inventory

    consonant = Consonant("b")
    vowel = Vowel("a")
    tone = Tone("á")
    syllograph = Syllograph("ka")
    inventory.add_consonant(consonant)
    inventory.add_vowel(vowel)
    inventory.add_tone(tone)
    inventory.add_syllograph(syllograph)

    assert settings.localization_table.get_message("unused") == ""
    assert Grapheme("x").get_marker() == "?"
    assert consonant.get_marker() == "C"
    assert vowel.get_marker() == "V"
    assert tone.get_marker() == ""
    assert syllograph.get_marker() == "S"

    tone.tone_bearing_unit = consonant
    assert tone.get_marker() == "C"
    tone.tone_bearing_unit = vowel
    assert tone.get_marker() == "V"
    tone.tone_bearing_unit = syllograph
    assert tone.get_marker() == "S"

    assert inventory.consonant_count() == 1
    assert inventory.vowel_count() == 1
    assert inventory.tone_count() == 1
    assert inventory.syllograph_count() == 1
    assert inventory.get_consonant(0) is consonant
    assert inventory.get_vowel(0) is vowel
    assert inventory.get_tone(0) is tone
    assert inventory.get_syllograph(0) is syllograph
    assert inventory.find_syllograph_index("ka") == 0
    assert Syllable("ka", settings).cv_pattern == "S"


def test_word_and_syllable_bounds_and_minimal_pair_length_mismatches() -> None:
    settings = create_english_settings()
    short = Word("at", settings)
    longer = Word("cat", settings)
    vowel = settings.grapheme_inventory.get_vowel(
        settings.grapheme_inventory.find_vowel_index("a")
    )
    consonant = settings.grapheme_inventory.get_consonant(
        settings.grapheme_inventory.find_consonant_index("c")
    )

    assert Syllable("at", settings).get_grapheme(99) is None
    assert longer.get_grapheme(99) is None
    assert longer.get_syllable(99) is None
    assert longer.is_minimal_pair(short, False) == -1
    assert not longer.is_minimal_pair_for_graphemes(
        short,
        False,
        consonant,
        vowel,
    )
    assert not longer.is_minimal_pair_harmony(short, False, consonant, vowel)


def test_paragraph_sentence_and_text_data_empty_paths() -> None:
    settings = create_english_settings()
    paragraph = Paragraph("", settings)
    sentence = Sentence("cat", settings)
    text_data = TextData(settings)

    assert paragraph.get_sentence(0) is None
    assert paragraph.word_count() == 0
    assert paragraph.syllable_count() == 0
    assert paragraph.max_number_of_words_in_sentences() == 0
    assert paragraph.max_number_of_syllables_in_words() == 0
    assert paragraph.avg_number_of_word_in_sentences() == 0
    assert paragraph.as_string() == ""

    paragraph.add_sentence(sentence)
    assert paragraph.get_sentence(0) is sentence
    assert sentence.get_word(99) is None
    assert sentence.as_string() == "cat"

    assert text_data.get_paragraph(0) is None
    assert text_data.paragraph_count() == 0
    assert text_data.sentence_count() == 0
    assert text_data.word_count() == 0
    assert text_data.syllable_count() == 0
    assert text_data.max_number_of_words_in_sentences() == 0
    assert text_data.max_number_of_syllables_in_words() == 0
    assert text_data.avg_number_of_words_in_sentences() == 0
    assert text_data.avg_number_of_syllables_in_words() == 0

    text_data.add_paragraph(paragraph)
    assert text_data.get_paragraph(0) is paragraph


def test_word_list_taught_order_and_sight_word_mutation_paths() -> None:
    settings = create_english_settings()
    word_list = WordList(settings)
    first = Word("cat", settings)
    second = Word("cat", settings)
    third = Word("cat", settings)

    word_list.add_word(first)
    word_list.add_word(second)
    word_list.add_word(third)

    assert word_list.word_count() == 3
    assert len({word_list.get_word_key(index) for index in range(3)}) == 3
    assert word_list.is_word_in_list("cat")
    assert word_list.is_word_in_list(Word("cat", settings))
    assert word_list.get_display_line_for_word(0, "c").startswith("<HL>cat</HL>")

    word_list.del_word(0)
    assert word_list.word_count() == 2

    taught_order = GraphemeTaughtOrder(settings)
    taught_order.add_grapheme("c")
    taught_order.add_grapheme("zz")
    assert taught_order.count() == 2
    assert taught_order.get_grapheme(0) == "c"
    assert taught_order.is_taught_grapheme("c")
    assert taught_order.get_missing_graphemes() == "zz"
    taught_order.del_grapheme(1)
    assert taught_order.get_missing_graphemes() == ""

    no_inventory_order = GraphemeTaughtOrder(Settings())
    no_inventory_order.add_grapheme("_x")
    assert no_inventory_order.get_missing_graphemes() == "_x"

    sight_words = SightWords(settings)
    sight_words.add_word("cat")
    assert sight_words.count() == 1
    assert sight_words.get_word(0) == "cat"
    assert sight_words.is_sight_word("cat")
    sight_words.del_word(0)
    assert sight_words.count() == 0


def test_cleaning_fallback_decomposition_and_counting_edges() -> None:
    settings = create_settings(["b"], ["a"])
    settings.option_settings.import_ignore_chars = "#"

    assert Word('[b]#"a!', settings).display_word == "ba"

    no_inventory = Settings()
    no_inventory.option_settings.max_size_grapheme = 2
    plain = Word("ab", no_inventory)
    assert plain.display_word == "ab"
    assert plain.cv_pattern == "??"

    tone_settings = create_tone_settings()
    tone_data = text_data_from_paragraphs(tone_settings, "má.")
    tone_counts = tone_data.update_grapheme_counts(
        tone_settings.grapheme_inventory,
        False,
        False,
    )
    high_tone = tone_counts.get_tone(tone_counts.find_tone_index("á"))
    assert high_tone.count_in_text_data == 1

    syllograph_settings = Settings()
    inventory = GraphemeInventory(syllograph_settings)
    syllograph = Syllograph("ka")
    inventory.add_syllograph(syllograph)
    syllograph_settings.grapheme_inventory = inventory
    syllograph_data = TextData(syllograph_settings)
    syllograph_data.add_paragraph(Paragraph("ka.", syllograph_settings))

    counted = syllograph_data.update_grapheme_counts(inventory, False, False)

    assert counted.get_syllograph(0).count_in_text_data == 1
