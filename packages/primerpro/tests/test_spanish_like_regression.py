from primerpro import Word

from .fixtures import (
    create_spanish_like_settings,
    grapheme_symbols,
    taught,
    text_data_from_paragraphs,
)


def test_spanish_style_inventory_decomposes_ch_ll_rr_qu_and_gu_as_graphemes():
    settings = create_spanish_like_settings()

    assert_word(settings, "queso", ["qu", "e", "s", "o"])
    assert_word(settings, "guitarra", ["gu", "i", "t", "a", "rr", "a"])
    assert_word(settings, "llama", ["ll", "a", "m", "a"])
    assert_word(settings, "chico", ["ch", "i", "c", "o"])
    assert_word(settings, "perro", ["p", "e", "rr", "o"])


def test_spanish_style_inventory_prefers_qu_and_gu_over_component_letters():
    settings = create_spanish_like_settings()

    assert_word(settings, "quema", ["qu", "e", "m", "a"])
    assert_word(settings, "guia", ["gu", "i", "a"])
    assert_word(settings, "carro", ["c", "a", "rr", "o"])


def test_spanish_style_uppercase_multigraphs_are_exact():
    settings = create_spanish_like_settings()

    assert grapheme_symbols(Word("QUEso", settings)) == ["qu", "e", "s", "o"]
    assert grapheme_symbols(Word("Queso", settings)) == ["q", "u", "e", "s", "o"]


def test_spanish_style_decodability_requires_taught_multigraph_graphemes():
    settings = create_spanish_like_settings()
    queso = Word("queso", settings)
    llama = Word("llama", settings)
    perro = Word("perro", settings)

    assert not queso.is_buildable_word(taught("q", "u", "e", "s", "o"))
    assert not llama.is_buildable_word(taught("l", "a", "m"))
    assert not perro.is_buildable_word(taught("p", "e", "r", "o"))
    assert queso.is_buildable_word(taught("qu", "e", "s", "o"))
    assert llama.is_buildable_word(taught("ll", "a", "m"))
    assert perro.is_buildable_word(taught("p", "e", "rr", "o"))


def test_spanish_style_staged_decodability_handles_qu_ch_ll_rr_and_gu():
    settings = create_spanish_like_settings()
    simple = taught("c", "a", "s", "o", "p", "e", "r", "t", "g")
    qu_ch = taught("c", "a", "s", "o", "p", "e", "r", "t", "g", "qu", "ch", "i")
    all_multigraphs = taught(
        "c",
        "a",
        "s",
        "o",
        "p",
        "e",
        "r",
        "t",
        "g",
        "qu",
        "ch",
        "i",
        "ll",
        "rr",
        "gu",
        "m",
    )

    assert Word("casa", settings).is_buildable_word(simple)
    assert not Word("queso", settings).is_buildable_word(simple)
    assert Word("queso", settings).is_buildable_word(qu_ch)
    assert Word("chico", settings).is_buildable_word(qu_ch)
    assert not Word("llama", settings).is_buildable_word(qu_ch)
    assert Word("llama", settings).is_buildable_word(all_multigraphs)
    assert Word("guitarra", settings).is_buildable_word(all_multigraphs)


def test_spanish_style_text_data_counts_story_structure_and_unique_words():
    settings = create_spanish_like_settings()
    text_data = text_data_from_paragraphs(
        settings,
        "Queso chico. Llama, llama!",
        "Guitarra perro?",
    )
    unique = text_data.build_word_list_with_no_duplicates()

    assert text_data.paragraph_count() == 2
    assert text_data.sentence_count() == 3
    assert text_data.word_count() == 6
    assert unique.word_count() == 5


def test_spanish_style_frequency_counts_use_decomposed_multigraph_graphemes():
    settings = create_spanish_like_settings()
    text_data = text_data_from_paragraphs(
        settings,
        "queso guitarra llama chico perro carro guia.",
    )
    counted = text_data.update_grapheme_counts(
        settings.grapheme_inventory, False, False
    )

    assert consonant_count(counted, "qu") == 1
    assert consonant_count(counted, "gu") == 2
    assert consonant_count(counted, "ll") == 1
    assert consonant_count(counted, "rr") == 3
    assert consonant_count(counted, "ch") == 1
    assert consonant_count(counted, "q") == 0


def assert_word(settings, text, expected):
    assert grapheme_symbols(Word(text, settings)) == expected


def consonant_count(inventory, symbol):
    return inventory.get_consonant(
        inventory.find_consonant_index(symbol)
    ).count_in_text_data
