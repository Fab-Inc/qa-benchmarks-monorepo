from primerpro import BuildableWordSearchWL, ResidueSearch

from .fixtures import (
    assert_contains_highlighted,
    create_spanish_like_settings,
    taught,
    text_data_from_paragraphs,
    word_list_from_words,
)


def test_spanish_style_buildable_search_models_a_qu_and_ch_teaching_stage():
    settings = create_spanish_like_settings()
    word_list = word_list_from_words(
        settings,
        "casa",
        "queso",
        "chico",
        "llama",
        "perro",
        "guitarra",
        "carro",
    )
    search = BuildableWordSearchWL(1, settings)
    search.graphemes = taught("c", "a", "s", "o", "qu", "e", "ch", "i")
    search.highlights = []

    search.execute_buildable_search(word_list)

    assert search.search_count == 3
    assert "casa" in search.search_results
    assert "queso" in search.search_results
    assert "chico" in search.search_results
    assert "llama" not in search.search_results
    assert "perro" not in search.search_results
    assert "guitarra" not in search.search_results
    assert "carro" not in search.search_results


def test_spanish_style_buildable_search_models_a_full_multigraph_teaching_stage():
    settings = create_spanish_like_settings()
    word_list = word_list_from_words(
        settings,
        "casa",
        "queso",
        "chico",
        "llama",
        "perro",
        "guitarra",
        "carro",
    )
    search = BuildableWordSearchWL(1, settings)
    search.graphemes = taught(
        "a",
        "c",
        "ch",
        "e",
        "gu",
        "i",
        "ll",
        "m",
        "o",
        "p",
        "qu",
        "rr",
        "s",
        "t",
    )
    search.highlights = []

    search.execute_buildable_search(word_list)

    assert search.search_count == 7
    for word in ["casa", "queso", "chico", "llama", "perro", "guitarra", "carro"]:
        assert word in search.search_results


def test_spanish_style_residue_search_reports_words_blocked_by_ll_rr_and_gu():
    settings = create_spanish_like_settings()
    text_data = text_data_from_paragraphs(
        settings,
        "casa queso guitarra llama chico perro.",
    )
    search = ResidueSearch(1, settings)
    search.graphemes = taught("c", "a", "s", "o", "qu", "e", "ch", "i", "p")

    search.execute_residue_search(text_data)

    assert search.search_count == 3
    assert_contains_highlighted("gu", search.search_results)
    assert_contains_highlighted("ll", search.search_results)
    assert_contains_highlighted("rr", search.search_results)
    assert "casa" not in search.search_results
    assert "queso" not in search.search_results
    assert "chico" not in search.search_results


def test_spanish_style_residue_search_can_ignore_an_untaught_sight_word():
    settings = create_spanish_like_settings()
    settings.sight_words.add_word("llama")
    text_data = text_data_from_paragraphs(settings, "llama queso.")
    search = ResidueSearch(1, settings)
    search.graphemes = taught("qu", "e", "s", "o")
    search.ignore_sight_words = True

    search.execute_residue_search(text_data)

    assert search.search_count == 0
