from primerpro import BuildableWordSearchWL, ResidueSearch

from .fixtures import (
    assert_contains_highlighted,
    create_english_settings,
    taught,
    text_data_from_paragraphs,
    word_list_from_words,
)


def test_english_buildable_search_models_an_early_short_vowel_teaching_stage():
    settings = create_english_settings()
    word_list = word_list_from_words(
        settings,
        "sat",
        "mat",
        "ship",
        "thin",
        "chain",
        "queen",
        "night",
        "sing",
    )
    search = BuildableWordSearchWL(1, settings)
    search.graphemes = taught("s", "a", "t", "m", "i", "n")
    search.highlights = []

    search.execute_buildable_search(word_list)

    assert search.search_count == 2
    assert "sat" in search.search_results
    assert "mat" in search.search_results
    assert "ship" not in search.search_results
    assert "thin" not in search.search_results
    assert "chain" not in search.search_results
    assert "queen" not in search.search_results
    assert "night" not in search.search_results
    assert "sing" not in search.search_results


def test_english_buildable_search_models_later_digraph_and_vowel_team_stage():
    settings = create_english_settings()
    word_list = word_list_from_words(
        settings,
        "sat",
        "ship",
        "thin",
        "chain",
        "queen",
        "night",
        "beach",
        "check",
        "book",
        "sing",
        "whale",
    )
    search = BuildableWordSearchWL(1, settings)
    search.graphemes = taught(
        "a",
        "ai",
        "b",
        "c",
        "ch",
        "ck",
        "e",
        "ea",
        "ee",
        "h",
        "i",
        "igh",
        "k",
        "l",
        "n",
        "ng",
        "oo",
        "p",
        "qu",
        "r",
        "s",
        "sh",
        "t",
        "th",
        "wh",
    )
    search.highlights = []

    search.execute_buildable_search(word_list)

    assert search.search_count == 11
    for word in [
        "sat",
        "ship",
        "thin",
        "chain",
        "queen",
        "night",
        "beach",
        "check",
        "book",
        "sing",
        "whale",
    ]:
        assert word in search.search_results


def test_english_residue_search_reports_words_blocked_by_untaught_multigraphs():
    settings = create_english_settings()
    text_data = text_data_from_paragraphs(settings, "sat ship queen night beach.")
    search = ResidueSearch(1, settings)
    search.graphemes = taught("s", "a", "t", "sh", "i", "p", "n", "b")

    search.execute_residue_search(text_data)

    assert search.search_count == 3
    assert_contains_highlighted("qu", search.search_results)
    assert_contains_highlighted("igh", search.search_results)
    assert_contains_highlighted("ea", search.search_results)
    assert "sat" not in search.search_results
    assert "ship" not in search.search_results


def test_english_residue_search_can_ignore_an_untaught_sight_word():
    settings = create_english_settings()
    settings.sight_words.add_word("the")
    text_data = text_data_from_paragraphs(settings, "the ship.")
    search = ResidueSearch(1, settings)
    search.graphemes = taught("sh", "i", "p")
    search.ignore_sight_words = True

    search.execute_residue_search(text_data)

    assert search.search_count == 0
