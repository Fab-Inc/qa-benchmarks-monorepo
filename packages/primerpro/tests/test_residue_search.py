from primerpro import ResidueSearch

from .fixtures import (
    assert_contains_highlighted,
    create_bantu_like_settings,
    taught,
    text_data_from_paragraphs,
)


def test_residue_search_counts_words_that_are_not_buildable_from_taught_graphemes():
    settings = create_bantu_like_settings()
    text_data = text_data_from_paragraphs(settings, "mbata ngoma kaana.")
    search = ResidueSearch(1, settings)
    search.graphemes = taught("mb", "a", "t", "k", "aa", "n")

    search.execute_residue_search(text_data)

    assert search.search_count == 1
    assert_contains_highlighted("ng", search.search_results)


def test_residue_search_can_ignore_configured_sight_words():
    settings = create_bantu_like_settings()
    settings.sight_words.add_word("ngoma")
    text_data = text_data_from_paragraphs(settings, "mbata ngoma kaana.")
    search = ResidueSearch(1, settings)
    search.graphemes = taught("mb", "a", "t", "k", "aa", "n")
    search.ignore_sight_words = True

    search.execute_residue_search(text_data)

    assert search.search_count == 0


def test_residue_search_can_append_reading_level_story_metrics():
    settings = create_bantu_like_settings()
    text_data = text_data_from_paragraphs(settings, "mbata ngoma. kaana ta.")
    search = ResidueSearch(1, settings)
    search.graphemes = taught("mb", "a", "t", "k", "aa", "n")
    search.reading_level_info = True

    search.execute_residue_search(text_data)

    assert "Number of words in story" in search.search_results
    assert "Number of sentences in story" in search.search_results
