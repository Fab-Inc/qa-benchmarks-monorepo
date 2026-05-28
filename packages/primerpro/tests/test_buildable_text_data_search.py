from primerpro import BuildableWordSearchTD

from .fixtures import (
    assert_contains_highlighted,
    create_english_settings,
    taught,
    text_data_from_paragraphs,
)


def test_buildable_text_data_search_lists_buildable_words_from_story_text():
    settings = create_english_settings()
    text_data = text_data_from_paragraphs(settings, "sat ship queen. mat chain.")
    search = BuildableWordSearchTD(1, settings)
    search.graphemes = taught("s", "a", "t", "m", "sh", "i", "p")
    search.highlights = []

    search.execute_buildable_word_search(text_data)

    assert search.search_count == 3
    assert "sat" in search.search_results
    assert "ship" in search.search_results
    assert "mat" in search.search_results
    assert "queen" not in search.search_results
    assert "chain" not in search.search_results


def test_buildable_text_data_search_can_remove_duplicate_display_words():
    settings = create_english_settings()
    text_data = text_data_from_paragraphs(settings, "sat ship sat. ship mat.")
    search = BuildableWordSearchTD(1, settings)
    search.graphemes = taught("s", "a", "t", "m", "sh", "i", "p")
    search.highlights = []
    search.no_duplicates = True

    search.execute_buildable_word_search(text_data)

    assert search.search_count == 3
    assert "sat" in search.search_results
    assert "ship" in search.search_results
    assert "mat" in search.search_results


def test_buildable_text_data_paragraph_format_highlights_buildable_words_in_context():
    settings = create_english_settings()
    text_data = text_data_from_paragraphs(settings, "sat queen ship.")
    search = BuildableWordSearchTD(1, settings)
    search.graphemes = taught("s", "a", "t", "sh", "i", "p")
    search.highlights = []
    search.para_format = True

    search.execute_buildable_word_search(text_data)

    assert search.search_count == 2
    assert_contains_highlighted("sat", search.search_results)
    assert "queen" in search.search_results
    assert_contains_highlighted("ship", search.search_results)
