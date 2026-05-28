from primerpro import BuildableWordSearchWL

from .fixtures import create_bantu_like_settings, taught, word_list_from_words


def test_buildable_word_list_search_includes_only_words_whose_graphemes_are_taught():
    settings = create_bantu_like_settings()
    word_list = word_list_from_words(settings, "mbata", "kaana", "ngoma", "tap")
    search = BuildableWordSearchWL(1, settings)
    search.graphemes = taught("mb", "a", "t", "k", "aa", "n")
    search.highlights = []

    search.execute_buildable_search(word_list)

    assert search.search_count == 2
    assert "mbata" in search.search_results
    assert "kaana" in search.search_results
    assert "ngoma" not in search.search_results
    assert "tap" not in search.search_results


def test_buildable_word_list_search_handles_positional_taught_graphemes():
    settings = create_bantu_like_settings()
    word_list = word_list_from_words(settings, "ta", "at", "ata", "ta.na")
    search = BuildableWordSearchWL(1, settings)
    search.graphemes = taught("t_", "_t", "n_", "a")
    search.highlights = []

    search.execute_buildable_search(word_list)

    assert search.search_count == 4
    assert "ta" in search.search_results
    assert "at" in search.search_results
    assert "ata" in search.search_results
    assert "tana" in search.search_results


def test_buildable_word_list_search_reports_no_results_for_an_empty_decodable_set():
    settings = create_bantu_like_settings()
    word_list = word_list_from_words(settings, "mbata", "ngoma")
    search = BuildableWordSearchWL(1, settings)
    search.graphemes = taught("z")
    search.highlights = []

    search.execute_buildable_search(word_list)

    assert search.search_count == 0
    assert "No Results" in search.search_results
