from primerpro import MinPairsSearch, Settings, TonePairsSearch

from .fixtures import (
    create_english_settings,
    create_tone_settings,
    word_list_from_words,
)


def test_minimal_pair_search_returns_without_inventory_or_target_graphemes() -> None:
    no_inventory = Settings()
    search = MinPairsSearch(0, no_inventory)
    word_list = word_list_from_words(create_english_settings(), "cat")

    search.execute_min_pairs_search(word_list)

    assert search.search_results == ""

    settings = create_english_settings()
    search = MinPairsSearch(0, settings)
    search.grapheme1 = "missing"
    search.grapheme2 = "a"
    search.execute_min_pairs_search(word_list_from_words(settings, "cat"))

    assert search.search_results == ""


def test_tone_pair_search_returns_without_inventory_or_target_tone() -> None:
    no_inventory = Settings()
    search = TonePairsSearch(0, no_inventory)
    search.execute_tone_pairs_search(word_list_from_words(create_tone_settings(), "má"))

    assert search.search_results == ""

    settings = create_tone_settings()
    search = TonePairsSearch(0, settings)
    search.grapheme = "missing"
    search.execute_tone_pairs_search(word_list_from_words(settings, "má"))

    assert search.search_results == ""


def test_tone_pair_search_reports_no_results_without_plain_counterpart() -> None:
    settings = create_tone_settings()
    word_list = word_list_from_words(settings, "mi", "má")
    search = TonePairsSearch(0, settings)
    search.grapheme = "á"

    search.execute_tone_pairs_search(word_list)

    assert search.search_count == 0
    assert search.search_results == "***No Results***"
