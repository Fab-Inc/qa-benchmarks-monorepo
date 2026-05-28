from primerpro import MinPairsSearch, ToneChartSearch, TonePairsSearch

from .fixtures import (
    create_english_settings,
    create_tone_settings,
    word_list_from_words,
)


def test_minimal_pair_search_reports_targeted_vowel_contrasts():
    settings = create_english_settings()
    word_list = word_list_from_words(settings, "mat", "mit", "sat")
    search = MinPairsSearch(0, settings)
    search.grapheme1 = "a"
    search.grapheme2 = "i"

    search.execute_min_pairs_search(word_list)

    assert search.search_count == 1
    assert "mat" in search.search_results
    assert "mit" in search.search_results


def test_minimal_pair_search_checks_requested_order_even_when_sort_order_differs():
    settings = create_english_settings()
    word_list = word_list_from_words(settings, "bat", "pat")
    search = MinPairsSearch(0, settings)
    search.grapheme1 = "p"
    search.grapheme2 = "b"

    search.execute_min_pairs_search(word_list)

    assert search.search_count == 1
    assert "pat" in search.search_results
    assert "bat" in search.search_results


def test_tone_chart_search_reports_configured_tones_levels_and_tone_bearing_units():
    settings = create_tone_settings()
    search = ToneChartSearch(0, settings)

    search.execute_tone_chart(settings.grapheme_inventory)

    assert "Tone" in search.search_results
    assert "TBU" in search.search_results
    assert "á\ta\tH" in search.search_results
    assert "à\ta\tL" in search.search_results


def test_tone_pairs_search_reports_tone_contrasts_against_tone_bearing_unit():
    settings = create_tone_settings()
    word_list = word_list_from_words(settings, "ma", "má", "mi")
    search = TonePairsSearch(0, settings)
    search.grapheme = "á"

    search.execute_tone_pairs_search(word_list)

    assert search.search_count == 1
    assert "ma" in search.search_results
    assert "má" in search.search_results
