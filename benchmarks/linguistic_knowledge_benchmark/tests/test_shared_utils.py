"""Tests for shared_utils."""

from linguistic_knowledge_benchmark import shared_utils


def test_english_language_name_maps_kiswahili_to_swahili():
    assert shared_utils.english_language_name("Kiswahili") == "Swahili"


def test_english_language_name_passes_through_unmapped_languages():
    assert shared_utils.english_language_name("Luganda") == "Luganda"
