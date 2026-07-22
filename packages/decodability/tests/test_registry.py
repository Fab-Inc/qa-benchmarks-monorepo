"""Tests for the language registry."""

import pytest
from decodability import DecodabilityObjects, get_decodability_objects


def test_get_decodability_objects_returns_kiswahili_bundle() -> None:
    """get_decodability_objects('sw') returns a populated Kiswahili bundle."""
    objs = get_decodability_objects("sw")
    assert isinstance(objs, DecodabilityObjects)
    assert objs.name == "Kiswahili"
    assert "score_known_graphemes_kiswahili" in objs.scoring_methods
    assert "kiswahili_kenya_tusome_aggregator" in objs.aggregations
    assert objs.extract_words("mama na") == ["mama", "na"]


def test_get_decodability_objects_unknown_raises_with_supported_list() -> None:
    """An unknown language code raises KeyError listing supported languages."""
    with pytest.raises(KeyError, match="sw"):
        get_decodability_objects("xx")
