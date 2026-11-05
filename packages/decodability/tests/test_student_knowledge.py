"""Tests for the shared student-knowledge rendering contract."""

import pytest
from decodability import DECODABILITY_REGISTRY
from decodability.student_knowledge import (
    DEFAULT_KNOWLEDGE_FORMAT,
    KNOWLEDGE_FORMATS,
    format_items,
    format_pairs,
)

LANGUAGES = sorted(DECODABILITY_REGISTRY)


def test_knowledge_formats_and_default() -> None:
    assert KNOWLEDGE_FORMATS == ("quoted", "plain")
    assert DEFAULT_KNOWLEDGE_FORMAT == "quoted"


@pytest.mark.parametrize("language", LANGUAGES)
def test_every_registered_model_renders_every_format(language: str) -> None:
    model = DECODABILITY_REGISTRY[language]["student_knowledge_model"]
    empty = model()
    for fmt in KNOWLEDGE_FORMATS:
        assert empty.render(fmt).startswith("- Known ")
    assert str(empty) == empty.render(DEFAULT_KNOWLEDGE_FORMAT)


@pytest.mark.parametrize("language", LANGUAGES)
def test_every_registered_model_rejects_unknown_formats(language: str) -> None:
    model = DECODABILITY_REGISTRY[language]["student_knowledge_model"]
    with pytest.raises(ValueError, match="Unknown knowledge format 'v2'"):
        model().render("v2")  # type: ignore[arg-type]


def test_format_items_sorts_and_quotes() -> None:
    assert format_items({"b", "a"}, "quoted") == '"a", "b"'
    assert format_items({"b", "a"}, "plain") == "a, b"
    assert format_items([], "quoted") == "None"
    assert format_items([], "plain") == "None"


def test_format_pairs_keeps_order() -> None:
    pairs = [("c", "k"), ("a", "æ")]
    assert format_pairs(pairs, "quoted") == '"c" → "k"; "a" → "æ"'
    assert format_pairs(pairs, "plain") == "c -> /k/; a -> /æ/"
    assert format_pairs([], "quoted") == "None"
    assert format_pairs([], "plain") == "None"


@pytest.mark.parametrize("helper", [format_items, format_pairs])
def test_helpers_reject_unknown_formats(helper) -> None:
    with pytest.raises(ValueError, match="Unknown knowledge format"):
        helper([], "v2")
