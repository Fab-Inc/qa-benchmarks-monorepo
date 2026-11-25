import json

import pytest
from reading_comprehension_question_generation.parsing import parse_qa_pairs


def _pairs(n: int) -> list[dict]:
    return [{"question": f"Q{i}?", "answer": f"A{i}"} for i in range(1, n + 1)]


def test_a_clean_array_of_three_pairs_is_ok() -> None:
    pairs, status = parse_qa_pairs(json.dumps(_pairs(3)))

    assert status == "ok"
    assert [(p.question, p.answer) for p in pairs] == [
        ("Q1?", "A1"),
        ("Q2?", "A2"),
        ("Q3?", "A3"),
    ]


@pytest.mark.parametrize(
    "trailer",
    ["\n```\nHope this helps.", "\n```\nSee note [1]."],
    ids=["trailing_text", "trailing_footnote_array"],
)
def test_preamble_code_fence_and_trailing_text_are_skipped(trailer: str) -> None:
    # A trailing "[1]" is itself a valid JSON array but holds no pair, so it must not
    # hide the real pairs (the design's "last array that holds a valid pair").
    response = (
        "Here are the pairs [as requested]:\n```json\n"
        + json.dumps(_pairs(3))
        + trailer
    )

    pairs, status = parse_qa_pairs(response)

    assert status == "ok"
    assert len(pairs) == 3


def test_invalid_elements_are_dropped_and_a_short_count_is_flagged() -> None:
    array = [
        {"question": " Q1? ", "answer": " A1 "},
        {"question": "Q2?", "answer": "A2", "extra": "ignored"},
        "stray string",
        {"question": "Q3?"},
        {"question": "  ", "answer": "A4"},
    ]

    pairs, status = parse_qa_pairs(json.dumps(array))

    assert status == "count_mismatch"
    assert [(p.question, p.answer) for p in pairs] == [("Q1?", "A1"), ("Q2?", "A2")]


def test_an_array_with_no_valid_pair_is_no_pairs() -> None:
    pairs, status = parse_qa_pairs('[{"question": "Q?"}, 5]')

    assert (pairs, status) == ([], "no_pairs")


def test_prose_and_reasoning_only_output_is_parse_failed() -> None:
    response = (
        '<think>draft [{"question": "d?", "answer": "x"}]</think>\nI could not answer.'
    )

    assert parse_qa_pairs(response) == ([], "parse_failed")


@pytest.mark.parametrize("response", ["", None], ids=["empty", "none"])
def test_an_empty_response_is_parse_failed(response) -> None:
    assert parse_qa_pairs(response) == ([], "parse_failed")
