"""Validation tests for the registered dataset JSON files.

Guards against corrupted questions that would otherwise fail silently: scoring
(score_multiple_choice) only ever reads expected_answers[0], so a
multiple_choice question with zero or more than one expected answer would
have the extra entries silently ignored.
"""

import pytest
from linguistic_knowledge_benchmark.cmd import DATASETS
from linguistic_knowledge_benchmark.run_bench import load_dataset

VALID_ANSWER_TYPES = {"multiple_choice", "short_form"}


def _all_questions():
    """Yield (language_key, question) for every registered dataset."""
    for language_key, entry in DATASETS.items():
        for question in load_dataset(entry["path"]):
            yield language_key, question


ALL_QUESTIONS = list(_all_questions())


@pytest.mark.parametrize(
    "language_key, question",
    ALL_QUESTIONS,
    ids=[f"{lang}-{q['question_id']}" for lang, q in ALL_QUESTIONS],
)
def test_answer_type_is_recognized(language_key, question):
    assert question["answer_type"] in VALID_ANSWER_TYPES


@pytest.mark.parametrize(
    "language_key, question",
    [(lang, q) for lang, q in ALL_QUESTIONS if q["answer_type"] == "multiple_choice"],
    ids=[
        f"{lang}-{q['question_id']}"
        for lang, q in ALL_QUESTIONS
        if q["answer_type"] == "multiple_choice"
    ],
)
def test_multiple_choice_has_exactly_one_expected_answer(language_key, question):
    assert len(question["expected_answers"]) == 1
    assert question["expected_answers"][0] in question.get("options", {})


@pytest.mark.parametrize(
    "language_key, question",
    [(lang, q) for lang, q in ALL_QUESTIONS if q["answer_type"] == "short_form"],
    ids=[
        f"{lang}-{q['question_id']}"
        for lang, q in ALL_QUESTIONS
        if q["answer_type"] == "short_form"
    ],
)
def test_short_form_has_at_least_one_expected_answer(language_key, question):
    assert len(question["expected_answers"]) >= 1
