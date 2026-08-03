"""Validation tests for the registered dataset JSON files.

Guards against corrupted questions that would otherwise fail silently: scoring
(score_multiple_choice / score_true_false_regex) only ever reads
expected_answers[0], so a multiple_choice or true_false question with zero or
more than one expected answer would have the extra entries silently ignored.
"""

import pytest
from linguistic_knowledge_benchmark.cmd import DATASETS
from linguistic_knowledge_benchmark.run_bench import load_dataset

VALID_ANSWER_TYPES = {"multiple_choice", "short_form", "true_false"}


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


def _assert_true_false_valid(question):
    assert len(question["expected_answers"]) == 1
    assert question["expected_answers"][0] in ("True", "False")


@pytest.mark.parametrize(
    "language_key, question",
    [(lang, q) for lang, q in ALL_QUESTIONS if q["answer_type"] == "true_false"],
    ids=[
        f"{lang}-{q['question_id']}"
        for lang, q in ALL_QUESTIONS
        if q["answer_type"] == "true_false"
    ],
)
def test_true_false_has_exactly_one_expected_answer_of_true_or_false(
    language_key, question
):
    _assert_true_false_valid(question)


def test_true_false_invariant_holds_for_a_wellformed_question():
    """No active dataset currently has true_false questions (see conftest's
    sample_questions), so the parametrized check above never runs its body.
    Exercise the same invariant directly against a synthetic true_false
    question so the check itself stays covered and verified.
    """
    from conftest import sample_questions

    question = next(q for q in sample_questions() if q["answer_type"] == "true_false")
    _assert_true_false_valid(question)


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
