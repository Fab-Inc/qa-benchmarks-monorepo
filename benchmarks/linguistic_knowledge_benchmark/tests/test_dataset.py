"""Validation tests for the registered dataset JSON files.

Guards against corrupted questions that would otherwise fail silently: scoring
(score_multiple_choice) only ever reads expected_answers[0], so a
multiple_choice question with zero or more than one expected answer would
have the extra entries silently ignored.
"""

from pathlib import Path

import pytest
from linguistic_knowledge_benchmark.cmd import DATASETS
from linguistic_knowledge_benchmark.run_bench import load_dataset
from linguistic_knowledge_benchmark.shared_utils import DATA_DIR

VALID_ANSWER_TYPES = {"multiple_choice", "short_form"}


def _dataset_path_for(language_key: str, entry: dict) -> Path | None:
    """The full item set if present locally, else the tracked 20-item public subset
    (data/hf_subset_<language>_20.json). The full sets are gitignored -- kept private
    ahead of open-sourcing the repo -- so they're absent on every fresh clone/CI run,
    not just occasionally; falling back to the subset keeps this test meaningful
    there instead of validating nothing. None if neither file exists."""
    full_path = Path(entry["path"])
    if full_path.exists():
        return full_path
    subset_path = DATA_DIR / f"hf_subset_{language_key}_20.json"
    return subset_path if subset_path.exists() else None


def _all_questions():
    """Yield (language_key, question) for every registered dataset with an available
    file (see _dataset_path_for) -- a language with neither file present is skipped
    rather than crashing collection for every other test in this file."""
    for language_key, entry in DATASETS.items():
        path = _dataset_path_for(language_key, entry)
        if path is None:
            continue
        for question in load_dataset(path):
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
