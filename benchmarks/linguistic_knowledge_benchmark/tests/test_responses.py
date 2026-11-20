"""Tests for responses.py: fab-bench JSONL rows -> legacy responses table."""

import json

import pandas as pd
import pytest
from conftest import VALID_MODEL, sample_questions
from linguistic_knowledge_benchmark.responses import (
    load_responses,
    responses_path,
)


def _experiment(tmp_path):
    (tmp_path / "responses").mkdir(parents=True)
    return tmp_path


def _write_rows(experiment_dir, rows):
    with open(responses_path(experiment_dir, VALID_MODEL), "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_load_responses_renames_and_joins(tmp_path):
    experiment_dir = _experiment(tmp_path)
    questions = sample_questions()
    _write_rows(
        experiment_dir,
        [
            {
                "benchmark_item_id": "1",
                "output": "B",
                "call_status": "success",
                "error_type": "",
                "error_message": "",
                "latency_seconds": 1.5,
                "tokens_used": 20,
                "tokens_used_completion": 8,
                "tokens_used_reasoning": 3,
            },
            {
                "benchmark_item_id": "2",
                "output": "",
                "call_status": "error",
                "error_type": "RuntimeError",
                "error_message": "boom",
            },
        ],
    )
    df = load_responses(experiment_dir, VALID_MODEL, questions).set_index("question_id")

    assert df.loc[1, "answer_type"] == "multiple_choice"
    assert df.loc[1, "domain"] == "morphology"
    assert df.loc[1, "expected_answers"] == "B"
    assert df.loc[1, "latency"] == 1.5
    assert df.loc[1, "total_tokens"] == 20
    assert df.loc[1, "completion_tokens"] == 8
    assert df.loc[1, "reasoning_tokens"] == 3
    assert df.loc[1, "call_status"] == "success"
    assert df.loc[2, "call_status"] == "error"
    assert df.loc[2, "error_type"] == "RuntimeError"
    assert df.loc[2, "expected_answers"] == "kintu"
    assert (df["retry_count"] == 0).all()


def test_load_responses_expected_answers_joined_with_pipe(tmp_path):
    experiment_dir = _experiment(tmp_path)
    questions = [
        {
            "question_id": 1,
            "question": "q",
            "answer_type": "short_form",
            "expected_answers": ["kintu", "ekintu"],
        }
    ]
    _write_rows(experiment_dir, [{"benchmark_item_id": "1", "output": "kintu"}])
    df = load_responses(experiment_dir, VALID_MODEL, questions)
    assert df.loc[0, "expected_answers"] == "kintu|ekintu"


def test_load_responses_partially_unmatched_warns(tmp_path):
    """Ids absent from the dataset (a wrong or stale --dataset) get empty
    metadata and a warning rather than silently becoming unscored rows."""
    experiment_dir = _experiment(tmp_path)
    _write_rows(
        experiment_dir,
        [
            {"benchmark_item_id": "1", "output": "B"},
            {"benchmark_item_id": "99", "output": "x"},
        ],
    )
    with pytest.warns(UserWarning, match="do not match the dataset"):
        df = load_responses(experiment_dir, VALID_MODEL, sample_questions())
    assert pd.isna(df.loc[1, "question_id"])
    assert pd.isna(df.loc[1, "answer_type"])


def test_load_responses_all_unmatched_raises(tmp_path):
    """Every row unmatched means the wrong dataset -- fail, don't score zeros."""
    experiment_dir = _experiment(tmp_path)
    _write_rows(experiment_dir, [{"benchmark_item_id": "99", "output": "x"}])
    with pytest.raises(ValueError, match="do not match the dataset"):
        load_responses(experiment_dir, VALID_MODEL, sample_questions())


def test_load_responses_preserves_question_id_type(tmp_path):
    """question_id comes back as the dataset's own value (int), matching the
    judged CSVs' historic schema."""
    experiment_dir = _experiment(tmp_path)
    questions = [{"question_id": 1.01, "question": "q", "answer_type": "short_form"}]
    _write_rows(experiment_dir, [{"benchmark_item_id": "1.01", "output": "x"}])
    df = load_responses(experiment_dir, VALID_MODEL, questions)
    assert df.loc[0, "question_id"] == 1.01
