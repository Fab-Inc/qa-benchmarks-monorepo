"""Tests for inputs.py: dataset questions -> fab-bench inputs file."""

import json

import pytest
from conftest import sample_questions, write_dataset
from linguistic_knowledge_benchmark.inputs import (
    _build_system_prompts,
    _build_user_message,
    build_inputs_rows,
    write_inputs_file,
)


def test_build_system_prompts_has_all_answer_types():
    prompts = _build_system_prompts("English")
    assert set(prompts) == {"multiple_choice", "short_form"}


def test_build_system_prompts_prefix_gives_language_context_for_all_types():
    prompts = _build_system_prompts("English")
    for prompt in prompts.values():
        assert "linguistics of English" in prompt


def test_build_system_prompts_normalizes_kiswahili_to_swahili():
    prompts = _build_system_prompts("Kiswahili")
    for prompt in prompts.values():
        assert "Swahili" in prompt
        assert "Kiswahili" not in prompt


def test_build_user_message_folds_mcq_options_into_text():
    question = {
        "question": "Which is right?",
        "answer_type": "multiple_choice",
        "options": {"A": "first", "B": "second"},
    }
    message = _build_user_message(question)
    assert message == "Which is right?\n\nA. first\nB. second"


def test_build_user_message_short_form_is_just_the_question():
    question = {"question": "Translate 'thing'.", "answer_type": "short_form"}
    assert _build_user_message(question) == "Translate 'thing'."


def test_build_inputs_rows_renders_every_question():
    rows = build_inputs_rows(sample_questions(), "Kiswahili")
    assert [row["benchmark_item_id"] for row in rows] == ["1", "2"]
    for row in rows:
        assert set(row) == {"benchmark_item_id", "prompt", "system_prompt"}
    # MCQ prompt folds the options in; every system prompt matches its type.
    assert "A. opt A" in rows[0]["prompt"]
    assert "letter" in rows[0]["system_prompt"]
    assert "1-3 words" in rows[1]["system_prompt"]


def test_build_inputs_rows_rejects_unknown_answer_type():
    with pytest.raises(ValueError, match="answer_type"):
        build_inputs_rows(
            [
                {
                    "question_id": 1,
                    "question": "q",
                    "answer_type": "essay",
                }
            ],
            "English",
        )


def test_write_inputs_file_writes_one_row_per_question(tmp_path):
    dataset = write_dataset(tmp_path / "ds.json", sample_questions())
    dest = write_inputs_file(dataset, "English", tmp_path / "out" / "inputs.jsonl")
    rows = [json.loads(line) for line in dest.read_text().splitlines()]
    assert len(rows) == len(sample_questions())
    assert rows[0]["benchmark_item_id"] == "1"
