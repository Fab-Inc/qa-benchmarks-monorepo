"""Tests for the prepared LLM trial dataset and prompt construction."""

import json

import pandas as pd

from benchmarks.phonics_benchmark.scripts.llm_output_trial import (
    BENCHMARK_ITEMS_PATH,
    build_decodability_prompt,
    process_model_response_into_result,
)

EXPECTED_COLUMNS = [
    "benchmark_item_id",
    "language_iso_code",
    "curriculum_stage",
    "grade",
    "content",
    "student_knowledge_id",
    "graphemes",
    "whole_words",
    "non_decodable_words",
]


def test_prepared_trial_dataset_is_complete_and_normalized():
    benchmark_items = pd.read_csv(BENCHMARK_ITEMS_PATH)

    assert benchmark_items.columns.tolist() == EXPECTED_COLUMNS
    assert len(benchmark_items) == 57
    assert not benchmark_items.isna().any().any()
    assert benchmark_items["benchmark_item_id"].is_unique
    assert not {"SW-KE-S17", "SW-KE-S18"} & set(benchmark_items["benchmark_item_id"])
    assert not {17, 18} & set(benchmark_items["curriculum_stage"])
    assert not benchmark_items["content"].str.contains("’", regex=False).any()

    for column in ("graphemes", "whole_words", "non_decodable_words"):
        parsed_values = benchmark_items[column].map(json.loads)
        assert parsed_values.map(lambda value: isinstance(value, list)).all()
        assert parsed_values.map(
            lambda value: all(isinstance(item, str) for item in value)
        ).all()

    negative_control = benchmark_items.loc[
        benchmark_items["benchmark_item_id"] == "SW-KE-S10-N"
    ].iloc[0]
    assert json.loads(negative_control["non_decodable_words"]) == []


def test_build_decodability_prompt_uses_prepared_case_data():
    benchmark_item = pd.read_csv(BENCHMARK_ITEMS_PATH).iloc[0]

    prompt = build_decodability_prompt(benchmark_item, language="sw")

    assert "language of Kiswahili" in prompt
    assert '- Known graphemes: "a", "k", "m", "u"' in prompt
    assert '- Known whole words: "ama", "anaona", "mama", "na", "paka"' in prompt
    assert "Mama anaona kuku na paka." in prompt
    assert "curriculum_stage" not in prompt
    assert benchmark_item["benchmark_item_id"] not in prompt
    assert benchmark_item["student_knowledge_id"] not in prompt


def test_process_model_response_normalizes_words_to_lowercase():
    call_record = {
        "model": "test-model",
        "response_text": '["KUKU", "Kaka"]',
        "max_tokens": 100,
        "tokens_used_completion": 5,
        "latency_seconds": 0.1,
        "timestamp_utc": "20260715T000000Z",
    }
    benchmark_item = {
        "benchmark_item_id": "SW-TEST",
        "curriculum_stage": 1,
        "student_knowledge_id": "student-test",
        "non_decodable_words": '["Kuku", "KAKA"]',
    }

    result = process_model_response_into_result(call_record, benchmark_item)

    assert result["response_parsed"] == ["kuku", "kaka"]
    assert result["correct_answer"] == ["kuku", "kaka"]
    assert result["perfect_match"]
