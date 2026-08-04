"""Tests for the prepared LLM trial dataset and prompt construction."""

import json

from benchmarks.phonics_benchmark.scripts.llm_output_trial import (
    BENCHMARK_ITEMS_PATH,
    build_decodability_prompt,
    process_model_response_into_result,
)

EXPECTED_KEYS = [
    "benchmark_item_id",
    "language_iso_code",
    "curriculum_stage",
    "grade",
    "content",
    "student_knowledge_id",
    "student_knowledge",
    "non_decodable_words",
]


def _load_benchmark_items():
    with BENCHMARK_ITEMS_PATH.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def test_prepared_trial_dataset_is_complete_and_normalized():
    benchmark_items = _load_benchmark_items()

    assert all(list(item) == EXPECTED_KEYS for item in benchmark_items)
    assert len(benchmark_items) == 57
    assert all(item["benchmark_item_id"] for item in benchmark_items)
    item_ids = [item["benchmark_item_id"] for item in benchmark_items]
    assert len(item_ids) == len(set(item_ids))
    assert not {"SW-KE-S17", "SW-KE-S18"} & set(item_ids)
    assert not {17, 18} & {item["curriculum_stage"] for item in benchmark_items}
    assert all("’" not in item["content"] for item in benchmark_items)

    for item in benchmark_items:
        assert isinstance(item["student_knowledge"], dict)
        for values in (
            *item["student_knowledge"].values(),
            item["non_decodable_words"],
        ):
            assert isinstance(values, list)
            assert all(isinstance(value, str) for value in values)

    negative_control = next(
        item for item in benchmark_items if item["benchmark_item_id"] == "SW-KE-S10-N"
    )
    assert negative_control["non_decodable_words"] == []


def test_build_decodability_prompt_uses_prepared_case_data():
    benchmark_item = _load_benchmark_items()[0]

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
        "non_decodable_words": ["Kuku", "KAKA"],
    }

    result = process_model_response_into_result(call_record, benchmark_item)

    assert result["response_parsed"] == ["kuku", "kaka"]
    assert result["correct_answer"] == ["kuku", "kaka"]
    assert result["perfect_match"]
