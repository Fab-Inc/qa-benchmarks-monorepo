"""Tests for the run module (no live LLM calls)."""

import json

import pandas as pd
from phonics_benchmark.run import (
    build_decodability_prompt,
    load_completed_keys,
    run_benchmark,
)

ITEMS = pd.DataFrame(
    {
        "benchmark_item_id": ["p1_t1", "p1_t2"],
        "student_knowledge_id": ["p1", "p1"],
        "content": ["Mama na paka.", "Kuku soko."],
        "student_knowledge": [
            {"graphemes": ["a", "m"], "whole_words": ["mama", "na", "paka"]},
            {"graphemes": ["a", "m"], "whole_words": ["mama", "na", "paka"]},
        ],
        "non_decodable_words": [[], ["kuku", "soko"]],
    }
)


def test_build_decodability_prompt_uses_human_language_name() -> None:
    """The rendered prompt names the language and includes the text."""
    prompt = build_decodability_prompt(ITEMS.iloc[0], "sw")
    assert "Kiswahili" in prompt
    assert "Mama na paka." in prompt


def test_run_benchmark_bypass_produces_one_record_per_item() -> None:
    """With bypass_llm, every (model, item) pair yields a well-formed record."""
    streamed: list[dict] = []
    results = run_benchmark(
        ITEMS,
        ["fake-model"],
        "sw",
        bypass_llm=True,
        on_result=streamed.append,
    )

    assert len(results) == len(ITEMS)
    assert streamed == results  # on_result saw every result
    first = results[0]
    assert first["model_name"] == "fake-model"
    assert first["benchmark_item_id"] == "p1_t1"
    assert first["response_raw"] == "BYPASS"
    assert set(first) >= {
        "model_name",
        "benchmark_item_id",
        "student_knowledge_id",
        "prompt",
        "response_parsed",
        "response_parse_ok",
        "correct_answer",
        "perfect_match",
    }


def test_run_benchmark_max_rows_caps_items() -> None:
    """max_rows_per_model limits how many items each model sees."""
    results = run_benchmark(
        ITEMS, ["fake-model"], "sw", bypass_llm=True, max_rows_per_model=1
    )
    assert len(results) == 1


def test_run_benchmark_skips_completed_keys() -> None:
    """Items whose (model, id) are in completed_keys are not re-run."""
    completed = {("fake-model", "p1_t1")}
    results = run_benchmark(
        ITEMS, ["fake-model"], "sw", bypass_llm=True, completed_keys=completed
    )
    assert len(results) == 1
    assert results[0]["benchmark_item_id"] == "p1_t2"


def test_run_benchmark_all_completed_makes_no_calls() -> None:
    """When every pair is already complete, nothing runs."""
    completed = {("fake-model", "p1_t1"), ("fake-model", "p1_t2")}
    results = run_benchmark(
        ITEMS, ["fake-model"], "sw", bypass_llm=True, completed_keys=completed
    )
    assert results == []


def test_load_completed_keys_round_trips(tmp_path) -> None:
    """load_completed_keys reads back the (model, id) pairs written to a JSONL."""
    path = tmp_path / "results.jsonl"
    path.write_text(
        json.dumps({"model_name": "m1", "benchmark_item_id": "a"})
        + "\n"
        + json.dumps({"model_name": "m1", "benchmark_item_id": "b"})
        + "\n",
        encoding="utf-8",
    )
    assert load_completed_keys(path) == {("m1", "a"), ("m1", "b")}


def test_load_completed_keys_missing_file_is_empty(tmp_path) -> None:
    """A non-existent results file yields an empty set (safe for fresh runs)."""
    assert load_completed_keys(tmp_path / "nope.jsonl") == set()
