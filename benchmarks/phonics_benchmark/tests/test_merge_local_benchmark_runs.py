"""Guard the data identity and response selection used by local reports."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/merge_local_benchmark_runs.py"
SPEC = importlib.util.spec_from_file_location("merge_local_runs", SCRIPT)
merge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(merge)


def entry(status="success", timestamp="20260916T100000_000000Z", output="[]"):
    return (
        {
            "model_call_id": "call-1",
            "language": "sw",
            "model": "example",
            "benchmark_item_id": "item-1",
            "prompt": "the exact prompt",
            "call_status": status,
            "timestamp_utc": timestamp,
            "output": output,
            "run_id": "original-run",
            "result_id": "original-result",
        },
        {"file": f"sources/{timestamp}.jsonl", "line": 1},
    )


def test_copied_response_keeps_original_ids_and_all_source_locations():
    original = entry()
    copied = (dict(original[0]), {"file": "sources/copied.jsonl", "line": 7})
    selected, audit = merge.deduplicate_calls([original, copied])
    assert len(selected) == 1
    row, _, origins = selected[0]
    assert row == original[0]
    assert len(origins) == 2
    assert audit[0]["conflicting_success_outputs"] is False


def test_later_error_does_not_replace_success():
    success = entry()
    error = entry("error", "20260917T100000_000000Z", "")
    selected, _ = merge.deduplicate_calls([success, error])
    assert selected[0][0] == success[0]


def test_latest_success_selected_even_if_its_output_does_not_parse():
    older = entry(output='["word"]')
    newer = entry(timestamp="20260917T100000_000000Z", output="cannot answer")
    selected, audit = merge.deduplicate_calls([newer, older])
    assert selected[0][0] == newer[0]
    assert audit[0]["conflicting_success_outputs"] is True


def test_conflicting_call_identity_fails():
    first = entry()
    conflicting = entry()
    conflicting[0]["prompt"] = "a different task"
    with pytest.raises(ValueError, match="Conflicting call identity"):
        merge.deduplicate_calls([first, conflicting])


def test_missing_call_id_fails():
    candidate = entry()
    del candidate[0]["model_call_id"]
    with pytest.raises(ValueError, match="Missing model_call_id"):
        merge.deduplicate_calls([candidate])


def test_duplicate_target_rows_are_collapsed_but_conflicting_labels_fail():
    item = {"benchmark_item_id": "one", "nondecodable_words": ["word"]}
    items, duplicates = merge.unique_items([item, dict(item)])
    assert list(items) == ["one"]
    assert duplicates == [{"benchmark_item_id": "one", "duplicate_line": 2}]
    with pytest.raises(ValueError, match="Conflicting item rows"):
        merge.unique_items([item, {**item, "nondecodable_words": []}])


def test_relabelled_gold_allowed_but_changed_model_input_fails():
    old = {
        "benchmark_item_id": "one",
        "content": "text",
        "student_knowledge": {"graphemes": ["t"]},
        "nondecodable_words": ["text"],
    }
    updated = {**old, "nondecodable_words": []}
    merge.validate_same_input(old, updated)
    with pytest.raises(ValueError, match="Changed content"):
        merge.validate_same_input(old, {**updated, "content": "other"})


def test_dataset_comes_from_destination_config_not_response_run_id():
    config = {
        "resolved_config": {
            "run_id": "destination",
            "items_path": "/workspace/items.jsonl",
            "metadata": {"items_source_uri": "gs://bucket/additional.jsonl"},
        },
        "raw_config": {"items_path": "gs://bucket/old.jsonl"},
    }
    assert merge.source_dataset_uri(config) == "gs://bucket/additional.jsonl"


def gold_item(item_id, words):
    return {
        "benchmark_item_id": item_id,
        "content": "The cat sat.",
        "student_knowledge": {"gpcs": [], "sight_words": []},
        "nondecodable_words": words,
        "num_total_words": 3,
        "num_nondecodable_words": len(words),
        "nondecodable_ratio": len(words) / 3,
    }


def test_gold_overlay_joins_by_id_and_recomputes_counts_without_changing_source():
    original = {"a": gold_item("a", ["cat"]), "b": gold_item("b", ["sat"])}
    updated = {"b": gold_item("b", ["sat"]), "a": gold_item("a", [])}
    updated["a"]["num_nondecodable_words"] = 99
    result, audit = merge.replace_gold_labels(original, updated, "en")
    assert result["a"]["nondecodable_words"] == []
    assert result["a"]["num_nondecodable_words"] == 0
    assert result["a"]["nondecodable_ratio"] == 0
    assert result["b"] == original["b"]
    assert original["a"]["nondecodable_words"] == ["cat"]
    assert audit[0]["benchmark_item_id"] == "a"
    assert audit[0]["removed_words"] == ["cat"]


def test_gold_overlay_rejects_missing_ids_changed_input_and_outside_words():
    original = {"a": gold_item("a", ["cat"])}
    with pytest.raises(ValueError, match="exactly the original item IDs"):
        merge.replace_gold_labels(original, {}, "en")
    with pytest.raises(ValueError, match="Changed content"):
        merge.replace_gold_labels(
            original, {"a": {**original["a"], "content": "dog"}}, "en"
        )
    with pytest.raises(ValueError, match="outside vocabulary"):
        merge.replace_gold_labels(original, {"a": gold_item("a", ["dog"])}, "en")
