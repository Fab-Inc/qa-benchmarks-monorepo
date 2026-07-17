"""Tests for phonics benchmark result persistence."""

import json
from pathlib import Path

import pandas as pd
from phonics_benchmark.results import save_results


def test_save_results_writes_jsonl_atomically(tmp_path: Path) -> None:
    """The canonical JSONL file is replaced via a temporary sibling file."""

    results_path = tmp_path / "responses" / "responses_model-a.jsonl"
    results = pd.DataFrame(
        [
            {"result_id": "r1", "output": "first"},
            {"result_id": "r2", "output": "second"},
        ]
    )

    saved_path = save_results(results, results_path)

    assert saved_path == results_path
    rows = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["result_id"] for row in rows] == ["r1", "r2"]
    assert list(results_path.parent.glob("*.tmp")) == []
    assert list(results_path.parent.glob(".*.tmp")) == []
"""Tests for the results module."""

import pandas as pd
from phonics_benchmark.results import (
    accuracy_by_nondecodable_count,
    add_word_level_counts,
    attach_item_metadata,
    sample_items_for_inspection,
    summarize_by_model,
)

ITEMS = pd.DataFrame(
    {
        "benchmark_item_id": ["p1_t1", "p1_t2"],
        "content": ["Mama na paka soko.", "Kuku soko kaka."],
        "nondecodable_ratio": [0.25, 0.33],
    }
)


def test_attach_item_metadata_adds_vocab_and_ratio() -> None:
    """The join brings in nondecodable_ratio and a lowercase vocab set."""
    results = pd.DataFrame({"benchmark_item_id": ["p1_t1"]})

    merged = attach_item_metadata(results, ITEMS, "sw")

    assert merged.loc[0, "nondecodable_ratio"] == 0.25
    assert merged.loc[0, "vocab"] == {"mama", "na", "paka", "soko"}


def test_add_word_level_counts_derives_confusion_counts() -> None:
    """tp/fp/fn/tn are derived from the predicted vs. actual word sets."""
    results = pd.DataFrame(
        {
            "response_parsed": [["soko", "kaka"]],  # predicted non-decodable
            "correct_answer": [["soko"]],  # actually non-decodable
            "vocab": [{"mama", "na", "paka", "soko", "kaka"}],
        }
    )

    scored = add_word_level_counts(results)

    assert scored.loc[0, "tp"] == 1  # soko: predicted and actually non-decodable
    assert scored.loc[0, "fp"] == 1  # kaka: predicted but decodable
    assert scored.loc[0, "fn"] == 0  # nothing actually non-decodable was missed
    assert scored.loc[0, "tn"] == 3  # mama, na, paka: correctly left off


def test_summarize_by_model_pools_counts_and_derives_rates() -> None:
    """Per-model accuracy plus precision/recall/f1 pooled across items."""
    results = pd.DataFrame(
        {
            "model_name": ["m1", "m1"],
            "benchmark_item_id": ["p1_t1", "p1_t2"],
            "perfect_match": [True, False],
            "tp": [1, 1],
            "fp": [0, 1],
            "fn": [0, 1],
            "tn": [3, 2],
        }
    )

    summary = summarize_by_model(results).set_index("model_name")

    assert summary.loc["m1", "n_items"] == 2
    assert summary.loc["m1", "accuracy"] == 0.5
    assert summary.loc["m1", "precision"] == 2 / 3  # tp=2, fp=1
    assert summary.loc["m1", "recall"] == 2 / 3  # tp=2, fn=1
    assert round(summary.loc["m1", "f1"], 4) == round(2 / 3, 4)


def test_accuracy_by_nondecodable_count_groups_by_model_and_count() -> None:
    """Accuracy is broken out per model and per item difficulty level."""
    results = pd.DataFrame(
        {
            "model_name": ["m1", "m1", "m1"],
            "num_nondecodable_words": [0, 0, 1],
            "perfect_match": [True, False, True],
        }
    )

    breakdown = accuracy_by_nondecodable_count(results).set_index(
        ["model_name", "num_nondecodable_words"]
    )

    assert breakdown.loc[("m1", 0), "accuracy"] == 0.5
    assert breakdown.loc[("m1", 0), "n_items"] == 2
    assert breakdown.loc[("m1", 1), "accuracy"] == 1.0


def test_sample_items_for_inspection_respects_bin_caps() -> None:
    """At most n_per_bin items are sampled from each ratio bin."""
    items = pd.DataFrame(
        {
            "benchmark_item_id": [f"i{i}" for i in range(10)],
            "nondecodable_ratio": [0.0] * 5 + [0.15] * 5,
        }
    )

    sampled = sample_items_for_inspection(
        items, bin_edges=(0.0, 0.05, 0.2), n_per_bin=2, random_state=1
    )

    assert len(sampled) == 4
    assert "ratio_bin" in sampled.columns


def test_sample_items_for_inspection_is_reproducible() -> None:
    """The same random_state yields the same sampled subset."""
    items = pd.DataFrame(
        {
            "benchmark_item_id": [f"i{i}" for i in range(10)],
            "nondecodable_ratio": [0.0] * 10,
        }
    )

    a = sample_items_for_inspection(items, n_per_bin=3, random_state=7)
    b = sample_items_for_inspection(items, n_per_bin=3, random_state=7)

    assert list(a["benchmark_item_id"]) == list(b["benchmark_item_id"])
