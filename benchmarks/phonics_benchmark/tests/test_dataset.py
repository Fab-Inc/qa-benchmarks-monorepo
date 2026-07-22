"""Tests for the dataset module."""

import json

import pandas as pd
from phonics_benchmark.dataset import (
    add_analysis_columns,
    build_benchmark_items,
    filter_benchmark_items,
)


def _write_jsonl(path, records) -> None:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )


def test_build_benchmark_items_is_full_cross_product(tmp_path) -> None:
    """Every profile is paired with every text, with a composite id."""
    profiles_path = tmp_path / "student_knowledge_profiles.jsonl"
    texts_path = tmp_path / "decodable_texts.jsonl"
    _write_jsonl(
        profiles_path,
        [
            {
                "student_knowledge_id": "p1",
                "language_iso_code": "sw",
                "taught_knowledge": {"graphemes": ["a"], "whole_words": ["mama"]},
            },
            {
                "student_knowledge_id": "p2",
                "language_iso_code": "sw",
                "taught_knowledge": {"graphemes": ["a", "m"], "whole_words": ["na"]},
            },
        ],
    )
    _write_jsonl(
        texts_path,
        [
            {"text_id": "t1", "language_iso_code": "sw", "content": "mama"},
            {"text_id": "t2", "language_iso_code": "sw", "content": "na"},
            {"text_id": "t3", "language_iso_code": "sw", "content": "paka"},
        ],
    )

    items = build_benchmark_items(profiles_path, texts_path)

    assert len(items) == 2 * 3
    assert "p1_t1" in set(items["benchmark_item_id"])
    assert "p2_t3" in set(items["benchmark_item_id"])
    # The raw JSONL key `taught_knowledge` is renamed to `student_knowledge`.
    assert "student_knowledge" in items.columns
    assert "taught_knowledge" not in items.columns


def test_add_analysis_columns_computes_ratio() -> None:
    """Analysis columns count words and derive the non-decodable ratio."""
    items = pd.DataFrame(
        {
            "content": ["mama na paka soko"],  # 4 words
            "non_decodable_words": [["soko"]],  # 1 non-decodable
            "vocab_set": [{"mama", "na", "paka", "soko"}],
        }
    )

    enriched = add_analysis_columns(items, "sw")

    assert enriched.loc[0, "num_nondecodable_words"] == 1
    assert enriched.loc[0, "num_total_words"] == 4
    assert enriched.loc[0, "nondecodable_ratio"] == 0.25


def test_filter_benchmark_items_respects_thresholds() -> None:
    """Sampling caps and the max-ratio drop are all applied."""
    items = pd.DataFrame(
        {
            "nondecodable_ratio": (
                [0.0] * 30  # 30 zero-ratio items -> capped
                + [0.05] * 50  # 50 one-word items -> capped
                + [0.5] * 5  # 5 too-hard items -> dropped
            ),
            "num_nondecodable_words": [0] * 30 + [1] * 50 + [10] * 5,
        }
    )

    filtered = filter_benchmark_items(
        items,
        max_zero_nondecodable=20,
        max_one_nondecodable=40,
        max_nondecodable_ratio=0.2,
        sample_size_other=None,
        random_state=42,
    )

    assert (filtered["nondecodable_ratio"] == 0.0).sum() == 20
    assert (filtered["num_nondecodable_words"] == 1).sum() == 40
    assert (filtered["nondecodable_ratio"] > 0.2).sum() == 0
    assert len(filtered) == 60


def test_filter_benchmark_items_is_reproducible() -> None:
    """The same random_state yields the same sampled subset."""
    items = pd.DataFrame(
        {
            "nondecodable_ratio": [0.0] * 100,
            "num_nondecodable_words": [0] * 100,
        }
    )
    a = filter_benchmark_items(items, random_state=7)
    b = filter_benchmark_items(items, random_state=7)
    assert list(a.index) == list(b.index)
