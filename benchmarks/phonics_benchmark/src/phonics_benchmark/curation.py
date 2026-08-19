"""Functions to curate benchmark items, including labeling and analysis."""

from __future__ import annotations

import warnings
from collections import OrderedDict
from typing import Any

import pandas as pd

from phonics_benchmark.labeling import get_vocabulary_set, label_nondecodable_words
from phonics_benchmark.utils import require_columns


@require_columns(["content", "student_knowledge"])
def add_labels_and_analysis(
    benchmark_items: pd.DataFrame,
    decodability_config: dict,
    language: str,
    *,
    score_column: str,
) -> pd.DataFrame:
    """Add num_nondecodable_words, num_total_words and nondecodable_ratio.

    ``score_column`` names the aggregation output that decides decodability and is
    passed straight to ``label_nondecodable_words``; it differs by language.

    Total word count is the number of distinct words in ``vocab_set``.
    ``vocab_set`` is dropped from the returned DataFrame as it's only used for
    intermediate calculations. Easily re-computed with ``get_vocabulary_set`` if needed.
    """
    benchmark_items = benchmark_items.copy()

    benchmark_items["vocab_set"] = get_vocabulary_set(benchmark_items, language)

    has_no_words = benchmark_items["vocab_set"].apply(len) == 0
    if has_no_words.any():
        dropped = benchmark_items.loc[has_no_words]
        # Identify the offenders by ID where we have one, else by row label.
        labels = (
            dropped["benchmark_item_id"].tolist()
            if "benchmark_item_id" in dropped.columns
            else dropped.index.tolist()
        )
        warnings.warn(
            f"Dropping {len(dropped)} benchmark item(s) whose content yields no "
            f"words, so the non-decodable ratio is undefined: "
            f"{', '.join(str(label) for label in labels)}",
            UserWarning,
            stacklevel=2,
        )
        benchmark_items = benchmark_items.loc[~has_no_words]

    benchmark_items["nondecodable_words"] = label_nondecodable_words(
        benchmark_items, decodability_config, language, score_column=score_column
    )

    benchmark_items["num_nondecodable_words"] = benchmark_items[
        "nondecodable_words"
    ].apply(len)
    benchmark_items["num_total_words"] = benchmark_items["vocab_set"].apply(len)
    benchmark_items["nondecodable_ratio"] = (
        benchmark_items["num_nondecodable_words"] / benchmark_items["num_total_words"]
    )

    return benchmark_items.drop(columns=["vocab_set"])


@require_columns(["nondecodable_ratio", "num_nondecodable_words"])
def get_nondecodable_word_analysis(benchmark_items: pd.DataFrame) -> dict[str, Any]:
    """Return a DataFrame with nondecodable word counts and ratio summary."""
    mask_zero = benchmark_items["nondecodable_ratio"] == 0.0
    mask_one = benchmark_items["num_nondecodable_words"] == 1

    analysis: OrderedDict[str, Any] = OrderedDict()
    analysis["Total items"] = len(benchmark_items)
    analysis["No non-decodable words"] = int(mask_zero.sum())
    analysis["One non-decodable word"] = int(mask_one.sum())

    bin_counts = pd.cut(
        benchmark_items["nondecodable_ratio"],
        bins=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 1.0],
    ).value_counts(sort=False)
    reformat_map = {
        interval: f"({interval.left:.2f}, {interval.right:.2f}]"
        for interval in bin_counts.index
    }
    bin_counts = bin_counts.rename(index=reformat_map).to_dict()
    analysis["Non-decodable ratio histogram"] = bin_counts

    threshold_counts = {}
    for threshold in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 1.0]:
        threshold_counts[f"({threshold:.2f}, 1.0]"] = int(
            (benchmark_items["nondecodable_ratio"] > threshold).sum()
        )
    analysis["Non-decodable ratio threshold count"] = threshold_counts

    return analysis


@require_columns(["nondecodable_ratio", "num_nondecodable_words"])
def filter_benchmark_items(
    benchmark_candidates: pd.DataFrame,
    max_zero_nondecodable: int | None = None,
    max_one_nondecodable: int | None = None,
    max_nondecodable_ratio: float | None = None,
    sample_size_other: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Down-sample to a manageable mix of easy and moderately hard items.

    - Keep at most ``max_zero_nondecodable`` items with a 0.0 ratio
    - Keep at most ``max_one_nondecodable`` items with exactly one non-decodable
      word.
    - Drop items with greater than ``max_nondecodable_ratio``
    - Keep at most ``sample_size_other`` items from other items.

    Sampling is seeded by ``random_state`` for reproducibility. This heuristic is
    expected to be refined as the item distribution becomes clearer.
    """
    mask_zero = benchmark_candidates["nondecodable_ratio"] == 0.0
    zero_nondecodable = benchmark_candidates[mask_zero]
    if (
        max_zero_nondecodable is not None
        and len(zero_nondecodable) > max_zero_nondecodable
    ):
        zero_nondecodable = zero_nondecodable.sample(
            max_zero_nondecodable, random_state=random_state
        )

    mask_one = benchmark_candidates["num_nondecodable_words"] == 1
    one_nondecodable = benchmark_candidates[mask_one]
    if (
        max_one_nondecodable is not None
        and len(one_nondecodable) > max_one_nondecodable
    ):
        one_nondecodable = one_nondecodable.sample(
            max_one_nondecodable, random_state=random_state
        )

    mask_too_hard = (
        benchmark_candidates["nondecodable_ratio"] > max_nondecodable_ratio
        if max_nondecodable_ratio is not None
        else pd.Series(False, index=benchmark_candidates.index)
    )

    mask_other = ~(mask_zero | mask_one | mask_too_hard)
    other_nondecodable = benchmark_candidates[mask_other]
    if sample_size_other is not None and len(other_nondecodable) > 0:
        n = min(sample_size_other, len(other_nondecodable))
        other_nondecodable = other_nondecodable.sample(n, random_state=random_state)

    benchmark_items = pd.concat(
        [zero_nondecodable, one_nondecodable, other_nondecodable]
    )
    return benchmark_items
