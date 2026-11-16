"""Plots related to model results, independent of hardware tiers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from phonics_benchmark.utils import require_columns


@require_columns(["model", "num_nondecodable_words", "accuracy"])
def save_accuracy_by_nondecodable_count_plot(
    accuracy_by_count: pd.DataFrame,
    output_path: Path,
    *,
    title: str = "Accuracy vs. Number of Non-Decodable Words in Item",
) -> None:
    """Save a per-model line plot of accuracy across item difficulty.

    ``accuracy_by_count`` is the long-format frame from
    ``results.accuracy_by_nondecodable_count`` (model,
    num_nondecodable_words, accuracy). A plain histogram would only show how
    many items fall at each difficulty level; plotting accuracy against it
    directly answers whether a model's exact-match rate drops off as items
    get harder.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for model_name, group in accuracy_by_count.groupby("model"):
        group = group.sort_values("num_nondecodable_words")
        plt.plot(
            group["num_nondecodable_words"],
            group["accuracy"],
            marker="o",
            label=model_name,
        )
    plt.title(title)
    plt.xlabel("Number of Non-Decodable Words in Item")
    plt.ylabel("Accuracy (Exact Match Rate)")
    plt.legend()
    plt.savefig(output_path)
    plt.close()
