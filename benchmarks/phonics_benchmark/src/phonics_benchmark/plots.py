"""Inspection plots for the benchmark item distribution and model results.

Kept separate from the run pipeline so it can be used ad hoc for manual analysis
— e.g. comparing the non-decodable ratio distribution before vs. after filtering,
or across different filter settings. Isolates the matplotlib dependency here.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_nondecodable_ratio_histogram(
    benchmark_items: pd.DataFrame,
    output_path: Path,
    *,
    column: str = "nondecodable_ratio",
    title: str = "Distribution of Non-Decodable Word Ratios",
    bins: int = 20,
) -> None:
    """Save a histogram of ``column`` to ``output_path``.

    The caller passes the full output path, so the same function serves the
    pre-filter, post-filter, or any other view by varying path and title.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    benchmark_items[column].hist(bins=bins)
    plt.title(title)
    plt.xlabel("Non-Decodable Word Ratio")
    plt.ylabel("Frequency")
    plt.savefig(output_path)
    plt.close()


def save_accuracy_by_nondecodable_count_plot(
    accuracy_by_count: pd.DataFrame,
    output_path: Path,
    *,
    title: str = "Accuracy vs. Number of Non-Decodable Words in Item",
) -> None:
    """Save a per-model line plot of accuracy across item difficulty.

    ``accuracy_by_count`` is the long-format frame from
    ``results.accuracy_by_nondecodable_count`` (model_name,
    num_nondecodable_words, accuracy). A plain histogram would only show how
    many items fall at each difficulty level; plotting accuracy against it
    directly answers whether a model's exact-match rate drops off as items
    get harder.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for model_name, group in accuracy_by_count.groupby("model_name"):
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


def save_confusion_bar_plot(
    summary_by_model: pd.DataFrame,
    output_path: Path,
    *,
    title: str = "Word-Level Errors by Model",
) -> None:
    """Save a stacked bar chart of pooled false positives/negatives per model.

    ``summary_by_model`` is the output of ``results.summarize_by_model``
    (must have model_name, fp, fn columns). Shows, per model, whether errors
    lean towards over-flagging decodable words (fp) or missing non-decodable
    ones (fn).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    models = summary_by_model["model_name"]
    plt.bar(models, summary_by_model["fp"], label="False Positives")
    plt.bar(
        models,
        summary_by_model["fn"],
        bottom=summary_by_model["fp"],
        label="False Negatives",
    )
    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel("Word Count (Pooled Across Items)")
    plt.xticks(rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
