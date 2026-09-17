"""Inspection plots for the benchmark item distribution and model results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from phonics_benchmark.utils import require_columns


@require_columns(["nondecodable_ratio"])
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


# The metrics `bootstrap.bootstrap_metric_cis` reports, with the axis label each
# one gets. Order sets the panel order left to right.
BOOTSTRAP_METRIC_LABELS = {
    "accuracy": "Accuracy (exact match rate)",
    "pooled_f1": "Pooled F1 (word level)",
    "pooled_f2": "Pooled F2 (word level)",
}


@require_columns(["model", "accuracy", "accuracy_ci_low", "accuracy_ci_high"])
def save_bootstrap_ci_plot(
    ci_report: pd.DataFrame,
    output_path: Path,
    *,
    title: str = "Per-model metrics with bootstrap confidence intervals",
) -> None:
    """Save one horizontal bar panel per metric, whiskered with its CI.

    ``ci_report`` is the frame from ``bootstrap.bootstrap_metric_cis``. Bars are
    horizontal and share one model axis because a run can carry 24 models whose
    names are too long to label vertical bars with, and the shared axis lets you
    read one model across all three metrics.

    Whiskers are asymmetric: a percentile interval is not centred on the point
    estimate, so the low and high ends are passed separately rather than as one
    half-width.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Worst accuracy first, so the best model ends up at the top of the axis.
    ordered = ci_report.sort_values("accuracy", ascending=True)
    metrics = [m for m in BOOTSTRAP_METRIC_LABELS if m in ordered.columns]
    positions = range(len(ordered))

    figure, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(5 * len(metrics), max(3.0, 0.32 * len(ordered) + 1.5)),
        sharey=True,
    )
    axes = axes if len(metrics) > 1 else [axes]

    for axis, metric in zip(axes, metrics, strict=True):
        point = ordered[metric]
        # clip(0): a point estimate sitting a hair outside its own percentile
        # bound would otherwise ask matplotlib for a negative whisker.
        lower = (point - ordered[f"{metric}_ci_low"]).clip(lower=0)
        upper = (ordered[f"{metric}_ci_high"] - point).clip(lower=0)
        axis.barh(positions, point, color="#40916C")
        axis.errorbar(
            point,
            positions,
            xerr=[lower, upper],
            fmt="none",
            ecolor="black",
            elinewidth=1.0,
            capsize=3,
        )
        axis.set_xlabel(BOOTSTRAP_METRIC_LABELS[metric])
        axis.set_xlim(0, 1)
        axis.grid(axis="x", linestyle=":", alpha=0.5)
        axis.set_axisbelow(True)

    axes[0].set_yticks(list(positions))
    axes[0].set_yticklabels(ordered["model"])
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
