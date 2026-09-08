"""Inspection plots for the benchmark item distribution and model results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from phonics_benchmark.curation import DIFFICULTY_BUCKETS
from phonics_benchmark.utils import require_columns

BUCKET_ORDER = [name for name, _ in DIFFICULTY_BUCKETS]


@require_columns(["model", "difficulty_bucket", "graded_score"])
def save_score_by_difficulty_plot(
    score_by_difficulty: pd.DataFrame,
    output_path: Path,
    *,
    title: str = "Decodable-Text Score vs. Student Difficulty",
) -> None:
    """Save a per-model line plot of graded score across difficulty buckets.

    Buckets are plotted in curriculum order rather than alphabetically, so the
    line reads left-to-right from the least-taught student to the most-taught.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    buckets_present = set(score_by_difficulty["difficulty_bucket"])
    present = [b for b in BUCKET_ORDER if b in buckets_present]
    positions = {bucket: index for index, bucket in enumerate(present)}

    plt.figure(figsize=(10, 6))
    for model_name, group in score_by_difficulty.groupby("model"):
        group = group[group["difficulty_bucket"].isin(positions)]
        group = group.assign(_x=group["difficulty_bucket"].map(positions)).sort_values(
            "_x"
        )
        plt.plot(group["_x"], group["graded_score"], marker="o", label=model_name)

    plt.xticks(range(len(present)), present)
    plt.title(title)
    plt.xlabel("Student Difficulty Bucket (graphemes taught)")
    plt.ylabel("Graded Score (mean)")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.savefig(output_path)
    plt.close()


@require_columns(["model", "scenario_id", "graded_score"])
def save_score_by_scenario_plot(
    score_by_scenario: pd.DataFrame,
    output_path: Path,
    *,
    title: str = "Decodable-Text Score by Generation Scenario",
) -> None:
    """Save a grouped bar chart of graded score per scenario, per model."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pivoted = score_by_scenario.pivot(
        index="scenario_id", columns="model", values="graded_score"
    ).sort_index()

    pivoted.plot(kind="bar", figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Scenario")
    plt.ylabel("Graded Score (mean)")
    plt.ylim(0.0, 1.0)
    plt.xticks(rotation=0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


@require_columns(["difficulty_bucket"])
def save_item_difficulty_histogram(
    items: pd.DataFrame,
    output_path: Path,
    *,
    title: str = "Benchmark Items per Difficulty Bucket",
) -> None:
    """Save a bar chart of how many items fall in each difficulty bucket."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    counts = items["difficulty_bucket"].value_counts()
    counts = counts.reindex([b for b in BUCKET_ORDER if b in counts.index])

    plt.figure(figsize=(10, 6))
    plt.bar(counts.index, counts.to_numpy())
    plt.title(title)
    plt.xlabel("Student Difficulty Bucket (graphemes taught)")
    plt.ylabel("Number of Items")
    plt.savefig(output_path)
    plt.close()
