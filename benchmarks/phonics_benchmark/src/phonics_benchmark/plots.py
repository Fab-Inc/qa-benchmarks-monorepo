"""Inspection plots for the benchmark item distribution and model results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from phonics_benchmark.curation import CURRICULUM_PHASES
from phonics_benchmark.utils import require_columns

PHASE_ORDER = [name for name, _ in CURRICULUM_PHASES]


@require_columns(["model", "curriculum_phase", "graded_score"])
def save_score_by_phase_plot(
    score_by_phase: pd.DataFrame,
    output_path: Path,
    *,
    title: str = "Decodable-Text Score by Curriculum Phase",
) -> None:
    """Save a per-model line plot of graded score across curriculum phases.

    Phases are plotted in curriculum order rather than alphabetically, so the
    line reads left-to-right from the least-taught student to the most-taught,
    i.e. hardest to easiest for the model.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    phases_present = set(score_by_phase["curriculum_phase"])
    present = [p for p in PHASE_ORDER if p in phases_present]
    positions = {phase: index for index, phase in enumerate(present)}

    plt.figure(figsize=(10, 6))
    for model_name, group in score_by_phase.groupby("model"):
        group = group[group["curriculum_phase"].isin(positions)]
        group = group.assign(_x=group["curriculum_phase"].map(positions)).sort_values(
            "_x"
        )
        plt.plot(group["_x"], group["graded_score"], marker="o", label=model_name)

    plt.xticks(range(len(present)), present)
    plt.title(title)
    plt.xlabel("Curriculum Phase (graphemes taught \u2014 earlier is harder)")
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


@require_columns(["curriculum_phase"])
def save_item_phase_histogram(
    items: pd.DataFrame,
    output_path: Path,
    *,
    title: str = "Benchmark Items per Curriculum Phase",
) -> None:
    """Save a bar chart of how many items fall in each curriculum phase."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    counts = items["curriculum_phase"].value_counts()
    counts = counts.reindex([p for p in PHASE_ORDER if p in counts.index])

    plt.figure(figsize=(10, 6))
    plt.bar(counts.index, counts.to_numpy())
    plt.title(title)
    plt.xlabel("Curriculum Phase (graphemes taught)")
    plt.ylabel("Number of Items")
    plt.savefig(output_path)
    plt.close()
