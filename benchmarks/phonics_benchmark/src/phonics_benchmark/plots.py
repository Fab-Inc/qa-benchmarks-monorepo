"""Inspection plots for the benchmark item distribution and model results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, PercentFormatter

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


# Leaderboards share one renderer; scores and saved bounds retain their units.
@dataclass(frozen=True)
class MetricSpec:
    label: str
    short_label: str
    axis_label: str
    scale: int = 1
    minimum: float = 0
    baseline: float | None = None


METRIC_SPECS = {
    "pooled_f1": MetricSpec("Pooled F1", "F1", "Pooled F1"),
    "pooled_f2": MetricSpec("Pooled F2", "F2", "Pooled F2"),
    "balanced_accuracy": MetricSpec(
        "Balanced accuracy", "BA", "Balanced accuracy", baseline=0.5
    ),
    "adjusted_balanced_accuracy": MetricSpec(
        "Adjusted balanced accuracy",
        "Adj. BA",
        "Adjusted balanced accuracy (0 = chance; 100 = perfect)",
        scale=100,
        minimum=-1,
        baseline=0,
    ),
}
SURFACE, INK_PRIMARY, INK_SECONDARY = "#fcfcfb", "#0b0b0b", "#52514e"
INK_MUTED, GRIDLINE, BASELINE = "#898781", "#e1e0d9", "#c3c2b7"


def plot_leaderboard(
    table: pd.DataFrame,
    colors: dict,
    language: str,
    *,
    metric: str = "pooled_f2",
    eligibility_label: str = "",
    gold_note: str = "",
    baseline: float | None = None,
):
    """Draw saved estimates and interval endpoints in the metric's stated units."""
    spec = METRIC_SPECS[metric]
    table = table.copy()
    values = table[["score", "ci_low", "ci_high"]] / spec.scale
    if not values.score.between(spec.minimum, 1).all():
        raise ValueError("Undefined or out-of-range leaderboard scores")
    finite = values.ci_low.notna() & values.ci_high.notna()
    if (
        not values.loc[finite, ["ci_low", "ci_high"]]
        .apply(lambda column: column.between(spec.minimum, 1))
        .all()
        .all()
    ):
        raise ValueError("Out-of-range confidence bounds")
    if not (values.loc[finite, "ci_low"] <= values.loc[finite, "ci_high"]).all():
        raise ValueError("Reversed confidence interval")
    table[["score", "ci_low", "ci_high"]] = values
    if table.empty:
        raise ValueError("No eligible models to plot")
    n = len(table)
    fig, ax = plt.subplots(figsize=(13, max(5, 2.0 + 0.34 * n)))
    fig.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    positions = np.arange(n)
    ax.barh(positions, table.score, height=0.65, color=table.color, zorder=3)
    # Endpoints are drawn directly: percentile bounds need not enclose the estimate.
    midpoint = (table.ci_low + table.ci_high) / 2
    bars = ax.errorbar(
        midpoint,
        positions,
        xerr=[midpoint - table.ci_low, table.ci_high - midpoint],
        fmt="none",
        ecolor=INK_PRIMARY,
        elinewidth=1,
        capsize=3,
        zorder=4,
    )
    for artist in (*bars.lines[1], *bars.lines[2]):
        artist.set_path_effects(
            [
                patheffects.Stroke(linewidth=2.5, foreground=SURFACE),
                patheffects.Normal(),
            ]
        )
    ax.set_yticks(positions, table.display_name)
    ax.set_ylim(n - 0.3, -0.8)
    leftmost = min(table.score.min(), table.ci_low.min())
    ax.set_xlim(min(-0.025, leftmost - 0.01) if spec.scale == 100 else 0, 1)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xlabel(spec.axis_label, color=INK_SECONDARY)
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, pos: f"{value * 100:.0f}")
        if spec.scale == 100
        else PercentFormatter(1)
    )
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, length=3, width=0.8, labelsize=9)
    for position, row in enumerate(table.itertuples()):
        ax.text(
            1.02,
            position,
            f"{row.score * 100:.1f}" if spec.scale == 100 else f"{row.score:.1%}",
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=9,
            color=INK_SECONDARY,
        )
        ax.text(
            1.145,
            position,
            f"{row.n_items:,}",
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=8.5,
            color=INK_MUTED,
        )
    ax.text(
        1.02,
        1.018,
        spec.short_label,
        transform=ax.transAxes,
        color=INK_SECONDARY,
        fontsize=9,
    )
    ax.text(
        1.145, 1.018, "Items", transform=ax.transAxes, color=INK_SECONDARY, fontsize=9
    )
    present = set(table.provider)
    ax.legend(
        handles=[
            Patch(facecolor=color, label=provider)
            for provider, color in colors.items()
            if provider in present
        ],
        title="Provider",
        loc="center left",
        bbox_to_anchor=(1.30, 0.5),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )
    level, reps, target = table.iloc[0][["ci_level", "n_bootstrap", "expected_items"]]
    ax.set_title(
        f"{language.title()} decodability · {spec.label}\nLeaderboard"
        + (f" · {eligibility_label}" if eligibility_label else ""),
        loc="left",
        fontsize=12,
        fontweight="semibold",
        color=INK_PRIMARY,
        pad=35,
    )
    ax.annotate(
        (
            f"{int(target):,} target items · {level:.0%} item-bootstrap CIs "
            f"({int(reps):,} resamples)"
            if reps
            else f"{int(target):,} target items · confidence intervals not computed"
        ),
        (0, 1),
        xytext=(0, 8),
        xycoords="axes fraction",
        textcoords="offset points",
        fontsize=8.5,
        color=INK_SECONDARY,
        va="bottom",
    )
    if baseline is None:
        baseline = spec.baseline
    footer = "Items = scorable responses per model; coverage varies."
    if baseline is not None:
        if metric == "pooled_f1":
            footer += f" Dashed line = full-set all-words baseline ({baseline:.1%})."
        else:
            footer += f" Dashed line = chance / flag-all / flag-none ({baseline:g})."
    ax.annotate(
        footer,
        (0, 0),
        xytext=(0, -43),
        xycoords="axes fraction",
        textcoords="offset points",
        fontsize=8,
        color=INK_SECONDARY,
        va="top",
    )
    if baseline is not None:
        ax.axvline(baseline / spec.scale, color="#454545", ls="--", lw=1, zorder=5)
    if gold_note:
        ax.annotate(
            gold_note,
            (0, 0),
            xytext=(0, -57),
            xycoords="axes fraction",
            textcoords="offset points",
            fontsize=8,
            color=INK_SECONDARY,
            va="top",
        )
    fig.tight_layout()
    return fig


def eligible_models(table: pd.DataFrame, min_items: int = 700) -> pd.DataFrame:
    """Select models by successfully scored items, not saved response rows."""
    if min_items < 1:
        raise ValueError("min_items must be positive")
    return table[table.scorable_items.ge(min_items)].reset_index(drop=True)


def plot_adjusted_leaderboard(table, colors, language, required, gold_note):
    """Convenience entry for adjusted BA inputs expressed in points."""
    return plot_leaderboard(
        table,
        colors,
        language,
        metric="adjusted_balanced_accuracy",
        eligibility_label=f"≥{required:,} evaluated items",
        gold_note=gold_note,
    )
