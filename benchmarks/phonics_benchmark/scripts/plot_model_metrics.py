r"""Plot model leaderboards or item F1 versus non-decodable ratio panels.

Run from the repository root (the default picks the newest local analysis)::

    uv run --frozen --package phonics-benchmark python \
        benchmarks/phonics_benchmark/scripts/plot_model_metrics.py --metric pooled_f1

Reads a `summary_by_model.csv` written by `analyze_finished_kiswahili_models.py`
and writes one horizontal bar per finished model, best at the top. Select a
metric with --metric; the default is pooled_f1. Bootstrap percentile intervals
are drawn when the selected metric's *_ci_low and *_ci_high columns are present.
Their confidence level is read from the accompanying analysis.json, if available.
Older summaries without intervals still work. Existing images are never overwritten.
Use --metric pooled_f2 for F2, which weights false negatives four times as much
as false positives in its denominator. Models with failed calls remain eligible;
their metrics use scorable responses, as recorded by the analysis script.

Use --metric item_averaged_f1 for the mean item F1 leaderboard. Use --plot item-panels
for one scatter panel per finished model (ordered by item-averaged F1 by default).
Panels read the accompanying results_with_scoring.csv, or --scored-csv. Empty/empty
items have undefined F1 and are omitted and counted; fully decodable false alarms
remain visible at (0, 0). Each point is one profile/text item; points may overlap.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
METRICS = {
    "perfect_match_accuracy": "Perfect-match accuracy",
    "pooled_f1": "Pooled F1",
    "item_averaged_f1": "Item-averaged F1",
    "pooled_f2": "Pooled F2",
}
METRIC_COLORS = {
    "perfect_match_accuracy": "accuracy",
    "pooled_f1": "f1",
    "item_averaged_f1": "f1",
    "pooled_f2": "f2",
}
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text_primary": "#0b0b0b",
        "text_secondary": "#52514e",
        "grid": "#dedcd6",
        "accuracy": "#2a78d6",
        "f1": "#eb6834",
        "f2": "#208567",
    },
    "dark": {
        "surface": "#1a1a19",
        "text_primary": "#ffffff",
        "text_secondary": "#c3c2b7",
        "grid": "#3a3a38",
        "accuracy": "#3987e5",
        "f1": "#d95926",
        "f2": "#39ac86",
    },
}


def latest_summary_csv() -> Path:
    """Return the summary CSV of the most recent local analysis directory."""
    candidates = list(OUTPUTS_DIR.glob("*-analysis-*/summary_by_model.csv"))
    if not candidates:
        raise FileNotFoundError(f"No analysis summary found under {OUTPUTS_DIR}")
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))


def load_summary(
    path: Path,
    sort_by: str,
    *,
    metric: str | None = None,
    include_undefined: bool = False,
) -> pd.DataFrame:
    """Load finished models, optionally retaining undefined metrics for panels."""
    metric = metric or sort_by
    summary = pd.read_csv(path)
    required = {"model", "finished", metric, sort_by}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    flags = summary["finished"].astype("string").str.strip().str.lower()
    if not flags.isin(["true", "false"]).all():
        raise ValueError(f"{path}: finished must contain true/false values")
    finished = summary.loc[flags.eq("true")].copy()
    for column in {metric, sort_by}:
        finished[column] = pd.to_numeric(finished[column], errors="raise")
    if not include_undefined:
        finished = finished.dropna(subset=[metric])
    if finished.empty:
        raise ValueError(f"{path} has no finished models with {metric}")
    if not finished[metric].dropna().between(0, 1).all():
        raise ValueError(f"{metric} must be between 0 and 1")
    if finished["model"].isna().any() or finished["model"].duplicated().any():
        raise ValueError(
            "Expected one result per model; select a single run for repeated models"
        )
    low, high = f"{metric}_ci_low", f"{metric}_ci_high"
    if (low in finished) != (high in finished):
        raise ValueError(f"Bootstrap intervals require both {low} and {high}")
    if low in finished:
        finished[[low, high]] = finished[[low, high]].apply(
            pd.to_numeric, errors="raise"
        )
        if finished[low].isna().ne(finished[high].isna()).any():
            raise ValueError("Bootstrap intervals must have both bounds or neither")
        intervals = finished.dropna(subset=[low, high])
        if not (
            intervals[low].between(0, 1)
            & intervals[high].between(0, 1)
            & intervals[low].le(intervals[high])
        ).all():
            raise ValueError("Bootstrap intervals must satisfy 0 <= low <= high <= 1")
    return finished.sort_values(sort_by, ascending=False, kind="stable")


def interval_note(summary: pd.DataFrame, metric: str, summary_path: Path) -> str:
    """Describe intervals without assuming a confidence level for a standalone CSV."""
    low = f"{metric}_ci_low"
    count = int(summary[low].notna().sum()) if low in summary else 0
    if not count:
        return "Bootstrap intervals unavailable"
    description = "Bootstrap intervals"
    metadata_path = summary_path.with_name("analysis.json")
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        bootstrap = metadata.get("bootstrap", {})
        level = bootstrap.get("ci_level")
        if isinstance(level, (int, float)) and 0 < level < 1:
            description = f"{level:.0%} bootstrap intervals"
        replicates = bootstrap.get("n_bootstrap")
        if isinstance(replicates, int) and replicates > 0:
            description += f" ({replicates:,} resamples)"
    if count < len(summary):
        description += f"; available for {count}/{len(summary)} models"
    return f"Whiskers: {description}"


def plot_metrics(
    summary: pd.DataFrame,
    *,
    theme: str,
    annotate: bool,
    title: str,
    subtitle: str,
    metric: str = "pooled_f1",
) -> plt.Figure:
    """Draw one bar per model with the selected metric's bootstrap interval."""
    colors = THEMES[theme]
    # Plot top-to-bottom in ranking order: matplotlib's y-axis grows upwards.
    rows = summary.iloc[::-1].reset_index(drop=True)
    positions = range(len(rows))

    fig, ax = plt.subplots(figsize=(11.5, max(4, 2.0 + 0.36 * len(rows))))
    fig.patch.set_facecolor(colors["surface"])
    ax.set_facecolor(colors["surface"])

    ax.barh(
        list(positions),
        rows[metric],
        height=0.65,
        color=colors[METRIC_COLORS[metric]],
        zorder=2,
    )
    low, high = f"{metric}_ci_low", f"{metric}_ci_high"
    if low in rows and high in rows:
        intervals = rows.dropna(subset=[low, high])
        # Draw the stored endpoints exactly. Percentile intervals can exclude the
        # point estimate, so treating estimate-low as xerr can become negative.
        midpoint = (intervals[low] + intervals[high]) / 2
        ax.errorbar(
            midpoint,
            intervals.index,
            xerr=[midpoint - intervals[low], intervals[high] - midpoint],
            fmt="none",
            ecolor=colors["text_primary"],
            elinewidth=1.2,
            capsize=3,
            capthick=1.2,
            zorder=3,
        )

    if annotate:
        # A separate value column keeps labels clear of bars and whiskers.
        for position, value in enumerate(rows[metric]):
            ax.text(
                1.02,
                position,
                f"{value:.2%}",
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="center",
                fontsize=9,
                color=colors["text_secondary"],
            )

    ax.set_yticks(list(positions), rows["model"], fontsize=9.5)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(0, 1)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_xlabel(METRICS[metric], fontsize=10, color=colors["text_secondary"])
    ax.xaxis.grid(True, color=colors["grid"], linewidth=1, linestyle="-")
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(colors=colors["text_secondary"], length=0)
    for label in ax.get_yticklabels():
        label.set_color(colors["text_primary"])
    for side, spine in ax.spines.items():
        spine.set_visible(side == "left")
        spine.set_color(colors["grid"])

    # Wrap the subtitle so it never runs past the axes; the title clears both lines.
    subtitle_lines = textwrap.wrap(
        subtitle, width=88, break_long_words=False, break_on_hyphens=False
    ) or [""]
    ax.set_title(
        title,
        fontsize=13,
        color=colors["text_primary"],
        loc="left",
        pad=16 + 12 * len(subtitle_lines),
    )
    ax.annotate(
        "\n".join(subtitle_lines),
        (0, 1),
        xytext=(0, 12),
        xycoords="axes fraction",
        textcoords="offset points",
        fontsize=9,
        color=colors["text_secondary"],
        va="bottom",
        linespacing=1.4,
    )
    fig.tight_layout()
    return fig


def load_item_scores(path: Path, summary: pd.DataFrame) -> pd.DataFrame:
    """Load per-item plot coordinates and check their coverage against the summary."""
    scored = pd.read_csv(path)
    keys = ["run_id", "model"]
    required = {*keys, "benchmark_item_id", "item_f1", "nondecodable_ratio"}
    missing = sorted(required - set(scored.columns))
    if missing:
        raise ValueError(
            f"{path} is missing columns: {', '.join(missing)}. "
            "Re-run the finished-model analysis into a new directory."
        )
    if not {*keys, "n_scored"}.issubset(summary.columns):
        raise ValueError(
            "Item panels require run_id, model and n_scored in the summary"
        )
    scored = scored.merge(summary[keys], on=keys, how="inner", validate="many_to_one")
    if scored[[*keys, "benchmark_item_id"]].isna().any().any():
        raise ValueError("Per-item scores must have non-null model/item identities")
    if scored.duplicated([*keys, "benchmark_item_id"]).any():
        raise ValueError("Per-item scores contain duplicate model/item rows")
    for column in ["item_f1", "nondecodable_ratio"]:
        scored[column] = pd.to_numeric(scored[column], errors="raise")
        values = scored[column].dropna() if column == "item_f1" else scored[column]
        if not values.between(0, 1).all():
            raise ValueError(f"{column} must be between 0 and 1")
    counts = scored.groupby(keys).size()
    for row in summary.itertuples(index=False):
        if counts.get((row.run_id, row.model), 0) != row.n_scored:
            raise ValueError(
                f"Per-item coverage differs from the summary for {row.model}"
            )
    return scored


def plot_item_f1_panels(
    scored: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    theme: str = "light",
    ncols: int = 5,
) -> plt.Figure:
    """Plot one point per defined item F1, on shared axes in summary order."""
    if ncols < 1 or summary.empty:
        raise ValueError(
            "Panels require a positive column count and at least one model"
        )
    colors = THEMES[theme]
    ncols = min(ncols, len(summary))
    nrows = (len(summary) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(max(11.5, 4.1 * ncols + 0.6), max(6.5, 3.1 * nrows + 2.0)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    fig.patch.set_facecolor(colors["surface"])
    max_ratio = scored["nondecodable_ratio"].max()
    xmax = max(0.05, float(max_ratio)) if pd.notna(max_ratio) else 0.05
    for ax, row in zip(axes.flat, summary.itertuples(index=False), strict=False):
        items = scored.loc[
            scored["run_id"].eq(row.run_id) & scored["model"].eq(row.model)
        ]
        points = items.dropna(subset=["item_f1"])
        ax.set_facecolor(colors["surface"])
        ax.scatter(
            points["nondecodable_ratio"],
            points["item_f1"],
            s=9,
            alpha=0.3,
            color=colors["f1"],
            edgecolors="none",
            zorder=3,
        )
        name = textwrap.fill(row.model, width=34, break_long_words=False)
        ax.set_title(
            f"{name}\nn={len(points):,} · undefined F1={len(items) - len(points):,}",
            loc="left",
            fontsize=9,
            color=colors["text_primary"],
            pad=9,
        )
        if points.empty:
            ax.text(
                0.5,
                0.5,
                "No defined item F1",
                transform=ax.transAxes,
                ha="center",
                color=colors["text_secondary"],
                fontsize=9,
            )
        ax.set_xlim(-0.025 * xmax, 1.025 * xmax)
        ax.set_ylim(-0.035, 1.035)
        ax.set_xticks([0, xmax / 2, xmax])
        ax.set_yticks([0, 0.5, 1])
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.tick_params(colors=colors["text_secondary"], labelsize=8, length=0, pad=5)
        ax.grid(color=colors["grid"], linewidth=0.7)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color(colors["grid"])
    for ax in list(axes.flat)[len(summary) :]:
        fig.delaxes(ax)
    # Keep tick labels on each occupied bottom panel, including incomplete rows.
    for index, ax in enumerate(fig.axes):
        if index + ncols >= len(summary):
            ax.tick_params(labelbottom=True)
    fig.suptitle(
        "Item F1 versus non-decodable word proportion",
        x=0.065,
        y=0.98,
        ha="left",
        fontsize=16,
        color=colors["text_primary"],
    )
    fig.text(
        0.065,
        0.95,
        "One panel per finished model · one point per scored item with defined F1 "
        "· shared axes",
        ha="left",
        fontsize=10,
        color=colors["text_secondary"],
    )
    fig.supxlabel(
        "Gold non-decodable words / all unique words in the item",
        y=0.06,
        fontsize=11,
        color=colors["text_secondary"],
    )
    fig.supylabel("Item F1", x=0.015, fontsize=11, color=colors["text_secondary"])
    fig.text(
        0.065,
        0.018,
        "Empty gold + empty prediction: undefined F1, omitted. "
        "False alarms on fully decodable items: F1=0, included.\n"
        "Failed or unparseable responses are excluded. "
        "Points can overlap; no jitter is applied.",
        ha="left",
        fontsize=9,
        color=colors["text_secondary"],
        linespacing=1.5,
    )
    fig.subplots_adjust(
        left=0.065, right=0.985, bottom=0.115, top=0.875, hspace=0.55, wspace=0.18
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="summary_by_model.csv to plot (default: newest local analysis)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="New PNG path (default: a name based on the plot beside the summary CSV)",
    )
    parser.add_argument(
        "--plot", choices=["leaderboard", "item-panels"], default="leaderboard"
    )
    parser.add_argument(
        "--scored-csv",
        type=Path,
        help="Per-item CSV for panels "
        "(default: results_with_scoring.csv beside the summary)",
    )
    parser.add_argument("--panel-columns", type=int, default=5)
    parser.add_argument(
        "--metric",
        choices=sorted(METRICS),
        default=None,
        help="Metric shown by each bar (default: --sort-by, or pooled_f1)",
    )
    parser.add_argument(
        "--sort-by",
        choices=sorted(METRICS),
        default=None,
        help="Optional ordering metric (default: the displayed metric)",
    )
    parser.add_argument("--theme", choices=sorted(THEMES), default="light")
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="add a right-hand column with each model's score",
    )
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    summary_csv = args.summary_csv or latest_summary_csv()
    panels = args.plot == "item-panels"
    if panels and args.panel_columns < 1:
        parser.error("--panel-columns must be positive")
    if panels and args.metric is not None:
        parser.error(
            "--metric selects leaderboard bars; item panels always show item F1"
        )
    metric = (
        "item_averaged_f1" if panels else args.metric or args.sort_by or "pooled_f1"
    )
    summary = load_summary(
        summary_csv,
        args.sort_by or metric,
        metric=metric,
        include_undefined=panels,
    )
    filename = (
        f"item_f1_vs_nondecodable_ratio_{args.theme}.png"
        if panels
        else f"leaderboard_{metric}_{args.theme}.png"
    )
    output = args.output or summary_csv.parent / filename
    if output.exists():
        parser.error(f"Output already exists: {output}. Choose a new --output path.")

    if panels:
        scored = load_item_scores(
            args.scored_csv or summary_csv.with_name("results_with_scoring.csv"),
            summary,
        )
        fig = plot_item_f1_panels(
            scored, summary, theme=args.theme, ncols=args.panel_columns
        )
    else:
        subtitle = (
            f"{len(summary)} finished models · scorable responses only · "
            f"{interval_note(summary, metric, summary_csv)}"
        )
        if metric == "item_averaged_f1":
            subtitle += (
                " · Equal item weights; empty/empty items have undefined F1 "
                "and are omitted"
            )
        elif metric == "pooled_f2":
            subtitle += " · Missed non-decodable words carry 4× the false-alarm penalty"
        fig = plot_metrics(
            summary,
            theme=args.theme,
            annotate=args.annotate,
            title=f"Decodability leaderboard: {METRICS[metric]}",
            subtitle=subtitle,
            metric=metric,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("xb") as handle:
            fig.savefig(
                handle, format="png", dpi=args.dpi, facecolor=fig.get_facecolor()
            )
    finally:
        plt.close(fig)

    columns = ["model", metric, f"{metric}_ci_low", f"{metric}_ci_high", "n_scored"]
    table = summary[[column for column in columns if column in summary]]
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nRead {summary_csv}\nWrote {output}")


if __name__ == "__main__":
    main()
