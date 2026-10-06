#!/usr/bin/env python3
"""
Paper-quality figures for the Linguistic Knowledge Benchmark.

Two views of the same quantity -- the short-form-vs-MCQ difficulty gap, signed as
SHORT-FORM MINUS MCQ so that it is NEGATIVE wherever short-form is the harder format
(the usual case). Both order models left to right by their ENGLISH overall accuracy
(best first), so the reader can see whether the gap is an artefact of weak models or
holds along the whole range.

    mcq-short-gap        Per model, the difference between short-form accuracy and
                         multiple-choice accuracy (in percentage points), for BOTH
                         languages on ONE axis: two markers per model joined by a
                         hairline.

    mcq-short-gap-bars   The same differences as BARS, in a 2x1 grid of small
                         multiples -- one panel per language, Kiswahili on top and
                         English at the bottom, sharing both the x order and the y
                         limits so the two panels (and their medians) are directly
                         comparable.

Why a paired-dot plot and not grouped bars: the quantity of interest is a signed
difference against a meaningful zero, for 59 models x 2 languages. Bars from a zero
baseline at that density read as noise; two markers per model joined by a hairline
keep the pairing visible and let the zero line do the work. Marker SHAPE doubles the
colour channel (circle = English, diamond = Kiswahili) so the figure survives
greyscale printing and colour-vision deficiency.

Inputs: the two per-language judge-bench summary CSVs (the `_fallback` ones, so the
MCQ accuracies include the rows rescued by the fallback judge). Both are located from
--experiment-id-english / --experiment-id-kiswahili, exactly like analyze_results.py locates its own
inputs; pass --summary-english / --summary-kiswahili to point at arbitrary files.

Every figure is written alongside a CSV of the plotted values (the table-view twin),
so no number in the figure is reachable only by reading pixels.

Run:
    uv run python benchmarks/linguistic_knowledge_benchmark/scripts/paper_visualizations.py mcq-short-gap
    uv run python .../paper_visualizations.py mcq-short-gap --format pdf --outdir /tmp/figs
    uv run python .../paper_visualizations.py mcq-short-gap \
        --summary-english results/summary_linguistic-knowledge_english_final-137-items_
        .csv \
        --summary-kiswahili results/summary_linguistic-knowledge_kiswahili_final-135-items_fallback.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from analyze_results import (
    _labels,
    _provider_legend_handles,
    load_display_names,
    load_provider_colors,
    load_summary,
)

# compute_frontier is the raw-Pareto frontier the leaderboard itself draws. Reused
# rather than reimplemented so the paper's frontier line and the website's are the
# same curve by construction.
from fab_benchmark_publisher.frontier import compute_frontier
from linguistic_knowledge_benchmark.shared_utils import MODELS_CSV, PROVIDERS_CSV
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# --------------------------------------------------------------------------- #
# Design tokens (see the dataviz palette; validated for CVD in light mode)
# --------------------------------------------------------------------------- #
SURFACE = "#fcfcfb"  # chart surface -- also the colour of marker rings
INK_PRIMARY = "#0b0b0b"  # titles
INK_SECONDARY = "#52514e"  # axis titles, annotations
INK_MUTED = "#898781"  # tick labels, connectors
GRIDLINE = "#e1e0d9"  # hairline grid, solid
BASELINE = "#c3c2b7"  # axis rules

# Categorical slots 1 and 2. Validated together (adjacent CVD dE 24.7, normal 33.6,
# both >= 3:1 on the surface) -- do not substitute without re-running the validator.
SERIES = {
    "english": {"color": "#2a78d6", "marker": "o", "label": "English"},
    "kiswahili": {"color": "#eb6834", "marker": "D", "label": "Kiswahili"},
}

# Default experiment-ids: the released experiments, scored with the MCQ fallback pass.
DEFAULT_EXPERIMENT_ID = {
    "english": "final-137-items_fallback",
    "kiswahili": "final-135-items_fallback",
}

LANGUAGE_LABEL = {"english": "English", "kiswahili": "Kiswahili"}

# Blended $/1M tokens = 0.75*input + 0.25*output (a 3:1 input:output mix). Identical to
# results_for_website.py's blend and to the CDPK leaderboard's, so a cost read off this
# figure matches the one on the website.
IO_PRICE_RATIO = (0.75, 0.25)

# The accuracy-vs-X scatters: one entry per x variable, so both figures are the same
# code path with different columns.
COST_SIZE_AXES = {
    "cost": {
        "column": "blended_price",
        "label": "Cost ($/1M tokens)",
        "title": "Accuracy vs Cost",
        "tick_fmt": lambda v, _: f"{v:g}" if v >= 0.01 else f"{v:g}",
    },
    "size": {
        "column": "size",
        "label": "Model size (# parameters)",
        "title": "Accuracy vs Model size",
        "tick_fmt": lambda v, _: _si_params(v),
    },
}

# The frontier staircase: dashed, and each segment takes the colour of the model at its
# top, so the line reads as "this provider owns the frontier through here".
FRONTIER_DASH = (0, (5, 3))
FRONTIER_WIDTH = 1.6

_RULE = "=" * 70


def _section(title: str) -> None:
    print(f"\n{_RULE}\n{title}\n{_RULE}")


def _kv(label: str, value) -> None:
    print(f"  {label:<30}: {value}")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def summary_path(root: str, language: str, experiment_id: str) -> str:
    return os.path.join(
        root, "results", f"summary_linguistic-knowledge_{language}_{experiment_id}.csv"
    )


def build_gap_table(
    summary_english: str, summary_kiswahili: str, models_csv: str
) -> pd.DataFrame:
    """One row per model present in BOTH experiments, ordered by English overall accuracy
    (best first), with each language's short-form - MCQ gap in percentage points.

    The subtraction runs short-form MINUS multiple-choice, so the gap is NEGATIVE where
    short-form items are the harder format -- which is the usual case. Signing it this
    way keeps the sign of the number and the direction of the effect aligned: a bar
    below zero means accuracy was lost on the harder format.

    Ordering on English accuracy is the point of the figure: it puts the x-axis on an
    axis of model strength, so a reader can see whether the gap is an artefact of weak
    models (it is not -- it holds along the whole range).
    """
    per_language = {}
    for language, path in (
        ("english", summary_english),
        ("kiswahili", summary_kiswahili),
    ):
        if not os.path.exists(path):
            raise SystemExit(
                f"Missing {language} summary CSV: {path}\n"
                "Point at it with --summary-english/--summary-kiswahili, or pass the "
                "--experiment-id-<language> that produced it."
            )
        df = load_summary(path)
        missing = [
            c
            for c in ("model", "accuracy", "accuracy_mcq", "accuracy_short_form")
            if c not in df.columns
        ]
        if missing:
            raise SystemExit(f"{path} is missing column(s) {missing}.")
        per_language[language] = df.set_index("model")

    english, kiswahili = per_language["english"], per_language["kiswahili"]
    common = english.index.intersection(kiswahili.index)
    dropped = sorted(set(english.index).symmetric_difference(kiswahili.index))
    if dropped:
        print(
            f"\n({len(dropped)} model(s) not in both experiments -- excluded: "
            f"{', '.join(dropped)})"
        )

    table = pd.DataFrame(
        {
            "model": common,
            "accuracy_english": english.loc[common, "accuracy"].to_numpy(),
            "accuracy_kiswahili": kiswahili.loc[common, "accuracy"].to_numpy(),
            "mcq_english": english.loc[common, "accuracy_mcq"].to_numpy(),
            "short_form_english": english.loc[common, "accuracy_short_form"].to_numpy(),
            "mcq_kiswahili": kiswahili.loc[common, "accuracy_mcq"].to_numpy(),
            "short_form_kiswahili": kiswahili.loc[
                common, "accuracy_short_form"
            ].to_numpy(),
        }
    )
    table["gap_english"] = table["short_form_english"] - table["mcq_english"]
    table["gap_kiswahili"] = table["short_form_kiswahili"] - table["mcq_kiswahili"]

    display_names = load_display_names(models_csv)
    table["display_name"] = _labels(table["model"].tolist(), display_names)

    return table.sort_values("accuracy_english", ascending=False).reset_index(drop=True)


def _si_params(value: float) -> str:
    """1e9 -> '1B', 3.5e10 -> '35B', 7e8 -> '700M'. Log-decade tick labels."""
    for scale, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if value >= scale:
            return f"{value / scale:g}{suffix}"
    return f"{value:g}"


def _draw_frontier_steps(
    ax, frontier: pd.DataFrame, x_column: str, provider_color: dict, x_max: float
) -> None:
    """Draw the Pareto frontier as a staircase, coloured by the model on top.

    For each frontier model i: a HORIZONTAL run at its accuracy, from its own x out to
    the next frontier model's x, in model i's provider colour -- i is the best thing
    available across that whole span, so it owns the segment. Then a VERTICAL riser at
    x_{i+1} from y_i up to y_{i+1}, in model i+1's colour, because that is the model
    the step arrives at.

    A staircase rather than straight segments between points: the frontier's claim is
    "nothing better exists at or below this x", which is a step function. Joining the
    points diagonally would imply intermediate models that do not exist. The last run
    extends to `x_max` because the final model is the best at any price on the chart.
    """
    xs = frontier[x_column].to_numpy()
    ys = frontier["accuracy"].to_numpy()
    colors = [provider_color.get(p, "#4e79a7") for p in frontier["provider"]]
    for i in range(len(xs)):
        x_end = xs[i + 1] if i + 1 < len(xs) else x_max
        if x_end > xs[i]:
            ax.plot(
                [xs[i], x_end],
                [ys[i], ys[i]],
                color=colors[i],
                linewidth=FRONTIER_WIDTH,
                linestyle=FRONTIER_DASH,
                zorder=2,
            )
        if i + 1 < len(xs):
            ax.plot(
                [xs[i + 1], xs[i + 1]],
                [ys[i], ys[i + 1]],
                color=colors[i + 1],
                linewidth=FRONTIER_WIDTH,
                linestyle=FRONTIER_DASH,
                zorder=2,
            )


def build_cost_size_table(
    summary: str, models_csv: str, providers_csv: str
) -> pd.DataFrame:
    """One row per scored model: accuracy, blended price, parameter count, provider.

    Cost and size come from models.csv, so a model is plottable on an axis only
    where that column is populated: closed-weight models rarely publish a parameter
    count, which is why the size figure carries far fewer points than the cost one.
    Rather than silently dropping them here, both columns are kept as NaN and each
    figure filters to its own axis, reporting how many it dropped.
    """
    if not os.path.exists(summary):
        raise SystemExit(
            f"Missing summary CSV: {summary}\n"
            "Point at it with --summary, or pass the --experiment-id that produced it."
        )
    df = load_summary(summary)
    for column in ("model", "accuracy"):
        if column not in df.columns:
            raise SystemExit(f"{summary} is missing column '{column}'.")

    meta = pd.read_csv(models_csv).set_index("model_id")
    known = df[df["model"].isin(meta.index)]
    unknown = sorted(set(df["model"]) - set(meta.index))
    if unknown:
        print(
            f"\n({len(unknown)} model(s) not in {os.path.basename(models_csv)} -- "
            f"excluded: {', '.join(unknown)})"
        )

    rows = meta.loc[known["model"]]
    inp, out = rows["input_cost"].to_numpy(), rows["output_cost"].to_numpy()
    model_provider, provider_color = load_provider_colors(models_csv, providers_csv)
    table = pd.DataFrame(
        {
            "model": known["model"].to_numpy(),
            "accuracy": known["accuracy"].to_numpy(),
            "blended_price": IO_PRICE_RATIO[0] * inp + IO_PRICE_RATIO[1] * out,
            "size": rows["size"].to_numpy(),
            "provider": [model_provider.get(m, "Unknown") for m in known["model"]],
        }
    )
    table["display_name"] = _labels(
        table["model"].tolist(), load_display_names(models_csv)
    )
    table.attrs["provider_color"] = provider_color
    table.attrs["model_provider"] = model_provider
    return table.sort_values("accuracy", ascending=False).reset_index(drop=True)


def plot_accuracy_vs(
    table: pd.DataFrame,
    out_path: str,
    *,
    axis: dict,
    language: str,
    dpi: int = 300,
) -> pd.DataFrame:
    """Accuracy against a log-scaled cost or size axis, with the Pareto frontier drawn.

    Returns the plotted subset (the CSV twin), which is smaller than `table` wherever
    the x column is missing.

    The frontier is the RAW Pareto set -- the best accuracy available at or below each
    x value -- computed with the same function the website uses, so it passes through
    real models and matches the leaderboard's own curve. See _draw_frontier_steps() for
    how it is drawn.
    """
    column = axis["column"]
    plotted = table[table[column].notna() & (table[column] > 0)].copy()
    dropped = len(table) - len(plotted)
    if plotted.empty:
        raise SystemExit(f"ERROR: no model has a usable '{column}' value to plot.")
    if dropped:
        print(f"  ({dropped} model(s) with no {column} -- not plotted)")

    provider_color = table.attrs["provider_color"]
    model_provider = table.attrs["model_provider"]

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    frontier = compute_frontier(
        plotted, acc_col="accuracy", sorting_col=column, smooth=False
    )
    _draw_frontier_steps(
        ax, frontier, column, provider_color, float(plotted[column].max())
    )
    ax.scatter(
        plotted[column],
        plotted["accuracy"],
        s=42,
        c=[provider_color.get(p, "#4e79a7") for p in plotted["provider"]],
        edgecolors=SURFACE,
        linewidths=0.9,
        zorder=3,
    )

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(axis["tick_fmt"]))
    # Decade rules, dashed: on a log axis the reader needs the order of magnitude, and
    # dashing keeps them from competing with the frontier line.
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8, linestyle=(0, (4, 4)))
    ax.set_xlabel(axis["label"], fontsize=10, color=INK_SECONDARY, labelpad=6)
    ax.set_ylabel("Accuracy (%)", fontsize=10, color=INK_SECONDARY, labelpad=6)
    ax.set_ylim(0, 100)
    _style_axes(ax)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8, linestyle=(0, (4, 4)))

    ax.set_title(
        f"{LANGUAGE_LABEL.get(language, language.capitalize())} Linguistic Knowledge "
        f"Benchmark\n{axis['title']}",
        fontsize=12,
        fontweight="semibold",
        color=INK_PRIMARY,
        pad=12,
        loc="left",
    )

    handles = _provider_legend_handles(
        plotted["model"].tolist(), model_provider, provider_color
    )
    if handles:
        legend = ax.legend(
            handles=handles,
            title="Provider",
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=False,
            fontsize=8,
            title_fontsize=9,
        )
        legend.get_title().set_color(INK_SECONDARY)
        for text in legend.get_texts():
            text.set_color(INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    plotted["on_frontier"] = plotted["model"].isin(frontier["model"])
    return plotted


def gap_stats(table: pd.DataFrame) -> dict:
    """Headline numbers the figure annotates itself with (and the caption can quote)."""
    stats = {"n_models": len(table)}
    for language in SERIES:
        gaps = table[f"gap_{language}"].dropna()
        stats[language] = {
            "median": float(gaps.median()),
            "mean": float(gaps.mean()),
            "min": float(gaps.min()),
            "max": float(gaps.max()),
            # The gap is short-form MINUS MCQ, so "short-form is harder" is the
            # NEGATIVE case -- that is the count worth reporting.
            "n_negative": int((gaps < 0).sum()),
            "n": int(len(gaps)),
        }
    both = table[["gap_english", "gap_kiswahili"]].dropna()
    stats["n_negative_both"] = int(
        ((both["gap_english"] < 0) & (both["gap_kiswahili"] < 0)).sum()
    )
    stats["n_both"] = int(len(both))
    return stats


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def _style_axes(ax) -> None:
    """Hairline solid grid on the value axis only; recessive spines and ticks."""
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, solid_capstyle="butt")
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK_MUTED, length=3, width=0.8)


def plot_mcq_short_gap(
    table: pd.DataFrame,
    stats: dict,
    out_path: str,
    *,
    dpi: int = 300,
) -> None:
    """Paired-dot plot of the MCQ - short-form gap, one pair of markers per model."""
    n = len(table)
    positions = list(range(n))
    labels = table["display_name"].tolist()

    fig, ax = plt.subplots(figsize=(max(11.0, 0.21 * n), 5.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # (1) hairline connector per model -- makes the pair read as one observation.
    for i, row in table.iterrows():
        ax.plot(
            [i, i],
            [row["gap_english"], row["gap_kiswahili"]],
            color=INK_MUTED,
            linewidth=0.8,
            alpha=0.55,
            zorder=1,
            solid_capstyle="round",
        )

    # (2) the zero baseline -- the only reference the reader must not miss.
    ax.axhline(0, color=INK_SECONDARY, linewidth=1.2, zorder=2)

    # (3) the markers. Shape carries identity alongside hue (greyscale/CVD safe);
    #     a 2px surface ring keeps overlapping pairs legible.
    for language, spec in SERIES.items():
        ax.plot(
            positions,
            table[f"gap_{language}"],
            linestyle="none",
            marker=spec["marker"],
            markersize=6.0,
            color=spec["color"],
            markeredgecolor=SURFACE,
            markeredgewidth=1.4,
            zorder=3,
            label=spec["label"],
        )

    # (4) per-language median, dashed, direct-labelled in the right margin. Dashing is
    #     reserved for these thresholds -- the grid stays solid.
    for language, spec in SERIES.items():
        median = stats[language]["median"]
        ax.axhline(
            median,
            color=spec["color"],
            linewidth=1.3,
            linestyle=(0, (5, 3)),
            alpha=0.85,
            zorder=2,
        )
        ax.annotate(
            f"{spec['label']} median {median:+.1f} pp",
            xy=(1.0, median),
            xycoords=("axes fraction", "data"),
            xytext=(6, 2),
            textcoords="offset points",
            fontsize=8,
            color=INK_SECONDARY,
            ha="left",
            va="bottom",
            annotation_clip=False,
        )

    # (5) breathing room above/below the extremes, so no marker sits on a spine.
    lo, hi = ax.get_ylim()
    pad = 0.08 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)

    # (6) chrome. Which side of zero means what rides on the axis title rather than
    #     a floating annotation -- at this density a rotated note collides with the
    #     tick labels.
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
    ax.set_xlim(-0.8, n - 0.2)
    ax.set_ylabel(
        "Short-form − multiple-choice accuracy (percentage points)\n"
        "below 0 = short-form harder",
        fontsize=9.5,
        color=INK_SECONDARY,
        labelpad=8,
    )
    ax.set_xlabel(
        "Models, ordered by English overall accuracy (best first)",
        fontsize=9.5,
        color=INK_SECONDARY,
        labelpad=6,
    )
    _style_axes(ax)

    # Descriptive only -- the interpretation belongs in the paper's figure caption.
    ax.set_title(
        "Per-model difference between short-form and multiple-choice accuracy, "
        "English and Kiswahili",
        fontsize=12,
        fontweight="semibold",
        color=INK_PRIMARY,
        pad=12,
        loc="left",
    )
    # No subtitle: the counts and medians go in the paper's figure caption instead
    # (they are printed to the console and written to the CSV twin).

    # A legend is mandatory for two series; marker shape is repeated in the key.
    handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=spec["marker"],
            markersize=6.0,
            color=spec["color"],
            markeredgecolor=SURFACE,
            markeredgewidth=1.4,
            label=spec["label"],
        )
        for spec in SERIES.values()
    ]
    legend = ax.legend(
        handles=handles,
        loc="upper right",
        frameon=False,
        fontsize=9,
        handletextpad=0.5,
        borderaxespad=0.2,
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_mcq_short_gap_bars(
    table: pd.DataFrame,
    stats: dict,
    out_path: str,
    *,
    dpi: int = 300,
) -> None:
    """Small multiples: one bar panel per language, stacked 2x1 with English at the
    bottom, sharing both scales.

    Same data as plot_mcq_short_gap(), read a different way. The paired-dot version
    makes the per-model pairing legible; bars off a common zero line make the SIZE of
    each gap and the balance above/below zero easier to read per language. Sharing the
    x order (English overall accuracy) and the y limits is what makes the two panels
    comparable -- so the reader can compare the two medians directly by their height.
    """
    n = len(table)
    positions = list(range(n))
    labels = table["display_name"].tolist()

    # English at the BOTTOM, so panels are ordered Kiswahili -> English top to bottom.
    panel_order = ["kiswahili", "english"]

    # Constrained layout, not tight_layout: it is the one that reserves space for the
    # figure-level suptitle/supylabel instead of letting them land on the tick labels.
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(max(11.0, 0.21 * n), 7.6),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    fig.patch.set_facecolor(SURFACE)

    # One shared y range over BOTH languages, padded -- the panels are only comparable
    # if the scale is identical, so it is computed once here rather than per panel.
    gaps = table[["gap_english", "gap_kiswahili"]]
    low, high = float(gaps.min().min()), float(gaps.max().max())
    pad = 0.10 * (high - low)

    for ax, language in zip(axes, panel_order):
        spec = SERIES[language]
        values = table[f"gap_{language}"]
        ax.set_facecolor(SURFACE)

        # Thin bars with air in the band; they grow from the shared zero baseline.
        ax.bar(positions, values, width=0.62, color=spec["color"], zorder=3)
        ax.axhline(0, color=INK_SECONDARY, linewidth=1.2, zorder=4)

        median = stats[language]["median"]
        ax.axhline(
            median,
            color=INK_PRIMARY,
            linewidth=1.3,
            linestyle=(0, (5, 3)),
            alpha=0.75,
            zorder=5,
        )
        ax.annotate(
            f"median {median:+.1f} pp",
            xy=(1.0, median),
            xycoords=("axes fraction", "data"),
            xytext=(6, 2),
            textcoords="offset points",
            fontsize=8,
            color=INK_SECONDARY,
            ha="left",
            va="bottom",
            annotation_clip=False,
        )

        # A single series per panel needs no legend -- the panel title names it.
        ax.set_title(
            spec["label"],
            fontsize=10.5,
            fontweight="semibold",
            color=INK_PRIMARY,
            loc="left",
            pad=6,
        )
        ax.set_ylim(low - pad, high + pad)
        ax.set_xlim(-0.8, n - 0.2)
        _style_axes(ax)

    axes[-1].set_xticks(positions)
    axes[-1].set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
    axes[-1].set_xlabel(
        "Models, ordered by English overall accuracy (best first)",
        fontsize=9.5,
        color=INK_SECONDARY,
        labelpad=6,
    )
    fig.supylabel(
        "Short-form − multiple-choice accuracy (percentage points)\n"
        "below 0 = short-form harder",
        fontsize=9.5,
        color=INK_SECONDARY,
    )
    # Descriptive only -- the interpretation belongs in the paper's figure caption.
    fig.suptitle(
        "Per-model difference between short-form and multiple-choice accuracy, "
        "by language",
        fontsize=12,
        fontweight="semibold",
        color=INK_PRIMARY,
    )

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def _resolve_summaries(args: argparse.Namespace, root: str) -> tuple[str, str]:
    """(english, kiswahili) summary CSV paths, from --summary-* or the experiment-ids."""
    return (
        args.summary_english
        or summary_path(root, "english", args.experiment_id_english),
        args.summary_kiswahili
        or summary_path(root, "kiswahili", args.experiment_id_kiswahili),
    )


def _print_gap_summary(table: pd.DataFrame, stats: dict) -> None:
    # ASCII only in console output -- Windows consoles default to cp1252.
    _section(
        "SHORT-FORM MINUS MCQ GAP (percentage points; negative = short-form harder)"
    )
    for language, spec in SERIES.items():
        s = stats[language]
        print(
            f"  {spec['label']:<10} median {s['median']:+5.1f}   mean {s['mean']:+5.1f} "
            f"  range [{s['min']:+5.1f}, {s['max']:+5.1f}]   "
            f"negative {s['n_negative']}/{s['n']}"
        )
    print(
        f"\n  negative in BOTH languages: {stats['n_negative_both']}/{stats['n_both']}"
    )
    reversed_rows = table[(table["gap_english"] >= 0) | (table["gap_kiswahili"] >= 0)]
    if len(reversed_rows):
        print("\n  models where the gap does NOT hold in one language:")
        for _, r in reversed_rows.iterrows():
            print(
                f"    {r['display_name']:<26} English {r['gap_english']:+5.1f}   "
                f"Kiswahili {r['gap_kiswahili']:+5.1f}"
            )


def filter_to_default_models(table: pd.DataFrame, models_json: str) -> pd.DataFrame:
    """Restrict a gap table to the models the leaderboard pre-selects.

    `models_json` is a benchmark_default_models.json written by
    results_for_website.py: benchmark_domain -> sorted model ids. This figure is
    cross-language, so it keeps a model that is a default in EITHER language (the union
    of LK-en and LK-sw) -- dropping a model that is a default in one panel but not the
    other would leave that panel's bar missing for no reason a reader could see. On the
    current lists the distinction is moot: the English set is a subset of the Kiswahili
    one, so union and intersection give the same models.
    """
    with open(models_json, encoding="utf-8") as fh:
        by_domain = json.load(fh)
    keep = {m for models in by_domain.values() for m in models}
    if not keep:
        raise SystemExit(f"ERROR: {models_json} lists no models")
    out = table[table["model"].isin(keep)].copy()
    if out.empty:
        raise SystemExit(
            f"ERROR: none of the {len(keep)} models in {models_json} appear in both "
            "experiments -- is it the right pair of experiment-ids?"
        )
    print(
        f"\nFiltered to {len(out)}/{len(table)} models "
        f"({os.path.basename(os.path.dirname(models_json))})"
    )
    return out.reset_index(drop=True)


def _run_gap_figure(
    args: argparse.Namespace, root: str, *, stem_prefix: str, plot
) -> None:
    """Shared body of both gap figures: locate inputs, build the table, plot, and
    write the CSV twin. `plot` is the panel-drawing function to use."""
    summary_english, summary_kiswahili = _resolve_summaries(args, root)

    _section("CONFIG")
    _kv("english summary", os.path.relpath(summary_english, root))
    _kv("kiswahili summary", os.path.relpath(summary_kiswahili, root))
    _kv("outdir", args.outdir)

    table = build_gap_table(summary_english, summary_kiswahili, args.models_csv)
    if args.models_json:
        _kv("models_json", os.path.relpath(args.models_json, root))
        table = filter_to_default_models(table, args.models_json)
    stats = gap_stats(table)
    _print_gap_summary(table, stats)

    os.makedirs(args.outdir, exist_ok=True)
    stem = f"{stem_prefix}_{args.label}"
    fig_path = os.path.join(args.outdir, f"{stem}.{args.format}")
    csv_path = os.path.join(args.outdir, f"{stem}.csv")

    plot(table, stats, fig_path, dpi=args.dpi)
    # The table-view twin: every plotted value, readable without the figure.
    table[
        [
            "model",
            "display_name",
            "accuracy_english",
            "mcq_english",
            "short_form_english",
            "gap_english",
            "accuracy_kiswahili",
            "mcq_kiswahili",
            "short_form_kiswahili",
            "gap_kiswahili",
        ]
    ].round(2).to_csv(csv_path, index=False)

    print(f"\nWrote:\n  {fig_path}\n  {csv_path}")


def run_mcq_short_gap(args: argparse.Namespace, root: str) -> None:
    _run_gap_figure(args, root, stem_prefix="mcq_short_gap", plot=plot_mcq_short_gap)


def run_mcq_short_gap_bars(args: argparse.Namespace, root: str) -> None:
    _run_gap_figure(
        args, root, stem_prefix="mcq_short_gap_bars", plot=plot_mcq_short_gap_bars
    )


def _run_cost_size_figure(args: argparse.Namespace, root: str, *, which: str) -> None:
    """Shared body of the two accuracy-vs-X scatters (cost and size)."""
    axis = COST_SIZE_AXES[which]
    summary = args.summary or summary_path(root, args.language, args.experiment_id)

    _section("CONFIG")
    _kv("language", args.language)
    _kv("summary", os.path.relpath(summary, root))
    _kv("x axis", axis["label"])
    _kv("outdir", args.outdir)

    table = build_cost_size_table(summary, args.models_csv, args.providers_csv)
    if args.models_json:
        _kv("models_json", os.path.relpath(args.models_json, root))
        table = filter_to_default_models(table, args.models_json)

    os.makedirs(args.outdir, exist_ok=True)
    stem = f"accuracy_vs_{which}_{args.language}_{args.label}"
    fig_path = os.path.join(args.outdir, f"{stem}.{args.format}")
    csv_path = os.path.join(args.outdir, f"{stem}.csv")

    plotted = plot_accuracy_vs(
        table, fig_path, axis=axis, language=args.language, dpi=args.dpi
    )

    _section(f"PARETO FRONTIER ({axis['label']})")
    for _, r in plotted[plotted["on_frontier"]].sort_values(axis["column"]).iterrows():
        x = r[axis["column"]]
        shown = f"{x:,.4g}" if which == "cost" else _si_params(x)
        print(f"  {r['display_name']:<30} {shown:>12}   {r['accuracy']:5.1f}%")

    plotted[
        ["model", "display_name", "provider", "accuracy", axis["column"], "on_frontier"]
    ].round(4).to_csv(csv_path, index=False)
    print(f"\nWrote:\n  {fig_path}\n  {csv_path}")


def run_accuracy_vs_cost(args: argparse.Namespace, root: str) -> None:
    _run_cost_size_figure(args, root, which="cost")


def run_accuracy_vs_size(args: argparse.Namespace, root: str) -> None:
    _run_cost_size_figure(args, root, which="size")


FIGURES = {
    "mcq-short-gap": run_mcq_short_gap,
    "mcq-short-gap-bars": run_mcq_short_gap_bars,
    "accuracy-vs-cost": run_accuracy_vs_cost,
    "accuracy-vs-size": run_accuracy_vs_size,
}


def build_parser(root: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--experiment-id-english",
        default=DEFAULT_EXPERIMENT_ID["english"],
        help=f"experiment-id of the English summary CSV (default {DEFAULT_EXPERIMENT_ID['english']})",
    )
    common.add_argument(
        "--experiment-id-kiswahili",
        default=DEFAULT_EXPERIMENT_ID["kiswahili"],
        help=f"experiment-id of the Kiswahili summary CSV "
        f"(default {DEFAULT_EXPERIMENT_ID['kiswahili']})",
    )
    common.add_argument(
        "--summary-english",
        default=None,
        help="explicit path to the English summary CSV (overrides --experiment-id-english)",
    )
    common.add_argument(
        "--summary-kiswahili",
        default=None,
        help="explicit path to the Kiswahili summary CSV (overrides --experiment-id-kiswahili)",
    )
    common.add_argument(
        "--outdir",
        default=os.path.join(root, "results", "paper_figures"),
        help="where the figure and its CSV twin are written",
    )
    common.add_argument(
        "--label", default="final", help="tag appended to the output filenames"
    )
    common.add_argument(
        "--format",
        default="png",
        choices=["png", "pdf", "svg"],
        help="figure format (pdf for LaTeX, png for drafts)",
    )
    common.add_argument("--dpi", type=int, default=300)
    common.add_argument(
        "--models-csv",
        dest="models_csv",
        default=str(MODELS_CSV),
        help="source of model display names, and of price/parameter-count for the "
        "accuracy-vs-cost / accuracy-vs-size figures",
    )
    common.add_argument(
        "--providers-csv",
        dest="providers_csv",
        default=str(PROVIDERS_CSV),
        help="maps provider -> marker colour",
    )
    common.add_argument(
        "--models-json",
        dest="models_json",
        default=None,
        help="plot only the models the leaderboard pre-selects, from a "
        "benchmark_default_models.json written by results_for_website.py (e.g. "
        "results/web/<timestamp>/benchmark_default_models.json). Each run folder has "
        "its own and they differ by curation setting, so name the one you mean. "
        "Remember to change --label too, or the filtered figure overwrites the "
        "all-models one.",
    )

    # The cost/size scatters are per-language, so they take --language/--experiment-id
    # rather than the cross-language pair the gap figures need.
    single = argparse.ArgumentParser(add_help=False)
    single.add_argument(
        "--language",
        default="kiswahili",
        choices=sorted(DEFAULT_EXPERIMENT_ID),
        help="which language's summary CSV to plot (default kiswahili)",
    )
    single.add_argument(
        "--experiment-id",
        default=None,
        help="experiment-id of that language's summary CSV (defaults to its entry in "
        "DEFAULT_EXPERIMENT_ID)",
    )
    single.add_argument(
        "--summary",
        default=None,
        help="explicit path to the summary CSV (overrides --language/--experiment-id)",
    )

    sub = ap.add_subparsers(dest="figure", required=True)
    sub.add_parser(
        "mcq-short-gap",
        parents=[common],
        help="per-model MCQ minus short-form accuracy gap, both languages on one axis",
    )
    sub.add_parser(
        "mcq-short-gap-bars",
        parents=[common],
        help="the same gap as bars, one panel per language (2x1, English at the "
        "bottom, shared scales)",
    )
    sub.add_parser(
        "accuracy-vs-cost",
        parents=[common, single],
        help="accuracy against blended $/1M tokens (log x), with the Pareto frontier",
    )
    sub.add_parser(
        "accuracy-vs-size",
        parents=[common, single],
        help="accuracy against parameter count (log x), with the Pareto frontier",
    )
    return ap


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    if here not in sys.path:  # analyze_results.py lives beside this file
        sys.path.insert(0, here)

    args = build_parser(root).parse_args()
    if getattr(args, "language", None) and args.experiment_id is None:
        args.experiment_id = DEFAULT_EXPERIMENT_ID[args.language]
    FIGURES[args.figure](args, root)


if __name__ == "__main__":
    main()
