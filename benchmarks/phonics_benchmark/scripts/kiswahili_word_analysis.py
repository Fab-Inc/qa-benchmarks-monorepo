"""Failure modes of models on the Kiswahili decodability benchmark

Usage, from repository root:
    uv run --frozen --package phonics_benchmark python \
        benchmarks/phonics_benchmark/scripts/kiswahili_word_analysis.py \
        --run outputs/phonics-relabelled-20261116 [--report-dir DIR]

The run folder needs reports/kiswahili/results_with_scoring.csv and the
hardware-tier outputs from ``report_on_benchmark_run.py --hardware-tiers``,
which choose the models each tier is averaged over (the same selection the
other hardware-tier error plots use).

Every item-word (one word in one benchmark item) is put in exactly one failure
mode: what, given the item's student profile, could make a model wrong about
it. Each mode holds words of a single label, so its error rate is a false
alarm rate (decodable modes) or a miss rate (nondecodable modes).

Outputs, in --report-dir (default: <run>/reports/kiswahili):
    failure_modes.csv          one row per (model, failure mode)
    failure_modes_by_tier.csv  one row per (hardware tier, failure mode), means
                               over each tier's top TOP_N models
    failure_modes.png          the tier table as a dot chart
    hallucination_summary.csv  one row per model: how often it predicts words
                               that are not in the item's vocabulary
"""

from __future__ import annotations

import ast
from pathlib import Path

import hardware_tier_errors as common
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from decodability.kiswahili import (
    KiswahiliStudentKnowledge,
    KiswahiliWordAnalysis,
    analyse_word_kiswahili,
)
from decodability.kiswahili.segment import (
    SYLLABIC_NASAL_EXPECTED_CLUSTERS,
    _consonant_runs,
    _nasal_verdict,
    decompose_word,
)
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from phonics_benchmark.hardware import CONNECTOR, GRID, INK, MUTED, SECONDARY, SURFACE

LANGUAGE = "kiswahili"

# Which hardware-tier outputs to read: the ranking metric, the coverage
# threshold they were made with (None for none), and N in the top-N selection
# file hardware_tiers_top{N}_models.csv. A tier with fewer than N eligible
# models has all of them selected.
METRIC = "adjusted_balanced_accuracy"
MIN_COVERAGE: float | None = None
TOP_N = 10

# The palette comes from phonics_benchmark.hardware. Tier colours are still this
# script's own; hardware_tier_errors.TIER_COLORS uses different ones.
TIER_ORDER = ["Cloud", "Mobile + Tablet"]
TIER_COLORS = {"Cloud": "#1f4e79", "Mobile + Tablet": "#b5651d"}
TIER_NAMES = {"Cloud": "Cloud", "Mobile + Tablet": "Mobile + tablet"}
# Below this many item-words, a mode is faded in the chart.
MIN_WORDS = 20

# Word length bins, in graphemes, for decodable words with no consonant run.
# 2-5 / 6-14 splits those item-words most evenly (12,094 / 11,084 on the
# 20261110 run); the next cut, 2-6 / 7-14, would split them 73 / 27.
GRAPHEME_BINS = pd.IntervalIndex.from_tuples([(1, 5), (5, 14)])


def length_label(span: pd.Interval) -> str:
    """'2-4' for the interval (1, 4]."""
    return f"{int(span.left) + 1}-{int(span.right)}"


def is_nasal(cluster: str) -> bool:
    """Nasal-initial cluster, excluding ng/ng'/ny, which are single sounds."""
    return cluster[:1] in "mn" and not cluster.startswith(("ng", "ng'", "ny"))


# --------------------------------------------------------------------------
# Failure modes
# --------------------------------------------------------------------------
# Rows in the order the chart draws them, grouped by the error that matters.
FAILURE_ROWS = {
    "Decodable words: share falsely flagged": [
        ("known whole word", "Known whole word"),
        *[
            (
                f"no consonant run, {length_label(b)} graphemes",
                f"No consonant run, {length_label(b)} graphemes",
            )
            for b in GRAPHEME_BINS
        ],
        ("taught cluster", "Every cluster taught"),
        ("syllabic nasal, m/n+C taught", "Syllabic nasal, m/n + C taught as a cluster"),
        ("syllabic nasal, m/n+C untaught", "Syllabic nasal, m/n + C not taught"),
    ],
    "Nondecodable words: share missed": [
        ("untaught unigraph", "Untaught single-letter grapheme"),
        ("untaught multigraph", "Untaught multigraph"),
        ("untaught non-nasal cluster", "Untaught non-nasal cluster, graphemes taught"),
        ("untaught nasal cluster", "Untaught nasal cluster, graphemes taught"),
        ("invalid grapheme", "Invalid grapheme"),
        ("several reasons", "More than one of the above"),
    ],
}


def failure_mode(
    analysis: KiswahiliWordAnalysis,
    taught_clusters: set[str],
    syllabic_onsets: list[tuple[str, str]],
    n_graphemes: int,
    has_run: bool,
    on_syllabic_list: bool,
) -> str:
    """The FAILURE_ROWS mode a (word, student) pair falls in.

    Nondecodable words are split by what the student has not been taught,
    with words untaught on more than one count kept apart so each other mode
    has a single cause. Decodable words are split by what could still trip a
    model: being a known whole word, a syllabic nasal, a taught cluster, or
    else only their length.

    A syllabic nasal is only a trap when the student has not been taught the
    nasal and the next consonant as a cluster (mt in mtoto): a model that
    misreads it as a cluster then wrongly finds it untaught. syllabic_onsets
    lists (nasal + next grapheme, whole run) for each syllabic run, and either
    being taught counts as the misreading being harmless.

    Nasal clusters include ambiguous boundaries, which the dataset labels as
    fused (gated).
    """
    if analysis.known_whole_word:
        return "known whole word"
    untaught = {
        "untaught unigraph": [
            g for g in analysis.unknown_graphemes if len(g.normalised) == 1
        ],
        "untaught multigraph": [
            g for g in analysis.unknown_graphemes if len(g.normalised) > 1
        ],
        "untaught nasal cluster": [
            c for c in analysis.unknown_clusters if is_nasal(c.normalised)
        ],
        "untaught non-nasal cluster": [
            c for c in analysis.unknown_clusters if not is_nasal(c.normalised)
        ],
        "invalid grapheme": analysis.invalid_graphemes,
    }
    reasons = [mode for mode, spans in untaught.items() if spans]
    if len(reasons) > 1:
        return "several reasons"
    if reasons:
        return reasons[0]
    # Every grapheme and cluster is taught from here on, so the word is decodable.
    if on_syllabic_list and syllabic_onsets:
        harmless = any(
            pair in taught_clusters or run in taught_clusters
            for pair, run in syllabic_onsets
        )
        return f"syllabic nasal, m/n+C {'taught' if harmless else 'untaught'}"
    if has_run:
        return "taught cluster"
    (span,) = [b for b in GRAPHEME_BINS if n_graphemes in b]
    return f"no consonant run, {length_label(span)} graphemes"


# --------------------------------------------------------------------------
# Prepare data
# --------------------------------------------------------------------------
def load_results(run: Path) -> pd.DataFrame:
    """Item-level results for tiered models, with the set columns parsed.

    Only models the tier file places are kept. Dropping them here rather than
    at each call site keeps the model set consistent: groupby discards rows
    with a NaN key, so a tally keyed on hardware_tier would lose the untiered
    models while one keyed on model alone would keep them.
    """
    # Same location hardware_tier_errors.load_scored_items reads from.
    results = pd.read_csv(run / "reports" / LANGUAGE / "results_with_scoring.csv")
    tiers = pd.read_csv(
        common.tier_dir(run, METRIC, MIN_COVERAGE) / f"hardware_tiers_{LANGUAGE}.csv"
    )
    results = results.merge(tiers[["model", "hardware_tier"]], on="model", how="inner")
    # parse_word_set reads the literal without executing it, unlike eval. The
    # response and label columns are written as lists; they become sets here.
    results["vocab_set"] = results.vocab_set.map(common.parse_word_set)
    for col in ["response_parsed", "nondecodable_words"]:
        results[col] = results[col].map(
            lambda cell: {w.lower() for w in common.parse_word_set(cell)}
        )
    return results


def hallucination_table(results: pd.DataFrame) -> pd.DataFrame:
    """Per model: items containing an out-of-vocabulary word, and the share of
    predicted words that are out of vocabulary."""
    frame = results.assign(
        n_hallucinated=[
            len(r - v) for r, v in zip(results.response_parsed, results.vocab_set)
        ],
        n_predicted=results.response_parsed.map(len),
    )
    frame["has_hallucination"] = frame.n_hallucinated > 0
    out = frame.groupby(["hardware_tier", "model"], observed=True).agg(
        n_items=("has_hallucination", "size"),
        items_with_hallucination=("has_hallucination", "sum"),
        n_predicted_words=("n_predicted", "sum"),
        n_hallucinated_words=("n_hallucinated", "sum"),
    )
    out["prop_items"] = out.items_with_hallucination / out.n_items
    out["prop_words"] = out.n_hallucinated_words / out.n_predicted_words.replace(
        0, np.nan
    )
    return out.reset_index().sort_values("prop_words", ascending=False)


def build(results: pd.DataFrame) -> pd.DataFrame:
    """Explode items to words and attach each (item, word)'s failure mode.

    results comes from load_results. Spelling-only inputs (graphemes,
    consonant runs, syllabic readings) are computed once per unique word.
    analyse_word_kiswahili also needs the student profile, which is fixed
    within a benchmark item, so it runs once per (word, profile) pair rather
    than once per row across all models.
    """
    item_words = (
        results[
            [
                "benchmark_item_id",
                "model",
                "hardware_tier",
                "vocab_set",
                "student_knowledge",
                "response_parsed",
                "nondecodable_words",
            ]
        ]
        .explode("vocab_set")
        .rename(columns={"vocab_set": "word"})
    )

    graphemes = {w: decompose_word(w) for w in item_words.word.unique()}
    # Runs of two or more consecutive consonants, as grapheme indices.
    runs = {
        w: [r for r in _consonant_runs(g) if len(r) >= 2] for w, g in graphemes.items()
    }
    # For each syllabic run: the nasal + next grapheme a model might misread as
    # a cluster (mt), and the whole run (mbw), both as normalised text.
    syllabic_onsets = {
        w: [
            (
                g[r[0]].normalised + g[r[1]].normalised,
                "".join(g[i].normalised for i in r),
            )
            for r in runs[w]
            if (v := _nasal_verdict(g, r)) is not None and v.value == "syllabic"
        ]
        for w, g in graphemes.items()
    }

    pairs = item_words[["word", "student_knowledge"]].drop_duplicates()
    profiles = {
        s: KiswahiliStudentKnowledge.model_validate(ast.literal_eval(s))
        for s in pairs.student_knowledge.unique()
    }
    pairs["failure_mode"] = [
        failure_mode(
            analyse_word_kiswahili(w, profiles[s]),
            profiles[s].all_known_clusters,
            syllabic_onsets[w],
            len(graphemes[w]),
            bool(runs[w]),
            w.lower() in SYLLABIC_NASAL_EXPECTED_CLUSTERS,
        )
        for w, s in zip(pairs.word, pairs.student_knowledge)
    ]
    item_words = item_words.merge(pairs, on=["word", "student_knowledge"])

    item_words["predicted"] = [
        w.lower() in r for w, r in zip(item_words.word, item_words.response_parsed)
    ]
    item_words["is_nondecodable"] = [
        w.lower() in n for w, n in zip(item_words.word, item_words.nondecodable_words)
    ]
    return item_words.drop(
        columns=["student_knowledge", "response_parsed", "nondecodable_words"]
    )


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def failure_mode_table(words: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, failure mode): error rate and share of the model's errors.

    Each mode holds one label, so error_rate is the false alarm rate for a
    decodable mode and the miss rate for a nondecodable one. share_of_errors
    is the mode's part of everything the model got wrong; the modes partition
    the item-words, so it sums to 1 per model.
    """
    out = (
        words.assign(wrong=words.predicted != words.is_nondecodable)
        .groupby(["hardware_tier", "model", "failure_mode"], observed=True)
        .agg(n=("wrong", "size"), wrong=("wrong", "sum"))
        .reset_index()
    )
    out["error_rate"] = out.wrong / out.n
    out["share_of_errors"] = out.wrong / out.groupby("model").wrong.transform("sum")
    return out


def tier_failure_table(failures: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    """One row per (tier, failure mode): means over each tier's selected models.

    selected is the model and hardware_tier selection from
    hardware_tier_errors.load_top_models. Each selected model counts equally.
    n is the mode's item-word count for the best-covered model, which does not
    depend on the tier.
    """
    out = (
        failures[failures.model.isin(selected.model)]
        .groupby(["hardware_tier", "failure_mode"], observed=True)
        .agg(
            error_rate=("error_rate", "mean"),
            share_of_errors=("share_of_errors", "mean"),
            n_models=("model", "nunique"),
        )
        .reset_index()
    )
    return out.merge(failures.groupby("failure_mode").n.max(), on="failure_mode")


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------
def style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(CONNECTOR)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, length=3, width=0.8, labelsize=9)


def plot_failure_modes(tier_failures: pd.DataFrame, selected: pd.DataFrame) -> Figure:
    """Error rate per failure mode, one dot per tier, plus each mode's share of errors.

    False alarms on decodable words sit in the top panel and misses on
    nondecodable words in the bottom, since the two are different errors.
    Rate says how often a model slips on that kind of word; share of errors
    says how much of the tier's total error it accounts for, which also
    depends on how common the words are.
    """
    rates = tier_failures.set_index(["failure_mode", "hardware_tier"])
    sizes = tier_failures.drop_duplicates("failure_mode").set_index("failure_mode").n
    groups = list(FAILURE_ROWS.items())
    fig, axes = plt.subplots(
        len(groups),
        1,
        sharex=True,
        figsize=(11.5, 8.2),
        height_ratios=[len(rows) for _, rows in groups],
        layout="constrained",
    )
    fig.set_facecolor(SURFACE)

    for ax, (title, rows) in zip(axes, groups):
        # y grows upward, so reverse to keep FAILURE_ROWS order top to bottom.
        rows = [row for row in rows[::-1] if row[0] in sizes.index]
        for y, (mode, _) in enumerate(rows):
            points = {
                tier: rates.error_rate[(mode, tier)] * 100
                for tier in TIER_ORDER
                if (mode, tier) in rates.index
            }
            faded = sizes[mode] < MIN_WORDS
            low, high = min(points.values()), max(points.values())
            ax.plot([low, high], [y, y], color=CONNECTOR, lw=2, zorder=1)
            for tier, rate in points.items():
                ax.scatter(
                    rate,
                    y,
                    s=110,
                    color=TIER_COLORS[tier],
                    edgecolors=SURFACE,
                    linewidths=1.5,
                    zorder=3,
                    alpha=0.4 if faded else 1,
                )
                # Lower value labelled on its left, higher on its right.
                left = rate == low and len(points) > 1
                ax.annotate(
                    f"{rate:.0f}%",
                    (rate, y),
                    xytext=(-9 if left else 9, 0),
                    textcoords="offset points",
                    ha="right" if left else "left",
                    va="center",
                    fontsize=9,
                    color=MUTED if faded else INK,
                )
            # Right-hand columns: item-words, then each tier's share of errors.
            ax.annotate(
                f"{sizes[mode]:,}",
                (1.02, y),
                xycoords=("axes fraction", "data"),
                va="center",
                fontsize=9,
                color=MUTED,
            )
            for x, tier in zip((1.14, 1.23), TIER_ORDER):
                share = rates.share_of_errors.get((mode, tier), np.nan)
                ax.annotate(
                    f"{share:.0%}",
                    (x, y),
                    xycoords=("axes fraction", "data"),
                    va="center",
                    fontsize=9,
                    color=TIER_COLORS[tier],
                )
        ax.set_yticks(range(len(rows)), [label for _, label in rows], fontsize=10)
        style(ax)
        # style() greys tick labels; mode names are the content, so restore them.
        for tick, (mode, _) in zip(ax.get_yticklabels(), rows):
            tick.set_color(MUTED if sizes[mode] < MIN_WORDS else INK)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.set_title(title, loc="left", fontsize=11, fontweight="semibold", color=INK)
    axes[0].annotate(
        "item-words",
        (1.02, 1.02),
        xycoords="axes fraction",
        fontsize=8.5,
        color=MUTED,
    )
    axes[0].annotate(
        "share of tier's errors",
        (1.14, 1.02),
        xycoords="axes fraction",
        fontsize=8.5,
        color=MUTED,
    )
    # Padding past 0 and 100 leaves room for the value labels.
    axes[-1].set_xlim(-9, 106)
    axes[-1].set_xticks([0, 20, 40, 60, 80, 100])
    axes[-1].set_xlabel(
        f"Error rate, mean over each tier's models {common.tier_title(selected)}",
        color=SECONDARY,
    )
    fig.legend(
        handles=[
            plt.Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=9,
                markerfacecolor=TIER_COLORS[tier],
                markeredgecolor=SURFACE,
                label=TIER_NAMES[tier],
            )
            for tier in TIER_ORDER
        ],
        loc="outside upper right",
        ncol=2,
        frameon=False,
        fontsize=10,
        labelcolor=SECONDARY,
    )
    fig.suptitle(
        "What do models get wrong, given what the student has been taught?",
        x=0.02,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.01,
        -0.01,
        "Every item-word sits in exactly one mode, judged against its item's "
        "student profile. Untaught modes hold words with that one problem only.\n"
        "Syllabic nasal (expert-reviewed list): a trap only when the student has "
        "not been taught m/n + the next consonant as a cluster (mt in mtoto).\n"
        "Nasal clusters include ambiguous boundaries, labelled as fused. "
        f"Faded: fewer than {MIN_WORDS} item-words.",
        va="top",
        fontsize=8,
        color=SECONDARY,
        linespacing=1.5,
    )
    return fig


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
def analyse_failure_modes(
    run: Path = typer.Option(
        ...,
        "--run",
        exists=True,
        file_okay=False,
        help=(
            "Run folder holding reports/kiswahili/results_with_scoring.csv "
            "and reports/hardware_tiers/."
        ),
    ),
    report_dir: Path | None = typer.Option(
        None,
        "--report-dir",
        file_okay=False,
        help="Where to write the outputs. Defaults to <run>/reports/kiswahili.",
    ),
) -> None:
    """Write the Kiswahili failure-mode tables, chart and hallucination summary."""
    report_dir = report_dir or run / "reports" / LANGUAGE
    report_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(run)
    words = build(results)
    # Hallucinations: predicted words that are not in the item's vocabulary.
    hallucination_table(results).to_csv(
        report_dir / "hallucination_summary.csv", index=False
    )

    selected = common.load_top_models(run, LANGUAGE, TOP_N, METRIC, MIN_COVERAGE)
    failures = failure_mode_table(words)
    tier_failures = tier_failure_table(failures, selected)
    failures.to_csv(report_dir / "failure_modes.csv", index=False)
    tier_failures.to_csv(report_dir / "failure_modes_by_tier.csv", index=False)
    plot_failure_modes(tier_failures, selected).savefig(
        report_dir / "failure_modes.png",
        dpi=300,
        bbox_inches="tight",
        facecolor=SURFACE,
    )

    # Top failure modes per tier, ranked by error rate, with share of errors.
    n_item_words = len(words.drop_duplicates(["benchmark_item_id", "word"]))
    typer.echo(f"models: {words.model.nunique()}   item-words: {n_item_words:,}")
    for tier, group in tier_failures.groupby("hardware_tier"):
        typer.echo(f"\n{tier}: top failure modes by error rate")
        for row in group.nlargest(5, "error_rate").itertuples():
            typer.echo(
                f"  {row.error_rate:6.1%}  {row.share_of_errors:5.1%} of errors"
                f"  n={row.n:<6,} {row.failure_mode}"
            )
    typer.echo(f"\nWrote to {report_dir}")


def main() -> None:
    """Entry point."""
    typer.run(analyse_failure_modes)


if __name__ == "__main__":
    main()
