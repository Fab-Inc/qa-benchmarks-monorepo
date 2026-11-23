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
mode.

Outputs, in --report-dir (default: <run>/reports/kiswahili/error_analysis):
    failure_modes.csv          one row per (model, failure mode)
    failure_modes_by_tier.csv  one row per (hardware tier, failure mode), means
                               over each tier's top TOP_N models
    failure_modes.png          the tier table as a dot chart
    hallucination_summary.csv  one row per model: how often it predicts words
                               that are not in the item's vocabulary
"""

from __future__ import annotations

import ast
from enum import StrEnum
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
    NasalReading,
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

TIER_ORDER = ["Cloud", "Mobile + Tablet"]
TIER_NAMES = {"Cloud": "Cloud", "Mobile + Tablet": "Mobile + tablet"}
# Below this many item-words, a mode is faded in the chart.
MIN_WORDS = 20
# Footnote markers on the chart's row labels: SYLLABIC_MARK on both syllabic
# nasal rows, FADED_MARK on any row with fewer than MIN_WORDS item-words.
SYLLABIC_MARK = "*"
FADED_MARK = "†"

# Word length bins, in graphemes, for decodable words with no consonant run.
GRAPHEME_BINS = pd.IntervalIndex.from_tuples([(1, 5), (5, 14)], closed="right")


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
class MissesFailureMode(StrEnum):
    UNTAUGHT_UNIGRAPH = "untaught unigraph"
    UNTAUGHT_MULTIGRAPH = "untaught multigraph"
    UNTAUGHT_NASAL_CLUSTER = "untaught nasal cluster"
    UNTAUGHT_NON_NASAL_CLUSTER = "untaught non-nasal cluster"
    INVALID_GRAPHEME = "invalid grapheme"
    SEVERAL_REASONS = "several reasons"  # TODO: split?


class FalseAlarmFailureMode(StrEnum):
    KNOWN_WHOLE_WORD = "known whole word"
    SYLLABIC_NASAL_TAUGHT = "syllabic nasal, m/n+C taught"
    SYLLABIC_NASAL_UNTAUGHT = "syllabic nasal, m/n+C untaught"
    TAUGHT_CLUSTER = "taught cluster"
    # length_label gives "2-5", not the interval's own "(1, 5]".
    NO_CONSONANT_RUN_SHORT = (
        f"no consonant run, {length_label(GRAPHEME_BINS[0])} graphemes"
    )
    NO_CONSONANT_RUN_LONG = (
        f"no consonant run, {length_label(GRAPHEME_BINS[1])} graphemes"
    )


FAILURE_ROWS = {
    "False alarm rate on decodable words": [
        (FalseAlarmFailureMode.KNOWN_WHOLE_WORD, "Known whole word"),
        # The footnote gives the grapheme range behind "short" and "long".
        (FalseAlarmFailureMode.NO_CONSONANT_RUN_SHORT, "Every grapheme taught - short"),
        (FalseAlarmFailureMode.NO_CONSONANT_RUN_LONG, "Every grapheme taught - long"),
        (FalseAlarmFailureMode.TAUGHT_CLUSTER, "Every cluster taught"),
        (
            FalseAlarmFailureMode.SYLLABIC_NASAL_TAUGHT,
            f"Syllabic nasal - m/n + C taught$^{{{SYLLABIC_MARK}}}$",
        ),
        (
            FalseAlarmFailureMode.SYLLABIC_NASAL_UNTAUGHT,
            f"Syllabic nasal - m/n + C not taught$^{{{SYLLABIC_MARK}}}$",
        ),
    ],
    "Miss rate on nondecodable words": [
        (MissesFailureMode.UNTAUGHT_UNIGRAPH, "Untaught single-letter grapheme"),
        (MissesFailureMode.UNTAUGHT_MULTIGRAPH, "Untaught multigraph"),
        (
            MissesFailureMode.UNTAUGHT_NON_NASAL_CLUSTER,
            "Untaught non-nasal cluster",
        ),
        (
            MissesFailureMode.UNTAUGHT_NASAL_CLUSTER,
            "Untaught nasal cluster",
        ),
        (MissesFailureMode.INVALID_GRAPHEME, "Non-Kiswahili grapheme"),
        (MissesFailureMode.SEVERAL_REASONS, "More than one of the above"),
    ],
}


def failure_mode(
    analysis: KiswahiliWordAnalysis,
    taught_clusters: set[str],
    syllabic_runs: list[str],
    n_graphemes: int,
    has_run: bool,
    on_syllabic_list: bool,
) -> MissesFailureMode | FalseAlarmFailureMode:
    """The FAILURE_ROWS mode a (word, student) pair falls in.

    The failure modes are grouped into two:
    1. Misses: missed non-decodable words
    2. False alarms: falsely flagged decodable words

    Failure modes for missed non-decodable words are split by what the
    student has not been taught. (Currently, words with multiple untaught
    reasons are grouped under "several reasons")

    Failure modes for falsely flagged decodable words are split by what
    could still trip a model:
    - being a known whole word
    - a syllabic nasal, split into cases where the nasal + the following
      grapheme/cluster as a unit is taught or untaught
    - a taught cluster
    - or else only their length.

    A syllabic nasal is a trap when the student has been taught the nasal and
    the consonant run after it (t in mtoto, bw in mbwa) but not the nasal
    plus that run as one cluster (mt, mbw). A model that misreads the nasal
    as fused needs that cluster, finds it untaught, and wrongly flags the
    word. The package treats a consonant run as one cluster that cannot be
    split, so knowing mb does not make mbw readable.

    `syllabic_runs` lists the whole run, nasal included, for each syllabic
    run in the word. A word is a trap if any of them is untaught. Other
    nasal runs in the word (the fused mb in mjomba) are ordinary clusters,
    already checked by the untaught modes above.

    Nasal clusters include ambiguous boundaries, which the dataset labels as
    fused (gated).
    """
    if analysis.known_whole_word:
        return FalseAlarmFailureMode.KNOWN_WHOLE_WORD

    untaught = {
        MissesFailureMode.UNTAUGHT_UNIGRAPH: [
            g for g in analysis.unknown_graphemes if len(g.normalised) == 1
        ],
        MissesFailureMode.UNTAUGHT_MULTIGRAPH: [
            g for g in analysis.unknown_graphemes if len(g.normalised) > 1
        ],
        MissesFailureMode.UNTAUGHT_NASAL_CLUSTER: [
            c for c in analysis.unknown_clusters if is_nasal(c.normalised)
        ],
        MissesFailureMode.UNTAUGHT_NON_NASAL_CLUSTER: [
            c for c in analysis.unknown_clusters if not is_nasal(c.normalised)
        ],
        MissesFailureMode.INVALID_GRAPHEME: analysis.invalid_graphemes,
    }

    reasons = [mode for mode, spans in untaught.items() if spans]

    if len(reasons) > 1:
        return (
            MissesFailureMode.SEVERAL_REASONS
        )  # TODO: do we want to list all reasons instead?

    if reasons:
        return reasons[0]

    # Every grapheme and cluster is taught from here on, so the word is decodable.
    if on_syllabic_list and syllabic_runs:
        # i.e. word contains a syllabic nasal
        fused_reading_taught = all(run in taught_clusters for run in syllabic_runs)
        if fused_reading_taught:
            return FalseAlarmFailureMode.SYLLABIC_NASAL_TAUGHT
        else:
            return FalseAlarmFailureMode.SYLLABIC_NASAL_UNTAUGHT

    if has_run:
        return FalseAlarmFailureMode.TAUGHT_CLUSTER

    (span,) = [b for b in GRAPHEME_BINS if n_graphemes in b]
    if span == GRAPHEME_BINS[0]:
        return FalseAlarmFailureMode.NO_CONSONANT_RUN_SHORT
    return FalseAlarmFailureMode.NO_CONSONANT_RUN_LONG


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

    `results` comes from `load_results` which has one row per model per
    benchmark item. This is exploded to have one row per (item, word) pair.

    Word-only inputs (graphemes, consonant runs, syllabic readings) are computed
    once per unique word.

    Decodability analysis runs once per (word, profile) pair.
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

    # Syllabic nasal consonant runs, as normalised text, including the nasal.
    # e.g. mt in mtoto, mbw in mbwa. These are the clusters a model would deem
    # required if it misread the syllabic run as gated.
    syllabic_runs = {
        w: [
            "".join(g[i].normalised for i in r)
            for r in runs[w]
            if _nasal_verdict(g, r) is NasalReading.SYLLABIC
        ]
        for w, g in graphemes.items()
    }

    word_profile_pairs = item_words[["word", "student_knowledge"]].drop_duplicates()
    # Sometimes, the same word is used in the same item multiple times.
    profiles = {
        s: KiswahiliStudentKnowledge.model_validate(ast.literal_eval(s))
        for s in word_profile_pairs.student_knowledge.unique()
    }
    word_profile_pairs["failure_mode"] = [
        failure_mode(
            analysis=analyse_word_kiswahili(w, profiles[s]),
            taught_clusters=profiles[s].all_known_clusters,
            syllabic_runs=syllabic_runs[w],
            n_graphemes=len(graphemes[w]),
            has_run=bool(runs[w]),
            on_syllabic_list=w.lower() in SYLLABIC_NASAL_EXPECTED_CLUSTERS,
        )
        for w, s in zip(word_profile_pairs.word, word_profile_pairs.student_knowledge)
    ]
    item_words = item_words.merge(word_profile_pairs, on=["word", "student_knowledge"])

    item_words["flagged"] = [
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
    """One row per (model, failure mode): item-words, errors and error rate.

    Each mode holds one label, so error_rate is the false alarm rate for a
    decodable mode and the miss rate for a nondecodable one.
    """
    out = (
        words.assign(wrong=words.flagged != words.is_nondecodable)
        .groupby(["hardware_tier", "model", "failure_mode"], observed=True)
        .agg(n=("wrong", "size"), wrong=("wrong", "sum"))
        .reset_index()
    )
    out["error_rate"] = out.wrong / out.n
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
    """Error rate per failure mode, one dot per tier, with each mode's item-words.

    False alarms on decodable words sit in the top panel and misses on
    nondecodable words in the bottom, since the two are different errors.
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
                    color=common.TIER_COLORS[tier],
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
            # Right-hand column: how many item-words the rate rests on.
            ax.annotate(
                f"{sizes[mode]:,}",
                (1.02, y),
                xycoords=("axes fraction", "data"),
                va="center",
                fontsize=9,
                color=MUTED,
            )
        # Faded rows get FADED_MARK as a superscript (mathtext), which the
        # footnote explains. Assumes labels contain no "$" of their own.
        labels = [
            label + (f"$^{{{FADED_MARK}}}$" if sizes[mode] < MIN_WORDS else "")
            for mode, label in rows
        ]
        ax.set_yticks(range(len(rows)), labels, fontsize=10)
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
                markerfacecolor=common.TIER_COLORS[tier],
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
        "What do models get wrong?",
        x=0.02,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color=INK,
    )
    # One footnote line per marker. The FADED_MARK line appears only when a row
    # is actually faded, so it never explains a mark the chart does not show.
    short, long = (length_label(b) for b in GRAPHEME_BINS)
    notes = [
        "Every item-word has exactly one failure mode. "
        f"Short: {short} graphemes; long: {long} graphemes.",
        f'{SYLLABIC_MARK} Some nasals ("m" or "n") preceding one or more consonants '
        "are syllabic, hence decodable without knowing the m/n + consonant(s) "
        "as a unit.\n"
        "   Knowing when such a nasal is syllabic depends on understanding the "
        "complexities of Kiswahili grammar.",
    ]
    if (sizes < MIN_WORDS).any():
        notes.append(f"{FADED_MARK} Faded: fewer than {MIN_WORDS} item-words.")
    fig.text(
        0.01,
        -0.01,
        "\n".join(notes),
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
        help=(
            "Where to write the outputs. "
            "Defaults to <run>/reports/kiswahili/error_analysis."
        ),
    ),
) -> None:
    """Write the Kiswahili failure-mode tables, chart and hallucination summary."""
    report_dir = report_dir or run / "reports" / LANGUAGE / "error_analysis"
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

    # Top failure modes per tier, ranked by error rate.
    n_item_words = len(words.drop_duplicates(["benchmark_item_id", "word"]))
    typer.echo(f"models: {words.model.nunique()}   item-words: {n_item_words:,}")
    for tier, group in tier_failures.groupby("hardware_tier"):
        typer.echo(f"\n{tier}: top failure modes by error rate")
        for row in group.nlargest(5, "error_rate").itertuples():
            typer.echo(f"  {row.error_rate:6.1%}  n={row.n:<6,} {row.failure_mode}")
    typer.echo(f"\nWrote to {report_dir}")


def main() -> None:
    """Entry point."""
    typer.run(analyse_failure_modes)


if __name__ == "__main__":
    main()
