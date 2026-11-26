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
    failure_modes_by_tier.md   the same tier means as two Markdown tables (false
                               alarms, misses), with an example word per mode
    failure_modes.png          the tier table as a dot chart
    hallucination_summary.csv  one row per model: how often it predicts words
                               that are not in the item's vocabulary

The CSVs keep every failure mode. The chart and the Markdown tables leave out
modes with fewer than MIN_WORDS item-words, and say which ones.
"""

from __future__ import annotations

import ast
import re
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
    GraphemeSpan,
    NasalReading,
    Span,
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

# Below this many item-words, a mode is left out of the chart and the Markdown
# table. It stays in the CSVs.
MIN_WORDS = 20

# Footnote markers
LENGTH_MARK = "*"
SYLLABIC_MARK = "†"

# Word length bins, in graphemes, for decodable words with no consonant run.
GRAPHEME_BINS = pd.IntervalIndex.from_tuples([(1, 5), (5, 14)], closed="right")


def length_label(span: pd.Interval) -> str:
    """'2-4' for the interval (1, 4]."""
    return f"{int(span.left) + 1}-{int(span.right)}"


# --------------------------------------------------------------------------
# Failure modes
# --------------------------------------------------------------------------
# Rows in the order the chart draws them, grouped by the error that matters.
class MissesFailureMode(StrEnum):
    UNTAUGHT_UNIGRAPH = "untaught unigraph"
    UNTAUGHT_MULTIGRAPH = "untaught multigraph"
    # A cluster of 3+ graphemes not taught as a unit, though a smaller run
    # inside it is (mb taught, mbw not).
    UNTAUGHT_COMPOUND_CLUSTER = "untaught compound cluster, part taught"
    # Every other untaught cluster.
    UNTAUGHT_CLUSTER = "untaught cluster"
    INVALID_GRAPHEME = "invalid grapheme"
    SEVERAL_REASONS = "several reasons"  # TODO: split?


class FalseAlarmFailureMode(StrEnum):
    KNOWN_WHOLE_WORD = "known whole word"
    # Holds syllabic nasal words whose fused reading is taught too (mt taught
    # in mtoto), since the fused misreading cannot cause an error there.
    TAUGHT_CLUSTER = "taught cluster"
    SYLLABIC_NASAL_UNTAUGHT = "syllabic nasal, m/n+C untaught"
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
        # The LENGTH_MARK footnote gives the grapheme counts behind short and long.
        (
            FalseAlarmFailureMode.NO_CONSONANT_RUN_SHORT,
            f"Short word$^{{{LENGTH_MARK}}}$",
        ),
        (
            FalseAlarmFailureMode.NO_CONSONANT_RUN_LONG,
            f"Long word$^{{{LENGTH_MARK}}}$",
        ),
        (FalseAlarmFailureMode.TAUGHT_CLUSTER, "Word with consonant cluster"),
        (
            FalseAlarmFailureMode.SYLLABIC_NASAL_UNTAUGHT,
            f"Word with a syllabic nasal$^{{{SYLLABIC_MARK}}}$",
        ),
    ],
    "Miss rate on nondecodable words": [
        (MissesFailureMode.UNTAUGHT_UNIGRAPH, "Untaught single-letter grapheme"),
        (MissesFailureMode.UNTAUGHT_MULTIGRAPH, "Untaught multigraph"),
        (MissesFailureMode.UNTAUGHT_CLUSTER, "Untaught cluster"),
        (
            MissesFailureMode.UNTAUGHT_COMPOUND_CLUSTER,
            "Untaught compound cluster",
        ),
        (MissesFailureMode.INVALID_GRAPHEME, "Non-Kiswahili grapheme"),
        (MissesFailureMode.SEVERAL_REASONS, "More than one of the above"),
    ],
}


def is_compound_with_taught_part(
    cluster: Span, graphemes: list[GraphemeSpan], taught_clusters: set[str]
) -> bool:
    """Whether an untaught cluster has 3+ graphemes and a smaller run inside
    it is taught: mb or bw for mbw, nj or jw for njw.

    Graphemes are counted, not letters: nch is n + ch, two graphemes. The
    smaller runs are the contiguous ones of 2 graphemes up to one short of
    the whole cluster.
    """
    inside = [g.normalised for g in graphemes if cluster.start <= g.start < cluster.end]
    return any(
        "".join(inside[i : i + size]) in taught_clusters
        for size in range(2, len(inside))
        for i in range(len(inside) - size + 1)
    )


def untaught_spans(
    analysis: KiswahiliWordAnalysis,
    graphemes: list[GraphemeSpan],
    taught_clusters: set[str],
) -> dict[MissesFailureMode, list[Span]]:
    """The parts of a word the student has not been taught, by miss mode.

    Every mode except SEVERAL_REASONS is a key; a word with spans under more
    than one key is a SEVERAL_REASONS word. Untaught clusters come from the
    strict view (ambiguous nasal boundaries fused), so mbw in mbwea is one
    cluster.
    """
    compound = [
        c
        for c in analysis.unknown_clusters
        if is_compound_with_taught_part(c, graphemes, taught_clusters)
    ]
    return {
        MissesFailureMode.UNTAUGHT_UNIGRAPH: [
            g for g in analysis.unknown_graphemes if len(g.normalised) == 1
        ],
        MissesFailureMode.UNTAUGHT_MULTIGRAPH: [
            g for g in analysis.unknown_graphemes if len(g.normalised) > 1
        ],
        MissesFailureMode.UNTAUGHT_COMPOUND_CLUSTER: compound,
        MissesFailureMode.UNTAUGHT_CLUSTER: [
            c for c in analysis.unknown_clusters if c not in compound
        ],
        MissesFailureMode.INVALID_GRAPHEME: analysis.invalid_graphemes,
    }


def find_syllabic_runs(graphemes: list[GraphemeSpan]) -> list[list[GraphemeSpan]]:
    """Consonant runs whose leading nasal is syllabic, nasal included.

    e.g. [m, t] in mtoto and [m, b, w] in mbwa. Runs read as fused or ambiguous
    are left out; they are ordinary clusters.
    """
    return [
        [graphemes[i] for i in run]
        for run in _consonant_runs(graphemes)
        if len(run) >= 2 and _nasal_verdict(graphemes, run) is NasalReading.SYLLABIC
    ]


def failure_mode(
    analysis: KiswahiliWordAnalysis,
    graphemes: list[GraphemeSpan],
    taught_clusters: set[str],
    syllabic_runs: list[str],
    has_run: bool,
    on_syllabic_list: bool,
) -> MissesFailureMode | FalseAlarmFailureMode:
    """The FAILURE_ROWS mode a (word, student) pair falls in.

    The failure modes are grouped into two:
    1. Misses: missed non-decodable words
    2. False alarms: falsely flagged decodable words

    Failure modes for missed non-decodable words are split by what the
    student has not been taught (see untaught_spans). Words untaught on more
    than one count are grouped under "several reasons".

    Failure modes for falsely flagged decodable words are split by what
    could still trip a model:
    - being a known whole word
    - a syllabic nasal whose fused reading is untaught (the trap below)
    - a consonant cluster, every one of them taught
    - or else only their length.

    A syllabic nasal is a trap when the student has been taught the nasal and
    the consonant run after it (t in mtoto, bw in mbwa) but not the nasal
    plus that run as one cluster (mt, mbw). A model that misreads the nasal
    as fused needs that cluster, finds it untaught, and wrongly flags the
    word. The package treats a consonant run as one cluster that cannot be
    split, so knowing mb does not make mbw readable. When the fused reading
    is taught too, the misreading cannot cause an error, so the word is an
    ordinary taught-cluster word.

    `syllabic_runs` lists the whole run, nasal included, for each syllabic
    run in the word. A word is a trap if any of them is untaught. Other
    nasal runs in the word (the fused mb in mjomba) are ordinary clusters,
    already checked by the untaught modes above.
    """
    if analysis.known_whole_word:
        return FalseAlarmFailureMode.KNOWN_WHOLE_WORD

    untaught = untaught_spans(analysis, graphemes, taught_clusters)
    reasons = [mode for mode, spans in untaught.items() if spans]

    if len(reasons) > 1:
        return (
            MissesFailureMode.SEVERAL_REASONS
        )  # TODO: do we want to list all reasons instead?

    if reasons:
        return reasons[0]

    # Every grapheme and cluster is taught from here on, so the word is decodable.
    if on_syllabic_list and syllabic_runs:
        # The word has a syllabic nasal. Its fused reading is harmless when
        # every syllabic run is taught as a cluster too.
        if not all(run in taught_clusters for run in syllabic_runs):
            return FalseAlarmFailureMode.SYLLABIC_NASAL_UNTAUGHT

    if has_run:
        return FalseAlarmFailureMode.TAUGHT_CLUSTER

    (span,) = [b for b in GRAPHEME_BINS if len(graphemes) in b]
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
            len(r - {w.lower() for w in v})
            for r, v in zip(results.response_parsed, results.vocab_set)
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
        w: ["".join(s.normalised for s in run) for run in find_syllabic_runs(g)]
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
            graphemes=graphemes[w],
            taught_clusters=profiles[s].all_known_clusters,
            syllabic_runs=syllabic_runs[w],
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
    n is the mode's item-word count for the highest-coverage, selected model, which
    does not depend on the tier.
    """
    chosen = failures[failures.model.isin(selected.model)]
    out = (
        chosen.groupby(["hardware_tier", "failure_mode"], observed=True)
        .agg(
            error_rate=("error_rate", "mean"),
            n_models=("model", "nunique"),
        )
        .reset_index()
    )
    return out.merge(chosen.groupby("failure_mode").n.max(), on="failure_mode")


# --------------------------------------------------------------------------
# Tier table as Markdown, with an example per failure mode
# --------------------------------------------------------------------------
# Keyed by failure mode, values are a tuple of (word, taught, untaught,
# explanation, chart line). Taught and untaught are cut down to what matters
# for the word and use KiswahiliStudentKnowledge's field names, so the tests
# can build each profile and check that the example lands in its mode.
# untaught is None when the student knows everything the word needs.
#
# The chart line is the short example drawn under the mode's name in
# failure_modes.png, written by hand since the profiles above are too long
# for a chart: "word: knows ..., not ...", with the part of the word that
# matters in mathtext bold ($\mathbf{...}$).
EXAMPLE_WORDS: dict[
    str, tuple[str, dict[str, list[str]], dict[str, list[str]] | None, str, str]
] = {
    FalseAlarmFailureMode.KNOWN_WHOLE_WORD: (
        "paka",
        {"whole_words": ["paka"], "graphemes": ["a", "k"]},
        {"graphemes": ["p"]},
        "The student doesn't know 'p' but they can decode by sight.",
        r"$\mathbf{p}$aka: knows paka as a whole word, not p",
    ),
    FalseAlarmFailureMode.NO_CONSONANT_RUN_SHORT: (
        "sana",
        {"graphemes": ["s", "a", "n"]},
        None,
        "A short word requiring no cluster knowledge, and the student knows every "
        "grapheme.",
        "sana: knows s, a, n",
    ),
    FalseAlarmFailureMode.NO_CONSONANT_RUN_LONG: (
        "watoto",
        {"graphemes": ["w", "a", "t", "o"]},
        None,
        "A longer word requiring no cluster knowledge, and the student knows every "
        "grapheme.",
        "watoto: knows w, a, t, o",
    ),
    FalseAlarmFailureMode.TAUGHT_CLUSTER: (
        "kwa",
        {"graphemes": ["k", "w", "a"], "clusters": ["kw"]},
        None,
        "The student knows every grapheme and consonant cluster in the word.",
        r"$\mathbf{kw}$a: knows k, w, a and kw",
    ),
    FalseAlarmFailureMode.SYLLABIC_NASAL_UNTAUGHT: (
        "mtoto",
        {"graphemes": ["m", "t", "o"]},
        {"clusters": ["mt"]},
        "The 'm' is syllabic, so knowing 'm', 't' and 'o' is enough to decode "
        "'mtoto'. 'mt' is not needed as a cluster, though a model that reads it "
        "as one finds it untaught.",
        # Same pattern as the untaught cluster line (njia), on purpose: the
        # only difference is that mt is not needed.
        r"$\mathbf{mt}$oto: knows m, t, o, not mt (not needed)",
    ),
    MissesFailureMode.UNTAUGHT_UNIGRAPH: (
        "maji",
        {"graphemes": ["m", "a", "i"]},
        {"graphemes": ["j"]},
        "The student doesn't know 'j' hence can't decode 'maji'.",
        r"ma$\mathbf{j}$i: knows m, a, i, not j",
    ),
    MissesFailureMode.UNTAUGHT_MULTIGRAPH: (
        "lenye",
        {"graphemes": ["l", "e", "n", "y"]},
        {"graphemes": ["ny"]},
        "Knowing 'n' and 'y' doesn't automatically grant the knowledge of 'ny', "
        "which is a digraph (two-letter grapheme) in Kiswahili whose sound is "
        "distinct from a simple blend of 'n' and 'y'.",
        r"le$\mathbf{ny}$e: knows n, y, not ny",
    ),
    MissesFailureMode.UNTAUGHT_COMPOUND_CLUSTER: (
        "magonjwa",
        {"graphemes": ["m", "a", "g", "o", "n", "j", "w"], "clusters": ["nj", "jw"]},
        {"clusters": ["njw"]},
        "The student knows 'nj' and 'jw' but not 'njw'. A consonant run is read as "
        "one unit, so knowing parts of it isn't enough to decode 'magonjwa'.",
        r"mago$\mathbf{njw}$a: knows nj, jw, not njw",
    ),
    MissesFailureMode.UNTAUGHT_CLUSTER: (
        "njia",
        {"graphemes": ["n", "j", "i", "a"]},
        {"clusters": ["nj"]},
        "The student knows 'n' and 'j' but not the prenasalised 'nj' sound, hence "
        "can't decode 'njia' properly.",
        r"$\mathbf{nj}$ia: knows n, j, not nj",
    ),
    MissesFailureMode.INVALID_GRAPHEME: (
        "taxi",
        {"graphemes": ["t", "a", "i"]},
        {"graphemes": ["x"]},
        "'x' is not in the Kiswahili grapheme inventory, hence 'taxi' should not "
        "be decodable to any student.",
        r"ta$\mathbf{x}$i: x is not a Kiswahili grapheme",
    ),
    MissesFailureMode.SEVERAL_REASONS: (
        "nyumbani",
        {"graphemes": ["u", "m", "b", "a", "n", "i"]},
        {"graphemes": ["ny"], "clusters": ["mb"]},
        "The student knows neither the digraph 'ny' nor the cluster 'mb', so "
        "'nyumbani' is untaught on more than one count.",
        r"$\mathbf{ny}$u$\mathbf{mb}$ani: knows neither ny nor mb",
    ),
}


def format_knowledge(knowledge: dict[str, list[str]] | None) -> str:
    """'whole words: paka; graphemes: a, k', or '–' when there is nothing."""
    if not knowledge:
        return "–"
    return "; ".join(
        f"{field.replace('_', ' ')}: {', '.join(items)}"
        for field, items in knowledge.items()
    )


def plain_label(label: str) -> str:
    """A FAILURE_ROWS label without its mathtext footnote marker."""
    return re.sub(r"\$\^\{[^}]*\}\$", "", label)


def reported_rows(
    tier_failures: pd.DataFrame,
) -> tuple[dict[str, list[tuple[str, str]]], list[str]]:
    """FAILURE_ROWS cut down to the modes the chart and Markdown table report.

    A mode is reported when it has at least MIN_WORDS item-words. Returns the
    reported rows by panel (empty panels dropped) and the plain labels of the
    modes left out for having too few. Modes with no item-words at all are
    left out silently.
    """
    sizes = tier_failures.drop_duplicates("failure_mode").set_index("failure_mode").n
    shown: dict[str, list[tuple[str, str]]] = {}
    too_few: list[str] = []
    for title, rows in FAILURE_ROWS.items():
        present = [(mode, label) for mode, label in rows if mode in sizes.index]
        kept = [(mode, label) for mode, label in present if sizes[mode] >= MIN_WORDS]
        too_few += [
            plain_label(label) for mode, label in present if sizes[mode] < MIN_WORDS
        ]
        if kept:
            shown[title] = kept
    return shown, too_few


def too_few_note(too_few: list[str]) -> str:
    """The note naming the modes left out of the chart and Markdown table."""
    return f"Not shown, with fewer than {MIN_WORDS} item-words: {', '.join(too_few)}"


def tier_failure_markdown(tier_failures: pd.DataFrame, selected: pd.DataFrame) -> str:
    """tier_failure_table as Markdown: one table per FAILURE_ROWS panel, with
    an EXAMPLE_WORDS example and each tier's mean rate on every row.

    Only the reported_rows modes appear. Markdown has no two-level headers, so
    the grouped columns (the example, the per-tier rates) carry their group in
    the column name.
    """
    rates = tier_failures.set_index(["failure_mode", "hardware_tier"]).error_rate
    shown, too_few = reported_rows(tier_failures)
    lines = [
        "# Failure modes by hardware tier",
        "",
        f"Rates are means over each tier's models {common.tier_title(selected)}. "
        "Example profiles are cut down to what matters for the example word.",
    ]
    if too_few:
        lines += ["", too_few_note(too_few)]
    for title, rows in shown.items():
        # The panel's error type decides the wording: misses are of
        # nondecodable words, false alarms of decodable ones.
        misses = isinstance(rows[0][0], MissesFailureMode)
        rate_name = "Mean miss rate" if misses else "Mean false alarm rate"
        header = [
            "Failure mode",
            "Example: word",
            "Example: student knowledge profile",
            "Example: missing knowledge",
            f"Why the word is {'not ' if misses else ''}decodable to the student",
            *[f"{rate_name}: {TIER_NAMES[tier]}" for tier in TIER_ORDER],
        ]
        lines += [
            "",
            f"## {title}",
            "",
            "| " + " | ".join(header) + " |",
            "|" + " --- |" * len(header),
        ]
        for mode, label in rows:
            word, taught, untaught, why, _ = EXAMPLE_WORDS[mode]
            cells = [
                plain_label(label),
                word,
                format_knowledge(taught),
                format_knowledge(untaught),
                why,
                *[
                    f"{rates[(mode, tier)]:.0%}" if (mode, tier) in rates.index else "–"
                    for tier in TIER_ORDER
                ],
            ]
            lines.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
    return "\n".join(lines) + "\n"


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
    Only the reported_rows modes are drawn; a footnote names the ones left
    out for having too few item-words.
    """
    rates = tier_failures.set_index(["failure_mode", "hardware_tier"])
    sizes = tier_failures.drop_duplicates("failure_mode").set_index("failure_mode").n
    shown, too_few = reported_rows(tier_failures)
    groups = list(shown.items())
    fig, axes = plt.subplots(
        len(groups),
        1,
        sharex=True,
        squeeze=False,  # an array of axes even if only one panel has rows
        figsize=(11.5, 8.2),
        height_ratios=[len(rows) for _, rows in groups],
        layout="constrained",
    )
    axes = axes[:, 0]
    fig.set_facecolor(SURFACE)

    for ax, (title, rows) in zip(axes, groups):
        # y grows upward, so reverse to keep FAILURE_ROWS order top to bottom.
        rows = rows[::-1]
        for y, (mode, _) in enumerate(rows):
            points = {
                tier: rates.error_rate[(mode, tier)] * 100
                for tier in TIER_ORDER
                if (mode, tier) in rates.index
            }
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
                    color=INK,
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
        # Each row is labelled in two lines: the mode's name just above the
        # row, and its EXAMPLE_WORDS chart line just below in smaller grey
        # text. A tick label has a single size and colour, so both lines are
        # annotations and the tick labels are left empty.
        ax.set_yticks(range(len(rows)), [""] * len(rows))
        style(ax)
        for y, (mode, label) in enumerate(rows):
            for text, offset, va, size, color in (
                (label, 2, "bottom", 10, INK),
                (EXAMPLE_WORDS[mode][4], -2, "top", 8.5, MUTED),
            ):
                ax.annotate(
                    text,
                    (0, y),
                    xycoords=("axes fraction", "data"),
                    xytext=(-8, offset),
                    textcoords="offset points",
                    ha="right",
                    va=va,
                    fontsize=size,
                    color=color,
                    annotation_clip=False,  # drawn left of the axes
                )
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
        "What kind of Kiswahili words do models get wrong?",
        x=0.02,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color=INK,
    )
    # One footnote line per marker, then a line naming any mode left out for
    # having too few item-words (only when there is one).
    # Grapheme counts come from GRAPHEME_BINS so the note follows the bins.
    short_max = int(GRAPHEME_BINS[0].right)
    long_min = int(GRAPHEME_BINS[1].left) + 1
    notes = [
        "Every item-word has exactly one failure mode.",
        f"{LENGTH_MARK} Short: up to {short_max} graphemes, "
        f"long: {long_min} or more graphemes.",
        f'{SYLLABIC_MARK} Some nasals ("m" or "n") preceding one or more consonants '
        "are syllabic, hence decodable without knowing the m/n + consonant(s) "
        "as a unit.\n"
        "   Knowing when such a nasal is syllabic depends on understanding the "
        "complexities of Kiswahili grammar.",
    ]
    if too_few:
        notes.append(too_few_note(too_few))
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
    (report_dir / "failure_modes_by_tier.md").write_text(
        tier_failure_markdown(tier_failures, selected)
    )
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
