#!/usr/bin/env python3
"""
Analyse a Linguistic-Knowledge benchmark experiment from its two result CSVs.

Inputs (both produced by `judge-bench`):
  * SUMMARY csv   — one row per model: accuracy, accuracy_mcq, accuracy_short_form.
  * DETAILED csv  — one row per question: question_id, domain, grade, answer_type,
                    question, expected_answers, then per model a `<model>_output` and
                    `<model>_correct` (True/False, blank = unscored) column.

If either CSV is missing (e.g. deleted after the experiment), it's rebuilt automatically
from
the per-model judged CSVs still under cache/<language>/experiments/<experiment-id>/responses_judged/
(and responses_judged_fallback/, if a --judge-fallback pass was run), using the same
judge_bench.compute_summary()/build_detailed() functions judge-bench itself uses --
no LLM judge calls, no re-running judge-bench. This needs the eval dataset the run was
actually scored against (--dataset; defaults to the language's standard dataset, wrong
for a experiment-id scored against a different pool -- see --dataset below). Before writing
anything, the rebuild cross-checks the judged question_ids/answer_types against
--dataset and ABORTS (no CSV written) if they don't line up -- e.g. the shipped,
IRT-selected dataset colliding on renumbered question_ids with a different item pool
-- since a wrong-but-plausible detailed CSV is worse than none.

Questions it answers:
  (a) Is there a difference between MCQ and short-form accuracy?  -> per-model grouped
      bars from the summary, plus the overall gap (also cross-checked from the detailed).
  (b) Which domain/grade are models getting wrong?  -> a domain x grade accuracy heatmap
      plus by-domain and by-grade bars, computed from the detailed correctness columns.
  (c) When the experiment used >=2 LLM judges (short_form items only): how much do the judges
      disagree, and what would each model's accuracy be if scored by only ONE judge
      instead of the ensemble?  -> per-model solo-judge accuracy deltas and a pairwise
      judge-agreement matrix, read from judge-bench's per-model judged CSVs under
      cache/<language>/experiments/<experiment-id>/responses_judged/ (not the two CSVs above, which
      only carry each model's final combined verdict). Silently skipped for single-judge
      experiments, since there's nothing to compare a lone judge against.

Outputs (under --outdir, tagged with --label):
    accuracy_by_model_<label>.png
    accuracy_mcq_vs_shortform_<label>.png
    accuracy_by_domain_grade_<label>.png
    accuracy_by_domain_grade_<label>.csv       the domain x grade accuracy table
    judge_analysis_<label>.csv                 (c) per-model ensemble vs solo-judge accuracy
    judge_pairwise_agreement_<label>.csv       (c) pairwise judge agreement, all models combined
    judge_accuracy_<label>.png                 (c) grouped bars: ensemble vs each judge, per model
    accuracy_by_model_size_<label>.png         only with --model-size: one accuracy barplot,
                                                with bars colored by estimated RAM tier
                                                (<4GB, 4-12GB, 12-35GB, >35GB) from
                                                models.csv's `size`; Cloud bars are grey
    accuracy_by_hardware_<label>.png           only with --hardware: four accuracy barplots,
                                                grouped by estimated RAM tier (<4GB, 4-12GB,
                                                12-35GB, >35GB) from models.csv's `size`
    accuracy_boxplot_<label>.png               only with --boxplot: four hardware-tier
                                                groups, each pairing English and Kiswahili
                                                model-accuracy distributions
    accuracy_swarplot_<label>.png              only with --swarplot: the paired boxplot
                                                with individual model accuracies jittered
                                                underneath

Run (--language and --experiment-id are mandatory; --experiment-id selects the judge-bench run's
CSVs, or pass --summary/--detailed to point at exact files instead):
    python benchmarks/linguistic_knowledge_benchmark/scripts/analyze_results.py --language english --experiment-id default
    python .../analyze_results.py --language kiswahili --experiment-id v10-full-dataset
    python .../analyze_results.py --language kiswahili --experiment-id default --summary ... --detailed ... --label ...
    python .../analyze_results.py --language kiswahili --experiment-id default --model-size   # also write accuracy_by_model_size_*.png

Restrict the domain×grade analysis to a band of models by overall accuracy
(--min-accuracy / --max-accuracy, in %); the band is stamped into the plot titles
and appended to the domain×grade output filenames (e.g. _acc90plus, _acc70-90):
    python .../analyze_results.py --language kiswahili --min-accuracy 90              # only the 90+% models
    python .../analyze_results.py --language kiswahili --min-accuracy 70 --max-accuracy 90   # the 70-90% band
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# Fallback bar color for models with no provider / no provider color.
DEFAULT_BAR_COLOR = "#4e79a7"

# Model-size buckets for --model-size and hardware tiers for --hardware, by total
# parameter count (models.csv's `size` column). Buckets use total params, not MoE
# active params.
SIZE_BUCKET_UNKNOWN = "12B+ / unknown"
SIZE_BUCKETS = [
    (0, 4e9, "<4B"),
    (4e9, 12e9, "4-12B"),
]
SIZE_BUCKET_COLORS = {
    "<4B": "#59a14f",
    "4-12B": "#4e79a7",
    SIZE_BUCKET_UNKNOWN: "#bab0ac",
}

HARDWARE_TIERS = [
    (0, 4e9, "Mobile (<4GB RAM)"),
    (4e9, 12e9, "Tablet (4-12GB RAM)"),
    (12e9, 35e9, "Workstation (12-35GB RAM)"),
    (35e9, float("inf"), "Cloud (>35GB RAM)"),
]
HARDWARE_TIER_COLORS = {
    "Mobile (<4GB RAM)": "#59a14f",
    "Tablet (4-12GB RAM)": "#4e79a7",
    "Workstation (12-35GB RAM)": "#f28e2b",
    "Cloud (>35GB RAM)": "#bab0ac",
}

# Fixed spine (same order as eda_pipeline.py) so empty domains/grades still show.
DOMAINS = [f"D{i}" for i in range(1, 10)]
GRADES = ["G0", "G1", "G2", "G3", "G4", "G5", "beyond_primary"]
GRADE_LABEL = {g: ("BP" if g == "beyond_primary" else g) for g in GRADES}
ACC_CMAP = "RdYlGn"  # low accuracy = red (getting it wrong), high = green
_CORRECT_MAP = {True: 1, False: 0, "True": 1, "False": 0}

# Per-model accuracy bar chart geometry (_plot_accuracy_bars). The figure width scales
# with the model count so each bar keeps the same slot however many models a run has:
# BAR_SLOT_INCHES is that slot, MIN_FIG_WIDTH_INCHES keeps a handful of models from
# rendering as a stubby chart, and MAX_FIG_WIDTH_INCHES caps the PNG's pixel width
# (at dpi=150, 44in is ~6600px) -- past that cap the fonts start shrinking again.
BAR_SLOT_INCHES = 0.30
MIN_FIG_WIDTH_INCHES = 10.0
MAX_FIG_WIDTH_INCHES = 44.0
# Fraction of each slot the bar itself fills (matplotlib's own default is 0.8). Near 1
# the bars nearly touch, leaving only a hairline of background between them; the bars
# stay distinguishable because adjacent models are usually different provider colours.
# Absolute bar width is BAR_WIDTH_FRACTION * BAR_SLOT_INCHES, so thinner bars AND tight
# spacing means lowering BAR_SLOT_INCHES -- which narrows the figure and, via slot_pt
# below, shrinks the fonts again.
BAR_WIDTH_FRACTION = 0.85

# Language keys match cmd.py's DATASETS and the result CSVs' own naming
# (summary_linguistic-knowledge_<language>_<label>.csv).
LANGUAGES = ["english", "kiswahili"]

# Default eval dataset per language, mirroring cmd.py's DATASETS registry (kept as a
# plain dict here, rather than importing cmd.py, so this script doesn't pull in its
# CLI-only dependencies (typer, dotenv, fab_benchmark_publisher) for the common case
# where a rebuild isn't needed). Only used by rebuild_summary_and_detailed() below;
# override with --dataset for a experiment-id scored against a different pool (e.g. the IRT
# full-dataset experiment, whose dataset is stage_3b_dataset_full.json, not this file).
DEFAULT_DATASET_PATH = {
    "english": os.path.join("data", "linguistic_knowledge_eval_english.json"),
    "kiswahili": os.path.join("data", "linguistic_knowledge_eval_kiswahili.json"),
}


def default_paths_for(language: str, results_dir: str, experiment_id: str) -> dict:
    return {
        "summary": os.path.join(
            results_dir, f"summary_linguistic-knowledge_{language}_{experiment_id}.csv"
        ),
        "detailed": os.path.join(
            results_dir,
            f"summary_linguistic-knowledge_{language}_{experiment_id}_detailed.csv",
        ),
        "label": f"{language}_{experiment_id}",
    }


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def load_summary(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["model"].notna()].copy()
    return df.sort_values("accuracy", ascending=False).reset_index(drop=True)


def _discover_judged_models(cache_experiment_dir: str) -> list[str]:
    """Model ids with a primary judged CSV under cache_experiment_dir/responses_judged/."""
    judged_dir = os.path.join(cache_experiment_dir, "responses_judged")
    if not os.path.isdir(judged_dir):
        return []
    prefix, suffix = "responses_judged_", ".csv"
    return sorted(
        f[len(prefix) : -len(suffix)]
        for f in os.listdir(judged_dir)
        if f.startswith(prefix) and f.endswith(suffix)
    )


def _load_full_judged(cache_experiment_dir: str, model: str) -> pd.DataFrame | None:
    """Load the complete judged frame for `model`: the fallback pass's CSV if one
    exists, else the primary judged CSV. judge_results_fallback() writes its output
    by reading the full primary CSV and adding fallback columns on top, so the
    fallback file is a strict superset -- unlike load_judged_model() below (which
    merges only the per-judge fallback columns for judge_analysis's narrower needs),
    this is the complete frame compute_summary()/build_detailed() expect.
    """
    fallback = os.path.join(
        cache_experiment_dir,
        "responses_judged_fallback",
        f"responses_judged_{model}.csv",
    )
    if os.path.exists(fallback):
        return pd.read_csv(fallback)
    primary = os.path.join(
        cache_experiment_dir, "responses_judged", f"responses_judged_{model}.csv"
    )
    if os.path.exists(primary):
        return pd.read_csv(primary)
    return None


def _check_dataset_matches_judged(
    judged_by_model: dict[str, pd.DataFrame], dataset_path: str
) -> None:
    """Abort with a warning if --dataset doesn't actually match what was judged.

    The dangerous case isn't a missing file (that already raises earlier) -- it's a
    same-ish-shaped but WRONG file, e.g. the shipped, IRT-selected dataset
    (renumbered question_id 1..135) colliding with a full item pool's own 1..N ids.
    build_detailed()'s merge would succeed either way, silently attaching the wrong
    domain/grade/question to a real judged answer -- so this runs BEFORE either CSV
    is written, and raises rather than just printing, since a wrong detailed CSV is
    worse than no detailed CSV. Two independent signals, since either alone can miss it:
      1. question_id sets that don't line up -- ids that were judged but aren't
         in the dataset would be silently dropped from the detailed table.
      2. for ids present in both, the judged CSV's own answer_type disagreeing
         with the dataset's answer_type for that same id -- catches a true id
         collision even when the id ranges happen to overlap cleanly.
    """
    dataset_rows = pd.read_json(dataset_path)
    dataset_types = dict(zip(dataset_rows["question_id"], dataset_rows["answer_type"]))
    dataset_ids = set(dataset_types)

    judged_ids: set[int] = set()
    checked = mismatched = 0
    for df in judged_by_model.values():
        for qid, atype in zip(df["question_id"], df["answer_type"]):
            qid = int(qid)
            judged_ids.add(qid)
            if qid in dataset_types:
                checked += 1
                if dataset_types[qid] != atype:
                    mismatched += 1

    missing_from_dataset = judged_ids - dataset_ids
    if not missing_from_dataset and not mismatched:
        return

    msg = [
        f"WARNING: --dataset {dataset_path} does not match what was actually judged "
        "-- aborting rebuild rather than writing a silently-wrong detailed CSV.",
        f"  {len(judged_ids)} distinct question_id(s) judged, "
        f"{len(dataset_ids)} in the dataset.",
    ]
    if missing_from_dataset:
        msg.append(
            f"  {len(missing_from_dataset)} judged question_id(s) not found in the "
            "dataset (would be dropped from the detailed table)."
        )
    if mismatched:
        msg.append(
            f"  {mismatched}/{checked} overlapping id(s) have a different "
            "answer_type in the dataset than what was actually judged -- likely a "
            "renumbered/different item pool reusing the same ids."
        )
    msg.append(
        "  If this run was scored against a different item pool (e.g. an IRT "
        "full-dataset run), pass the matching --dataset and re-run."
    )
    raise SystemExit("\n".join(msg))


def rebuild_summary_and_detailed(
    language: str,
    experiment_id: str,
    root: str,
    dataset_path: str,
    summary_path: str,
    detailed_path: str,
) -> None:
    """Recreate a deleted summary/detailed CSV pair from the per-model judged CSVs
    still under cache/<language>/experiments/<experiment-id>/, using the exact same functions
    cmd.py's judge-bench command uses to build them the first time
    (judge_bench.compute_summary / judge_bench.build_detailed) -- no LLM judge
    calls, no re-running judge-bench, just recomputing from what's already cached.
    """
    from linguistic_knowledge_benchmark.judge_bench import (
        build_detailed,
        compute_summary,
    )

    cache_experiment_dir = os.path.join(
        root, "cache", language, "experiments", experiment_id
    )
    model_ids = _discover_judged_models(cache_experiment_dir)
    if not model_ids:
        raise SystemExit(
            f"Cannot rebuild {os.path.basename(summary_path)}: no judged CSVs found "
            f"under {os.path.join(cache_experiment_dir, 'responses_judged')}.\n"
            "Run 'judge-bench' for this language/experiment-id first."
        )
    if not os.path.exists(dataset_path):
        raise SystemExit(
            f"Cannot rebuild {os.path.basename(detailed_path)}: dataset not found "
            f"at {dataset_path}.\nPass --dataset to point at the JSON this run's "
            "responses were actually scored against (e.g. the IRT full-dataset "
            "pool, if --experiment-id is an experiment like 'v10-full-dataset')."
        )

    judged_by_model: dict[str, pd.DataFrame] = {}
    summaries = []
    for model in model_ids:
        df = _load_full_judged(cache_experiment_dir, model)
        if df is None:
            continue
        judged_by_model[model] = df
        summaries.append(compute_summary(df, model))

    # Checked before either CSV is written: a wrong --dataset must abort the whole
    # rebuild, not just skip the detailed CSV -- see _check_dataset_matches_judged.
    _check_dataset_matches_judged(judged_by_model, dataset_path)

    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    pd.DataFrame(summaries).to_csv(summary_path, index=False)

    detailed_df = build_detailed(judged_by_model, dataset_path)
    os.makedirs(os.path.dirname(detailed_path) or ".", exist_ok=True)
    detailed_df.to_csv(detailed_path, index=False)

    print(
        f"Rebuilt from {len(judged_by_model)} cached judged CSV(s) in "
        f"{os.path.join(cache_experiment_dir, 'responses_judged')}:\n"
        f"  {summary_path}\n  {detailed_path}"
    )


def load_display_names(models_csv: str) -> dict:
    """model_id -> display_name, from data/models.csv (empty dict if the file is absent)."""
    if not os.path.exists(models_csv):
        return {}
    df = pd.read_csv(models_csv)
    return dict(zip(df["model_id"], df["display_name"]))


def _labels(models: list[str], display_names: dict | None) -> list[str]:
    """Map model ids to their models.csv display_name; unmapped ids are shown as-is."""
    display_names = display_names or {}
    return [display_names.get(m, m) for m in models]


def load_provider_colors(models_csv: str, providers_csv: str) -> tuple[dict, dict]:
    """Return (model_id -> provider, provider -> color) from models.csv & providers.csv.

    Either map is empty if its file is missing; callers fall back to DEFAULT_BAR_COLOR.
    """
    model_provider: dict = {}
    if os.path.exists(models_csv):
        m = pd.read_csv(models_csv)
        if "provider" in m.columns:
            model_provider = dict(zip(m["model_id"], m["provider"]))
    provider_color: dict = {}
    if os.path.exists(providers_csv):
        p = pd.read_csv(providers_csv)
        if {"provider", "color"}.issubset(p.columns):
            provider_color = dict(zip(p["provider"], p["color"]))
    return model_provider, provider_color


def _bar_colors(
    models: list[str], model_provider: dict, provider_color: dict
) -> list[str]:
    """One bar color per model = its provider's color (DEFAULT_BAR_COLOR if unknown)."""
    return [
        provider_color.get(model_provider.get(m), DEFAULT_BAR_COLOR) for m in models
    ]


def _provider_legend_handles(
    models: list[str], model_provider: dict, provider_color: dict
) -> list[Patch]:
    """Legend patches for the providers present among `models`, in providers.csv order.

    providers.csv's row order IS the canonical provider order: that file already maps
    provider -> colour, so one file decides both, and re-ordering any legend is a
    one-line edit there. Ordering by anything read off the plotted models instead
    (first-seen order, model count, best accuracy) gives the same provider a
    different legend slot in every figure -- bars are sorted by accuracy, which
    differs per language and per experiment -- so the English and Kiswahili legends
    could not be read against each other. Same shape as the hardware-tier legends:
    canonical order, filtered to what is actually present (see plot_accuracy_by_size).

    Providers missing from providers.csv are dropped -- with no colour there is no
    patch to draw.
    """
    present = {model_provider.get(m) for m in models} & provider_color.keys()
    return [
        Patch(facecolor=provider_color[p], label=p)
        for p in provider_color
        if p in present
    ]


def load_model_sizes(models_csv: str) -> dict:
    """model_id -> total parameter count (float), from models.csv's `size` column.

    Empty dict if the file or column is missing. A model present in the file with a
    blank `size` (e.g. a closed-source model whose parameter count isn't published)
    maps to None -- distinct from being absent, both bucket as SIZE_BUCKET_UNKNOWN.
    """
    if not os.path.exists(models_csv):
        return {}
    m = pd.read_csv(models_csv)
    if "size" not in m.columns:
        return {}
    return {
        row.model_id: (None if pd.isna(row.size) else float(row.size))
        for row in m.itertuples()
    }


def size_bucket(size: float | None) -> str:
    """Bucket label for a parameter count (see SIZE_BUCKETS), or SIZE_BUCKET_UNKNOWN
    if size is None/NaN."""
    if size is None or pd.isna(size):
        return SIZE_BUCKET_UNKNOWN
    for lo, hi, label in SIZE_BUCKETS:
        if lo <= size < hi:
            return label
    return SIZE_BUCKET_UNKNOWN


def hardware_tier(size: float | None) -> str:
    """Return the estimated RAM tier for a total parameter count."""
    if size is None or pd.isna(size):
        return HARDWARE_TIERS[-1][2]
    for lo, hi, label in HARDWARE_TIERS:
        if lo <= size < hi:
            return label
    return HARDWARE_TIERS[-1][2]


def _bar_colors_by_size(models: list[str], sizes: dict) -> list[str]:
    """One bar color per model = its parameter-count bucket's color."""
    return [SIZE_BUCKET_COLORS[size_bucket(sizes.get(m))] for m in models]


def _size_bucket_legend_handles(models: list[str], sizes: dict) -> list[Patch]:
    """Legend patches for the size buckets present among `models`, in bucket order
    (smallest to largest, SIZE_BUCKET_UNKNOWN last)."""
    order = [label for *_, label in SIZE_BUCKETS] + [SIZE_BUCKET_UNKNOWN]
    present = {size_bucket(sizes.get(m)) for m in models}
    return [
        Patch(facecolor=SIZE_BUCKET_COLORS[label], label=label)
        for label in order
        if label in present
    ]


def load_detailed(path: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return (wide detailed df, long correctness df, model list).

    The long df has one row per (question, model): question_id, domain, grade,
    answer_type, model, correct (1/0/NaN).
    """
    df = pd.read_csv(path)
    corr_cols = [c for c in df.columns if c.endswith("_correct")]
    models = [c[: -len("_correct")] for c in corr_cols]
    id_cols = [
        c for c in ["question_id", "domain", "grade", "answer_type"] if c in df.columns
    ]
    long = df.melt(
        id_vars=id_cols, value_vars=corr_cols, var_name="model", value_name="correct"
    )
    long["model"] = long["model"].str.replace(r"_correct$", "", regex=True)
    long["correct"] = long["correct"].map(_CORRECT_MAP)
    return df, long, models


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _plot_accuracy_bars(
    summary: pd.DataFrame,
    out: str,
    title: str,
    display_names: dict | None,
    colors: list[str],
    legend_handles: list[Patch] | None,
    legend_title: str | None,
) -> None:
    """Shared renderer for the per-model accuracy bar chart: same bars/labels/style
    for plot_accuracy() and plot_accuracy_by_size(), which differ only in how bars
    are colored and what (if any) legend they show.

    The figure WIDTH grows with the number of models (BAR_SLOT_INCHES each, floored at
    MIN_FIG_WIDTH_INCHES) instead of being fixed: at a fixed width, 120 models leave
    each bar ~1.5mm and the rotated names and value labels collide into an unreadable
    band. Growing the canvas keeps the per-model slot constant, so the fonts below can
    stay at a readable size however many models are plotted.
    """
    s = summary.sort_values("accuracy", ascending=False)
    models = s["model"].tolist()
    x = np.arange(len(models))

    n_models = max(len(models), 1)
    fig_width = min(
        MAX_FIG_WIDTH_INCHES, max(MIN_FIG_WIDTH_INCHES, n_models * BAR_SLOT_INCHES)
    )
    # Points of horizontal room per model, once the width above is fixed (72pt = 1in).
    # Fonts are sized from that, so they shrink only when the width cap is what's
    # binding (very many models) rather than on every added model.
    slot_pt = 72.0 * fig_width / n_models
    label_fontsize = float(np.clip(slot_pt * 0.55, 5.0, 11.0))
    value_fontsize = float(np.clip(slot_pt * 0.50, 4.5, 10.0))
    # Grow the height with the width too: the rotated name band below the axes is a
    # fixed height, so a wide-but-short figure spends most of its canvas on labels and
    # squashes the bars themselves into a ribbon.
    fig_height = float(np.clip(fig_width * 0.24, 6.0, 12.0))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    bars = ax.bar(x, s["accuracy"], width=BAR_WIDTH_FRACTION, color=colors)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 106)
    ax.set_xticks(x)
    ax.set_xticklabels(
        _labels(models, display_names),
        rotation=35,
        ha="right",
        fontsize=label_fontsize,
    )
    ax.set_title(title, fontweight="bold", pad=16, fontsize=15)
    for b, a in zip(bars, s["accuracy"]):
        ax.text(
            b.get_x() + b.get_width() / 2,
            a + 1,
            f"{a:.0f}",
            ha="center",
            fontsize=value_fontsize,
        )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#eeeeee")
    ax.set_axisbelow(True)

    # Colour legend, placed to the right of the axes.
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            title=legend_title,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=False,
            fontsize=8,
            title_fontsize=9,
        )
    fig.tight_layout()
    # bbox_inches="tight" so the right-hand legend is not clipped.
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy(
    summary: pd.DataFrame,
    out: str,
    title: str = "Linguistic Knowledge Benchmark Performance",
    display_names: dict | None = None,
    model_provider: dict | None = None,
    provider_color: dict | None = None,
) -> None:
    model_provider = model_provider or {}
    provider_color = provider_color or {}
    models = summary.sort_values("accuracy", ascending=False)["model"].tolist()
    colors = _bar_colors(models, model_provider, provider_color)
    handles = _provider_legend_handles(models, model_provider, provider_color)
    _plot_accuracy_bars(
        summary, out, title, display_names, colors, handles or None, "Provider"
    )


def plot_accuracy_by_size(
    summary: pd.DataFrame,
    out: str,
    title: str = "Linguistic Knowledge Benchmark Performance — by hardware tier",
    display_names: dict | None = None,
    sizes: dict | None = None,
    model_provider: dict | None = None,
    provider_color: dict | None = None,
) -> None:
    """Plot one accuracy barplot with bars colored by hardware/RAM tier."""
    sizes = sizes or {}
    ordered = summary.sort_values("accuracy", ascending=False)
    models = ordered["model"].tolist()
    colors = [HARDWARE_TIER_COLORS[hardware_tier(sizes.get(model))] for model in models]
    present = {hardware_tier(sizes.get(model)) for model in models}
    handles = [
        Patch(facecolor=HARDWARE_TIER_COLORS[label], label=label)
        for _, _, label in HARDWARE_TIERS
        if label in present
    ]
    _plot_accuracy_bars(
        summary,
        out,
        title,
        display_names,
        colors,
        handles or None,
        "Hardware tier",
    )


def plot_accuracy_by_hardware(
    summary: pd.DataFrame,
    out: str,
    title: str = "Linguistic Knowledge Benchmark Performance — by hardware tier",
    display_names: dict | None = None,
    sizes: dict | None = None,
    model_provider: dict | None = None,
    provider_color: dict | None = None,
    cloud_grey: bool = False,
) -> None:
    """Plot one accuracy barplot per estimated hardware/RAM tier."""
    sizes = sizes or {}
    model_provider = model_provider or {}
    provider_color = provider_color or {}
    ordered = summary.sort_values("accuracy", ascending=False)
    fig, axes = plt.subplots(1, len(HARDWARE_TIERS), figsize=(18, 5), sharey=True)

    for ax, (_, _, tier_label) in zip(axes, HARDWARE_TIERS):
        tier_summary = ordered[
            ordered["model"].map(lambda model: hardware_tier(sizes.get(model)))
            == tier_label
        ]
        models = tier_summary["model"].tolist()
        x = np.arange(len(models))
        colors = _bar_colors(models, model_provider, provider_color)
        if cloud_grey and tier_label == HARDWARE_TIERS[-1][2]:
            colors = [SIZE_BUCKET_COLORS[SIZE_BUCKET_UNKNOWN]] * len(models)
        ax.bar(x, tier_summary["accuracy"], color=colors)
        ax.set_title(tier_label, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 106)
        ax.set_xticks(x)
        ax.set_xticklabels(
            _labels(models, display_names), rotation=45, ha="right", fontsize=7
        )
        ax.grid(axis="y", color="#eeeeee")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        for xi, accuracy in zip(x, tier_summary["accuracy"]):
            ax.text(xi, accuracy + 1, f"{accuracy:.0f}", ha="center", fontsize=7)

    axes[0].set_ylabel("Accuracy (%)")
    handles = _provider_legend_handles(
        ordered["model"].tolist(), model_provider, provider_color
    )
    if handles:
        fig.legend(
            handles=handles,
            title="Provider",
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            frameon=False,
            fontsize=8,
            title_fontsize=9,
        )
    fig.suptitle(title, fontweight="bold", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_boxplot(
    summaries: dict[str, pd.DataFrame],
    out: str,
    title: str = "Linguistic Knowledge Benchmark Performance — by hardware tier",
    sizes: dict | None = None,
) -> None:
    """Plot boxplots whose whiskers are bootstrap 95% CIs for the mean."""
    sizes = sizes or {}
    tier_labels = [label for _, _, label in HARDWARE_TIERS]
    languages = ["english", "kiswahili"]
    language_labels = {"english": "English", "kiswahili": "Kiswahili"}
    language_colors = {"english": "#4e79a7", "kiswahili": "#f28e2b"}
    positions = np.arange(len(tier_labels)) * 3.0
    width = 0.8
    rng = np.random.default_rng(42)

    def bootstrap_stats(values: list[float]) -> dict:
        values_array = np.asarray(values, dtype=float)
        if len(values_array) == 1:
            ci_low = ci_high = float(values_array[0])
        else:
            samples = rng.choice(
                values_array, size=(1000, len(values_array)), replace=True
            )
            means = samples.mean(axis=1)
            ci_low, ci_high = np.percentile(means, [2.5, 97.5])
        return {
            "label": "",
            "q1": float(np.percentile(values_array, 25)),
            "med": float(np.percentile(values_array, 50)),
            "q3": float(np.percentile(values_array, 75)),
            "whislo": float(ci_low),
            "whishi": float(ci_high),
            "fliers": [],
        }

    stats_by_language = {}
    for language in languages:
        summary = summaries[language]
        stats_by_language[language] = []
        for tier_label in tier_labels:
            values = summary.loc[
                summary["model"].map(lambda model: hardware_tier(sizes.get(model)))
                == tier_label,
                "accuracy",
            ].tolist()
            stats_by_language[language].append(bootstrap_stats(values))

    fig, ax = plt.subplots(figsize=(12, 6))
    for offset, language in enumerate(languages):
        position = positions + (offset - 0.5) * width
        color = language_colors[language]
        boxplot = ax.bxp(
            stats_by_language[language],
            positions=position,
            widths=width,
            patch_artist=True,
            showfliers=False,
            capprops={"color": color, "linewidth": 1.2},
            whiskerprops={"color": color, "linewidth": 1.2},
            boxprops={"facecolor": color, "alpha": 0.35},
            medianprops={"color": "#222222", "linewidth": 1.5},
        )
        for patch in boxplot["boxes"]:
            patch.set_label(language_labels[language])

    ax.set_xticks(positions)
    ax.set_xticklabels(tier_labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 106)
    ax.set_title(
        f"{title} (whiskers = 95% bootstrap CI for mean)",
        fontweight="bold",
        pad=16,
        fontsize=15,
    )
    handles = [
        Patch(
            facecolor=language_colors[language],
            label=language_labels[language],
            alpha=0.35,
        )
        for language in languages
    ]
    ax.legend(handles=handles, frameon=False, title="Language")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#eeeeee")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_swarplot(
    summaries: dict[str, pd.DataFrame],
    out: str,
    title: str = "Linguistic Knowledge Benchmark Performance — by hardware tier",
    sizes: dict | None = None,
) -> None:
    """Plot jittered model accuracies with transparent English/Kiswahili boxplots."""
    sizes = sizes or {}
    tier_labels = [label for _, _, label in HARDWARE_TIERS]
    languages = ["english", "kiswahili"]
    language_labels = {"english": "English", "kiswahili": "Kiswahili"}
    language_colors = {"english": "#4e79a7", "kiswahili": "#f28e2b"}
    positions = np.arange(len(tier_labels)) * 3.0
    width = 0.8
    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(12, 6))
    for offset, language in enumerate(languages):
        summary = summaries[language]
        values_by_tier = []
        for tier_label in tier_labels:
            values = summary.loc[
                summary["model"].map(lambda model: hardware_tier(sizes.get(model)))
                == tier_label,
                "accuracy",
            ].tolist()
            values_by_tier.append(values)

        positions_for_language = positions + (offset - 0.5) * width
        color = language_colors[language]
        for position, values in zip(positions_for_language, values_by_tier):
            if values:
                jitter = rng.uniform(-width * 0.28, width * 0.28, len(values))
                ax.scatter(
                    position + jitter,
                    values,
                    s=24,
                    color=color,
                    alpha=0.65,
                    edgecolors="white",
                    linewidths=0.5,
                    zorder=2,
                )

        boxplot = ax.boxplot(
            values_by_tier,
            positions=positions_for_language,
            widths=width,
            patch_artist=True,
            manage_ticks=False,
            medianprops={"color": "#222222", "linewidth": 1.5},
            boxprops={"facecolor": color, "alpha": 0.18, "zorder": 3},
            whiskerprops={"color": color, "zorder": 3},
            capprops={"color": color, "zorder": 3},
            flierprops={"marker": "", "alpha": 0},
        )
        for patch in boxplot["boxes"]:
            patch.set_label(language_labels[language])

    ax.set_xticks(positions)
    ax.set_xticklabels(tier_labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 106)
    ax.set_title(title, fontweight="bold", pad=16, fontsize=15)
    handles = [
        Patch(
            facecolor=language_colors[language],
            label=language_labels[language],
            alpha=0.45,
        )
        for language in languages
    ]
    ax.legend(handles=handles, frameon=False, title="Language")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#eeeeee")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mcq_vs_short(
    summary: pd.DataFrame,
    out: str,
    title: str = "MCQ vs short-form accuracy per model  (label = MCQ − short)",
    display_names: dict | None = None,
) -> None:
    """Grouped bars: MCQ vs short-form accuracy per model, ordered by overall accuracy."""
    s = summary.sort_values("accuracy", ascending=False)
    models = s["model"].tolist()
    mc = s["accuracy_mcq"].to_numpy(dtype=float)
    sf = s["accuracy_short_form"].to_numpy(dtype=float)
    x = np.arange(len(models))
    w = 0.4

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(x - w / 2, mc, w, label="multiple_choice", color="#4e79a7")
    ax.bar(x + w / 2, sf, w, label="short_form", color="#f28e2b")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 106)
    ax.set_xticks(x)
    ax.set_xticklabels(
        _labels(models, display_names), rotation=35, ha="right", fontsize=8
    )
    ax.set_title(title, fontweight="bold", pad=16, fontsize=15)
    ax.legend(frameon=False)
    for xi, (m, sfx) in enumerate(zip(mc, sf)):
        gap = m - sfx
        ax.text(
            xi,
            max(m, sfx) + 1.5,
            f"{gap:+.0f}",
            ha="center",
            fontsize=8,
            color="#b30000" if gap > 0 else "#1a7a1a",
        )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#eeeeee")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_domain_grade(
    long: pd.DataFrame, out: str, note: str = "all models"
) -> pd.DataFrame:
    """Heatmap of mean accuracy by domain x grade, plus per-domain and per-grade bars.
    `note` describes the model subset (shown in the titles). Returns the table."""
    scored = long.dropna(subset=["correct"])
    mat = (
        scored.groupby(["domain", "grade"])["correct"]
        .mean()
        .mul(100)
        .unstack("grade")
        .reindex(index=DOMAINS, columns=GRADES)
    )

    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[2.1, 1.0],
        height_ratios=[1, 1],
        hspace=0.45,
        wspace=0.25,
        left=0.08,
        right=0.97,
        top=0.9,
        bottom=0.12,
    )

    # (1) domain x grade heatmap
    ax = fig.add_subplot(gs[:, 0])
    data = mat.to_numpy(dtype=float)
    im = ax.imshow(data, cmap=ACC_CMAP, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(GRADES)))
    ax.set_xticklabels([GRADE_LABEL[g] for g in GRADES], fontsize=9)
    ax.set_yticks(range(len(DOMAINS)))
    ax.set_yticklabels(DOMAINS, fontsize=9)
    ax.set_title(
        f"Mean accuracy by domain × grade ({note})  · = no items",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            v = data[r, c]
            ax.text(
                c,
                r,
                "·" if np.isnan(v) else f"{v:.0f}",
                ha="center",
                va="center",
                fontsize=8,
                color="#222222" if (np.isnan(v) or 25 < v < 80) else "white",
            )
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="accuracy %")

    # (2) accuracy by domain
    by_dom = scored.groupby("domain")["correct"].mean().mul(100).reindex(DOMAINS)
    ax = fig.add_subplot(gs[0, 1])
    ax.barh(DOMAINS, by_dom.to_numpy(dtype=float), color="#4e79a7")
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_title("Accuracy by domain", fontsize=11, fontweight="bold", pad=6)
    for i, v in enumerate(by_dom.to_numpy(dtype=float)):
        if not np.isnan(v):
            ax.text(v + 1, i, f"{v:.0f}", va="center", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # (3) accuracy by grade
    by_grade = scored.groupby("grade")["correct"].mean().mul(100).reindex(GRADES)
    ax = fig.add_subplot(gs[1, 1])
    ax.bar(
        [GRADE_LABEL[g] for g in GRADES],
        by_grade.to_numpy(dtype=float),
        color="#fd8d3c",
    )
    ax.set_ylim(0, 100)
    ax.set_title("Accuracy by grade", fontsize=11, fontweight="bold", pad=6)
    for i, v in enumerate(by_grade.to_numpy(dtype=float)):
        if not np.isnan(v):
            ax.text(i, v + 1, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Where the models struggle — accuracy across the grid  ({note})",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return mat


# --------------------------------------------------------------------------- #
# Judge disagreement / solo-judge accuracy analysis
#
# The summary/detailed CSVs only carry each model's FINAL combined verdict
# (majority vote across judges, for short_form items); the per-judge verdicts
# only exist in judge-bench's per-model judged CSVs under
# cache/<language>/experiments/<experiment-id>/responses_judged/. This section reads those
# directly to answer: how much do judges disagree, and how would each model's
# accuracy shift if only one judge (instead of the ensemble) had scored it?
# --------------------------------------------------------------------------- #
def judge_names_from_columns(columns) -> list[str]:
    """Judge names present as primary-pass 'is_correct_<judge>' columns.

    Excludes the combined 'is_correct_fallback' column and the per-judge
    fallback-pass columns ('is_correct_fallback_<judge>').
    """
    names = []
    for c in columns:
        if not c.startswith("is_correct_"):
            continue
        if c == "is_correct_fallback" or c.startswith("is_correct_fallback_"):
            continue
        names.append(c[len("is_correct_") :])
    return names


def load_judged_model(cache_experiment_dir: str, model: str) -> pd.DataFrame | None:
    """Load a model's judged CSV (primary pass), merged with its fallback pass's
    per-judge columns if a fallback pass exists. None if no judged CSV exists."""
    primary = os.path.join(
        cache_experiment_dir, "responses_judged", f"responses_judged_{model}.csv"
    )
    if not os.path.exists(primary):
        return None
    df = pd.read_csv(primary)
    fallback = os.path.join(
        cache_experiment_dir,
        "responses_judged_fallback",
        f"responses_judged_{model}.csv",
    )
    if os.path.exists(fallback):
        fb = pd.read_csv(fallback)
        fb_cols = [c for c in fb.columns if c.startswith("is_correct_fallback_")]
        if fb_cols:
            df = df.merge(fb[["question_id", *fb_cols]], on="question_id", how="left")
    return df


def solo_judge_accuracy(df: pd.DataFrame, judge: str) -> float | None:
    """Accuracy if `judge` alone had scored every short_form item.

    MCQ rows keep their regex-scored `is_correct` either way, since
    judges never touch those; only short_form rows swap in the named judge's
    own verdict (fallback-pass verdict takes priority when both exist).
    """
    primary_col = f"is_correct_{judge}"
    fallback_col = f"is_correct_fallback_{judge}"
    if primary_col not in df.columns:
        return None
    verdict = df[primary_col]
    if fallback_col in df.columns:
        verdict = df[fallback_col].fillna(verdict)
    solo = df["is_correct"].where(df["answer_type"] != "short_form", verdict)
    scored = solo.notna()
    if not scored.any():
        return None
    return float(solo[scored].mean()) * 100.0


def pairwise_judge_agreement(df: pd.DataFrame, judges: list[str]) -> pd.DataFrame:
    """Fraction of short_form rows where each judge pair's verdicts agree
    (rows where either judge didn't respond are excluded, not counted as agreeing)."""
    sf = df[df["answer_type"] == "short_form"]
    mat = pd.DataFrame(index=judges, columns=judges, dtype=float)
    for j1 in judges:
        for j2 in judges:
            if j1 == j2:
                mat.loc[j1, j2] = 1.0
                continue
            c1, c2 = f"is_correct_{j1}", f"is_correct_{j2}"
            if c1 not in sf.columns or c2 not in sf.columns:
                mat.loc[j1, j2] = float("nan")
                continue
            both = sf[[c1, c2]].dropna()
            mat.loc[j1, j2] = (
                (both[c1] == both[c2]).mean() if len(both) else float("nan")
            )
    return mat


def plot_judge_accuracy(
    report: pd.DataFrame, judges: list[str], out: str, language_name: str = ""
) -> None:
    """Grouped bars: ensemble accuracy vs each judge's solo accuracy, per model."""
    models = report["model"].tolist()
    x = np.arange(len(models))
    n_bars = len(judges) + 1
    w = 0.8 / n_bars
    palette = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948"]

    fig, ax = plt.subplots(figsize=(max(12, len(models) * 0.6), 5))
    ax.bar(
        x - 0.4 + w / 2,
        report["ensemble_accuracy"],
        w,
        label="ensemble (majority vote)",
        color="#333333",
    )
    for i, j in enumerate(judges, start=1):
        vals = report[f"accuracy_solo_{j}"].to_numpy(dtype=float)
        ax.bar(
            x - 0.4 + w / 2 + i * w, vals, w, label=j, color=palette[i % len(palette)]
        )
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 106)
    prefix = (
        f"{language_name} Linguistic Knowledge Benchmark Performance — "
        if language_name
        else ""
    )
    ax.set_title(
        f"{prefix}overall accuracy per model\n"
        f"(ensemble = majority vote across {len(judges)} LLM judges; judges only "
        "score short-form items — MCQs are regex-scored, unaffected)",
        fontweight="bold",
        pad=12,
        fontsize=11,
    )
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#eeeeee")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def judge_analysis(
    models: list[str],
    cache_experiment_dir: str,
    outdir: str,
    label: str,
    language_name: str = "",
) -> None:
    """Per-model ensemble-vs-solo-judge accuracy + judge (dis)agreement.

    Silently no-ops (with a one-line note) for models scored by fewer than 2 LLM
    judges — single-judge experiments have nothing to compare against.
    """
    rows = []
    judged_dfs: dict[str, pd.DataFrame] = {}
    all_judges: set[str] = set()
    for model in models:
        df = load_judged_model(cache_experiment_dir, model)
        if df is None:
            continue
        judges = judge_names_from_columns(df.columns)
        if len(judges) < 2:
            continue
        judged_dfs[model] = df
        all_judges.update(judges)

        sf = df[df["answer_type"] == "short_form"]
        ensemble_acc = float(df["is_correct"].dropna().mean()) * 100.0
        row = {
            "model": model,
            "n_judges": len(judges),
            "ensemble_accuracy": ensemble_acc,
            "judge_agreement_mean": sf["judge_agreement"].mean()
            if "judge_agreement" in sf.columns
            else float("nan"),
            "disagreement_rate": sf["judge_disagreement_flag"].mean() * 100.0
            if "judge_disagreement_flag" in sf.columns
            else float("nan"),
        }
        for j in judges:
            acc = solo_judge_accuracy(df, j)
            row[f"accuracy_solo_{j}"] = acc
            row[f"delta_solo_{j}"] = None if acc is None else acc - ensemble_acc
        rows.append(row)

    if not rows:
        print(
            "\n(c) Judge analysis: no model in this experiment was scored by >=2 LLM "
            "judges - skipping (nothing to compare a single judge against)."
        )
        return

    report = pd.DataFrame(rows).sort_values("ensemble_accuracy", ascending=False)
    judges_sorted = sorted(all_judges)

    print(
        f"\n(c) Judge analysis - {len(report)} model(s) scored by "
        f"{len(judges_sorted)} judges: {', '.join(judges_sorted)}"
    )
    for _, r in report.iterrows():
        parts = [f"ensemble {r['ensemble_accuracy']:5.1f}%"]
        for j in judges_sorted:
            acc = r.get(f"accuracy_solo_{j}")
            delta = r.get(f"delta_solo_{j}")
            if acc is not None and not pd.isna(acc):
                parts.append(f"{j} {acc:5.1f}% ({delta:+.1f})")
        agree = r["judge_agreement_mean"]
        disagree = r["disagreement_rate"]
        agree_str = f"{agree:.2f}" if pd.notna(agree) else "n/a"
        disagree_str = f"{disagree:.1f}%" if pd.notna(disagree) else "n/a"
        print(
            f"  {r['model']:<32} {'   '.join(parts)}   "
            f"agreement={agree_str}  disagreement={disagree_str}"
        )

    report_path = os.path.join(outdir, f"judge_analysis_{label}.csv")
    report.round(2).to_csv(report_path, index=False)

    all_df = pd.concat(judged_dfs.values(), ignore_index=True)
    pw = pairwise_judge_agreement(all_df, judges_sorted)
    print("\nPairwise judge agreement (all models combined, short_form rows):")
    print(pw.round(3).to_string())
    pw_path = os.path.join(outdir, f"judge_pairwise_agreement_{label}.csv")
    pw.round(3).to_csv(pw_path)

    plot_path = os.path.join(outdir, f"judge_accuracy_{label}.png")
    plot_judge_accuracy(report, judges_sorted, plot_path, language_name)

    print("\nWrote:")
    for p in [report_path, pw_path, plot_path]:
        print("  ", p)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--language",
        required=True,
        choices=LANGUAGES,
        help="which language's result CSVs to analyse",
    )
    ap.add_argument(
        "--experiment-id",
        required=True,
        help="experiment-id passed to judge-bench; selects "
        "summary_linguistic-knowledge_<language>_<experiment-id>[_detailed].csv",
    )
    # --summary/--detailed/--label default to that language's --experiment-id
    # (see default_paths_for above); override individually for a specific experiment.
    ap.add_argument("--summary", default=None)
    ap.add_argument("--detailed", default=None)
    ap.add_argument(
        "--dataset",
        default=None,
        help="eval dataset JSON this experiment's responses were scored against; only used "
        "to rebuild --summary/--detailed if they're missing (defaults to the "
        "language's standard dataset -- override for a experiment-id scored against a "
        "different pool, e.g. an IRT full-dataset experiment)",
    )
    ap.add_argument("--outdir", default=os.path.join(root, "results", "analysis"))
    ap.add_argument("--label", default=None, help="tag for output filenames")
    ap.add_argument(
        "--models-csv",
        dest="models_csv",
        default=os.path.join(root, "data", "models.csv"),
        help="maps model_id -> display_name (x-axis labels) and model_id -> provider (bar colours)",
    )
    ap.add_argument(
        "--providers-csv",
        dest="providers_csv",
        default=os.path.join(root, "data", "providers.csv"),
        help="maps provider -> color for the per-model bar colours and legend",
    )
    ap.add_argument(
        "--min-accuracy",
        type=float,
        default=0.0,
        help="only include models with overall accuracy >= this (%%) in the "
        "domain×grade analysis, e.g. 90 for the 90+%% models",
    )
    ap.add_argument(
        "--max-accuracy",
        type=float,
        default=100.0,
        help="only include models with overall accuracy <= this (%%)",
    )
    ap.add_argument(
        "--model-size",
        action="store_true",
        help="also write accuracy_by_model_size_<label>.png: the same accuracy bars, "
        "colored by parameter-count bucket (<4B, 4-12B; everything else grey) from "
        "models.csv's `size` column instead of by provider",
    )
    ap.add_argument(
        "--hardware",
        action="store_true",
        help="also write accuracy_by_hardware_<label>.png: four accuracy barplots "
        "grouped by estimated RAM tier (<4GB, 4-12GB, 12-35GB, >35GB) from "
        "models.csv's `size` column",
    )
    ap.add_argument(
        "--boxplot",
        action="store_true",
        help="also write accuracy_boxplot_<label>.png: paired English/Kiswahili "
        "model-accuracy distributions grouped by hardware tier",
    )
    ap.add_argument(
        "--swarplot",
        action="store_true",
        help="also write accuracy_swarplot_<label>.png: paired English/Kiswahili "
        "transparent boxplots over individual model accuracies by hardware tier",
    )
    ap.add_argument(
        "--english-experiment-id",
        default=None,
        help="experiment-id for the English summary used by --boxplot (defaults to --experiment-id)",
    )
    ap.add_argument(
        "--kiswahili-experiment-id",
        default=None,
        help="experiment-id for the Kiswahili summary used by --boxplot (defaults to --experiment-id)",
    )
    args = ap.parse_args()

    defaults = default_paths_for(
        args.language, os.path.join(root, "results"), args.experiment_id
    )
    for key in ("summary", "detailed", "label"):
        if getattr(args, key) is None:
            setattr(args, key, defaults[key])
    if args.dataset is None:
        args.dataset = os.path.join(root, DEFAULT_DATASET_PATH[args.language])

    os.makedirs(args.outdir, exist_ok=True)

    def out(name: str) -> str:
        return os.path.join(args.outdir, name)

    # Rebuild deleted summary/detailed CSVs from the still-cached judged CSVs rather
    # than requiring a full judge-bench re-run (see rebuild_summary_and_detailed()).
    if not os.path.exists(args.summary) or not os.path.exists(args.detailed):
        print(
            f"{args.summary if not os.path.exists(args.summary) else args.detailed} "
            "not found -- rebuilding from cached judged CSVs instead of re-judging."
        )
        rebuild_summary_and_detailed(
            args.language,
            args.experiment_id,
            root,
            args.dataset,
            args.summary,
            args.detailed,
        )

    summary = load_summary(args.summary)
    _, long, models = load_detailed(args.detailed)
    display_names = load_display_names(args.models_csv)
    model_provider, provider_color = load_provider_colors(
        args.models_csv, args.providers_csv
    )

    # ---- model selection for the domain×grade analysis (by overall accuracy) ----
    sel = summary[
        (summary["accuracy"] >= args.min_accuracy)
        & (summary["accuracy"] <= args.max_accuracy)
    ]
    sel_models = [m for m in sel["model"] if m in models]
    if not sel_models:
        raise SystemExit(
            f"no models in accuracy band [{args.min_accuracy}, {args.max_accuracy}]"
        )
    filtered = args.min_accuracy > 0 or args.max_accuracy < 100
    if not filtered:
        note, tag = "all models", ""
    elif args.max_accuracy >= 100:
        note = f"{len(sel_models)} models · accuracy ≥{args.min_accuracy:g}%"
        tag = f"_acc{args.min_accuracy:g}plus"
    else:
        note = f"{len(sel_models)} models · accuracy {args.min_accuracy:g}-{args.max_accuracy:g}%"
        tag = f"_acc{args.min_accuracy:g}-{args.max_accuracy:g}"
    long_sel = long[long["model"].isin(sel_models)]

    # ---- plots ----
    # Prefix the plot titles with the language ("Kiswahili" for kiswahili, else capitalized).
    language_name = {"kiswahili": "Kiswahili"}.get(
        args.language, args.language.capitalize()
    )
    plot_accuracy(
        summary,
        out(f"accuracy_by_model_{args.label}.png"),
        title=f"{language_name} Linguistic Knowledge Benchmark Performance",
        display_names=display_names,
        model_provider=model_provider,
        provider_color=provider_color,
    )
    plot_mcq_vs_short(
        summary,
        out(f"accuracy_mcq_vs_shortform_{args.label}.png"),
        title=f"{language_name} — MCQ vs short-form accuracy per model  (label = MCQ − short)",
        display_names=display_names,
    )
    if args.model_size:
        sizes = load_model_sizes(args.models_csv)
        plot_accuracy_by_size(
            summary,
            out(f"accuracy_by_model_size_{args.label}.png"),
            title=f"{language_name} Linguistic Knowledge Benchmark Performance — by hardware tier",
            display_names=display_names,
            sizes=sizes,
            model_provider=model_provider,
            provider_color=provider_color,
        )
    if args.hardware:
        sizes = load_model_sizes(args.models_csv)
        plot_accuracy_by_hardware(
            summary,
            out(f"accuracy_by_hardware_{args.label}.png"),
            title=f"{language_name} Linguistic Knowledge Benchmark Performance — by hardware tier",
            display_names=display_names,
            sizes=sizes,
            model_provider=model_provider,
            provider_color=provider_color,
        )
    if args.boxplot:
        experiment_ids = {
            "english": args.english_experiment_id or args.experiment_id,
            "kiswahili": args.kiswahili_experiment_id or args.experiment_id,
        }
        boxplot_summaries = {}
        for language in LANGUAGES:
            summary_path = default_paths_for(
                language, os.path.join(root, "results"), experiment_ids[language]
            )["summary"]
            if not os.path.exists(summary_path):
                raise SystemExit(
                    f"Cannot create --boxplot: missing {language} summary at "
                    f"{summary_path}; pass --{language}-experiment-id to select it"
                )
            boxplot_summaries[language] = load_summary(summary_path)
        sizes = load_model_sizes(args.models_csv)
        plot_accuracy_boxplot(
            boxplot_summaries,
            out(f"accuracy_boxplot_{args.label}.png"),
            title="English and Kiswahili Linguistic Knowledge Benchmark Performance — by hardware tier",
            sizes=sizes,
        )
    if args.swarplot:
        if "boxplot_summaries" not in locals():
            experiment_ids = {
                "english": args.english_experiment_id or args.experiment_id,
                "kiswahili": args.kiswahili_experiment_id or args.experiment_id,
            }
            boxplot_summaries = {}
            for language in LANGUAGES:
                summary_path = default_paths_for(
                    language, os.path.join(root, "results"), experiment_ids[language]
                )["summary"]
                if not os.path.exists(summary_path):
                    raise SystemExit(
                        f"Cannot create --swarplot: missing {language} summary at "
                        f"{summary_path}; pass --{language}-experiment-id to select it"
                    )
                boxplot_summaries[language] = load_summary(summary_path)
        sizes = load_model_sizes(args.models_csv)
        plot_accuracy_swarplot(
            boxplot_summaries,
            out(f"accuracy_swarplot_{args.label}.png"),
            title="English and Kiswahili Linguistic Knowledge Benchmark Performance — by hardware tier",
            sizes=sizes,
        )
    cell = plot_domain_grade(
        long_sel, out(f"accuracy_by_domain_grade_{args.label}{tag}.png"), note
    )
    cell.round(1).to_csv(out(f"accuracy_by_domain_grade_{args.label}{tag}.csv"))

    # ---- (a) MCQ vs short-form ----
    print(f"models: {len(models)}   questions: {long['question_id'].nunique()}")
    print("\n(a) MCQ vs short-form accuracy (%, gap = MCQ - short):")
    for _, r in summary.iterrows():
        gap = r["accuracy_mcq"] - r["accuracy_short_form"]
        print(
            f"  {r['model']:<32} MCQ {r['accuracy_mcq']:5.1f}   short {r['accuracy_short_form']:5.1f}   gap {gap:+5.1f}"
        )
    mc_mean = summary["accuracy_mcq"].mean()
    sf_mean = summary["accuracy_short_form"].mean()
    print(
        f"  {'MEAN':<32} MCQ {mc_mean:5.1f}   short {sf_mean:5.1f}   gap {mc_mean - sf_mean:+5.1f}"
    )
    # cross-check straight from the detailed correctness columns
    at = (
        long.dropna(subset=["correct"])
        .groupby("answer_type")["correct"]
        .mean()
        .mul(100)
    )
    print(
        "  cross-check (detailed): " + "  ".join(f"{k} {v:.1f}%" for k, v in at.items())
    )

    # ---- (b) domain / grade the models get wrong ----
    scored = long_sel.dropna(subset=["correct"])
    print(f"\n(b) domain/grade analysis over {note}: {', '.join(sel_models)}")
    print("    Accuracy by domain:")
    for dom, v in (
        scored.groupby("domain")["correct"].mean().mul(100).reindex(DOMAINS).items()
    ):
        if not np.isnan(v):
            print(f"  {dom}  {v:5.1f}%")
    print("\n    Accuracy by grade:")
    for g, v in (
        scored.groupby("grade")["correct"].mean().mul(100).reindex(GRADES).items()
    ):
        if not np.isnan(v):
            print(f"  {GRADE_LABEL[g]:<3} {v:5.1f}%")
    print("\n    Hardest domain×grade cells:")
    for (dom, grade), v in cell.stack().sort_values().head(8).items():
        print(f"  {dom} / {GRADE_LABEL.get(grade, grade):<3}  {v:5.1f}%")

    print("\nWrote:")
    written = [
        out(f"accuracy_by_model_{args.label}.png"),
        out(f"accuracy_mcq_vs_shortform_{args.label}.png"),
        out(f"accuracy_by_domain_grade_{args.label}{tag}.png"),
        out(f"accuracy_by_domain_grade_{args.label}{tag}.csv"),
    ]
    if args.model_size:
        written.append(out(f"accuracy_by_model_size_{args.label}.png"))
    if args.hardware:
        written.append(out(f"accuracy_by_hardware_{args.label}.png"))
    if args.boxplot:
        written.append(out(f"accuracy_boxplot_{args.label}.png"))
    if args.swarplot:
        written.append(out(f"accuracy_swarplot_{args.label}.png"))
    for p in written:
        print("  ", p)

    # ---- (c) judge disagreement / solo-judge accuracy (needs >=2 LLM judges) ----
    cache_experiment_dir = os.path.join(
        root, "cache", args.language, "experiments", args.experiment_id
    )
    judge_analysis(models, cache_experiment_dir, args.outdir, args.label, language_name)


if __name__ == "__main__":
    main()
