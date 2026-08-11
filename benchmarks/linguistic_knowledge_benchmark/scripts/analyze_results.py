#!/usr/bin/env python3
"""
Analyse a Linguistic-Knowledge benchmark run from its two result CSVs.

Inputs (both produced by `judge-bench`):
  * SUMMARY csv   — one row per model: accuracy, accuracy_mcq, accuracy_short_form.
  * DETAILED csv  — one row per question: question_id, domain, grade, answer_type,
                    question, expected_answers, then per model a `<model>_output` and
                    `<model>_correct` (True/False, blank = unscored) column.

If either CSV is missing (e.g. deleted after the run), it's rebuilt automatically from
the per-model judged CSVs still under cache/<language>/runs/<run-id>/responses_judged/
(and responses_judged_fallback/, if a --judge-fallback pass was run), using the same
judge_bench.compute_summary()/build_detailed() functions judge-bench itself uses --
no LLM judge calls, no re-running judge-bench. This needs the eval dataset the run was
actually scored against (--dataset; defaults to the language's standard dataset, wrong
for a run-id scored against a different pool -- see --dataset below). Before writing
anything, the rebuild cross-checks the judged question_ids/answer_types against
--dataset and ABORTS (no CSV written) if they don't line up -- e.g. the shipped,
IRT-selected dataset colliding on renumbered question_ids with a different item pool
-- since a wrong-but-plausible detailed CSV is worse than none.

Questions it answers:
  (a) Is there a difference between MCQ and short-form accuracy?  -> per-model grouped
      bars from the summary, plus the overall gap (also cross-checked from the detailed).
  (b) Which domain/grade are models getting wrong?  -> a domain x grade accuracy heatmap
      plus by-domain and by-grade bars, computed from the detailed correctness columns.
  (c) When the run used >=2 LLM judges (short_form items only): how much do the judges
      disagree, and what would each model's accuracy be if scored by only ONE judge
      instead of the ensemble?  -> per-model solo-judge accuracy deltas and a pairwise
      judge-agreement matrix, read from judge-bench's per-model judged CSVs under
      cache/<language>/runs/<run-id>/responses_judged/ (not the two CSVs above, which
      only carry each model's final combined verdict). Silently skipped for single-judge
      runs, since there's nothing to compare a lone judge against.

Outputs (under --outdir, tagged with --label):
    accuracy_by_model_<label>.png
    accuracy_mcq_vs_shortform_<label>.png
    accuracy_by_domain_grade_<label>.png
    accuracy_by_domain_grade_<label>.csv       the domain x grade accuracy table
    judge_analysis_<label>.csv                 (c) per-model ensemble vs solo-judge accuracy
    judge_pairwise_agreement_<label>.csv       (c) pairwise judge agreement, all models combined
    judge_accuracy_<label>.png                 (c) grouped bars: ensemble vs each judge, per model

Run (--language and --run-id are mandatory; --run-id selects the judge-bench run's
CSVs, or pass --summary/--detailed to point at exact files instead):
    python benchmarks/linguistic_knowledge_benchmark/scripts/analyze_results.py --language english --run-id default
    python .../analyze_results.py --language kiswahili --run-id v10-full-dataset
    python .../analyze_results.py --language kiswahili --run-id default --summary ... --detailed ... --label ...

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

# Fixed spine (same order as eda_pipeline.py) so empty domains/grades still show.
DOMAINS = [f"D{i}" for i in range(1, 10)]
GRADES = ["G0", "G1", "G2", "G3", "G4", "G5", "beyond_primary"]
GRADE_LABEL = {g: ("BP" if g == "beyond_primary" else g) for g in GRADES}
ACC_CMAP = "RdYlGn"  # low accuracy = red (getting it wrong), high = green
_CORRECT_MAP = {True: 1, False: 0, "True": 1, "False": 0}

# Language keys match cmd.py's DATASETS and the result CSVs' own naming
# (summary_linguistic-knowledge_<language>_<label>.csv).
LANGUAGES = ["english", "kiswahili"]

# Default eval dataset per language, mirroring cmd.py's DATASETS registry (kept as a
# plain dict here, rather than importing cmd.py, so this script doesn't pull in its
# CLI-only dependencies (typer, dotenv, fab_benchmark_publisher) for the common case
# where a rebuild isn't needed). Only used by rebuild_summary_and_detailed() below;
# override with --dataset for a run-id scored against a different pool (e.g. the IRT
# full-dataset run, whose dataset is stage_3b_dataset_full.json, not this file).
DEFAULT_DATASET_PATH = {
    "english": os.path.join("data", "linguistic_knowledge_eval_english.json"),
    "kiswahili": os.path.join("data", "linguistic_knowledge_eval_kiswahili.json"),
}


def default_paths_for(language: str, results_dir: str, run_id: str) -> dict:
    return {
        "summary": os.path.join(
            results_dir, f"summary_linguistic-knowledge_{language}_{run_id}.csv"
        ),
        "detailed": os.path.join(
            results_dir,
            f"summary_linguistic-knowledge_{language}_{run_id}_detailed.csv",
        ),
        "label": f"{language}_{run_id}",
    }


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def load_summary(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["model"].notna()].copy()
    return df.sort_values("accuracy", ascending=False).reset_index(drop=True)


def _discover_judged_models(cache_run_dir: str) -> list[str]:
    """Model ids with a primary judged CSV under cache_run_dir/responses_judged/."""
    judged_dir = os.path.join(cache_run_dir, "responses_judged")
    if not os.path.isdir(judged_dir):
        return []
    prefix, suffix = "responses_judged_", ".csv"
    return sorted(
        f[len(prefix) : -len(suffix)]
        for f in os.listdir(judged_dir)
        if f.startswith(prefix) and f.endswith(suffix)
    )


def _load_full_judged(cache_run_dir: str, model: str) -> pd.DataFrame | None:
    """Load the complete judged frame for `model`: the fallback pass's CSV if one
    exists, else the primary judged CSV. judge_results_fallback() writes its output
    by reading the full primary CSV and adding fallback columns on top, so the
    fallback file is a strict superset -- unlike load_judged_model() below (which
    merges only the per-judge fallback columns for judge_analysis's narrower needs),
    this is the complete frame compute_summary()/build_detailed() expect.
    """
    fallback = os.path.join(
        cache_run_dir, "responses_judged_fallback", f"responses_judged_{model}.csv"
    )
    if os.path.exists(fallback):
        return pd.read_csv(fallback)
    primary = os.path.join(
        cache_run_dir, "responses_judged", f"responses_judged_{model}.csv"
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
    run_id: str,
    root: str,
    dataset_path: str,
    summary_path: str,
    detailed_path: str,
) -> None:
    """Recreate a deleted summary/detailed CSV pair from the per-model judged CSVs
    still under cache/<language>/runs/<run-id>/, using the exact same functions
    cmd.py's judge-bench command uses to build them the first time
    (judge_bench.compute_summary / judge_bench.build_detailed) -- no LLM judge
    calls, no re-running judge-bench, just recomputing from what's already cached.
    """
    from linguistic_knowledge_benchmark.judge_bench import (
        build_detailed,
        compute_summary,
    )

    cache_run_dir = os.path.join(root, "cache", language, "runs", run_id)
    model_ids = _discover_judged_models(cache_run_dir)
    if not model_ids:
        raise SystemExit(
            f"Cannot rebuild {os.path.basename(summary_path)}: no judged CSVs found "
            f"under {os.path.join(cache_run_dir, 'responses_judged')}.\n"
            "Run 'judge-bench' for this language/run-id first."
        )
    if not os.path.exists(dataset_path):
        raise SystemExit(
            f"Cannot rebuild {os.path.basename(detailed_path)}: dataset not found "
            f"at {dataset_path}.\nPass --dataset to point at the JSON this run's "
            "responses were actually scored against (e.g. the IRT full-dataset "
            "pool, if --run-id is a run like 'v10-full-dataset')."
        )

    judged_by_model: dict[str, pd.DataFrame] = {}
    summaries = []
    for model in model_ids:
        df = _load_full_judged(cache_run_dir, model)
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
        f"{os.path.join(cache_run_dir, 'responses_judged')}:\n"
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
    """Legend patches for the providers present among `models`, in first-seen order."""
    seen: list[str] = []
    for m in models:
        prov = model_provider.get(m)
        if prov in provider_color and prov not in seen:
            seen.append(prov)
    return [Patch(facecolor=provider_color[p], label=p) for p in seen]


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
def plot_accuracy(
    summary: pd.DataFrame,
    out: str,
    title: str = "Linguistic Knowledge Benchmark Performance",
    display_names: dict | None = None,
    model_provider: dict | None = None,
    provider_color: dict | None = None,
) -> None:
    s = summary.sort_values("accuracy", ascending=False)
    models = s["model"].tolist()
    x = np.arange(len(models))
    model_provider = model_provider or {}
    provider_color = provider_color or {}

    fig, ax = plt.subplots(figsize=(12, 4.5))
    colors = _bar_colors(models, model_provider, provider_color)
    bars = ax.bar(x, s["accuracy"], color=colors)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 106)
    ax.set_xticks(x)
    ax.set_xticklabels(
        _labels(models, display_names), rotation=35, ha="right", fontsize=8
    )
    ax.set_title(title, fontweight="bold", pad=16, fontsize=15)
    for b, a in zip(bars, s["accuracy"]):
        ax.text(
            b.get_x() + b.get_width() / 2, a + 1, f"{a:.0f}", ha="center", fontsize=8
        )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#eeeeee")
    ax.set_axisbelow(True)

    # Provider colour legend, placed to the right of the axes.
    handles = _provider_legend_handles(models, model_provider, provider_color)
    if handles:
        ax.legend(
            handles=handles,
            title="Provider",
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
# cache/<language>/runs/<run-id>/responses_judged/. This section reads those
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


def load_judged_model(cache_run_dir: str, model: str) -> pd.DataFrame | None:
    """Load a model's judged CSV (primary pass), merged with its fallback pass's
    per-judge columns if a fallback run exists. None if no judged CSV exists."""
    primary = os.path.join(
        cache_run_dir, "responses_judged", f"responses_judged_{model}.csv"
    )
    if not os.path.exists(primary):
        return None
    df = pd.read_csv(primary)
    fallback = os.path.join(
        cache_run_dir, "responses_judged_fallback", f"responses_judged_{model}.csv"
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
    cache_run_dir: str,
    outdir: str,
    label: str,
    language_name: str = "",
) -> None:
    """Per-model ensemble-vs-solo-judge accuracy + judge (dis)agreement.

    Silently no-ops (with a one-line note) for models scored by fewer than 2 LLM
    judges — single-judge runs have nothing to compare against.
    """
    rows = []
    judged_dfs: dict[str, pd.DataFrame] = {}
    all_judges: set[str] = set()
    for model in models:
        df = load_judged_model(cache_run_dir, model)
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
            "\n(c) Judge analysis: no model in this run was scored by >=2 LLM "
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
        "--run-id",
        required=True,
        help="run-id passed to judge-bench; selects "
        "summary_linguistic-knowledge_<language>_<run-id>[_detailed].csv",
    )
    # --summary/--detailed/--label default to that language's --run-id run
    # (see default_paths_for above); override individually for a specific run.
    ap.add_argument("--summary", default=None)
    ap.add_argument("--detailed", default=None)
    ap.add_argument(
        "--dataset",
        default=None,
        help="eval dataset JSON this run's responses were scored against; only used "
        "to rebuild --summary/--detailed if they're missing (defaults to the "
        "language's standard dataset -- override for a run-id scored against a "
        "different pool, e.g. an IRT full-dataset run)",
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
    args = ap.parse_args()

    defaults = default_paths_for(
        args.language, os.path.join(root, "results"), args.run_id
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
            args.language, args.run_id, root, args.dataset, args.summary, args.detailed
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
    for p in [
        out(f"accuracy_by_model_{args.label}.png"),
        out(f"accuracy_mcq_vs_shortform_{args.label}.png"),
        out(f"accuracy_by_domain_grade_{args.label}{tag}.png"),
        out(f"accuracy_by_domain_grade_{args.label}{tag}.csv"),
    ]:
        print("  ", p)

    # ---- (c) judge disagreement / solo-judge accuracy (needs >=2 LLM judges) ----
    cache_run_dir = os.path.join(root, "cache", args.language, "runs", args.run_id)
    judge_analysis(models, cache_run_dir, args.outdir, args.label, language_name)


if __name__ == "__main__":
    main()
