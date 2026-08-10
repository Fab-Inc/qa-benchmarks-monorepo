#!/usr/bin/env python3
"""
Fit Item Response Theory to a Linguistic-Knowledge run and recommend a trimmed set.

Where it sits: this IS Stage 3b — the selection step (it replaced the old
feature-map/hard-coded-quota assembler, stage_3b_assemble_v10.py). You run every
QC-passed, de-duplicated candidate (3a's pool, collapsed to one item per dup_group)
against a panel of models with the eval harness, then feed the per-item correctness
matrix here to select the final dataset.

Input: the DETAILED results CSV (one row per question, produced by `judge-bench`):
    question_id, domain, grade, answer_type, question, expected_answers,
    then per model a `<model>_output` and `<model>_correct` (True/False, blank = unscored).
Each model is treated as an IRT respondent and `<model>_correct` as its binary response.

What it computes, per item:
  * Rasch/1PL difficulty (GIRTH marginal-maximum-likelihood) and per-model ability (EAP);
  * p-value (fraction of models correct) and a corrected point-biserial DISCRIMINATION
    (correlation of the item's correctness with each model's rest-score). With only a
    handful of model respondents the point-biserial is far more stable than a 2PL slope,
    so selection leans on it; the Rasch difficulty is reported alongside.

How it selects a trimmed dataset (three goals):
  1. No saturation — keep the top model's accuracy on the kept set as low as the pool
     allows (drop items every model gets right; prefer harder, discriminating items).
  2. Differentiation — favour high-discrimination items so models spread out; drop
     always-right and weakly/negatively discriminating items (point-biserial below
     MIN_DISCRIMINATION). ALWAYS-WRONG items are KEPT, not dropped: no model answers
     them, so they are either genuinely hard (they fight saturation) or flawed — either
     way the human-review gate should look at them. (Set --min-pvalue above 0 to drop them.)
  3. Coverage — keep a good spread across domains (D1..D9) and grades by filling a
     per-domain x grade quota proportional to the discriminating pool, allocated exactly
     via largest-remainder rounding (quotas always sum to min(--target-n, pool size) —
     no off-by-a-few from independent per-cell rounding).

Within each domain x grade cell, items are further ranked by SELECTION STABILITY before
discrimination/difficulty: `selection_freq` is the fraction of leave-`--leave-out`-out
model subpanels (all C(n_models, leave_out) of them, each with p-value/discrimination
recomputed for just that subpanel — no Rasch refit needed) that would independently pick
this item. Ranking on this first means the kept set favours items that stay competitive
across many reasonable model panels, not just the exact panel on hand — with only ~10
model respondents, point-biserial/rank estimates can shift when the panel changes.

The target-n CEILING is a SOFT target: the script reports the resulting top-model
accuracy against `--ceiling` rather than hard-forcing every model under it (on this data
the strongest model fails so few items that a hard cap would shrink the set drastically
— see the printed summary and tune `--target-n` / `--max-pvalue` from there).

It also prints two validation diagnostics (cf. Ramprasath 2026, IRT redundancy analysis):
  * ranking recovery — Spearman rho / Kendall tau-b / top-k agreement between the model
    ranking on the FULL set and on the SELECTED set. Near 1.0 means trimming widened the
    spread without scrambling who beats whom.
  * discrimination concentration — the median point-biserial and its Gini over the pool.
    High median = items genuinely separate models; high Gini (>0.6) = a few items carry
    most of the signal (very reducible), low Gini = signal spread over many items.

Outputs — one directory per run, never overwritten (see --run-id below):
  <outdir>/irt/<language>_runs/<run-id>/
    items.csv              per-item difficulty, p-value, point-biserial, selection_freq, flags
    models.csv              per-model ability, full vs selected accuracy, ranks
    selected_ids.json       the kept question_ids (sorted)
    selected_dataset.json   the kept items, filtered from --dataset (ready to use)
    selected_dataset_cleaned.json  same items, question_id renumbered 1..n (contiguous)
    diagnostics.png         difficulty vs discrimination; item difficulty spread
    accuracy_shift.png      per-model accuracy, full set vs selected set
    coverage.png            domain x grade counts of the selected set
    coverage_comparison.png 5-panel domain x grade coverage: full bank, selected set,
                            items dropped, plus %% of each domain's and each grade's
                            own before-cut total that got dropped -- answers "which
                            domains/grades lost coverage in the cut?" directly (the
                            printed per-domain coverage table below answers the same
                            question for domains in text form)
    selection_freq.png      selection_freq distribution over the candidate pool
--outdir defaults to results/; --run-id defaults to "default". If any of the above files
already exist in that run's directory, the script aborts rather than overwrite them —
pass a different --run-id (or delete the old run's directory) to rerun.

Run (--language is mandatory; defaults target that language's v10 full-dataset run):
    python benchmarks/linguistic_knowledge_benchmark/scripts/estimate_irt.py --language kiswahili
    python .../estimate_irt.py --language english --run-id target150 --target-n 150
    python .../estimate_irt.py --language kiswahili --max-pvalue 0.9   # also drop items >90%% of models ace
    python .../estimate_irt.py --language kiswahili --leave-out 1   # leave-one-out stability instead
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analyze_results import LANGUAGES
from girth import ability_eap, rasch_mml

# Fixed spine (same order as analyze_results.py / eda_pipeline.py) so empty cells still show.
DOMAINS = [f"D{i}" for i in range(1, 10)]
GRADES = ["G0", "G1", "G2", "G3", "G4", "G5", "beyond_primary"]
GRADE_LABEL = {g: ("BP" if g == "beyond_primary" else g) for g in GRADES}
DISCRIMINATION = 1.0  # Rasch fixes the slope at 1.
_CORRECT_MAP = {"true": 1, "false": 0, "1": 1, "0": 0}

# The result CSVs and cmd.py's DATASETS both key on "kiswahili"; the pipeline's
# working-data directory for that language is named "kiswahili_workflow" too.
LANGUAGE_WORKFLOW_DIR = {
    "english": "english_workflow",
    "kiswahili": "kiswahili_workflow",
}

# The run tag differs per language: Kiswahili's full-pool eval run is tagged
# "v10-full-dataset" (528 detailed rows -> 307 items scored); English's equivalent run
# (its whole 377-item stage_3b_dataset_full pool) is simply tagged "default".
DEFAULT_RUN_LABEL = {
    "english": "default",
    "kiswahili": "v10-full-dataset",
}


def default_paths_for(language: str, root: str) -> dict:
    run_label = DEFAULT_RUN_LABEL[language]
    return {
        "detailed": os.path.join(
            root,
            "results",
            f"summary_linguistic-knowledge_{language}_{run_label}_detailed.csv",
        ),
        "dataset": os.path.join(
            root,
            "data",
            LANGUAGE_WORKFLOW_DIR[language],
            "process",
            "stage_3b_dataset_full.json",
        ),
    }


# Fixed filenames written into each run's own directory (see main()'s --run-id / rundir).
OUTPUT_FILENAMES = [
    "items.csv",
    "models.csv",
    "selected_ids.json",
    "selected_dataset.json",
    "selected_dataset_cleaned.json",
    "diagnostics.png",
    "accuracy_shift.png",
    "coverage.png",
    "coverage_comparison.png",
    "selection_freq.png",
]


def _refuse_if_exists(rundir: str, filenames: list[str]) -> None:
    """Abort rather than silently overwrite a previous run's output files."""
    if not os.path.isdir(rundir):
        return
    clashes = [f for f in filenames if os.path.exists(os.path.join(rundir, f))]
    if clashes:
        raise SystemExit(
            f"refusing to overwrite existing run output in {rundir}:\n"
            + "\n".join(f"  - {f}" for f in clashes)
            + "\nUse a different --run-id, or delete these files first."
        )


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def _to01(value) -> float:
    """Map a `<model>_correct` cell to 1.0 / 0.0, or NaN when blank/unscored."""
    return _CORRECT_MAP.get(str(value).strip().lower(), np.nan)


def load_matrix(detailed_path: str):
    """Load the wide detailed CSV into an item x model correctness matrix.

    Returns (meta, models, matrix, scored_mask) where `meta` is a DataFrame with
    question_id/domain/grade/answer_type per item (full row order), `matrix` is a float
    ndarray (items x models) with NaN for unscored cells, and `scored_mask` marks the rows
    that are fully scored (usable for the IRT fit).
    """
    df = pd.read_csv(detailed_path)
    corr_cols = [c for c in df.columns if c.endswith("_correct")]
    if not corr_cols:
        raise SystemExit(f"no '<model>_correct' columns found in {detailed_path}")
    models = [c[: -len("_correct")] for c in corr_cols]

    meta = df[
        [
            c
            for c in ["question_id", "domain", "grade", "answer_type"]
            if c in df.columns
        ]
    ].copy()
    matrix = df[corr_cols].apply(lambda col: col.map(_to01)).to_numpy(dtype=float)
    scored_mask = ~np.isnan(matrix).any(axis=1)
    return meta, models, matrix, scored_mask


# --------------------------------------------------------------------------- #
# IRT + classical item statistics
# --------------------------------------------------------------------------- #
def fit_rasch(scored: np.ndarray):
    """Fit Rasch/1PL to a complete items x respondents 0/1 matrix.

    Returns (difficulty[item], ability[model]). Items answered identically by every
    model have no finite difficulty; GIRTH returns optimiser bounds for those (they are
    flagged `boundary` downstream and never selected).
    """
    estimates = rasch_mml(scored, discrimination=DISCRIMINATION)
    difficulty = np.asarray(estimates["Difficulty"], dtype=float)
    ability = np.asarray(ability_eap(scored, difficulty, DISCRIMINATION), dtype=float)
    return difficulty, ability


def point_biserial(scored: np.ndarray) -> np.ndarray:
    """Corrected item-total (point-biserial) discrimination for each item.

    For item i, correlate its 0/1 responses across models with each model's REST score
    (total correct excluding item i). Correcting for the item removes the spurious
    self-correlation. Boundary items (all-right / all-wrong) have zero variance -> NaN.
    """
    totals = scored.sum(axis=0)  # per-model total correct (over scored items)
    out = np.full(scored.shape[0], np.nan)
    for i in range(scored.shape[0]):
        item = scored[i, :]
        rest = totals - item
        if item.std() == 0 or rest.std() == 0:
            continue  # boundary item or degenerate rest score
        out[i] = np.corrcoef(item, rest)[0, 1]
    return out


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def allocate_quotas_lr(cell_counts: pd.Series, target_n: int) -> pd.Series:
    """Largest-remainder allocation of `target_n` across cells, proportional to
    `cell_counts` and capped by each cell's availability.

    Unlike independent per-cell rounding, the returned quotas always sum to exactly
    `min(target_n, cell_counts.sum())`: each cell first gets `floor(share)`, then the
    leftover seats go one-by-one to the cells with the largest fractional remainder
    (a stable sort keeps this deterministic for tied remainders).
    """
    total_pool = int(cell_counts.sum())
    n_target = min(target_n, total_pool)
    if total_pool == 0 or n_target <= 0:
        return pd.Series(0, index=cell_counts.index, dtype=int)

    raw = cell_counts / total_pool * n_target
    quotas = np.floor(raw).astype(int)
    remainder = raw - quotas
    leftover = n_target - int(quotas.sum())
    if leftover > 0:
        order = remainder.sort_values(ascending=False, kind="stable").index
        quotas.loc[order[:leftover]] += 1
    return quotas


def select_items(
    items: pd.DataFrame,
    target_n: int,
    min_disc: float,
    max_pvalue: float,
    min_pvalue: float,
    primary_rank_col: str | None = None,
) -> pd.Series:
    """Choose a balanced, discriminating subset. Returns a boolean `selected` Series.

    Pool = scored items that are NOT always-right and have p-value within
    [min_pvalue, max_pvalue], keeping either point_biserial >= min_disc OR an always-wrong
    item (no model correct — kept as a hard/anti-saturation candidate for human review).
    Each domain x grade cell gets an exact quota (largest-remainder allocation, see
    `allocate_quotas_lr`) proportional to its share of the pool, scaled to target_n and
    capped by availability. Within a cell, always-wrong items are taken first (scarce,
    anti-saturation, human-review-gated), then — if `primary_rank_col` is given (e.g.
    "selection_freq") — items ranked highest on that column, then the most discriminating
    (then hardest) items. Deterministic given the same inputs.
    """
    pool = (
        items["scored"]
        & ~items["all_right"]
        & (items["all_wrong"] | (items["point_biserial"] >= min_disc))
        & (items["p_value"] <= max_pvalue)
        & (items["p_value"] >= min_pvalue)
    )
    selected = pd.Series(False, index=items.index)
    pool_items = items[pool]
    total_pool = len(pool_items)
    if total_pool == 0:
        return selected

    cell_counts = pool_items.groupby(["domain", "grade"]).size()
    quotas = allocate_quotas_lr(cell_counts, target_n)
    for (domain, grade), quota in quotas.items():
        if quota <= 0:
            continue
        cell = pool_items[
            (pool_items["domain"] == domain) & (pool_items["grade"] == grade)
        ]
        # Always-wrong items first, then (optionally) selection stability, then most
        # discriminating, then hardest, then lowest p-value, then stable id.
        # Always-wrong discrimination is NaN -> treat as 0.
        cell = cell.assign(_disc=cell["point_biserial"].fillna(0.0))
        sort_cols = ["all_wrong"]
        ascending = [False]
        if primary_rank_col is not None:
            sort_cols.append(primary_rank_col)
            ascending.append(False)
        sort_cols += ["_disc", "difficulty_logit", "p_value", "question_id"]
        ascending += [False, False, True, True]
        ranked = cell.sort_values(by=sort_cols, ascending=ascending)
        selected.loc[ranked.index[:quota]] = True
    return selected


def _stats_for_columns(
    meta: pd.DataFrame, matrix: np.ndarray, cols: list[int]
) -> pd.DataFrame:
    """Rebuild a per-item stats table (p-value, point-biserial, all-right/-wrong flags)
    using only the model columns in `cols` — i.e. one leave-`k`-out subpanel.

    No Rasch refit here (see module docstring): `difficulty_logit` is stood in with
    `-p_value`, which is rank-equivalent for `select_items`'s tie-break under a common
    Rasch discrimination. The subpanel's own scored mask is recomputed from `cols` alone,
    since dropping columns can turn a row that was unscored over the full panel into a
    fully-scored one for this subpanel.
    """
    sub = matrix[:, cols]
    scored_mask = ~np.isnan(sub).any(axis=1)
    scored = sub[scored_mask].astype(int)
    n_sub = len(cols)

    items = meta[["question_id", "domain", "grade"]].copy()
    items["scored"] = scored_mask
    p_value = np.full(len(items), np.nan)
    pb = np.full(len(items), np.nan)
    if scored.shape[0] > 0:
        p_value[scored_mask] = scored.sum(axis=1) / n_sub
        pb[scored_mask] = point_biserial(scored)
    items["p_value"] = p_value
    items["point_biserial"] = pb
    items["difficulty_logit"] = -items["p_value"]
    items["all_right"] = items["scored"] & (items["p_value"] == 1.0)
    items["all_wrong"] = items["scored"] & (items["p_value"] == 0.0)
    return items


def selection_frequency(
    meta: pd.DataFrame,
    matrix: np.ndarray,
    n_models: int,
    target_n: int,
    min_disc: float,
    max_pvalue: float,
    min_pvalue: float,
    leave_out: int = 2,
) -> pd.Series:
    """Fraction of leave-`leave_out`-out model subpanels that would select each item.

    Iterates every C(n_models, leave_out) subpanel, recomputes p-value/discrimination
    for just that subpanel (`_stats_for_columns`), reruns `select_items` on it, and
    averages how often each item gets picked. An item's `selection_freq` close to 1.0
    means it keeps getting chosen regardless of exactly which models are on the panel.
    """
    omit_sets = list(combinations(range(n_models), leave_out))
    counts = pd.Series(0, index=meta.index, dtype=int)
    for omitted in omit_sets:
        keep = [j for j in range(n_models) if j not in omitted]
        sub_items = _stats_for_columns(meta, matrix, keep)
        sel = select_items(sub_items, target_n, min_disc, max_pvalue, min_pvalue)
        counts += sel.astype(int)
    return counts / len(omit_sets)


# --------------------------------------------------------------------------- #
# Per-model accuracy (full vs selected)
# --------------------------------------------------------------------------- #
def model_table(
    models: list[str],
    matrix: np.ndarray,
    scored_mask: np.ndarray,
    selected_row_mask: np.ndarray,
    ability: np.ndarray,
) -> pd.DataFrame:
    """Per-model ability plus accuracy on the full scored set and on the selected set."""
    scored = matrix[scored_mask]
    sel = matrix[selected_row_mask]
    rows = []
    for j, m in enumerate(models):
        full_correct = int(np.nansum(scored[:, j]))
        full_total = scored.shape[0]
        sel_correct = int(np.nansum(sel[:, j]))
        sel_total = sel.shape[0]
        rows.append(
            {
                "model": m,
                "ability_logit": ability[j],
                "n_items_full": full_total,
                "correct_full": full_correct,
                "accuracy_full": 100.0 * full_correct / full_total
                if full_total
                else np.nan,
                "n_items_selected": sel_total,
                "correct_selected": sel_correct,
                "accuracy_selected": 100.0 * sel_correct / sel_total
                if sel_total
                else np.nan,
            }
        )
    tbl = pd.DataFrame(rows)
    tbl["rank_full"] = (
        tbl["accuracy_full"].rank(method="min", ascending=False).astype(int)
    )
    tbl["rank_selected"] = (
        tbl["accuracy_selected"].rank(method="min", ascending=False).astype(int)
    )
    return tbl.sort_values("accuracy_full", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Validation diagnostics (ranking recovery + discrimination concentration)
# --------------------------------------------------------------------------- #
def _spearman(a, b) -> float:
    """Spearman rho = Pearson correlation of the rank-transformed values (ties averaged)."""
    ra = pd.Series(np.asarray(a, dtype=float)).rank().to_numpy()
    rb = pd.Series(np.asarray(b, dtype=float)).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _kendall_tau_b(a, b) -> float:
    """Kendall tau-b (tie-corrected) over all pairs; fine for the handful of models here."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    c = d = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            da, db = a[i] - a[j], b[i] - b[j]
            if da == 0 and db == 0:
                continue
            if da == 0:
                tx += 1
            elif db == 0:
                ty += 1
            elif (da > 0) == (db > 0):
                c += 1
            else:
                d += 1
    denom = math.sqrt((c + d + tx) * (c + d + ty))
    return (c - d) / denom if denom > 0 else float("nan")


def _topk_agreement(full_order: list[str], sel_order: list[str], k: int) -> float:
    """Fraction of the full set's top-k models that are also in the selected set's top-k."""
    k = min(k, len(full_order))
    if k == 0:
        return float("nan")
    return len(set(full_order[:k]) & set(sel_order[:k])) / k


def _gini(x: np.ndarray) -> float:
    """Gini concentration of a non-negative array (0 = perfectly even, ->1 = concentrated)."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return float("nan")
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_diagnostics(items: pd.DataFrame, out: str) -> None:
    """Left: difficulty vs discrimination, coloured by pool/selected. Right: item spread."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Always-wrong / always-right items have undefined discrimination (NaN) -> plot at 0.
    d = items.copy()
    d["_pb"] = d["point_biserial"].fillna(0.0)
    excluded = d[~d["in_pool"]]
    pool_only = d[d["in_pool"] & ~d["selected"]]
    chosen = d[d["selected"]]
    ax1.scatter(
        excluded["difficulty_logit"],
        excluded["_pb"],
        s=18,
        color="#cccccc",
        label="excluded",
        zorder=1,
    )
    ax1.scatter(
        pool_only["difficulty_logit"],
        pool_only["_pb"],
        s=22,
        color="#4e79a7",
        alpha=0.7,
        label="pool, not kept",
        zorder=2,
    )
    ax1.scatter(
        chosen["difficulty_logit"],
        chosen["_pb"],
        s=30,
        color="#e15759",
        label="selected",
        zorder=3,
    )
    ax1.axhline(0, color="#888888", lw=0.8)
    ax1.set_xlabel("Rasch difficulty (logit)")
    ax1.set_ylabel("point-biserial discrimination (NaN → 0)")
    ax1.set_title("Item difficulty vs discrimination", fontweight="bold")
    ax1.legend(frameon=False, fontsize=8)
    ax1.spines[["top", "right"]].set_visible(False)

    # Item spread by #models correct, selected overlaid.
    n_models = int(items["n_models_scored"].max())
    bins = np.arange(-0.5, n_models + 1.5, 1)
    ax2.hist(
        items.loc[items["scored"], "n_correct"],
        bins=bins,
        color="#dddddd",
        label="all scored items",
    )
    ax2.hist(
        items.loc[items["selected"], "n_correct"],
        bins=bins,
        color="#e15759",
        label="selected",
    )
    ax2.set_xlabel(f"# models correct (of {n_models})")
    ax2.set_ylabel("items")
    ax2.set_title("Difficulty spread (item p-value)", fontweight="bold")
    ax2.set_xticks(range(n_models + 1))
    ax2.legend(frameon=False, fontsize=8)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_accuracy_shift(tbl: pd.DataFrame, ceiling: float, out: str) -> None:
    """Per-model accuracy on the full set vs the selected set (grouped horizontal bars)."""
    s = tbl.sort_values("accuracy_full")  # best ends on top after barh
    y = np.arange(len(s))
    h = 0.4
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(y + h / 2, s["accuracy_full"], h, color="#bab0ac", label="full set")
    ax.barh(y - h / 2, s["accuracy_selected"], h, color="#e15759", label="selected set")
    ax.axvline(ceiling, color="#333333", ls="--", lw=1, label=f"ceiling {ceiling:g}%")
    ax.set_yticks(y)
    ax.set_yticklabels(s["model"], fontsize=8)
    ax.set_xlabel("Accuracy (%)")
    ax.set_xlim(0, 100)
    ax.set_title("Model accuracy — full vs selected set", fontweight="bold")
    for yi, (af, asel) in enumerate(zip(s["accuracy_full"], s["accuracy_selected"])):
        ax.text(
            af + 0.8, yi + h / 2, f"{af:.0f}", va="center", fontsize=7, color="#555555"
        )
        ax.text(
            asel + 0.8,
            yi - h / 2,
            f"{asel:.0f}",
            va="center",
            fontsize=7,
            color="#e15759",
        )
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#eeeeee")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _domain_grade_matrix(df: pd.DataFrame) -> np.ndarray:
    """Domain x grade item counts for `df`, reindexed onto the fixed DOMAINS/GRADES
    spine (so empty cells are 0 rather than absent)."""
    mat = (
        df.groupby(["domain", "grade"])
        .size()
        .unstack("grade")
        .reindex(index=DOMAINS, columns=GRADES)
    )
    return np.nan_to_num(mat.to_numpy(dtype=float))


def _draw_coverage_heatmap(ax, data: np.ndarray, title: str, cmap: str) -> None:
    """Shared domain x grade heatmap rendering (axes/ticks/cell-value annotations)."""
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0)
    ax.set_xticks(range(len(GRADES)))
    ax.set_xticklabels([GRADE_LABEL[g] for g in GRADES], fontsize=9)
    ax.set_yticks(range(len(DOMAINS)))
    ax.set_yticklabels(DOMAINS, fontsize=9)
    ax.set_title(f"{title}  (n={int(data.sum())})", fontweight="bold", pad=8)
    vmax = data.max() if data.max() > 0 else 1.0
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            n = int(data[r, c])
            ax.text(
                c,
                r,
                "·" if n == 0 else str(n),
                ha="center",
                va="center",
                fontsize=8,
                color="#222222" if data[r, c] < vmax * 0.6 else "white",
            )
    return im


def plot_coverage(items: pd.DataFrame, out: str) -> None:
    """Domain x grade counts of the selected set (heatmap)."""
    data = _domain_grade_matrix(items[items["selected"]])
    fig, ax = plt.subplots(figsize=(9, 6))
    im = _draw_coverage_heatmap(ax, data, "Selected items by domain × grade", "Blues")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="items")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _pct_dropped(before_totals: np.ndarray, after_totals: np.ndarray) -> np.ndarray:
    """% of each row/column's own before-cut total that got dropped; NaN where
    the before-cut total is 0 (nothing to drop a percentage of)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(
            before_totals > 0,
            100.0 * (before_totals - after_totals) / before_totals,
            np.nan,
        )


def plot_coverage_comparison(items: pd.DataFrame, out: str) -> None:
    """Domain x grade heatmap: the full item bank, the selected set, the difference
    (items dropped per cell), then -- on the right -- each domain's %% dropped and
    each grade's %% dropped, each proportional to its OWN before-cut total (not the
    pool total). Those last two panels directly answer "which domains/grades lost
    coverage in the cut?": the per-cell diff heatmap shows raw counts, which favours
    large domains/grades just for having more items to lose; normalizing by each
    one's starting size means a small domain/grade that got wiped out shows up as
    clearly as a large one.

    `items` is every item that entered this IRT run (scored or not); "before"
    counts all of them, "after" counts only those with `selected=True`. Since
    selected is always a subset, the diff panel is >= 0 everywhere (cells never
    gain items).
    """
    before = _domain_grade_matrix(items)
    after = _domain_grade_matrix(items[items["selected"]])
    diff = before - after

    pct_dropped_domain = _pct_dropped(before.sum(axis=1), after.sum(axis=1))
    pct_dropped_grade = _pct_dropped(before.sum(axis=0), after.sum(axis=0))

    fig, axes = plt.subplots(
        1, 5, figsize=(30, 6), gridspec_kw={"width_ratios": [1, 1, 1, 0.7, 0.7]}
    )
    im0 = _draw_coverage_heatmap(axes[0], before, "Full item bank (before)", "Blues")
    im1 = _draw_coverage_heatmap(axes[1], after, "Selected set (after)", "Blues")
    im2 = _draw_coverage_heatmap(
        axes[2], diff, "Dropped in the cut (before − after)", "Reds"
    )
    for ax, im in zip(axes[:3], [im0, im1, im2]):
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="items")

    # 4th panel: % of each domain's own before-cut total that got dropped.
    # Row-aligned with the heatmaps (same DOMAINS order, D1 at top) so a row reads
    # as one continuous story: before -> after -> raw dropped -> %% of that domain lost.
    ax = axes[3]
    y = np.arange(len(DOMAINS))
    colors = plt.get_cmap("Reds")(
        np.clip(np.nan_to_num(pct_dropped_domain) / 100.0, 0.15, 1.0)
    )
    ax.barh(y, np.nan_to_num(pct_dropped_domain), color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(DOMAINS, fontsize=9)
    ax.invert_yaxis()  # D1 at top, matching the heatmaps' row order
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of domain dropped")
    ax.set_title(
        "% dropped\n(of domain's own total)", fontweight="bold", fontsize=10, pad=8
    )
    for i, pct in enumerate(pct_dropped_domain):
        label = "n/a" if np.isnan(pct) else f"{pct:.0f}%"
        x = 0 if np.isnan(pct) else pct
        ax.text(x + 2, i, label, va="center", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # 5th panel: % of each grade's own before-cut total that got dropped, grade on
    # the x-axis -- column-aligned with the heatmaps' GRADES order (G0 at left).
    ax = axes[4]
    x = np.arange(len(GRADES))
    colors = plt.get_cmap("Reds")(
        np.clip(np.nan_to_num(pct_dropped_grade) / 100.0, 0.15, 1.0)
    )
    ax.bar(x, np.nan_to_num(pct_dropped_grade), color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels([GRADE_LABEL[g] for g in GRADES], fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of grade dropped")
    ax.set_title(
        "% dropped\n(of grade's own total)", fontweight="bold", fontsize=10, pad=8
    )
    for i, pct in enumerate(pct_dropped_grade):
        label = "n/a" if np.isnan(pct) else f"{pct:.0f}%"
        y_ = 0 if np.isnan(pct) else pct
        ax.text(i, y_ + 2, label, ha="center", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Domain × grade coverage: before vs after the IRT cut", fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_selection_freq(items: pd.DataFrame, leave_out: int, out: str) -> None:
    """Distribution of `selection_freq` (leave-`leave_out`-out stability) over the
    candidate pool, selected items overlaid."""
    pool = items[items["in_pool"]]
    bins = np.arange(-0.05, 1.15, 0.1)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(pool["selection_freq"], bins=bins, color="#dddddd", label="candidate pool")
    ax.hist(
        pool.loc[pool["selected"], "selection_freq"],
        bins=bins,
        color="#e15759",
        label="selected",
    )
    ax.set_xlabel(f"selection_freq (leave-{leave_out}-out)")
    ax.set_ylabel("items")
    ax.set_xlim(-0.05, 1.05)
    ax.set_title(
        "How often would each item be picked under a different model panel?",
        fontweight="bold",
    )
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


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
        help="which language's v10 full-dataset run to analyse",
    )
    # --detailed/--dataset default to that language's v10-full-dataset run (see
    # default_paths_for above); override individually for a specific run.
    ap.add_argument("--detailed", default=None)
    ap.add_argument(
        "--dataset",
        default=None,
        help="full item pool to filter into the selected dataset (matched on question_id)",
    )
    ap.add_argument(
        "--outdir",
        default=os.path.join(root, "results"),
        help="base results dir; this run's files go under "
        "<outdir>/irt/<language>_runs/<run-id>/",
    )
    ap.add_argument(
        "--run-id",
        default="default",
        help="tag for this run's output directory (results/irt/<language>_runs/<run-id>/); "
        "pass a specific name, or 'timestamp' to auto-generate one (YYYYmmdd-HHMMSS); "
        "refuses to overwrite an existing run's files — pick a new --run-id instead",
    )
    ap.add_argument(
        "--target-n", type=int, default=150, help="target size of the selected set"
    )
    ap.add_argument(
        "--ceiling",
        type=float,
        default=90.0,
        help="top-accuracy ceiling to report against (soft; not hard-enforced)",
    )
    ap.add_argument(
        "--min-discrimination",
        type=float,
        default=0.0,
        help="drop items whose point-biserial is below this (keep only positive by default)",
    )
    ap.add_argument(
        "--max-pvalue",
        type=float,
        default=1.0,
        help="drop items more than this fraction of models get right (e.g. 0.9)",
    )
    ap.add_argument(
        "--min-pvalue",
        type=float,
        default=0.0,
        help="drop items fewer than this fraction of models get right",
    )
    ap.add_argument(
        "--leave-out",
        type=int,
        default=2,
        help="models omitted per subpanel when computing selection_freq "
        "(default 2 = leave-two-out; try 1 for leave-one-out)",
    )
    args = ap.parse_args()

    defaults = default_paths_for(args.language, root)
    for key in ("detailed", "dataset"):
        if getattr(args, key) is None:
            setattr(args, key, defaults[key])

    if args.run_id == "timestamp":
        args.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    elif os.sep in args.run_id or "/" in args.run_id:
        raise SystemExit(f"--run-id must be a bare name, not a path: {args.run_id!r}")

    rundir = os.path.join(args.outdir, "irt", f"{args.language}_runs", args.run_id)
    _refuse_if_exists(rundir, OUTPUT_FILENAMES)
    os.makedirs(rundir, exist_ok=True)

    def out(name: str) -> str:
        return os.path.join(rundir, name)

    meta, models, matrix, scored_mask = load_matrix(args.detailed)
    n_models = len(models)
    n_unscored = int((~scored_mask).sum())

    # Fit Rasch on the fully scored items only.
    scored = matrix[scored_mask].astype(int)
    difficulty, ability = fit_rasch(scored)
    pb = point_biserial(scored)

    # Assemble a per-item table over ALL items (unscored ones carry NaN stats, never pooled).
    items = meta.copy()
    items["scored"] = scored_mask
    items["n_models_scored"] = np.where(scored_mask, n_models, 0)
    n_correct = np.full(len(items), np.nan)
    p_value = np.full(len(items), np.nan)
    diff_col = np.full(len(items), np.nan)
    pb_col = np.full(len(items), np.nan)
    n_correct[scored_mask] = scored.sum(axis=1)
    p_value[scored_mask] = scored.sum(axis=1) / n_models
    diff_col[scored_mask] = difficulty
    pb_col[scored_mask] = pb
    items["n_correct"] = n_correct
    items["p_value"] = p_value
    items["difficulty_logit"] = diff_col
    items["point_biserial"] = pb_col
    items["all_right"] = items["scored"] & (items["n_correct"] == n_models)
    items["all_wrong"] = items["scored"] & (items["n_correct"] == 0)
    # 'boundary' = Rasch difficulty is only an optimiser bound (all-right / all-wrong);
    # kept for the CSV/summary. Always-right items are dropped; always-wrong are kept.
    items["boundary"] = (
        items["all_right"]
        | items["all_wrong"]
        | (items["scored"] & items["point_biserial"].isna())
    )

    items["selection_freq"] = selection_frequency(
        meta,
        matrix,
        n_models,
        args.target_n,
        args.min_discrimination,
        args.max_pvalue,
        args.min_pvalue,
        args.leave_out,
    )
    items["selected"] = select_items(
        items,
        args.target_n,
        args.min_discrimination,
        args.max_pvalue,
        args.min_pvalue,
        primary_rank_col="selection_freq",
    )
    items["in_pool"] = (
        items["scored"]
        & ~items["all_right"]
        & (items["all_wrong"] | (items["point_biserial"] >= args.min_discrimination))
        & (items["p_value"] <= args.max_pvalue)
        & (items["p_value"] >= args.min_pvalue)
    )

    selected_row_mask = items["selected"].to_numpy()
    tbl = model_table(models, matrix, scored_mask, selected_row_mask, ability)

    # ---- write per-item / per-model CSVs ----
    item_cols = [
        "question_id",
        "domain",
        "grade",
        "answer_type",
        "scored",
        "n_models_scored",
        "n_correct",
        "p_value",
        "difficulty_logit",
        "point_biserial",
        "selection_freq",
        "all_right",
        "all_wrong",
        "boundary",
        "in_pool",
        "selected",
    ]
    items_out = items[item_cols].copy()
    items_out["p_value"] = items_out["p_value"].round(3)
    items_out["difficulty_logit"] = items_out["difficulty_logit"].round(3)
    items_out["point_biserial"] = items_out["point_biserial"].round(3)
    items_out["selection_freq"] = items_out["selection_freq"].round(3)
    items_out.to_csv(out("items.csv"), index=False)

    tbl_out = tbl.copy()
    for c in ["ability_logit", "accuracy_full", "accuracy_selected"]:
        tbl_out[c] = tbl_out[c].round(3)
    tbl_out.to_csv(out("models.csv"), index=False)

    # ---- selected ids + filtered dataset ----
    selected_ids = sorted(int(q) for q in items.loc[items["selected"], "question_id"])
    with open(out("selected_ids.json"), "w", encoding="utf-8") as fh:
        json.dump(selected_ids, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    n_selected_dataset = 0
    if os.path.exists(args.dataset):
        with open(args.dataset, encoding="utf-8") as fh:
            full = json.load(fh)
        keep = set(selected_ids)
        subset = [it for it in full if it.get("question_id") in keep]
        subset.sort(key=lambda it: it.get("question_id", 0))
        with open(out("selected_dataset.json"), "w", encoding="utf-8") as fh:
            json.dump(subset, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        n_selected_dataset = len(subset)

        # Same items, renumbered question_id 1..n so the benchmark ships with a clean,
        # contiguous id range instead of the original (sparse) Stage-3b ids.
        cleaned = [
            {**it, "question_id": new_id} for new_id, it in enumerate(subset, start=1)
        ]
        with open(out("selected_dataset_cleaned.json"), "w", encoding="utf-8") as fh:
            json.dump(cleaned, fh, ensure_ascii=False, indent=2)
            fh.write("\n")

    # ---- plots ----
    plot_diagnostics(items, out("diagnostics.png"))
    plot_accuracy_shift(tbl, args.ceiling, out("accuracy_shift.png"))
    plot_coverage(items, out("coverage.png"))
    plot_coverage_comparison(items, out("coverage_comparison.png"))
    plot_selection_freq(items, args.leave_out, out("selection_freq.png"))

    # ---- console summary ----
    n_sel = int(items["selected"].sum())
    top_full = tbl["accuracy_full"].max()
    top_sel = tbl["accuracy_selected"].max()
    spread_full = tbl["accuracy_full"].max() - tbl["accuracy_full"].min()
    spread_sel = tbl["accuracy_selected"].max() - tbl["accuracy_selected"].min()
    all_right = int(items["all_right"].sum())
    all_wrong = int(items["all_wrong"].sum())
    n_sel_wrong = int((items["selected"] & items["all_wrong"]).sum())
    print(
        f"models={n_models}  items={len(items)}  unscored(dropped from IRT)={n_unscored}  "
        f"all-right(dropped)={all_right}  all-wrong(kept)={all_wrong}"
    )
    n_pool = int(items["in_pool"].sum())
    n_target = min(args.target_n, n_pool)
    print(
        f"pool={n_pool}  selected={n_sel}  (target {args.target_n})  "
        f"filtered_dataset={n_selected_dataset}  of which always-wrong={n_sel_wrong}"
    )
    print(
        f"exact quota check: selected == min(target_n, pool) -> {n_sel} == {n_target}  "
        f"{'OK' if n_sel == n_target else 'MISMATCH'}"
    )
    print(
        f"top accuracy: full={top_full:.1f}%  selected={top_sel:.1f}%  "
        f"(ceiling {args.ceiling:g}% — {'MET' if top_sel < args.ceiling else 'NOT met, soft'})"
    )
    print(
        f"accuracy spread (max-min): full={spread_full:.1f}pp  selected={spread_sel:.1f}pp"
    )

    # ---- selection stability (leave-`leave_out`-out) ----
    n_subpanels = math.comb(n_models, args.leave_out)
    sel_freq = items.loc[items["selected"], "selection_freq"]
    n_unstable = int((sel_freq < 0.5).sum())
    print(
        f"selection stability: leave-{args.leave_out}-out over {n_models} models "
        f"({n_subpanels} subpanels)  selection_freq among selected: "
        f"min={sel_freq.min():.2f}  median={sel_freq.median():.2f}  max={sel_freq.max():.2f}  "
        f"(<0.5 i.e. quota-filled rather than robustly picked: {n_unstable})"
    )

    # ---- validation diagnostics (ranking recovery + discrimination concentration) ----
    rho = _spearman(tbl["accuracy_full"], tbl["accuracy_selected"])
    tau = _kendall_tau_b(tbl["accuracy_full"], tbl["accuracy_selected"])
    full_order = tbl.sort_values("accuracy_full", ascending=False)["model"].tolist()
    sel_order = tbl.sort_values("accuracy_selected", ascending=False)["model"].tolist()
    top3 = _topk_agreement(full_order, sel_order, 3)
    top5 = _topk_agreement(full_order, sel_order, 5)
    pb_pool = items.loc[
        items["in_pool"] & items["point_biserial"].notna(), "point_biserial"
    ]
    pb_pool = pb_pool[pb_pool >= 0].to_numpy()
    gini = _gini(pb_pool)
    med_pb = float(np.median(pb_pool)) if len(pb_pool) else float("nan")
    print(
        f"ranking recovery (full vs selected): Spearman rho={rho:.3f}  Kendall tau-b={tau:.3f}  "
        f"top-3={top3:.2f}  top-5={top5:.2f}"
    )
    print(
        f"discrimination (pool, n={len(pb_pool)}): median point-biserial={med_pb:.3f}  Gini={gini:.3f}  "
        f"(Gini>0.6 => a few items dominate; low => signal spread out)"
    )

    print("\nper-model accuracy (full -> selected):")
    for _, r in tbl.iterrows():
        print(
            f"  {r['model']:<32} {r['accuracy_full']:5.1f}% -> {r['accuracy_selected']:5.1f}%"
        )

    # ---- per-domain coverage loss (before vs after the cut) ----
    before_by_domain = items.groupby("domain").size().reindex(DOMAINS, fill_value=0)
    after_by_domain = (
        items.loc[items["selected"]]
        .groupby("domain")
        .size()
        .reindex(DOMAINS, fill_value=0)
    )
    print("\nper-domain coverage (before -> after, dropped, %kept):")
    for d in DOMAINS:
        b, a = int(before_by_domain[d]), int(after_by_domain[d])
        if b == 0:
            continue
        pct_kept = 100.0 * a / b
        print(f"  {d}  {b:4d} -> {a:4d}   dropped={b - a:4d}   kept={pct_kept:5.1f}%")
    print("\nWrote:")
    for name in [
        "items.csv",
        "models.csv",
        "selected_ids.json",
        "selected_dataset.json",
        "selected_dataset_cleaned.json",
        "diagnostics.png",
        "accuracy_shift.png",
        "coverage.png",
        "coverage_comparison.png",
        "selection_freq.png",
    ]:
        print("  ", out(name))


if __name__ == "__main__":
    main()
