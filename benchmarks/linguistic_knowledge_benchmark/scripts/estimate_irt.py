#!/usr/bin/env python3
"""
Fit Item Response Theory to a Linguistic-Knowledge experiment and recommend a trimmed
set.

Where it sits: this IS Stage 3b — the selection step (it replaced the old
feature-map/hard-coded-quota assembler, stage_3b_assemble_v10.py). You run every
QC-passed, de-duplicated candidate (3a's pool, collapsed to one item per dup_group)
against a panel of models with the eval harness, then feed the per-item correctness
matrix here to select the final dataset.

--language is the only CLI-configurable input; its two data sources are fixed, per-
language constants (RESULTS_DETAILED_PATH / DATASET_PATH below), not CLI flags -- edit
those dicts directly if the experiment being selected from changes:
  RESULTS_DETAILED_PATH[language]  the judged results CSV (one row per question,
      produced by judge-bench): question_id, domain, grade, answer_type, question,
      expected_answers, then per model a `<model>_output` and `<model>_correct`
      (True/False, blank = unscored). Each model is treated as an IRT respondent and
      `<model>_correct` as its binary response -- this is what IRT is actually fit on.
  DATASET_PATH[language]  the full item-pool JSON RESULTS_DETAILED_PATH's questions were
      drawn from (question_id, domain, question, options, expected_answers, ...). Used to
      attach question text/options/domain/grade to the selected question_ids when
      writing selected_dataset.json / selected_dataset_cleaned.json below, matched purely
      on question_id. It must be the exact pool RESULTS_DETAILED_PATH was scored against:
      a same-shaped-but-different pool can collide on question_id and silently produce a
      wrong-but-plausible selected_dataset.json -- so before any output is written, the
      script cross-checks the two paths' question_ids/answer_types and ABORTS if they
      don't line up (a safety net against DATASET_PATH/RESULTS_DETAILED_PATH drifting out
      of sync with each other, not just a CLI-input check).

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
     way the human-review gate should look at them. (Raise MIN_PVALUE above 0 to drop them.)
  3. Coverage — keep a good spread across domains (D1..D9) and grades by filling a
     per-domain x grade quota proportional to the discriminating pool, allocated exactly
     via largest-remainder rounding (quotas always sum to min(TARGET_N, pool size) —
     no off-by-a-few from independent per-cell rounding).

Within each domain x grade cell, items are further ranked by SELECTION STABILITY before
discrimination/difficulty: `selection_freq` is the fraction of leave-LEAVE_OUT-out model
subpanels (all C(n_models, LEAVE_OUT) of them, each with p-value/discrimination
recomputed for just that subpanel — no Rasch refit needed) that would independently pick
this item. Ranking on this first means the kept set favours items that stay competitive
across many reasonable model panels, not just the exact panel on hand — with only ~10
model respondents, point-biserial/rank estimates can shift when the panel changes.

The TARGET_N CEILING is a SOFT target: the script reports the resulting top-model
accuracy against CEILING rather than hard-forcing every model under it (on this data
the strongest model fails so few items that a hard cap would shrink the set drastically
— see the printed summary and tune TARGET_N / MAX_PVALUE from there).

It also prints two validation diagnostics (cf. Ramprasath 2026, IRT redundancy analysis):
  * ranking recovery — Spearman rho / Kendall tau-b / top-k agreement between the model
    ranking on the FULL set and on the SELECTED set. Near 1.0 means trimming widened the
    spread without scrambling who beats whom.
  * discrimination concentration — the median point-biserial and its Gini over the pool.
    High median = items genuinely separate models; high Gini (>0.6) = a few items carry
    most of the signal (very reducible), low Gini = signal spread over many items.

Outputs — one directory per experiment, never overwritten (see --experiment-id below):
  <outdir>/irt/<language>_experiments/<experiment-id>/
    items.csv              per-item difficulty, p-value, point-biserial, selection_freq, flags
    models.csv              per-model ability, full vs selected accuracy, ranks
    selected_ids.json       the kept question_ids (sorted)
    selected_dataset.json   the kept items, filtered from DATASET_PATH (ready to use)
    selected_dataset_cleaned.json  same items, question_id renumbered 1..n (contiguous)
    diagnostics.png         difficulty vs discrimination; item difficulty spread
    accuracy_shift.png      per-model accuracy, full set vs selected set
    coverage.png            domain x grade counts of the selected set
    coverage_comparison.png 5-panel domain x grade coverage: full bank, selected set,
                            items dropped, plus % of each domain's and each grade's
                            own before-cut total that got dropped -- answers "which
                            domains/grades lost coverage in the cut?" directly (the
                            printed per-domain coverage table below answers the same
                            question for domains in text form)
    selection_freq.png      selection_freq distribution over the candidate pool
--outdir defaults to results/; --experiment-id defaults to "default". If any of the above files
already exist in that experiment's directory, the script aborts rather than overwrite
them —
pass a different --experiment-id (or delete the old run's directory) to rerun.

Selection tuning (TARGET_N, CEILING, MIN_DISCRIMINATION, MAX_PVALUE, MIN_PVALUE,
LEAVE_OUT above) edit those constants directly and
rerun with a new --experiment-id.

Run (--language is the only required flag):
    uv run python benchmarks/linguistic_knowledge_benchmark/scripts/estimate_irt.py --language kiswahili
    uv run python .../estimate_irt.py --language english --experiment-id target150
    uv run python .../estimate_irt.py --language kiswahili --outdir /tmp/irt-experiments
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

# The two data sources this script always uses -- one pair of paths per language, not a
# CLI flag (see module docstring's Inputs section). Edit these directly to point at a
# different experiment; --language just selects which entry.
#
# Kiswahili's full-pool eval experiment is tagged "v10-full-dataset" (528 detailed rows -> 307
# items scored); English's equivalent experiment (its whole 377-item
# stage_3b_dataset_full pool)
# is tagged "full-run". Paths are relative to the package root (joined in main()).
# NOTE: these must point at the FULL POOL, not a final trimmed set -- the
# final-137-items CSVs hold only the already-selected items (and 120 models, not the
# 10-model selection panel), so pointing here at one of those would silently re-select
# from a set that has already been selected.
RESULTS_DETAILED_PATH = {
    "english": os.path.join(
        "results", "summary_linguistic-knowledge_english_full-run_detailed.csv"
    ),
    "kiswahili": os.path.join(
        "results",
        "summary_linguistic-knowledge_kiswahili_v10-full-dataset_detailed.csv",
    ),
}

DATASET_PATH = {
    "english": os.path.join(
        "data", "english_workflow", "process", "stage_3b_dataset_full.json"
    ),
    "kiswahili": os.path.join(
        "data", "kiswahili_workflow", "process", "stage_3b_dataset_full.json"
    ),
}

# Selection tuning, also not CLI flags -- edit these directly to change how an
# experiment is
# trimmed (see "How it selects a trimmed dataset" above for what each one controls).
TARGET_N = 150  # target size of the selected set
CEILING = 90.0  # top-accuracy ceiling to report against (soft; not hard-enforced)
MIN_DISCRIMINATION = 0.0  # drop items whose point-biserial is below this
MAX_PVALUE = 1.0  # drop items more than this fraction of models get right (e.g. 0.9)
MIN_PVALUE = 0.0  # drop items fewer than this fraction of models get right
LEAVE_OUT = 2  # models omitted per selection_freq subpanel (2 = leave-two-out)

# Font sizes for coverage_comparison.png, in one place. Matplotlib's defaults (and
# _draw_coverage_heatmap's) are sized for a normal-width plot; on this 30-inch-wide
# five-panel strip, printed into a report, they come out unreadably small.
FS_SUPTITLE = 30
FS_PANEL_TITLE = 20
FS_TICK = 16
FS_VALUE = 15  # heatmap cell counts and bar value labels
FS_AXIS_LABEL = 16


def default_results_detailed_for(language: str, root: str) -> str:
    """A language's default results-detailed CSV path (see RESULTS_DETAILED_PATH above).

    estimate_irt.py's own main() reads RESULTS_DETAILED_PATH directly (--language is its
    only CLI parameter); this exists for other scripts (e.g. plot_selected_accuracy.py)
    that still want a same-language guess to fall back to.
    """
    return os.path.join(root, RESULTS_DETAILED_PATH[language])


# Fixed filenames written into each experiment's own directory (see main()'s --experiment-id / experiment_dir).
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


def _refuse_if_exists(experiment_dir: str, filenames: list[str]) -> None:
    """Abort rather than silently overwrite a previous experiment's output files."""
    if not os.path.isdir(experiment_dir):
        return
    clashes = [f for f in filenames if os.path.exists(os.path.join(experiment_dir, f))]
    if clashes:
        raise SystemExit(
            f"refusing to overwrite existing experiment output in {experiment_dir}:\n"
            + "\n".join(f"  - {f}" for f in clashes)
            + "\nUse a different --experiment-id, or delete these files first."
        )


# --------------------------------------------------------------------------- #
# Console summary formatting
# --------------------------------------------------------------------------- #
_RULE = "=" * 70


def _section(title: str) -> None:
    """Print a section header for the console summary."""
    print(f"\n{_RULE}\n{title}\n{_RULE}")


def _kv(label: str, value) -> None:
    """Print one left-aligned 'label : value' row for the console summary."""
    print(f"  {label:<22}: {value}")


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


def load_dataset_items(dataset_path: str) -> list[dict]:
    """Load DATASET_PATH's item-pool JSON, or abort with a clear error if it's missing."""
    if not os.path.exists(dataset_path):
        raise SystemExit(f"DATASET_PATH entry not found on disk: {dataset_path}")
    with open(dataset_path, encoding="utf-8") as fh:
        return json.load(fh)


def check_dataset_matches_results(
    meta: pd.DataFrame, dataset_items: list[dict]
) -> None:
    """Abort if DATASET_PATH doesn't match what RESULTS_DETAILED_PATH was scored on.

    The dangerous case isn't a missing file (load_dataset_items already catches that) --
    it's a same-shaped-but-WRONG pool, e.g. two item pools that happen to reuse the same
    question_id range. Filtering selected_ids against the wrong pool would silently
    attach the wrong question text/options/domain/grade to a real selected id, so this
    runs BEFORE any output is written and raises rather than warns -- a wrong
    selected_dataset.json is worse than none (same hazard, same two-signal approach as
    analyze_results.py's --dataset check). Catches DATASET_PATH/RESULTS_DETAILED_PATH
    drifting out of sync with each other, since neither is a CLI input anymore:
      1. question_ids present in the results CSV but missing from the dataset JSON --
         those items would be silently dropped from selected_dataset.json if selected.
      2. for ids present in both, the dataset JSON's answer_type disagreeing with the
         results CSV's own answer_type for that id -- catches a true id collision even
         when the id ranges happen to overlap cleanly.
    """
    dataset_types = {
        it["question_id"]: it.get("answer_type")
        for it in dataset_items
        if "question_id" in it
    }
    dataset_ids = set(dataset_types)

    results_ids = {int(q) for q in meta["question_id"]}
    missing_from_dataset = results_ids - dataset_ids

    checked = mismatched = 0
    for qid, atype in zip(meta["question_id"], meta["answer_type"]):
        qid = int(qid)
        if qid in dataset_types:
            checked += 1
            if dataset_types[qid] != atype:
                mismatched += 1

    if not missing_from_dataset and not mismatched:
        return

    msg = [
        "DATASET_PATH does not match RESULTS_DETAILED_PATH for this language -- "
        "aborting rather than writing a silently-wrong selected_dataset.json.",
        f"  {len(results_ids)} distinct question_id(s) in the results CSV, "
        f"{len(dataset_ids)} in the dataset JSON.",
    ]
    if missing_from_dataset:
        msg.append(
            f"  {len(missing_from_dataset)} question_id(s) in the results CSV not found "
            "in the dataset JSON (would be dropped from selected_dataset.json if selected)."
        )
    if mismatched:
        msg.append(
            f"  {mismatched}/{checked} overlapping id(s) have a different answer_type in "
            "the dataset JSON than in the results CSV -- likely a different item pool "
            "reusing the same ids."
        )
    msg.append(
        "  Fix RESULTS_DETAILED_PATH / DATASET_PATH at the top of estimate_irt.py so "
        "they refer to the same experiment for this language."
    )
    raise SystemExit("\n".join(msg))


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


def _draw_coverage_heatmap(
    ax,
    data: np.ndarray,
    title: str,
    cmap: str,
    *,
    tick_fontsize: int = 9,
    cell_fontsize: int = 8,
    title_fontsize: int | None = None,
) -> None:
    """Shared domain x grade heatmap rendering (axes/ticks/cell-value annotations).

    The font sizes are parameters because this helper serves two very differently
    sized figures: plot_coverage()'s single 9x6 panel (the defaults) and
    plot_coverage_comparison()'s 30-inch-wide five-panel strip, where the defaults
    render far too small to read. Passing sizes keeps the small figure unchanged.
    """
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0)
    ax.set_xticks(range(len(GRADES)))
    ax.set_xticklabels([GRADE_LABEL[g] for g in GRADES], fontsize=tick_fontsize)
    ax.set_yticks(range(len(DOMAINS)))
    ax.set_yticklabels(DOMAINS, fontsize=tick_fontsize)
    ax.set_title(
        f"{title}  (n={int(data.sum())})",
        fontweight="bold",
        pad=8,
        fontsize=title_fontsize,
    )
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
                fontsize=cell_fontsize,
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


def plot_coverage_comparison(
    items: pd.DataFrame, out: str, language: str | None = None
) -> None:
    """Domain x grade heatmap: the full item bank, the selected set, the difference
    (items dropped per cell), then -- on the right -- each domain's % dropped and
    each grade's % dropped, each proportional to its OWN before-cut total (not the
    pool total). Those last two panels directly answer "which domains/grades lost
    coverage in the cut?": the per-cell diff heatmap shows raw counts, which favours
    large domains/grades just for having more items to lose; normalizing by each
    one's starting size means a small domain/grade that got wiped out shows up as
    clearly as a large one.

    `items` is every item that entered this IRT experiment (scored or not); "before"
    counts all of them, "after" counts only those with `selected=True`. Since
    selected is always a subset, the diff panel is >= 0 everywhere (cells never
    gain items). `language`, if given, is prefixed onto the title -- the figure is
    otherwise identical between languages and impossible to tell apart in a report.
    """
    before = _domain_grade_matrix(items)
    after = _domain_grade_matrix(items[items["selected"]])
    diff = before - after

    pct_dropped_domain = _pct_dropped(before.sum(axis=1), after.sum(axis=1))
    pct_dropped_grade = _pct_dropped(before.sum(axis=0), after.sum(axis=0))

    fig, axes = plt.subplots(
        1, 5, figsize=(30, 7.5), gridspec_kw={"width_ratios": [1, 1, 1, 0.7, 0.7]}
    )
    heat_kw = dict(
        tick_fontsize=FS_TICK,
        cell_fontsize=FS_VALUE,
        title_fontsize=FS_PANEL_TITLE,
    )
    im0 = _draw_coverage_heatmap(
        axes[0], before, "Full item bank (before)", "Blues", **heat_kw
    )
    im1 = _draw_coverage_heatmap(
        axes[1], after, "Selected set (after)", "Blues", **heat_kw
    )
    im2 = _draw_coverage_heatmap(
        axes[2], diff, "Dropped in the cut (before − after)", "Reds", **heat_kw
    )
    for ax, im in zip(axes[:3], [im0, im1, im2]):
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("items", fontsize=FS_AXIS_LABEL)
        cbar.ax.tick_params(labelsize=FS_TICK)

    # 4th panel: % of each domain's own before-cut total that got dropped.
    # Row-aligned with the heatmaps (same DOMAINS order, D1 at top) so a row reads
    # as one continuous story: before -> after -> raw dropped -> % of that domain lost.
    ax = axes[3]
    y = np.arange(len(DOMAINS))
    colors = plt.get_cmap("Reds")(
        np.clip(np.nan_to_num(pct_dropped_domain) / 100.0, 0.15, 1.0)
    )
    ax.barh(y, np.nan_to_num(pct_dropped_domain), color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(DOMAINS, fontsize=FS_TICK)
    ax.invert_yaxis()  # D1 at top, matching the heatmaps' row order
    # Headroom past 100 so a 100%-dropped bar's value label still fits inside the
    # axes instead of colliding with the neighbouring panel.
    ax.set_xlim(0, 118)
    ax.set_xlabel("% of domain dropped", fontsize=FS_AXIS_LABEL)
    ax.tick_params(axis="x", labelsize=FS_TICK)
    ax.set_title(
        "% dropped\n(of domain's own total)",
        fontweight="bold",
        fontsize=FS_PANEL_TITLE,
        pad=8,
    )
    for i, pct in enumerate(pct_dropped_domain):
        label = "n/a" if np.isnan(pct) else f"{pct:.0f}%"
        x = 0 if np.isnan(pct) else pct
        ax.text(x + 2, i, label, va="center", fontsize=FS_VALUE)
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
    ax.set_xticklabels([GRADE_LABEL[g] for g in GRADES], fontsize=FS_TICK)
    # Same headroom as the domain panel: a 100% bar's label would otherwise run
    # into the title.
    ax.set_ylim(0, 118)
    ax.set_ylabel("% of grade dropped", fontsize=FS_AXIS_LABEL)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.set_title(
        "% dropped\n(of grade's own total)",
        fontweight="bold",
        fontsize=FS_PANEL_TITLE,
        pad=8,
    )
    for i, pct in enumerate(pct_dropped_grade):
        label = "n/a" if np.isnan(pct) else f"{pct:.0f}%"
        y_ = 0 if np.isnan(pct) else pct
        ax.text(i, y_ + 2, label, ha="center", fontsize=FS_VALUE)
    ax.spines[["top", "right"]].set_visible(False)

    title = "Domain × grade coverage: before vs after item selection"
    if language:
        title = f"{language.capitalize()} Linguistic Knowledge Benchmark -- {title}"
    fig.suptitle(title, fontweight="bold", fontsize=FS_SUPTITLE)
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
        help="which language's experiment to analyse -- selects RESULTS_DETAILED_PATH / "
        "DATASET_PATH above, the only two data sources this script reads",
    )
    ap.add_argument(
        "--outdir",
        default=os.path.join(root, "results"),
        help="base results dir; this run's files go under "
        "<outdir>/irt/<language>_experiments/<experiment-id>/",
    )
    ap.add_argument(
        "--experiment-id",
        default="default",
        help="tag for this run's output directory (results/irt/<language>_experiments/<experiment-id>/); "
        "pass a specific name, or 'timestamp' to auto-generate one (YYYYmmdd-HHMMSS); "
        "refuses to overwrite an existing experiment's files — pick a new --experiment-id instead",
    )
    args = ap.parse_args()
    args.results_detailed = os.path.join(root, RESULTS_DETAILED_PATH[args.language])
    args.dataset = os.path.join(root, DATASET_PATH[args.language])

    if args.experiment_id == "timestamp":
        args.experiment_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    elif os.sep in args.experiment_id or "/" in args.experiment_id:
        raise SystemExit(
            f"--experiment-id must be a bare name, not a path: {args.experiment_id!r}"
        )

    experiment_dir = os.path.join(
        args.outdir, "irt", f"{args.language}_experiments", args.experiment_id
    )
    _refuse_if_exists(experiment_dir, OUTPUT_FILENAMES)
    os.makedirs(experiment_dir, exist_ok=True)

    def out(name: str) -> str:
        return os.path.join(experiment_dir, name)

    _section("CONFIG")
    _kv("language", args.language)
    _kv("results_detailed", args.results_detailed)
    _kv("dataset", args.dataset)
    _kv("outdir", args.outdir)
    _kv("experiment_id", args.experiment_id)
    _kv("experiment_dir", experiment_dir)
    _kv("TARGET_N", TARGET_N)
    _kv("CEILING", CEILING)
    _kv("MIN_DISCRIMINATION", MIN_DISCRIMINATION)
    _kv("MAX_PVALUE", MAX_PVALUE)
    _kv("MIN_PVALUE", MIN_PVALUE)
    _kv("LEAVE_OUT", LEAVE_OUT)

    meta, models, matrix, scored_mask = load_matrix(args.results_detailed)
    n_models = len(models)
    n_unscored = int((~scored_mask).sum())

    # Validated up front, before any (expensive) fitting or output writing -- see
    # check_dataset_matches_results' docstring for why a silent mismatch is dangerous.
    dataset_items = load_dataset_items(args.dataset)
    check_dataset_matches_results(meta, dataset_items)

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
        TARGET_N,
        MIN_DISCRIMINATION,
        MAX_PVALUE,
        MIN_PVALUE,
        LEAVE_OUT,
    )
    items["selected"] = select_items(
        items,
        TARGET_N,
        MIN_DISCRIMINATION,
        MAX_PVALUE,
        MIN_PVALUE,
        primary_rank_col="selection_freq",
    )
    items["in_pool"] = (
        items["scored"]
        & ~items["all_right"]
        & (items["all_wrong"] | (items["point_biserial"] >= MIN_DISCRIMINATION))
        & (items["p_value"] <= MAX_PVALUE)
        & (items["p_value"] >= MIN_PVALUE)
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

    keep = set(selected_ids)
    subset = [it for it in dataset_items if it.get("question_id") in keep]
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
    plot_accuracy_shift(tbl, CEILING, out("accuracy_shift.png"))
    plot_coverage(items, out("coverage.png"))
    plot_coverage_comparison(
        items, out("coverage_comparison.png"), language=args.language
    )
    plot_selection_freq(items, LEAVE_OUT, out("selection_freq.png"))

    # ---- console summary ----
    n_sel = int(items["selected"].sum())
    top_full = tbl["accuracy_full"].max()
    top_sel = tbl["accuracy_selected"].max()
    spread_full = tbl["accuracy_full"].max() - tbl["accuracy_full"].min()
    spread_sel = tbl["accuracy_selected"].max() - tbl["accuracy_selected"].min()
    all_right = int(items["all_right"].sum())
    all_wrong = int(items["all_wrong"].sum())
    n_sel_wrong = int((items["selected"] & items["all_wrong"]).sum())
    n_pool = int(items["in_pool"].sum())
    n_target = min(TARGET_N, n_pool)

    _section("ITEM POOL")
    _kv("models", n_models)
    _kv("items", len(items))
    _kv("unscored (dropped from IRT)", n_unscored)
    _kv("all-right (dropped)", all_right)
    _kv("all-wrong (kept)", all_wrong)

    _section("SELECTION")
    _kv("pool", n_pool)
    _kv("selected", f"{n_sel}  (target {TARGET_N})")
    _kv(
        "filtered_dataset",
        f"{n_selected_dataset}  (of which always-wrong: {n_sel_wrong})",
    )
    _kv(
        "exact quota check",
        f"selected == min(TARGET_N, pool) -> {n_sel} == {n_target}  "
        f"{'OK' if n_sel == n_target else 'MISMATCH'}",
    )

    _section("ACCURACY")
    _kv(
        "top accuracy",
        f"full={top_full:.1f}%  selected={top_sel:.1f}%  "
        f"(ceiling {CEILING:g}% — {'MET' if top_sel < CEILING else 'NOT met, soft'})",
    )
    _kv(
        "accuracy spread (max-min)",
        f"full={spread_full:.1f}pp  selected={spread_sel:.1f}pp",
    )

    # ---- selection stability (leave-`leave_out`-out) ----
    n_subpanels = math.comb(n_models, LEAVE_OUT)
    sel_freq = items.loc[items["selected"], "selection_freq"]
    n_unstable = int((sel_freq < 0.5).sum())
    _section(f"SELECTION STABILITY (leave-{LEAVE_OUT}-out)")
    _kv("subpanels", f"{n_subpanels}  (over {n_models} models)")
    _kv(
        "selection_freq (selected)",
        f"min={sel_freq.min():.2f}  median={sel_freq.median():.2f}  max={sel_freq.max():.2f}",
    )
    _kv("quota-filled (freq<0.5)", n_unstable)

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
    _section("VALIDATION DIAGNOSTICS")
    _kv(
        "ranking recovery (full vs sel)",
        f"Spearman rho={rho:.3f}  Kendall tau-b={tau:.3f}  top-3={top3:.2f}  top-5={top5:.2f}",
    )
    _kv(
        "discrimination (pool)",
        f"n={len(pb_pool)}  median point-biserial={med_pb:.3f}  Gini={gini:.3f}  "
        f"(>0.6 => a few items dominate; low => signal spread out)",
    )

    _section("PER-MODEL ACCURACY (full -> selected)")
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
    _section("PER-DOMAIN COVERAGE (before -> after)")
    for d in DOMAINS:
        b, a = int(before_by_domain[d]), int(after_by_domain[d])
        if b == 0:
            continue
        pct_kept = 100.0 * a / b
        print(f"  {d}  {b:4d} -> {a:4d}   dropped={b - a:4d}   kept={pct_kept:5.1f}%")

    _section("OUTPUT FILES")
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
