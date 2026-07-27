#!/usr/bin/env python3
"""
Fit Item Response Theory to a Linguistic-Knowledge run and recommend a trimmed set.

Where it sits: this is a Stage-5 *analysis* step. You run every QC-passed, deduplicated
question (the Stage-3b `dataset_full`) against a panel of models with the eval harness,
then feed the per-item correctness matrix here to decide which items are worth keeping.

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
     per-domain x grade quota proportional to the discriminating pool (a CEILING per cell).
The CEILING is a SOFT target: the script reports the resulting top-model accuracy against
`--ceiling` rather than hard-forcing every model under it (on this data the strongest
model fails so few items that a hard cap would shrink the set drastically — see the
printed summary and tune `--target-n` / `--max-pvalue` from there).

It also prints two validation diagnostics (cf. Ramprasath 2026, IRT redundancy analysis):
  * ranking recovery — Spearman rho / Kendall tau-b / top-k agreement between the model
    ranking on the FULL set and on the SELECTED set. Near 1.0 means trimming widened the
    spread without scrambling who beats whom.
  * discrimination concentration — the median point-biserial and its Gini over the pool.
    High median = items genuinely separate models; high Gini (>0.6) = a few items carry
    most of the signal (very reducible), low Gini = signal spread over many items.

Outputs (under --outdir, tagged with --label):
    irt_items_<label>.csv             per-item difficulty, p-value, point-biserial, flags
    irt_models_<label>.csv            per-model ability, full vs selected accuracy, ranks
    irt_selected_ids_<label>.json     the kept question_ids (sorted)
    irt_selected_dataset_<label>.json the kept items, filtered from --dataset (ready to use)
    irt_diagnostics_<label>.png       difficulty vs discrimination; item difficulty spread
    irt_accuracy_shift_<label>.png    per-model accuracy, full set vs selected set
    irt_coverage_<label>.png          domain x grade counts of the selected set

Run (defaults target the Swahili v10 full-dataset run):
    python benchmarks/linguistic_knowledge_benchmark/scripts/estimate_irt.py
    python .../estimate_irt.py --target-n 150 --min-discrimination 0.10
    python .../estimate_irt.py --max-pvalue 0.9   # also drop items >90%% of models ace
"""

from __future__ import annotations

import argparse
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from girth import ability_eap, rasch_mml

# Fixed spine (same order as analyze_results.py / eda_pipeline.py) so empty cells still show.
DOMAINS = [f"D{i}" for i in range(1, 10)]
GRADES = ["G0", "G1", "G2", "G3", "G4", "G5", "beyond_primary"]
GRADE_LABEL = {g: ("BP" if g == "beyond_primary" else g) for g in GRADES}
DISCRIMINATION = 1.0  # Rasch fixes the slope at 1.
_CORRECT_MAP = {"true": 1, "false": 0, "1": 1, "0": 0}


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
def _round_half_up(x: float) -> int:
    """Round halves up (0.5 -> 1) for non-negative x, matching a spreadsheet ROUND."""
    return math.floor(x + 0.5)


def select_items(
    items: pd.DataFrame,
    target_n: int,
    min_disc: float,
    max_pvalue: float,
    min_pvalue: float,
) -> pd.Series:
    """Choose a balanced, discriminating subset. Returns a boolean `selected` Series.

    Pool = scored items that are NOT always-right and have p-value within
    [min_pvalue, max_pvalue], keeping either point_biserial >= min_disc OR an always-wrong
    item (no model correct — kept as a hard/anti-saturation candidate for human review).
    Each domain x grade cell gets a quota proportional to its share of the pool (scaled to
    target_n, capped by availability). Within a cell, always-wrong items are taken first
    (scarce, anti-saturation, human-review-gated), then the most discriminating (then
    hardest) items. Deterministic given the same inputs.
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
    for (domain, grade), avail in cell_counts.items():
        quota = min(avail, _round_half_up(avail / total_pool * target_n))
        if quota <= 0:
            continue
        cell = pool_items[
            (pool_items["domain"] == domain) & (pool_items["grade"] == grade)
        ]
        # Always-wrong items first, then most discriminating, then hardest, then lowest
        # p-value, then stable id. Always-wrong discrimination is NaN -> treat as 0.
        cell = cell.assign(_disc=cell["point_biserial"].fillna(0.0))
        ranked = cell.sort_values(
            by=["all_wrong", "_disc", "difficulty_logit", "p_value", "question_id"],
            ascending=[False, False, False, True, True],
        )
        selected.loc[ranked.index[:quota]] = True
    return selected


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


def plot_coverage(items: pd.DataFrame, out: str) -> None:
    """Domain x grade counts of the selected set (heatmap)."""
    sel = items[items["selected"]]
    mat = (
        sel.groupby(["domain", "grade"])
        .size()
        .unstack("grade")
        .reindex(index=DOMAINS, columns=GRADES)
    )
    data = mat.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(np.nan_to_num(data), cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(GRADES)))
    ax.set_xticklabels([GRADE_LABEL[g] for g in GRADES], fontsize=9)
    ax.set_yticks(range(len(DOMAINS)))
    ax.set_yticklabels(DOMAINS, fontsize=9)
    total = int(np.nansum(data))
    ax.set_title(
        f"Selected items by domain × grade  (n={total})", fontweight="bold", pad=8
    )
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            v = data[r, c]
            n = 0 if np.isnan(v) else int(v)
            ax.text(
                c,
                r,
                "·" if n == 0 else str(n),
                ha="center",
                va="center",
                fontsize=8,
                color="#222222" if n < 4 else "white",
            )
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="items")
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
        "--detailed",
        default=os.path.join(
            root,
            "results",
            "summary_linguistic-knowledge_swahili_v10-full-dataset_detailed.csv",
        ),
    )
    ap.add_argument(
        "--dataset",
        default=os.path.join(
            root, "data", "kiswahili_workflow", "process", "stage_3b_dataset_full.json"
        ),
        help="full item pool to filter into the selected dataset (matched on question_id)",
    )
    ap.add_argument("--outdir", default=os.path.join(root, "results", "analysis"))
    ap.add_argument(
        "--label", default="swahili_v10-full-dataset", help="tag for output filenames"
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
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    def out(name: str) -> str:
        return os.path.join(args.outdir, name)

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

    items["selected"] = select_items(
        items, args.target_n, args.min_discrimination, args.max_pvalue, args.min_pvalue
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
    items_out.to_csv(out(f"irt_items_{args.label}.csv"), index=False)

    tbl_out = tbl.copy()
    for c in ["ability_logit", "accuracy_full", "accuracy_selected"]:
        tbl_out[c] = tbl_out[c].round(3)
    tbl_out.to_csv(out(f"irt_models_{args.label}.csv"), index=False)

    # ---- selected ids + filtered dataset ----
    selected_ids = sorted(int(q) for q in items.loc[items["selected"], "question_id"])
    with open(out(f"irt_selected_ids_{args.label}.json"), "w", encoding="utf-8") as fh:
        json.dump(selected_ids, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    n_selected_dataset = 0
    if os.path.exists(args.dataset):
        with open(args.dataset, encoding="utf-8") as fh:
            full = json.load(fh)
        keep = set(selected_ids)
        subset = [it for it in full if it.get("question_id") in keep]
        subset.sort(key=lambda it: it.get("question_id", 0))
        with open(
            out(f"irt_selected_dataset_{args.label}.json"), "w", encoding="utf-8"
        ) as fh:
            json.dump(subset, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        n_selected_dataset = len(subset)

    # ---- plots ----
    plot_diagnostics(items, out(f"irt_diagnostics_{args.label}.png"))
    plot_accuracy_shift(tbl, args.ceiling, out(f"irt_accuracy_shift_{args.label}.png"))
    plot_coverage(items, out(f"irt_coverage_{args.label}.png"))

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
    print(
        f"pool={int(items['in_pool'].sum())}  selected={n_sel}  (target {args.target_n})  "
        f"filtered_dataset={n_selected_dataset}  of which always-wrong={n_sel_wrong}"
    )
    print(
        f"top accuracy: full={top_full:.1f}%  selected={top_sel:.1f}%  "
        f"(ceiling {args.ceiling:g}% — {'MET' if top_sel < args.ceiling else 'NOT met, soft'})"
    )
    print(
        f"accuracy spread (max-min): full={spread_full:.1f}pp  selected={spread_sel:.1f}pp"
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
    print("\nWrote:")
    for name in [
        f"irt_items_{args.label}.csv",
        f"irt_models_{args.label}.csv",
        f"irt_selected_ids_{args.label}.json",
        f"irt_selected_dataset_{args.label}.json",
        f"irt_diagnostics_{args.label}.png",
        f"irt_accuracy_shift_{args.label}.png",
        f"irt_coverage_{args.label}.png",
    ]:
        print("  ", out(name))


if __name__ == "__main__":
    main()
