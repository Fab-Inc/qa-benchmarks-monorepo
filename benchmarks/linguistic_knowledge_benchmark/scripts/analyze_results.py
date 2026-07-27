#!/usr/bin/env python3
"""
Analyse a Linguistic-Knowledge benchmark run from its two result CSVs.

Inputs (both produced by `judge-bench`):
  * SUMMARY csv   — one row per model: accuracy, accuracy_mc, accuracy_short_form.
  * DETAILED csv  — one row per question: question_id, domain, grade, answer_type,
                    question, expected_answers, then per model a `<model>_output` and
                    `<model>_correct` (True/False, blank = unscored) column.

Questions it answers:
  (a) Is there a difference between MCQ and short-form accuracy?  -> per-model grouped
      bars from the summary, plus the overall gap (also cross-checked from the detailed).
  (b) Which domain/grade are models getting wrong?  -> a domain x grade accuracy heatmap
      plus by-domain and by-grade bars, computed from the detailed correctness columns.

Outputs (under --outdir, tagged with --label):
    accuracy_by_model_<label>.png
    accuracy_mcq_vs_shortform_<label>.png
    accuracy_by_domain_grade_<label>.png
    accuracy_by_domain_grade_<label>.csv   the domain x grade accuracy table

Run (--language is mandatory; defaults target that language's "default" run):
    python benchmarks/linguistic_knowledge_benchmark/scripts/analyze_results.py --language english
    python benchmarks/linguistic_knowledge_benchmark/scripts/analyze_results.py --language swahili
    python .../analyze_results.py --language swahili --summary ... --detailed ... --label ...

Restrict the domain×grade analysis to a band of models by overall accuracy
(--min-accuracy / --max-accuracy, in %); the band is stamped into the plot titles
and appended to the domain×grade output filenames (e.g. _acc90plus, _acc70-90):
    python .../analyze_results.py --language swahili --min-accuracy 90              # only the 90+%% models
    python .../analyze_results.py --language swahili --min-accuracy 70 --max-accuracy 90   # the 70-90%% band
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Fixed spine (same order as eda_pipeline.py) so empty domains/grades still show.
DOMAINS = [f"D{i}" for i in range(1, 10)]
GRADES = ["G0", "G1", "G2", "G3", "G4", "G5", "beyond_primary"]
GRADE_LABEL = {g: ("BP" if g == "beyond_primary" else g) for g in GRADES}
ACC_CMAP = "RdYlGn"  # low accuracy = red (getting it wrong), high = green
_CORRECT_MAP = {True: 1, False: 0, "True": 1, "False": 0}

# Language keys match cmd.py's DATASETS and the result CSVs' own naming
# (summary_linguistic-knowledge_<language>_<label>.csv).
LANGUAGES = ["english", "swahili"]


def default_paths_for(language: str, results_dir: str) -> dict:
    return {
        "summary": os.path.join(
            results_dir, f"summary_linguistic-knowledge_{language}_default.csv"
        ),
        "detailed": os.path.join(
            results_dir, f"summary_linguistic-knowledge_{language}_default_detailed.csv"
        ),
        "label": f"{language}_default",
    }


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def load_summary(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["model"].notna()].copy()
    return df.sort_values("accuracy", ascending=False).reset_index(drop=True)


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
def plot_accuracy(summary: pd.DataFrame, out: str) -> None:
    s = summary.sort_values("accuracy")  # ascending -> best on top after barh
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(s["model"], s["accuracy"], color="#4e79a7")
    ax.set_xlabel("Accuracy (%)")
    ax.set_xlim(0, 100)
    ax.set_title("Linguistic Knowledge — accuracy by model", fontweight="bold")
    for b, a in zip(bars, s["accuracy"]):
        ax.text(
            a + 0.8, b.get_y() + b.get_height() / 2, f"{a:.1f}", va="center", fontsize=9
        )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#eeeeee")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_mcq_vs_short(summary: pd.DataFrame, out: str) -> None:
    """Grouped bars: MCQ vs short-form accuracy per model, ordered by overall accuracy."""
    s = summary.sort_values("accuracy", ascending=False)
    models = s["model"].tolist()
    mc = s["accuracy_mc"].to_numpy(dtype=float)
    sf = s["accuracy_short_form"].to_numpy(dtype=float)
    x = np.arange(len(models))
    w = 0.4

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.bar(x - w / 2, mc, w, label="multiple_choice", color="#4e79a7")
    ax.bar(x + w / 2, sf, w, label="short_form", color="#f28e2b")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 106)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=35, ha="right", fontsize=8)
    ax.set_title(
        "MCQ vs short-form accuracy per model  (label = MCQ − short)",
        fontweight="bold",
        pad=16,
    )
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
    # --summary/--detailed/--label default to that language's "default" run
    # (see default_paths_for above); override individually for a specific run.
    ap.add_argument("--summary", default=None)
    ap.add_argument("--detailed", default=None)
    ap.add_argument("--outdir", default=os.path.join(root, "results", "analysis"))
    ap.add_argument("--label", default=None, help="tag for output filenames")
    ap.add_argument(
        "--min-accuracy",
        type=float,
        default=0.0,
        help="only include models with overall accuracy >= this (%) in the "
        "domain×grade analysis, e.g. 90 for the 90+%% models",
    )
    ap.add_argument(
        "--max-accuracy",
        type=float,
        default=100.0,
        help="only include models with overall accuracy <= this (%)",
    )
    args = ap.parse_args()

    defaults = default_paths_for(args.language, os.path.join(root, "results"))
    for key in ("summary", "detailed", "label"):
        if getattr(args, key) is None:
            setattr(args, key, defaults[key])

    os.makedirs(args.outdir, exist_ok=True)

    def out(name: str) -> str:
        return os.path.join(args.outdir, name)

    summary = load_summary(args.summary)
    _, long, models = load_detailed(args.detailed)

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
    plot_accuracy(summary, out(f"accuracy_by_model_{args.label}.png"))
    plot_mcq_vs_short(summary, out(f"accuracy_mcq_vs_shortform_{args.label}.png"))
    cell = plot_domain_grade(
        long_sel, out(f"accuracy_by_domain_grade_{args.label}{tag}.png"), note
    )
    cell.round(1).to_csv(out(f"accuracy_by_domain_grade_{args.label}{tag}.csv"))

    # ---- (a) MCQ vs short-form ----
    print(f"models: {len(models)}   questions: {long['question_id'].nunique()}")
    print("\n(a) MCQ vs short-form accuracy (%, gap = MCQ - short):")
    for _, r in summary.iterrows():
        gap = r["accuracy_mc"] - r["accuracy_short_form"]
        print(
            f"  {r['model']:<32} MCQ {r['accuracy_mc']:5.1f}   short {r['accuracy_short_form']:5.1f}   gap {gap:+5.1f}"
        )
    mc_mean = summary["accuracy_mc"].mean()
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


if __name__ == "__main__":
    main()
