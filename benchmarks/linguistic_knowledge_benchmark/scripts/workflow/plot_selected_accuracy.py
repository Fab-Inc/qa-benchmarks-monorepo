#!/usr/bin/env python3
"""
Plot per-model accuracy on a *selected* (trimmed) item subset — e.g. the IRT-selected
dataset from estimate_irt.py — rather than the full experiment.

The summary csv from `judge-bench` scores the FULL experiment; this recomputes accuracy
from
the DETAILED csv restricted to --select-ids, then reuses analyze_results.py's plotting
functions so the styling matches the full-experiment plots exactly.

--select-ids defaults to that --language's --experiment-id under estimate_irt.py's output tree
(results/irt/<language>_experiments/<experiment-id>/selected_ids.json) — pass just --language and
--experiment-id to plot any of its experiments; override --select-ids directly for
anything else.

Outputs (under --outdir, tagged with --label):
    accuracy_by_model_<label>.png            overall accuracy per model
    accuracy_mcq_vs_shortform_<label>.png    MCQ vs short-form accuracy per model

Run:
    python benchmarks/linguistic_knowledge_benchmark/scripts/workflow/plot_selected_accuracy.py --language kiswahili
    python .../plot_selected_accuracy.py --language kiswahili --experiment-id target150
    python .../plot_selected_accuracy.py --language kiswahili --select-ids ... --detailed ... --label ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

# analyze_results.py / estimate_irt.py live in the parent scripts/ dir, not here.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyze_results import LANGUAGES, load_detailed, plot_accuracy, plot_mcq_vs_short
from estimate_irt import default_results_detailed_for
from linguistic_knowledge_benchmark.shared_utils import MODELS_CSV


def load_display_names(models_csv: str) -> dict:
    """model_id -> display_name, from models.csv."""
    df = pd.read_csv(models_csv)
    return dict(zip(df["model_id"], df["display_name"]))


def summary_for_selection(long_df, selected_ids, display_names=None):
    """Recompute a summary-shaped df (model, accuracy, accuracy_mcq, accuracy_short_form)
    from the long detailed df, restricted to `selected_ids`. `model` is renamed to its
    models.csv `display_name` when a mapping is given (unmapped ids are left as-is)."""
    sel = long_df[long_df["question_id"].isin(selected_ids)]
    scored = sel.dropna(subset=["correct"])
    overall = scored.groupby("model")["correct"].mean().mul(100)
    by_type = (
        scored.groupby(["model", "answer_type"])["correct"]
        .mean()
        .mul(100)
        .unstack("answer_type")
    )
    summary = overall.rename("accuracy").to_frame()
    summary["accuracy_mcq"] = by_type.get("multiple_choice")
    summary["accuracy_short_form"] = by_type.get("short_form")
    summary = summary.reset_index()
    if display_names:
        unmapped = sorted(set(summary["model"]) - set(display_names))
        if unmapped:
            print(
                f"  [warning] no display_name in models.csv for: {', '.join(unmapped)}"
            )
        summary["model"] = summary["model"].map(lambda m: display_names.get(m, m))
    return summary.sort_values("accuracy", ascending=False).reset_index(drop=True)


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--language", required=True, choices=LANGUAGES)
    ap.add_argument(
        "--experiment-id",
        dest="experiment_id",
        default="default",
        help="which estimate_irt.py experiment to plot (results/irt/<language>_experiments/<experiment-id>/); "
        "ignored if --select-ids is passed explicitly",
    )
    ap.add_argument("--detailed", default=None)
    ap.add_argument(
        "--select-ids",
        dest="select_ids",
        default=None,
        help="JSON array of question_ids to restrict accuracy to (default: "
        "results/irt/<language>_experiments/<experiment-id>/selected_ids.json)",
    )
    ap.add_argument("--outdir", default=os.path.join(root, "results", "analysis"))
    ap.add_argument("--label", default=None, help="tag for output filenames")
    ap.add_argument(
        "--models-csv",
        dest="models_csv",
        default=str(MODELS_CSV),
        help="maps model_id -> display_name for the plot labels",
    )
    args = ap.parse_args()

    if args.detailed is None:
        args.detailed = default_results_detailed_for(args.language, root)
    if args.select_ids is None:
        args.select_ids = os.path.join(
            root,
            "results",
            "irt",
            f"{args.language}_experiments",
            args.experiment_id,
            "selected_ids.json",
        )
    if not os.path.exists(args.select_ids):
        experiments_dir = os.path.join(
            root, "results", "irt", f"{args.language}_experiments"
        )
        available = (
            sorted(os.listdir(experiments_dir))
            if os.path.isdir(experiments_dir)
            else []
        )
        raise SystemExit(
            f"no such --select-ids file: {args.select_ids}\n"
            f"available --experiment-id values for --language {args.language}: "
            + (", ".join(available) if available else "(none found)")
        )
    if args.label is None:
        args.label = f"{args.language}_{args.experiment_id}"

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.select_ids, encoding="utf-8") as fh:
        selected_ids = set(json.load(fh))

    display_names = load_display_names(args.models_csv)
    _, long_df, _ = load_detailed(args.detailed)
    summary = summary_for_selection(long_df, selected_ids, display_names)

    def out(name: str) -> str:
        return os.path.join(args.outdir, name)

    n = len(selected_ids)
    sel_meta = long_df[long_df["question_id"].isin(selected_ids)].drop_duplicates(
        "question_id"
    )
    n_mcq = int((sel_meta["answer_type"] == "multiple_choice").sum())
    n_sf = int((sel_meta["answer_type"] == "short_form").sum())

    # Display "Kiswahili" for the kiswahili experiment, otherwise the capitalized language
    # name.
    language_name = {"kiswahili": "Kiswahili"}.get(
        args.language, args.language.capitalize()
    )
    title_acc = f"{language_name} Linguistic Knowledge Benchmark Performance"
    title_mcq = (
        "Linguistic Knowledge Benchmark — MCQ vs short-form accuracy by model\n"
        f"{language_name}, IRT-selected set (n={n}: {n_mcq} MCQ, {n_sf} short-form)"
    )
    plot_accuracy(summary, out(f"accuracy_by_model_{args.label}.png"), title=title_acc)
    plot_mcq_vs_short(
        summary, out(f"accuracy_mcq_vs_shortform_{args.label}.png"), title=title_mcq
    )

    print(f"{len(selected_ids)} selected questions · {len(summary)} models")
    for _, r in summary.iterrows():
        print(
            f"  {r['model']:<32} overall {r['accuracy']:5.1f}   "
            f"MCQ {r['accuracy_mcq']:5.1f}   short {r['accuracy_short_form']:5.1f}"
        )
    print("\nWrote:")
    print("  ", out(f"accuracy_by_model_{args.label}.png"))
    print("  ", out(f"accuracy_mcq_vs_shortform_{args.label}.png"))


if __name__ == "__main__":
    main()
