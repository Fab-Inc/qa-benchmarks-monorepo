#!/usr/bin/env python3
"""
Write this benchmark's result files, one set per language.

Output (local only -- nothing is uploaded), <out-root> defaulting to the repo's
results/:

    <out-root>/linguistic-knowledge/LK-en/
        LK-en_accuracy_final-137-items.csv              metrics
        LK-en_accuracy_final-137-items_frontiers.json   frontiers (price, size)
        benchmark_metadata.json                         benchmark + metric metadata
    <out-root>/linguistic-knowledge/LK-sw/
        ... the same for Kiswahili

The cluster metadata above them is written separately, by
packages/fab_benchmark_publisher/scripts/write_cluster_metadata.py.

Run from the repo root:

    # both languages
    uv run python benchmarks/linguistic_knowledge_benchmark/scripts/website_metrics_frontiers.py

    # just one
    uv run python benchmarks/linguistic_knowledge_benchmark/scripts/website_metrics_frontiers.py --pair LK-en

INPUTS, per language (see PAIRS below for which experiment each one reads):

  results/summary_linguistic-knowledge_<lang>_<experiment>[_fallback]_detailed.csv
      item scores: one row per question, one <model>_correct column per model.
      True -> 1, False -> 0, empty -> badly formatted (left out of accuracy, counted
      in n_items_bad_format), which is how the summary CSV scores it too.
  results/summary_linguistic-knowledge_<lang>_<experiment>[_fallback].csv
      cross-check only: the new accuracy must match its accuracy column to within
      its 2-decimal rounding, for the same set of models.
  models.csv (repo root)
      price and size for the frontiers.

No subsets: the website shows LK as a single score, so each model gets only its
overall row and each language gets one price and one size frontier.

LEFT EMPTY FOR NOW -- the response cache has per-item token counts, but what
"completion" includes differs by provider and the cache is local-only, so how to
fill these is not decided yet:
  input_tokens_per_repeat_mean, output_total_tokens_per_repeat_mean,
  output_reason_tokens_per_repeat_mean, output_non_reason_tokens_per_repeat_mean,
  cost_usd_per_repeat_mean, cost_usd_per_item_mean

This script is standalone on purpose: nothing imports it, so deleting it (and its
test, tests/test_website_metrics_frontiers.py) removes it cleanly.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from fab_benchmark_publisher.benchmark_frontiers import build_frontiers
from fab_benchmark_publisher.benchmark_results import write_benchmark_results
from fab_benchmark_publisher.metrics import DEFAULT_N_BOOTSTRAP, aggregate_item_scores
from fab_benchmark_publisher.result_metadata import BenchmarkMeta, MetricMeta

HERE = Path(__file__).resolve().parent
BENCH_DIR = HERE.parent
REPO_ROOT = BENCH_DIR.parent.parent

# Local folder grouping both languages; on GCS this sits under
# website/benchmark-clusters/foundational-literacy/.
BENCHMARK_GROUP = "linguistic-knowledge"

ACCURACY = MetricMeta(
    metric_slug="accuracy",
    display_name="Accuracy",
    display_description=(
        "Percentage of questions answered correctly. Badly formatted answers are "
        "left out of the score and counted separately."
    ),
    unit="%",
    is_higher_better=True,
    is_bad_format_scored=False,
)

# One row per language. `experiment` is the experiment_slug (the --experiment-id
# given to run-bench) and the middle of the source file names; `fallback` picks the
# *_fallback summaries, which are the ones the leaderboard publishes today.
PAIRS: list[dict] = [
    {
        "benchmark_slug": "LK-en",
        "display_name": "LK English",
        "language": "english",
        "experiment": "final-137-items",
        "fallback": True,
    },
    {
        "benchmark_slug": "LK-sw",
        "display_name": "LK Kiswahili",
        "language": "kiswahili",
        "experiment": "final-135-items",
        "fallback": True,
    },
]

# Summary accuracy is rounded to 2 decimals; anything further off means the reshape
# below misread the detailed CSV.
MAX_SUMMARY_DIFF = 0.01


def source_paths(pair: dict, results_dir: Path) -> tuple[Path, Path]:
    """(detailed CSV, summary CSV) for one pair."""
    stem = f"summary_linguistic-knowledge_{pair['language']}_{pair['experiment']}"
    if pair["fallback"]:
        stem += "_fallback"
    return results_dir / f"{stem}_detailed.csv", results_dir / f"{stem}.csv"


def item_scores_from_detailed(path: Path) -> pd.DataFrame:
    """Wide detailed CSV -> one row per (model_slug, item_id)."""
    detailed = pd.read_csv(path, dtype=str)
    correct_cols = [c for c in detailed.columns if c.endswith("_correct")]
    if not correct_cols:
        raise SystemExit(f"ERROR: no <model>_correct columns in {path}")

    long = detailed.melt(
        id_vars=["question_id"],
        value_vars=correct_cols,
        var_name="model_slug",
        value_name="correct",
    )
    unexpected = set(long["correct"].dropna()) - {"True", "False"}
    if unexpected:
        raise SystemExit(f"ERROR: unexpected _correct values in {path}: {unexpected}")

    long["model_slug"] = long["model_slug"].str.removesuffix("_correct")
    long["is_bad_format"] = long["correct"].isna()
    long["item_score"] = long["correct"].map({"True": 1.0, "False": 0.0})
    return long.rename(columns={"question_id": "item_id"})[
        ["model_slug", "item_id", "item_score", "is_bad_format"]
    ]


def check_against_summary(metrics: pd.DataFrame, summary_csv: Path) -> None:
    """The new accuracy must reproduce the summary CSV's for every model."""
    summary = pd.read_csv(summary_csv, dtype={"model": str})
    new = metrics.set_index("model_slug")["metric_value"].astype(float)
    old = summary.set_index("model")["accuracy"].astype(float)
    mismatched = sorted(set(new.index) ^ set(old.index))
    if mismatched:
        raise SystemExit(
            f"ERROR: detailed and summary CSVs disagree on models: {mismatched}"
        )
    diff = (new - old.reindex(new.index)).abs()
    bad = diff[diff > MAX_SUMMARY_DIFF]
    if not bad.empty:
        raise SystemExit(
            "ERROR: accuracy differs from the summary CSV for "
            + ", ".join(f"{m} ({d:.3f})" for m, d in bad.items())
        )


def write_pair(
    pair: dict,
    *,
    results_dir: Path,
    models_csv: Path,
    out_root: Path,
    n_bootstrap: int,
    seed: int,
) -> Path:
    detailed_csv, summary_csv = source_paths(pair, results_dir)
    for path in (detailed_csv, summary_csv):
        if not path.is_file():
            raise SystemExit(f"ERROR: {pair['benchmark_slug']}: not found: {path}")

    meta = BenchmarkMeta(
        benchmark_slug=pair["benchmark_slug"],
        display_name=pair["display_name"],
        metrics=(ACCURACY,),
    )
    metrics = aggregate_item_scores(
        item_scores_from_detailed(detailed_csv),
        benchmark_slug=meta.benchmark_slug,
        metric=ACCURACY,
        experiment_slug=pair["experiment"],
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    check_against_summary(metrics, summary_csv)
    frontiers = build_frontiers(metrics, metric=ACCURACY, models_csv=models_csv)

    outdir = out_root / BENCHMARK_GROUP / meta.benchmark_slug
    paths = write_benchmark_results(
        outdir, metrics=metrics, frontiers=frontiers, meta=meta
    )

    print(f"{meta.benchmark_slug}: {len(metrics)} models from {detailed_csv.name}")
    for f in frontiers["frontiers"]:
        print(f"  {f['axis']} frontier: {len(f['points'])} models")
    if frontiers["models_without_metadata"]:
        print(
            "  not in models.csv (no frontier): "
            + ", ".join(frontiers["models_without_metadata"])
        )
    for name in paths:
        print(f"  -> {outdir / name}")
    return outdir


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--pair",
        action="append",
        choices=[p["benchmark_slug"] for p in PAIRS],
        help="only write this pair (repeatable; default: all)",
    )
    parser.add_argument("--results-dir", type=Path, default=BENCH_DIR / "results")
    parser.add_argument("--models-csv", type=Path, default=REPO_ROOT / "models.csv")
    parser.add_argument("--out-root", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    selected = [p for p in PAIRS if not args.pair or p["benchmark_slug"] in args.pair]
    for pair in selected:
        write_pair(
            pair,
            results_dir=args.results_dir,
            models_csv=args.models_csv,
            out_root=args.out_root,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
