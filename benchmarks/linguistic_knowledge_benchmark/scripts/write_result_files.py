#!/usr/bin/env python3
"""
Write this benchmark's result files, one set per language.

Output (local only -- nothing is uploaded), <out-root> defaulting to the repo's
results/:

    <out-root>/linguistic-knowledge/LK-en/
        LK-en_accuracy.csv              metrics (overall, then per subset)
        LK-en_accuracy_metadata.json    metric metadata
        LK-en_accuracy_frontiers.json   price and size frontiers (overall, then per
                                        subset)
    <out-root>/linguistic-knowledge/LK-sw/
        ... the same for Kiswahili

Run from the repo root:

    # both languages, from their default judged folders (BENCHMARKS below)
    uv run python benchmarks/linguistic_knowledge_benchmark/scripts/write_result_files.py

    # just one, from another judged folder
    uv run python benchmarks/linguistic_knowledge_benchmark/scripts/write_result_files.py \
        --benchmark LK-en --judged-dir LK-en <experiment folder>/<judging pass folder>

Steps, per language:
  1. metrics   the judged folder: one <JUDGED_FILE_PREFIX><model>.csv per model, in
               <experiment folder>/<judging pass folder> (shared_utils.experiment_dir);
               overall, then per value of each benchmark_subsets key of the
               benchmark in the leaderboard config (a column of the judged files,
               e.g. domain; LK uses none for now)
               -> LK-<lang>_accuracy.csv
  2. metadata  the metric's definition in the leaderboard config
               (leaderboard_cli.DEFAULT_CONFIG_FILE, at the repo root), plus the
               benchmark's benchmark_subsets
               -> LK-<lang>_accuracy_metadata.json
  3. frontier  LK-<lang>_accuracy.csv (read back from disk) + the models CSV
               (shared_utils.MODELS_CSV)
               -> LK-<lang>_accuracy_frontiers.json

Step 1 reads the judged response files directly, and only models active in the
model list (MODELS_LIST_YAML in shared_utils.py) are published: a model dropped from
that list can still have files in the cache. experiment_slug is the name of the
experiment folder holding the judged folder, so it always says which run the numbers
came from.

Accuracy follows the metric's is_bad_format_scored in the leaderboard config. Each
row is one of (see accuracy_from_judged):
  scored       has a verdict (the fallback judge's where it gave one, else
               is_correct) -- always in the accuracy
  bad format   the model was called fine but its answer got no verdict (empty, or
               unreadable even to the fallback judge) -- counted as wrong when
               is_bad_format_scored is true, left out when false
  call error   call_status == "error": the provider failed, the model never
               answered -- always left out, whatever the flag says

This script is standalone on purpose: nothing imports it, so deleting it (and its
test, tests/test_write_result_files.py) removes it cleanly.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd
from fab_benchmark_publisher.benchmark_frontiers import build_frontier
from fab_benchmark_publisher.leaderboard_cli import DEFAULT_CONFIG_FILE
from fab_benchmark_publisher.metrics import (
    build_metrics,
    canonical_subset,
    load_benchmark_subsets,
    load_metric_definition,
    result_file_name,
)
from linguistic_knowledge_benchmark.run_bench import CALL_STATUS_ERROR
from linguistic_knowledge_benchmark.shared_utils import (
    EVAL_MODELS,
    MODELS_CSV,
    MODELS_LIST_YAML,
    REPO_ROOT,
    experiment_dir,
)

# <out-root>/<RESULTS_FOLDER>/<benchmark_slug>/ holds each language's files.
RESULTS_FOLDER = "linguistic-knowledge"
# The one metric this benchmark publishes; its definition is in the leaderboard config.
METRIC_SLUG = "accuracy"

# The names judge_bench.py writes judged responses under, inside an experiment folder:
# one folder per judging pass, one <JUDGED_FILE_PREFIX><model>.csv per model.
JUDGED_FALLBACK_DIR_NAME = "responses_judged_fallback"
JUDGED_FILE_PREFIX = "responses_judged_"

# One row per language: the exact judged folder to publish is
# experiment_dir(language, experiment) / judged_pass. JUDGED_FALLBACK_DIR_NAME is the
# pass the leaderboard publishes today. Override the folder per run with --judged-dir.
BENCHMARKS: list[dict] = [
    {
        "benchmark_slug": "LK-en",
        "language": "english",
        "experiment": "final-137-items",
        "judged_pass": JUDGED_FALLBACK_DIR_NAME,
    },
    {
        "benchmark_slug": "LK-sw",
        "language": "kiswahili",
        "experiment": "final-135-items",
        "judged_pass": JUDGED_FALLBACK_DIR_NAME,
    },
]


def accuracy_from_judged(
    judged: pd.DataFrame, *, is_bad_format_scored: bool
) -> float | None:
    """Compute one model's accuracy from its judged responses.

    Each row is scored, bad format or call error, and counted as the module
    docstring describes; is_bad_format_scored decides whether bad-format rows count
    as wrong or are left out.

    Args:
        judged: one model's judged CSV (is_correct, optionally is_correct_fallback
            and call_status).
        is_bad_format_scored: the metric's flag from the leaderboard config.

    Returns:
        The accuracy in % rounded to 2 decimals, or None when no row counts.
    """
    verdict = judged["is_correct"]
    if "is_correct_fallback" in judged.columns:
        verdict = judged["is_correct_fallback"].fillna(verdict)
    is_call_error = (
        judged.get("call_status", pd.Series(index=judged.index))
        .astype(str)
        .eq(CALL_STATUS_ERROR)
    )
    is_scored = verdict.notna()
    is_bad_format = ~is_scored & ~is_call_error

    n_correct = int(verdict[is_scored].astype(str).str.lower().eq("true").sum())
    n_counted = int(is_scored.sum())
    if is_bad_format_scored:
        n_counted += int(is_bad_format.sum())
    return round(n_correct / n_counted * 100, 2) if n_counted else None


def accuracy_by_model(
    judged_dir: Path,
    listed_models: Iterable[str],
    subset_keys: Sequence[str],
    *,
    is_bad_format_scored: bool,
) -> tuple[dict[str, float], dict[str, dict[str, float]], list[str], list[str]]:
    """Compute the accuracy of every listed model judged in one folder, overall and
    per subset.

    Only models in `listed_models` are published; the two mismatches are returned so
    the run can report them instead of failing.

    Args:
        judged_dir: the judging-pass folder, holding <JUDGED_FILE_PREFIX><model>.csv.
        listed_models: the models allowed on the leaderboard (the model list).
        subset_keys: the benchmark's benchmark_subsets, each a column of the judged
            files (e.g. "domain"); every value of that column is one subset.
        is_bad_format_scored: passed through to accuracy_from_judged.

    Returns:
        ({model: accuracy} for listed models with a counted row,
         {benchmark_subset: {model: accuracy}} for each key, then each sorted value,
         listed models with no judged file, judged models that are not listed).

    Raises ValueError if a subset key is not a column of a judged file.
    """
    judged_file_by_model = {
        path.stem.removeprefix(JUDGED_FILE_PREFIX): path
        for path in judged_dir.glob(f"{JUDGED_FILE_PREFIX}*.csv")
    }
    listed_models = list(listed_models)
    accuracy_of_model: dict[str, float] = {}
    # (index of the key in subset_keys, value) -> {model: accuracy}
    accuracy_by_key_value: dict[tuple[int, str], dict[str, float]] = {}
    for model in listed_models:
        if model not in judged_file_by_model:
            continue
        judged = pd.read_csv(judged_file_by_model[model])
        accuracy = accuracy_from_judged(
            judged, is_bad_format_scored=is_bad_format_scored
        )
        if accuracy is not None:
            accuracy_of_model[model] = accuracy
        for key_index, key in enumerate(subset_keys):
            if key not in judged.columns:
                raise ValueError(
                    f"benchmark_subsets: '{key}' is not a column of "
                    f"{judged_file_by_model[model].name}"
                )
            for value, subset_rows in judged.groupby(key):
                accuracy = accuracy_from_judged(
                    subset_rows, is_bad_format_scored=is_bad_format_scored
                )
                if accuracy is not None:
                    accuracy_by_key_value.setdefault((key_index, str(value)), {})[
                        model
                    ] = accuracy
    accuracy_by_subset = {
        canonical_subset({subset_keys[key_index]: value}): accuracy_by_key_value[
            (key_index, value)
        ]
        for key_index, value in sorted(accuracy_by_key_value)
    }
    listed_not_judged = [m for m in listed_models if m not in judged_file_by_model]
    judged_not_listed = sorted(set(judged_file_by_model) - set(listed_models))
    return accuracy_of_model, accuracy_by_subset, listed_not_judged, judged_not_listed


def write_result_files(
    benchmark_slug: str,
    judged_dir: Path,
    *,
    listed_models: Iterable[str],
    models_csv: Path,
    config_path: Path,
    out_root: Path,
) -> None:
    """Write the metrics, metadata and frontier files of one benchmark.

    Runs the three steps of the module docstring for one language, writes the files
    to <out_root>/<RESULTS_FOLDER>/<benchmark_slug>/ and prints what was published
    and which models were left out. Stops the run (SystemExit) when no listed model
    has a scored result, e.g. a wrong or missing judged folder.

    Args:
        benchmark_slug: the benchmark to write, e.g. "LK-en".
        judged_dir: the judging-pass folder to read; its parent folder's name
            becomes the experiment_slug.
        listed_models: the models allowed on the leaderboard.
        models_csv: model price, size and open weights, for the frontier.
        config_path: the leaderboard config holding the metric's definition.
        out_root: root folder the result files are written under.
    """
    metric_definition = load_metric_definition(config_path, benchmark_slug, METRIC_SLUG)
    subset_keys = load_benchmark_subsets(config_path, benchmark_slug)
    accuracy_of_model, accuracy_by_subset, listed_not_judged, judged_not_listed = (
        accuracy_by_model(
            judged_dir,
            listed_models,
            subset_keys,
            is_bad_format_scored=metric_definition["is_bad_format_scored"],
        )
    )
    if not accuracy_of_model:
        raise SystemExit(f"ERROR: {benchmark_slug}: no scored model in {judged_dir}")
    outdir = out_root / RESULTS_FOLDER / benchmark_slug
    outdir.mkdir(parents=True, exist_ok=True)
    metrics_path, metadata_path, frontier_path = (
        outdir / result_file_name(benchmark_slug, METRIC_SLUG, suffix)
        for suffix in (".csv", "_metadata.json", "_frontiers.json")
    )

    build_metrics(
        accuracy_of_model,
        benchmark_slug=benchmark_slug,
        metric_slug=METRIC_SLUG,
        # a judged folder sits directly in its experiment folder
        experiment_slug=judged_dir.parent.name,
        metric_value_by_subset=accuracy_by_subset,
    ).to_csv(metrics_path, index=False)
    metadata = {
        "benchmark_slug": benchmark_slug,
        "benchmark_subsets": subset_keys,
        **metric_definition,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    # the frontier is built from the metrics file just written, not from memory
    metrics = pd.read_csv(metrics_path)
    frontier = build_frontier(
        metrics,
        is_higher_better=metric_definition["is_higher_better"],
        models_csv=models_csv,
    )
    frontier_path.write_text(json.dumps(frontier, indent=2) + "\n", encoding="utf-8")

    print(f"{benchmark_slug}: {len(accuracy_of_model)} models from {judged_dir}")
    model_list_name = MODELS_LIST_YAML.name
    if listed_not_judged:
        print(
            f"  in {model_list_name} but not judged here: "
            + ", ".join(listed_not_judged)
        )
    if judged_not_listed:
        print(
            f"  judged but not in {model_list_name} (left out): "
            + ", ".join(judged_not_listed)
        )
    for axis_frontier in frontier["frontiers"]:
        subset = (
            ", ".join(f"{k}={v}" for k, v in axis_frontier["benchmark_subset"].items())
            or "overall"
        )
        print(
            f"  {subset} {axis_frontier['axis']} frontier: "
            f"{len(axis_frontier['points'])} models"
        )
    if frontier["models_without_metadata"]:
        print(
            f"  not in {models_csv.name} (no frontier): "
            + ", ".join(frontier["models_without_metadata"])
        )
    for path in (metrics_path, metadata_path, frontier_path):
        print(f"  -> {path}")


def main(argv: Sequence[str] | None = None) -> None:
    """Parse the command line and write the result files of each selected benchmark.

    `argv` defaults to the real command line; tests pass their own.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        choices=[b["benchmark_slug"] for b in BENCHMARKS],
        help="only write this benchmark (repeatable; default: all)",
    )
    parser.add_argument(
        "--judged-dir",
        action="append",
        nargs=2,
        metavar=("BENCHMARK", "FOLDER"),
        default=[],
        help="judged responses folder to read for one benchmark, instead of its "
        "default in BENCHMARKS (repeatable)",
    )
    parser.add_argument("--models-csv", type=Path, default=MODELS_CSV)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / DEFAULT_CONFIG_FILE)
    parser.add_argument("--out-root", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args(argv)
    judged_dir_overrides = {slug: Path(folder) for slug, folder in args.judged_dir}
    unknown_slugs = set(judged_dir_overrides) - {
        b["benchmark_slug"] for b in BENCHMARKS
    }
    if unknown_slugs:
        parser.error(f"--judged-dir: unknown benchmark(s) {', '.join(unknown_slugs)}")

    for benchmark in BENCHMARKS:
        benchmark_slug = benchmark["benchmark_slug"]
        if args.benchmark and benchmark_slug not in args.benchmark:
            continue
        default_judged_dir = (
            experiment_dir(benchmark["language"], benchmark["experiment"])
            / benchmark["judged_pass"]
        )
        write_result_files(
            benchmark_slug,
            judged_dir_overrides.get(benchmark_slug, default_judged_dir),
            listed_models=EVAL_MODELS,
            models_csv=args.models_csv,
            config_path=args.config,
            out_root=args.out_root,
        )


if __name__ == "__main__":
    main()
