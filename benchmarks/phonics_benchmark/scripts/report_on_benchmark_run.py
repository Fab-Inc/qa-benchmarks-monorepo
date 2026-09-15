"""Score one benchmark run and report per-model metrics.

Reads a run directory written by ``phonics-benchmark run-decodability`` and writes
its scoring outputs back into the same directory.

The language and the benchmark item file are read from the run's own
``config.json`` snapshot rather than passed in, so a run is always scored against
the exact item set it was executed on. If that file has changed since the run,
the recorded ``items_sha256`` no longer matches and the script says so.

Scoring is only on responses with successful provider calls and output parsing.
If any responses are excluded, the script prints how many, per model.

Run from the repository root, e.g.:

    uv run --package phonics-benchmark python \\
        benchmarks/phonics_benchmark/scripts/report_on_benchmark_run.py \\
        --run-id sw-tusome-260810 \\
        --inspect-sample-size 20

Writes, into `{run directory}/report`:

- ``results_with_scoring.csv`` — every model/item row with its confusion counts
- ``summary_by_model.csv`` — one row per model
- ``bootstrap_ci.csv`` — per-model accuracy, pooled F1 and pooled F2 with
  confidence intervals, skipped by ``--n-bootstrap 0``
- ``accuracy_by_nondecodable_count.png`` — accuracy against item difficulty
- ``sampled_items_for_inspection.csv`` — only with ``--inspect-sample-size``

The bootstrap resamples benchmark items with replacement, so its interval covers
item-sampling error and nothing else: not run-to-run variation in a model's own
output, and not whether the item set represents Kiswahili texts in general. Each
item counts as one independent draw, which understates the interval on runs where
one text backs many items.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pandas as pd
import typer
from phonics_benchmark.bootstrap import bootstrap_metric_cis
from phonics_benchmark.parsing import parse_word_list
from phonics_benchmark.plots import save_accuracy_by_nondecodable_count_plot
from phonics_benchmark.results import CALL_STATUS_SUCCESS, load_results
from phonics_benchmark.scoring import (
    accuracy_by_nondecodable_count,
    add_per_item_confusion_matrix,
    summarize_by_model,
)
from phonics_benchmark.utils import require_columns

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "outputs"

# Fixed rather than exposed as a flag: the sample is for eyeballing, and a stable
# seed means re-running the script shows you the same items.
INSPECTION_RANDOM_STATE = 42

# Item metadata joined onto each response row. `nondecodable_words` is the gold
# answer; the rest are the difficulty columns the breakdowns group by.
ITEM_METADATA_COLUMNS = [
    "benchmark_item_id",
    "student_knowledge",
    "content",
    "nondecodable_words",
    "nondecodable_ratio",
    "num_nondecodable_words",
    "num_total_words",
]

# Enough to judge a response by eye without the noise of the full row.
INSPECTION_COLUMNS = [
    "run_id",
    "benchmark_item_id",
    "model",
    "prompt",
    "latency_seconds",
    "tokens_used_completion",
    "content",
    "student_knowledge",
    "nondecodable_words",
    "response_parsed",
    "tp_count",
    "fp_count",
    "fn_count",
    "tn",
    "perfect_match",
]


def resolve_run_dir(run_id: str, output_root: Path) -> Path:
    """Return the run's output directory, checking it looks like one."""
    run_dir = output_root / run_id
    if not run_dir.is_dir():
        raise typer.BadParameter(f"Run directory does not exist: {run_dir}")
    if not (run_dir / "responses").is_dir():
        raise typer.BadParameter(
            f"{run_dir} has no responses/ subdirectory — is it a run directory?"
        )
    return run_dir


def read_run_snapshot(run_dir: Path) -> dict:
    """Return the ``resolved_config`` block of a run's config snapshot."""
    snapshot_path = run_dir / "config.json"
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise typer.BadParameter(
            f"Missing {snapshot_path}; the run's language and item set are read "
            "from it. Was this run written by an older version of the runner?"
        ) from None
    return snapshot["resolved_config"]


def load_run_responses(run_dir: Path) -> pd.DataFrame:
    """Concatenate every model's responses JSONL and parse each raw response."""
    response_file_paths = sorted((run_dir / "responses").glob("*.jsonl"))
    if not response_file_paths:
        raise typer.BadParameter(f"No responses/*.jsonl found in {run_dir}")

    frames = [load_results(path) for path in response_file_paths]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise typer.BadParameter(f"Every responses file in {run_dir} was empty")

    results = pd.concat(frames, ignore_index=True)

    parsed = results["output"].apply(parse_word_list)
    results["response_parsed"] = [result.words for result in parsed]
    results["parse_ok"] = [result.parse_ok for result in parsed]

    return results


@require_columns(["call_status", "parse_ok"])
def split_scorable_responses(
    responses: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split loaded responses into included vs. excluded rows for scoring.

    Two types of rows are excluded:

    * ``call_status`` is not ``success``
    * ``parse_ok`` is False

    Returns ``(scorable, excluded)`` with ``excluded`` having an ``exclusion_reason``
    column.
    """
    call_failed = responses["call_status"].ne(CALL_STATUS_SUCCESS)
    parse_failed = ~responses["parse_ok"] & ~call_failed

    reason = pd.Series(index=responses.index, dtype="object")
    reason[parse_failed] = "parse_failed"
    reason[call_failed] = "call_failed"

    excluded = responses[reason.notna()].copy()
    excluded["exclusion_reason"] = reason[reason.notna()]
    scorable = responses[reason.isna()].copy()
    return scorable, excluded


def warn_about_exclusions(excluded: pd.DataFrame) -> None:
    """Print a per-model, per-reason count of the rows left out of scoring.

    Counts only: the excluded rows themselves stay in the run's own
    ``responses/*.jsonl``, so the report has no reason to copy them.
    """
    breakdown = (
        excluded.groupby(["model", "exclusion_reason"]).size().unstack(fill_value=0)
    )
    typer.secho(
        f"WARNING: excluded {len(excluded)} response row(s) from scoring.\n",
        fg=typer.colors.YELLOW,
        err=True,
    )
    typer.secho(breakdown.to_string(), fg=typer.colors.YELLOW, err=True)


def load_run_items(resolved_config: dict) -> pd.DataFrame:
    """Load the exact item set the run was executed against.

    Warns if the file's bytes don't hash to the ``items_sha256`` recorded at
    run time, which would mean the scores are being computed against different
    items than the models saw.
    """
    items_path = Path(resolved_config["items_path"])
    if not items_path.is_file():
        raise typer.BadParameter(f"The run's items file no longer exists: {items_path}")

    recorded_sha256 = resolved_config.get("items_sha256")
    # actual sha256 from items.py:24
    actual_sha256 = sha256(items_path.read_bytes()).hexdigest()
    if recorded_sha256 and recorded_sha256 != actual_sha256:
        typer.secho(
            f"WARNING: {items_path} has changed since this run.\n"
            f"  recorded items_sha256: {recorded_sha256}\n"
            f"  current  items_sha256: {actual_sha256}",
            fg=typer.colors.YELLOW,
            err=True,
        )

    items = pd.read_json(items_path, lines=True)
    return items


def sample_items_for_inspection(
    scored_results: pd.DataFrame,
    sample_size: int,
) -> pd.DataFrame:
    """Return every model's row for a random sample of benchmark items."""
    item_ids = pd.Series(scored_results["benchmark_item_id"].unique())
    sampled_ids = item_ids.sample(
        min(sample_size, len(item_ids)), random_state=INSPECTION_RANDOM_STATE
    )
    sampled = scored_results[scored_results["benchmark_item_id"].isin(sampled_ids)]
    columns = [column for column in INSPECTION_COLUMNS if column in sampled.columns]
    return sampled[columns].sort_values(["benchmark_item_id", "model"])


def report_on_run(
    run_id: str = typer.Option(
        ...,
        "--run-id",
        help="Run ID to report on, resolved under the output root.",
    ),
    output_root: Path = typer.Option(
        DEFAULT_OUTPUT_ROOT,
        "--output-root",
        file_okay=False,
        help="Where --run-id is looked up. Defaults to the package's outputs/ folder.",
    ),
    inspect_sample_size: int | None = typer.Option(
        None,
        "--inspect-sample-size",
        min=1,
        help=(
            "Number of benchmark items to sample, with every model's response, "
            "for manual inspection. Omit to skip the sample."
        ),
    ),
    n_bootstrap: int = typer.Option(
        1000,
        "--n-bootstrap",
        min=0,
        help="Bootstrap replicates for the confidence intervals. 0 skips them.",
    ),
    ci_level: float = typer.Option(
        0.95,
        "--ci-level",
        min=0.0,
        max=1.0,
        help="Confidence level for the bootstrap intervals.",
    ),
    seed: int = typer.Option(
        0,
        "--seed",
        help="Bootstrap seed. Fixed by default so reruns give the same interval.",
    ),
) -> None:
    """Report on a benchmark run and write its per-model summary and plots."""
    run_dir = resolve_run_dir(run_id, output_root)
    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = read_run_snapshot(run_dir)
    language = resolved_config["language"]

    responses, excluded = split_scorable_responses(load_run_responses(run_dir))
    written: list[str] = []
    if not excluded.empty:
        warn_about_exclusions(excluded)
    if responses.empty:
        raise typer.BadParameter(
            f"Nothing to score. Every response row in {run_dir} failed or could"
            "not be parsed."
        )

    items = load_run_items(resolved_config)

    results = responses.merge(
        items[ITEM_METADATA_COLUMNS], on="benchmark_item_id", how="left"
    )
    missing_metadata = int(results["nondecodable_words"].isna().sum())
    if missing_metadata:
        typer.secho(
            f"WARNING: {missing_metadata} response row(s) matched no benchmark "
            "item and cannot be scored.",
            fg=typer.colors.YELLOW,
            err=True,
        )

    results = add_per_item_confusion_matrix(results, language)
    scored_path = report_dir / "results_with_scoring.csv"
    results.to_csv(scored_path, index=False)
    written.append(scored_path.name)

    # Per-model accuracy + pooled word-level confusion (precision/recall/f1).
    summary = summarize_by_model(results)
    typer.echo(f"\nPer-model summary for {resolved_config['run_id']}:")
    typer.echo(summary.to_string(index=False))
    summary.to_csv(report_dir / "summary_by_model.csv", index=False)
    written.append("summary_by_model.csv")

    # How much of the gap between two models is item-sampling noise? See
    # bootstrap.py for what the interval does and does not cover.
    if n_bootstrap:
        ci_report = bootstrap_metric_cis(
            results,
            n_bootstrap=n_bootstrap,
            ci_level=ci_level,
            seed=seed,
        )
        typer.echo(
            f"\nBootstrap {ci_level:.0%} CIs from {n_bootstrap} replicates, "
            f"resampling {ci_report['n_items_resampled'].iloc[0]} items:"
        )
        typer.echo(ci_report.to_string(index=False))
        ci_report.to_csv(report_dir / "bootstrap_ci.csv", index=False)
        written.append("bootstrap_ci.csv")

    # Does accuracy degrade as items get harder (more non-decodable words)?
    accuracy_by_count = accuracy_by_nondecodable_count(results)
    plot_path = report_dir / "accuracy_by_nondecodable_count.png"
    save_accuracy_by_nondecodable_count_plot(accuracy_by_count, plot_path)
    written.append(plot_path.name)

    # Sample a few items for inspection, if requested.
    if inspect_sample_size:
        sampled = sample_items_for_inspection(results, inspect_sample_size)
        sampled_path = report_dir / "sampled_items_for_inspection.csv"
        sampled.to_csv(sampled_path, index=False)
        written.append(sampled_path.name)
        typer.echo(
            f"\nSampled {len(sampled['benchmark_item_id'].unique())} items for "
            f"inspection in {sampled_path}."
        )
        typer.echo("  Use --inspect-sample-size to change the number of items.")

    typer.echo(f"\nWrote to {report_dir}:")
    for name in written:
        typer.echo(f"  - {name}")


def main() -> None:
    """Entry point."""
    typer.run(report_on_run)


if __name__ == "__main__":
    main()
