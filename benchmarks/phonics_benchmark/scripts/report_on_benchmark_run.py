"""Score benchmark runs against the items they executed on, and report metrics.

Reads run directories written by ``phonics-benchmark run-decodability`` and
writes the scoring outputs back into them.

The language and the benchmark item file are read from each run's own
``config.json`` snapshot, so a run is always scored against the exact item set
it was executed on. If that file has changed since the run, the recorded
``items_sha256`` no longer matches and the script says so. ``--items-path``
points at the items instead, for a run whose recorded path is unreachable from
here — a cloud run records the container's — and must still hash to that
``items_sha256``. To score against a *revised* gold standard on purpose, where
that hash cannot match, use ``regrade_benchmark_run.py``.

Repeating ``--run-id`` scores several runs that were executed on the same items
— a benchmark split across runs by provider, say — and pools them into one
combined report as well as writing each run's own. ``--combined-name`` names the
directory for it under ``<output root>/analysis/``, which is outside the run-id
namespace: a directory beside the runs would resolve as a ``--run-id`` but hold
no responses. Pooling re-scores the runs together rather than merging their
reports, because a bootstrap interval has to resample one item set across every
model for the intervals to be comparable between them.

Where one model was scored on the same item in two of the runs, only the run
that scored it on more items goes into the pool — a rerun normally supersedes a
truncated first attempt. The drop is announced, recorded in the combined
``scoring_provenance.json``, and applies to the pool alone: each run's own
report still covers everything that run scored.

Scoring is only on responses with successful provider calls and output parsing.
If any responses are excluded, the script prints how many, per model.

Run from the repository root, e.g.:

    uv run --frozen --package phonics-benchmark python \\
        benchmarks/phonics_benchmark/scripts/report_on_benchmark_run.py \\
        --run-id sw-tusome-260810 \\
        --inspect-sample-size 20

Writes, into `{run directory}/report` or whatever ``--report-subdir`` names:

- ``results_with_scoring.csv`` — every model/item row with its confusion counts
- ``summary_by_model.csv`` — one row per model, plus an ``is_baseline`` row each
  for a classifier that flags every word non-decodable and one that flags none
- ``bootstrap_ci.csv`` — per-model accuracy, pooled precision, recall, F1 and F2
  with confidence intervals, plus the same two ``is_baseline`` rows, skipped by
  ``--n-bootstrap 0``
- ``accuracy_by_nondecodable_count.png`` — accuracy against item difficulty
- ``sampled_items_for_inspection.csv`` — only with ``--inspect-sample-size``
- ``scoring_provenance.json`` — which gold file and hashes produced these numbers

The bootstrap resamples benchmark items with replacement, so its interval covers
item-sampling error and nothing else: not run-to-run variation in a model's own
output, and not whether the item set represents Kiswahili texts in general. Each
item counts as one independent draw, which understates the interval on runs where
one text backs many items.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from phonics_benchmark.reporting import (
    DEFAULT_OUTPUT_ROOT,
    ScoredRun,
    build_scored_run,
    load_run,
    load_run_items,
    write_report,
)


def score_one_run(
    run_id: str,
    output_root: Path,
    items_path: Path | None,
) -> ScoredRun:
    """Load and score one run against the items it executed on."""
    loaded = load_run(run_id, output_root)
    items = load_run_items(loaded.resolved_config, items_path)
    gold_path = items_path or Path(loaded.resolved_config["items_path"])
    return build_scored_run(loaded, items, gold_path)


def verify_runs_are_comparable(scored_runs: list[ScoredRun]) -> None:
    """Refuse to pool runs whose numbers do not belong on one axis.

    Pooling assumes the runs are slices of one benchmark, sliced by model, by
    item, or both. Three differences make that false and cannot be worked
    around: a different language, which means a different decodability rule;
    different items at run time, which means the models were not shown the same
    texts; and a different gold standard, which makes the metrics answer a
    different question.

    A model scored twice on the same item would also break the pool, but that
    one is repairable rather than fatal — see pool_results.
    """
    languages = {run.language for run in scored_runs}
    if len(languages) > 1:
        raise typer.BadParameter(
            f"These runs span {len(languages)} languages ({sorted(languages)}). "
            "Decodability is language-specific, so they cannot be pooled."
        )

    # What the models were shown, as opposed to what they are scored against.
    # Runs executed on different item sets answered different exams, and that
    # stays true even when both are later scored against one gold file.
    executed_on = {
        run.run_id: run.provenance.get("run_items_sha256") for run in scored_runs
    }
    distinct_executed = {digest for digest in executed_on.values() if digest}
    if len(distinct_executed) > 1:
        detail = "\n".join(
            f"  {run_id}: {str(digest)[:12]}"
            for run_id, digest in sorted(executed_on.items())
        )
        raise typer.BadParameter(
            "These runs were executed on different item sets, so their models "
            f"were not shown the same texts and cannot be pooled:\n{detail}"
        )

    golds = {run.provenance["gold_items_sha256"] for run in scored_runs}
    if len(golds) > 1:
        detail = "\n".join(
            f"  {run.run_id}: {run.provenance['gold_items_sha256'][:12]}"
            for run in scored_runs
        )
        raise typer.BadParameter(
            f"These runs were scored against {len(golds)} different gold "
            f"standards, so their metrics cannot be pooled:\n{detail}"
        )


def items_scored(results: pd.DataFrame, model: str) -> set:
    """The distinct benchmark items one model was scored on in one run."""
    return set(results[results["model"] == model]["benchmark_item_id"].unique())


def pool_results(scored_runs: list[ScoredRun]) -> tuple[pd.DataFrame, list[dict]]:
    """Concatenate the runs' rows, keeping one run's copy of any repeated model.

    A model scored on the same item in two runs would have that item counted
    twice in its pooled confusion totals. Rather than refuse, drop the copy from
    the run that scored that model on fewer items: when a model is rerun it is
    normally because the first attempt was truncated, so the longer one
    supersedes it. The usual case is a strict superset, where nothing is lost.

    A whole model-run is dropped rather than just the repeated items, so every
    pooled row for a model comes from one execution. Where that discards items
    the kept run does not have, the caller is told how many.

    Returns the pooled frame and a record of what was dropped, which the
    combined report keeps in its ``scoring_provenance.json`` — a leaderboard
    should not be the only trace of a model-run having been set aside.
    """
    frames = {run.run_id: run.results for run in scored_runs}
    coverage = pd.concat(
        [
            results[["model", "benchmark_item_id"]].assign(run_id=run_id)
            for run_id, results in frames.items()
        ],
        ignore_index=True,
    )
    repeated = coverage[coverage.duplicated(["model", "benchmark_item_id"], keep=False)]

    dropped: list[dict] = []
    drop_by_run: dict[str, set[str]] = {}
    for model in sorted(repeated["model"].unique()):
        # Stable sort, and groupby orders run_ids alphabetically, so an exact
        # tie in item count resolves the same way on every rerun.
        counts = (
            coverage[coverage["model"] == model]
            .groupby("run_id")["benchmark_item_id"]
            .nunique()
            .sort_values(ascending=False, kind="stable")
        )
        kept_run = str(counts.index[0])
        kept_items = items_scored(frames[kept_run], model)
        for run_id in map(str, counts.index[1:]):
            run_items = items_scored(frames[run_id], model)
            drop_by_run.setdefault(run_id, set()).add(model)
            dropped.append(
                {
                    "model": model,
                    "kept_run": kept_run,
                    "kept_items": len(kept_items),
                    "dropped_run": run_id,
                    "dropped_items": len(run_items),
                    "items_lost": len(run_items - kept_items),
                    "tied_on_item_count": len(run_items) == len(kept_items),
                }
            )

    announce_dropped_model_runs(dropped)

    pooled = pd.concat(
        [
            results[~results["model"].isin(drop_by_run.get(run_id, set()))]
            for run_id, results in frames.items()
        ],
        ignore_index=True,
    )
    return pooled, dropped


def announce_dropped_model_runs(dropped: list[dict]) -> None:
    """Say which model-runs were left out of the pool, and what it cost."""
    if not dropped:
        return
    typer.secho(
        f"Dropped {len(dropped)} model-run(s) from the pool as superseded:",
        fg=typer.colors.CYAN,
        err=True,
    )
    for record in dropped:
        line = (
            f"  {record['model']}: kept {record['kept_run']} "
            f"({record['kept_items']} items), dropped {record['dropped_run']} "
            f"({record['dropped_items']} items)"
        )
        if record["tied_on_item_count"]:
            line += " — same item count, kept the first by run ID"
        if record["items_lost"]:
            line += (
                f"\n    WARNING: this discards {record['items_lost']} item(s) "
                "the kept run does not cover"
            )
        typer.secho(
            line,
            fg=typer.colors.YELLOW if record["items_lost"] else typer.colors.CYAN,
            err=True,
        )


def warn_about_partial_coverage(pooled: pd.DataFrame) -> None:
    """Name models that the pool does not cover evenly.

    Coverage is counted per model over the pooled rows, not per run: a model
    split across item shards is complete even though neither shard holds all of
    it. Failed and unparseable responses are already excluded, so a short count
    here is a real gap in what that model was scored on, and its metrics rest on
    a different item subset from everyone else's.
    """
    pooled_items = pooled["benchmark_item_id"].nunique()
    per_model = pooled.groupby("model")["benchmark_item_id"].nunique()
    short = per_model[per_model < pooled_items]
    if short.empty:
        return
    detail = "\n".join(
        f"  {model}: {count} of {pooled_items} items" for model, count in short.items()
    )
    typer.secho(
        "WARNING: these models were not scored on every pooled item, so their "
        f"metrics rest on a different item subset:\n{detail}",
        fg=typer.colors.YELLOW,
        err=True,
    )


def report_on_run(
    run_ids: list[str] = typer.Option(
        ...,
        "--run-id",
        help=(
            "Run ID to report on, resolved under the output root. Repeat it to "
            "score several runs that share an item set and pool them into one "
            "combined report as well as their own."
        ),
    ),
    combined_name: str | None = typer.Option(
        None,
        "--combined-name",
        help=(
            "Directory name for the combined report, written under "
            "<output root>/analysis/. Required when more than one --run-id is "
            "given, and unused for a single run."
        ),
    ),
    output_root: Path = typer.Option(
        DEFAULT_OUTPUT_ROOT,
        "--output-root",
        file_okay=False,
        help="Where --run-id is looked up. Defaults to the package's outputs/ folder.",
    ),
    items_path: Path | None = typer.Option(
        None,
        "--items-path",
        exists=True,
        dir_okay=False,
        readable=True,
        help=(
            "Local copy of the run's benchmark items, for a run whose recorded "
            "items_path is not reachable from here (a cloud run records the "
            "container's path). Must hash to the run's items_sha256, and to "
            "every run's when several are given."
        ),
    ),
    report_subdir: str = typer.Option(
        "report",
        "--report-subdir",
        help="Directory under each run to write that run's report into.",
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
    """Report on one or more benchmark runs, and pool them if there are several."""
    duplicates = sorted({run_id for run_id in run_ids if run_ids.count(run_id) > 1})
    if duplicates:
        raise typer.BadParameter(
            f"--run-id was given more than once for {duplicates}. A run can only "
            "contribute to the pool once."
        )

    combined_dir: Path | None = None
    if len(run_ids) > 1:
        if combined_name is None:
            raise typer.BadParameter(
                "--combined-name is required with more than one --run-id: it "
                "names the directory under analysis/ that the combined report "
                "goes in."
            )
        combined_dir = output_root / "analysis" / combined_name
    elif combined_name is not None:
        typer.secho(
            "WARNING: --combined-name is ignored with a single --run-id; that "
            "run's report goes in its own directory as usual.",
            fg=typer.colors.YELLOW,
            err=True,
        )

    # Score every run before writing anything, so a run that does not belong in
    # the pool fails while the previous reports are still untouched.
    scored_runs = [score_one_run(run_id, output_root, items_path) for run_id in run_ids]
    if len(scored_runs) > 1:
        verify_runs_are_comparable(scored_runs)

    for run in scored_runs:
        write_report(
            run.results,
            run.language,
            run.run_dir / report_subdir,
            run.provenance,
            run.run_id,
            n_bootstrap=n_bootstrap,
            ci_level=ci_level,
            seed=seed,
            inspect_sample_size=inspect_sample_size,
        )

    if combined_dir is None:
        return

    # Only the pool drops a superseded model-run; each run's own report above
    # still reports everything that run actually scored.
    # Every row already carries its own run_id, so the pooled CSV stays traceable
    # back to the run each response came from.
    pooled, dropped = pool_results(scored_runs)
    warn_about_partial_coverage(pooled)
    combined_provenance = {
        "run_ids": [run.run_id for run in scored_runs],
        # Identical across runs — verify_runs_are_comparable refused otherwise.
        "gold_items_path": scored_runs[0].provenance["gold_items_path"],
        "gold_items_sha256": scored_runs[0].provenance["gold_items_sha256"],
        "dropped_as_superseded": dropped,
        "runs": {run.run_id: run.provenance for run in scored_runs},
    }
    write_report(
        pooled,
        scored_runs[0].language,
        combined_dir,
        combined_provenance,
        f"{len(scored_runs)} pooled runs ({', '.join(run_ids)})",
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        seed=seed,
        inspect_sample_size=inspect_sample_size,
    )


def main() -> None:
    """Entry point."""
    typer.run(report_on_run)


if __name__ == "__main__":
    main()
