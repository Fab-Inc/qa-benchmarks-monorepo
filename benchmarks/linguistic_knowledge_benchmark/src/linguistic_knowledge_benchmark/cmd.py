"""Command-line interface for linguistic_knowledge_benchmark."""

import json
import re
from datetime import UTC, datetime

import pandas as pd
import typer
from dotenv import load_dotenv
from fab_benchmark_publisher import download_file
from fdllm import register_models

from linguistic_knowledge_benchmark.judge_bench import (
    DEFAULT_JUDGE_MODELS,
    build_detailed,
    compute_judge_agreement,
    compute_summary,
    judge_results,
    judge_results_fallback,
)
from linguistic_knowledge_benchmark.run_bench import run_benchmark
from linguistic_knowledge_benchmark.shared_utils import (
    BENCH_DIR,
    DATA_DIR,
    EVAL_MODELS,
    REPO_ROOT,
    run_dir,
)

# Add a new language by adding an entry here and dropping its JSON in data/.
# The medium of evaluation is English for every language: questions are written
# in English (only their subject changes from language to language).
#
# This is the NORMAL way run-bench/judge-bench pick a dataset: each language points
# directly at one fixed file here, no other configuration needed for everyday use.
# `--dataset` on both commands (below) is an EXCEPTIONAL escape hatch that overrides
# this for a single run -- e.g. scoring against the repeated dataset built by
# scripts/build_consistency_dataset.py for the "consistency" validation check.
DATASETS: dict[str, dict] = {
    "kiswahili": {
        "path": DATA_DIR / "linguistic_knowledge_eval_kiswahili.json",
        "language_name": "Kiswahili",
    },
    "english": {
        "path": DATA_DIR / "linguistic_knowledge_eval_english.json",
        "language_name": "English",
    },
}

# run-id must be alphanumeric + hyphens only (no underscores) so that underscores
# can be used as field separators in output filenames.
_RUN_ID_RE = re.compile(r"^[a-zA-Z0-9-]+$")


def _timestamp_run_id() -> str:
    """Return a filesystem-safe UTC timestamp for non-overwriting runs."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _validate_run_id(run_id: str | None) -> None:
    """Abort with a clear message if run_id contains invalid characters."""
    if run_id and not _RUN_ID_RE.match(run_id):
        typer.echo(
            f"ERROR: --run-id '{run_id}' is invalid. "
            "Use only alphanumeric characters and hyphens (no underscores).",
            err=True,
        )
        raise typer.Exit(1)


app = typer.Typer(
    name="linguistic-knowledge-benchmark",
    help="Benchmark system for evaluating LLM performance on language knowledge evals",
    no_args_is_help=True,
)


@app.callback()
def setup() -> None:
    """Linguistic Knowledge Benchmark"""
    # Load provider API keys from the repo-root .env into os.environ so fdllm
    # can find them. Existing environment variables take precedence (override=False).
    load_dotenv(REPO_ROOT / ".env")
    # custom_models.yaml is gitignored (it's synced from fab-open-configs, not
    # committed), so it must be fetched here on every run -- without this, a fresh
    # checkout (e.g. CI) has no file for register_models() to read and crashes.
    download_file(
        bucket_name="fab-open-configs",
        file_name="custom_models.yaml",
        write_path=REPO_ROOT / "custom_models.yaml",
        require_auth=False,
    )
    register_models(str(REPO_ROOT / "custom_models.yaml"))


@app.command("run-bench")
def run_bench_cmd(
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Model name to run evaluation on",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-run even if results already exist",
    ),
    retry_errors: bool = typer.Option(
        False,
        "--retry-errors",
        help="Retry only rows with call_status=error in an existing responses CSV",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help=(
            "Write/read responses under cache/<language>/runs/<run-id>/. "
            "Alphanumeric and hyphens only. Defaults to 'default'."
        ),
    ),
    timestamped_output: bool = typer.Option(
        False,
        "--timestamped-output",
        help="Generate a UTC timestamp run ID for a non-overwriting run",
    ),
    language: str = typer.Option(
        ...,
        "--language",
        "-l",
        help=f"Language to evaluate: one of {', '.join(DATASETS)}",
    ),
    dataset: str | None = typer.Option(
        None,
        "--dataset",
        help=(
            "EXCEPTIONAL override: score against a different dataset JSON than this "
            "language's normal one (the fixed path in the DATASETS registry above). "
            "FILENAME ONLY, no path -- it's always resolved as DATA_DIR/<dataset>, "
            "i.e. the file must live directly in data/ alongside the normal "
            "per-language datasets (e.g. --dataset "
            "linguistic_knowledge_eval_english_20x.json, not "
            "data/linguistic_knowledge_eval_english_20x.json or an absolute path). "
            "Normal usage never needs this. Exists for special one-off runs, e.g. the "
            "repeated dataset scripts/build_consistency_dataset.py builds for the "
            "'consistency' validation check."
        ),
    ),
) -> None:
    """Run Linguistic Knowledge evaluation on LLM models.

    If --model is not provided, runs all models from list_models.yaml.
    """
    if force and retry_errors:
        typer.echo("ERROR: --force and --retry-errors cannot be combined.", err=True)
        raise typer.Exit(1)

    if language not in DATASETS:
        typer.echo(f"ERROR: Invalid language '{language}'.", err=True)
        typer.echo(f"Available: {', '.join(DATASETS)}", err=True)
        raise typer.Exit(1)

    _validate_run_id(run_id)

    if model:
        if model not in EVAL_MODELS:
            typer.echo(
                f"ERROR: Model '{model}' not found in list_models.yaml.", err=True
            )
            typer.echo(f"Available: {', '.join(EVAL_MODELS.keys())}", err=True)
            raise typer.Exit(1)
        models_to_run = [model]
    else:
        models_to_run = list(EVAL_MODELS.keys())

    entry = DATASETS[language]
    dataset_path = DATA_DIR / dataset if dataset is not None else entry["path"]
    if timestamped_output and run_id is None:
        run_id = _timestamp_run_id()
    typer.echo(f"\n=== Running Linguistic Knowledge Benchmark ({language}) ===")
    if run_id:
        typer.echo(f"Run ID: {run_id}")
    else:
        typer.echo("Run ID: default (no --run-id supplied)")
    if dataset is not None:
        typer.echo(
            f"NOTE: --dataset override in effect -- scoring against {dataset_path} "
            f"instead of this language's normal dataset ({entry['path']})."
        )

    for current_model in models_to_run:
        try:
            run_benchmark(
                model_name=current_model,
                language_name=entry["language_name"],
                dataset_path=dataset_path,
                force=force,
                retry_errors=retry_errors,
                run_id=run_id,
            )
        except Exception as e:
            typer.echo(f"ERROR [{current_model}]: {e}", err=True)
            raise typer.Exit(1)


@app.command("judge-bench")
def judge_bench_cmd(
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Model name whose results to judge",
    ),
    judge_model: list[str] = typer.Option(
        None,
        "--judge-model",
        "-j",
        help=(
            "Judge model to use; repeat for an explicit ensemble "
            "(e.g. -j model-a -j model-b). Overrides --num-judges. Defaults to "
            f"the first --num-judges of DEFAULT_JUDGE_MODELS ({DEFAULT_JUDGE_MODELS})."
        ),
    ),
    num_judges: int = typer.Option(
        1,
        "--num-judges",
        "-n",
        help=(
            "How many of DEFAULT_JUDGE_MODELS to use as independent judges, when "
            "--judge-model isn't given. 1 (default) = single judge, "
            f"DEFAULT_JUDGE_MODELS[0]; up to {len(DEFAULT_JUDGE_MODELS)} for an "
            "ensemble with majority-vote + agreement columns."
        ),
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-judge even if judged results exist",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help=(
            "Read/write judged artifacts under cache/<language>/runs/<run-id>/. "
            "Alphanumeric and hyphens only. Defaults to 'default'."
        ),
    ),
    language: str = typer.Option(
        ...,
        "--language",
        "-l",
        help=f"Language to judge: one of {', '.join(DATASETS)}",
    ),
    judge_fallback: bool = typer.Option(
        False,
        "--judge-fallback",
        help=(
            "Use LLM judge as fallback for MCQ rows where "
            "regex parsing failed (is_correct is null). Reads from "
            "responses_judged/ and writes to responses_judged_fallback/."
        ),
    ),
    dataset: str | None = typer.Option(
        None,
        "--dataset",
        help=(
            "EXCEPTIONAL override: score against a different dataset JSON than this "
            "language's normal one (the fixed path in the DATASETS registry above). "
            "FILENAME ONLY, no path -- always resolved as DATA_DIR/<dataset>, so the "
            "file must live directly in data/ (e.g. --dataset "
            "linguistic_knowledge_eval_english_20x.json). Must match whatever "
            "--dataset (if any) run-bench used to produce this run-id's responses, "
            "since this also drives the detailed CSV's question metadata merge. "
            "Normal usage never needs this; see run-bench --dataset's help for when "
            "it does."
        ),
    ),
) -> None:
    """Judge Linguistic Knowledge evaluation results.

    Scores MCQ responses with regex and short_form with an LLM judge.
    If --model is not provided, judges all models that have step 1 results.
    """
    if language not in DATASETS:
        typer.echo(f"ERROR: Invalid language '{language}'.", err=True)
        typer.echo(f"Available: {', '.join(DATASETS)}", err=True)
        raise typer.Exit(1)

    _validate_run_id(run_id)

    if model:
        if model not in EVAL_MODELS:
            typer.echo(
                f"ERROR: Model '{model}' not found in list_models.yaml.", err=True
            )
            typer.echo(f"Available: {', '.join(EVAL_MODELS.keys())}", err=True)
            raise typer.Exit(1)
        models_to_judge = [model]
    else:
        models_to_judge = list(EVAL_MODELS.keys())

    if judge_model:
        judge_models = list(judge_model)
    else:
        if not 1 <= num_judges <= len(DEFAULT_JUDGE_MODELS):
            typer.echo(
                f"ERROR: --num-judges must be between 1 and "
                f"{len(DEFAULT_JUDGE_MODELS)}.",
                err=True,
            )
            raise typer.Exit(1)
        judge_models = DEFAULT_JUDGE_MODELS[:num_judges]

    entry = DATASETS[language]
    mode = "fallback" if judge_fallback else "standard"
    effective_run_id = run_id or "default"
    # Report the directory this mode actually reads from: standard judging reads
    # the raw run-bench responses, while --judge-fallback re-scores the already-judged
    # CSVs, so its source is responses_judged/ (produced by a prior standard pass).
    input_subdir = "responses_judged" if judge_fallback else "responses"
    input_dir = run_dir(language, effective_run_id) / input_subdir

    typer.echo(f"\n=== Judging Linguistic Knowledge Results ({language}, {mode}) ===")
    typer.echo(f"Judge model(s): {', '.join(judge_models)}")
    if run_id:
        typer.echo(f"Run ID: {effective_run_id}")
    else:
        typer.echo(f"Run ID: {effective_run_id} (no --run-id supplied)")
    typer.echo(f"Judging responses from: {input_dir}")

    # Capture the dataset path now: `entry` is reassigned later when merging log fields.
    dataset_path = DATA_DIR / dataset if dataset is not None else entry["path"]
    if dataset is not None:
        typer.echo(
            f"NOTE: --dataset override in effect -- scoring against {dataset_path} "
            f"instead of this language's normal dataset ({entry['path']})."
        )

    summaries: list[dict[str, object]] = []
    judged_by_model: dict[str, pd.DataFrame] = {}
    skipped_models: list[str] = []

    for current_model in models_to_judge:
        try:
            if judge_fallback:
                judged_df = judge_results_fallback(
                    model_name=current_model,
                    language_name=entry["language_name"],
                    dataset_path=dataset_path,
                    judge_models=judge_models,
                    force=force,
                    run_id=run_id,
                )
            else:
                judged_df = judge_results(
                    model_name=current_model,
                    language_name=entry["language_name"],
                    dataset_path=dataset_path,
                    judge_models=judge_models,
                    force=force,
                    run_id=run_id,
                )

            col_prefix = "is_correct_fallback_" if judge_fallback else "is_correct_"
            missing_judges = [
                jm
                for jm in judge_models
                if f"{col_prefix}{jm}" not in judged_df.columns
            ]
            if missing_judges:
                typer.echo(
                    f"\nERROR: {current_model} was already judged but only "
                    f"has results for {len(judge_models) - len(missing_judges)} "
                    f"of the {len(judge_models)} requested judge(s) "
                    f"(missing: {', '.join(missing_judges)}).\n"
                    "Options:\n"
                    "  - Re-run with --force to rejudge this model with the "
                    "requested judge(s).\n"
                    "  - Re-run with a matching --num-judges (or --judge-model) "
                    "selection if you don't need a different judge set.\n"
                    "  - Use a different --run-id to start a fresh run.",
                    err=True,
                )
                raise typer.Exit(1)

            summary = compute_summary(judged_df, current_model)
            summaries.append(summary)
            judged_by_model[current_model] = judged_df

            typer.echo(f"\n{current_model}:")
            typer.echo(f"  Overall accuracy: {summary['accuracy']}%")
            typer.echo(f"  MCQ accuracy: {summary['accuracy_mcq']}%")
            typer.echo(f"  Short-form accuracy: {summary['accuracy_short_form']}%")
            if len(judge_models) > 1 and summary["judge_agreement_mean"] is not None:
                typer.echo(
                    f"  Judge agreement: {summary['judge_agreement_mean']:.2%} "
                    f"({summary['judge_disagreement_count']} disagreement(s))"
                )

        except typer.Exit:
            raise
        except FileNotFoundError:
            skipped_models.append(current_model)
        except Exception as e:
            typer.echo(f"ERROR [{current_model}]: {e}", err=True)
            raise typer.Exit(1)

    # If run directory missing, show only the "directory not found" diagnostic.
    if not summaries and not input_dir.exists():
        typer.echo(
            f"\nNothing judged: expected directory not found — {input_dir}\n"
            f"No run '{effective_run_id}' exists for {language}. If you meant a "
            "timestamped run, re-run with its --run-id; otherwise run 'run-bench' "
            "first."
        )
        return

    if skipped_models:
        typer.echo(
            f"\nSkipped {len(skipped_models)} model(s) with no results to judge: "
            f"{', '.join(skipped_models)}"
        )

    if not summaries:
        typer.echo(
            f"\nNothing judged: no result files in {input_dir}\n"
            "The directory exists but is empty. Run 'run-bench' for this run-id first."
        )
        return

    results_dir = BENCH_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    log_path = run_dir(language, effective_run_id) / "log.json"
    log_data: dict[str, dict[str, str | None]] = {}
    if log_path.exists():
        with open(log_path, encoding="utf-8") as _f:
            log_data = json.load(_f)
    log_fields = [
        "responses_timestamp-utc",
        "responses_git-hash",
        "responses_retry_timestamp-utc",
        "responses_retry_git-hash",
        "responses_judged_timestamp-utc",
        "responses_judged_git-hash",
        "llm_judges",
        "responses_judged_fallback_timestamp-utc",
        "responses_judged_fallback_git-hash",
        "llm_judges_fallback",
    ]
    for s in summaries:
        entry = log_data.get(str(s.get("model", "")), {})
        for field in log_fields:
            s[field] = entry.get(field)
    summary_df = pd.DataFrame(summaries)
    fallback_suffix = "_fallback" if judge_fallback else ""
    summary_name = (
        f"summary_linguistic-knowledge_{language}"
        f"_{effective_run_id}{fallback_suffix}.csv"
    )
    summary_path = results_dir / summary_name
    summary_df.to_csv(summary_path, index=False)
    typer.echo(f"\nSummary saved to {summary_path}")

    # Detailed breakdown: one row per question, per-model output + correctness.
    detailed_df = build_detailed(judged_by_model, dataset_path)
    detailed_name = (
        f"summary_linguistic-knowledge_{language}"
        f"_{effective_run_id}{fallback_suffix}_detailed.csv"
    )
    detailed_path = results_dir / detailed_name
    detailed_df.to_csv(detailed_path, index=False)
    typer.echo(f"Detailed results saved to {detailed_path}")

    # Pairwise judge agreement across every judged row this run, only meaningful
    # with an ensemble (single-judge agreement is trivially 1.0).
    if len(judge_models) > 1 and judged_by_model:
        all_judged = pd.concat(judged_by_model.values(), ignore_index=True)
        try:
            agreement_matrix = compute_judge_agreement(
                all_judged, judge_models, fallback=judge_fallback
            )
        except KeyError:
            agreement_matrix = None
        if agreement_matrix is not None:
            typer.echo("\nPairwise judge agreement (across all judged rows this run):")
            typer.echo(agreement_matrix.round(3).to_string())
            agreement_name = (
                f"summary_linguistic-knowledge_{language}"
                f"_{effective_run_id}{fallback_suffix}_judge_agreement.csv"
            )
            agreement_path = results_dir / agreement_name
            agreement_matrix.to_csv(agreement_path)
            typer.echo(f"Judge agreement matrix saved to {agreement_path}")


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    app()  # pragma: no cover
