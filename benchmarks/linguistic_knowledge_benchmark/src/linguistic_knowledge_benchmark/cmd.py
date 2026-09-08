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
    MAX_CALL_ERROR_RATE,
    TooManyCallErrorsError,
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
    experiment_dir,
)

# Add a new language by adding an entry here and dropping its JSON in data/.
# The medium of evaluation is English for every language: questions are written
# in English (only their subject changes from language to language).
#
# This is the NORMAL way run-bench/judge-bench pick a dataset: each language points
# directly at one fixed file here, no other configuration needed for everyday use.
# `--dataset` on both commands (below) is an EXCEPTIONAL escape hatch that overrides
# this for a single experiment -- e.g. scoring against the repeated dataset built by
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

# experiment-id must be alphanumeric + hyphens only (no underscores) so that underscores
# can be used as field separators in output filenames.
_EXPERIMENT_ID_RE = re.compile(r"^[a-zA-Z0-9-]+$")


def _timestamp_experiment_id() -> str:
    """Return a filesystem-safe UTC timestamp for non-overwriting experiments."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _format_accuracy(value: float | None) -> str:
    """Accuracy for console display; 'n/a' when there was no scored row to average."""
    return "n/a" if value is None else f"{value}%"


def _validate_experiment_id(experiment_id: str | None) -> None:
    """Abort with a clear message if experiment_id contains invalid characters."""
    if experiment_id and not _EXPERIMENT_ID_RE.match(experiment_id):
        typer.echo(
            f"ERROR: --experiment-id '{experiment_id}' is invalid. "
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
    # committed), so it must be fetched here on every invocation -- without this, a
    # fresh checkout (e.g. CI) has no file for register_models() to read and crashes.
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
    experiment_id: str | None = typer.Option(
        None,
        "--experiment-id",
        help=(
            "Write/read responses under cache/<language>/experiments/<experiment-id>/. "
            "Alphanumeric and hyphens only. Defaults to 'default'."
        ),
    ),
    timestamped_output: bool = typer.Option(
        False,
        "--timestamped-output",
        help="Generate a UTC timestamp experiment ID for a non-overwriting experiment",
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
            "Normal usage never needs this. Exists for special one-off experiments, "
            "e.g. the repeated dataset scripts/build_consistency_dataset.py builds for "
            "the "
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

    _validate_experiment_id(experiment_id)

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
    if timestamped_output and experiment_id is None:
        experiment_id = _timestamp_experiment_id()
    typer.echo(f"\n=== Running Linguistic Knowledge Benchmark ({language}) ===")
    if experiment_id:
        typer.echo(f"Experiment ID: {experiment_id}")
    else:
        typer.echo("Experiment ID: default (no --experiment-id supplied)")
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
                experiment_id=experiment_id,
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
    experiment_id: str | None = typer.Option(
        None,
        "--experiment-id",
        help=(
            "Read/write judged artifacts under "
            "cache/<language>/experiments/<experiment-id>/. "
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
            "--dataset (if any) run-bench used to produce this experiment-id's "
            "responses, "
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

    _validate_experiment_id(experiment_id)

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
    effective_experiment_id = experiment_id or "default"
    # Report the directory this mode actually reads from: standard judging reads
    # the raw run-bench responses, while --judge-fallback re-scores the already-judged
    # CSVs, so its source is responses_judged/ (produced by a prior standard pass).
    input_subdir = "responses_judged" if judge_fallback else "responses"
    input_dir = experiment_dir(language, effective_experiment_id) / input_subdir

    typer.echo(f"\n=== Judging Linguistic Knowledge Results ({language}, {mode}) ===")
    typer.echo(f"Judge model(s): {', '.join(judge_models)}")
    if experiment_id:
        typer.echo(f"Experiment ID: {effective_experiment_id}")
    else:
        typer.echo(
            f"Experiment ID: {effective_experiment_id} (no --experiment-id supplied)"
        )
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
    error_rate_skipped: list[str] = []

    for current_model in models_to_judge:
        try:
            if judge_fallback:
                judged_df = judge_results_fallback(
                    model_name=current_model,
                    language_name=entry["language_name"],
                    dataset_path=dataset_path,
                    judge_models=judge_models,
                    force=force,
                    experiment_id=experiment_id,
                )
            else:
                judged_df = judge_results(
                    model_name=current_model,
                    language_name=entry["language_name"],
                    dataset_path=dataset_path,
                    judge_models=judge_models,
                    force=force,
                    experiment_id=experiment_id,
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
                    "  - Use a different --experiment-id to start a fresh experiment.",
                    err=True,
                )
                raise typer.Exit(1)

            summary = compute_summary(judged_df, current_model)
            summaries.append(summary)
            judged_by_model[current_model] = judged_df

            typer.echo(f"\n{current_model}:")
            typer.echo(f"  Overall accuracy: {_format_accuracy(summary['accuracy'])}")
            typer.echo(f"  MCQ accuracy: {_format_accuracy(summary['accuracy_mcq'])}")
            typer.echo(
                "  Short-form accuracy: "
                f"{_format_accuracy(summary['accuracy_short_form'])}"
            )
            # One line per judging pass. .get() because compute_summary() omits the
            # mcq fallback keys entirely for a standard pass, and gated on the mean
            # rather than len(judge_models): a --judge-fallback invocation carries the
            # short_form pass's columns through, so its own judge count says nothing
            # about whether those were an ensemble.
            for label, mean_key, count_key in (
                (
                    "Short-form judge agreement",
                    "short_form_judge_agreement_mean",
                    "short_form_judge_disagreement_count",
                ),
                (
                    "MCQ fallback judge agreement",
                    "mcq_judge_fallback_agreement_mean",
                    "mcq_judge_fallback_disagreement_count",
                ),
            ):
                mean = summary.get(mean_key)
                if mean is not None:
                    typer.echo(
                        f"  {label}: {mean:.2%} "
                        f"({summary.get(count_key, 0)} disagreement(s))"
                    )

        except typer.Exit:
            raise
        except TooManyCallErrorsError as e:
            # Loud, per-model, as it happens: this is a model whose results are
            # incomplete enough to be misleading, so it is judged nowhere and appears
            # in neither the summary nor the detailed CSV.
            typer.echo(f"\n!! SKIPPING {current_model}: {e}", err=True)
            error_rate_skipped.append(current_model)
        except FileNotFoundError:
            skipped_models.append(current_model)
        except Exception as e:
            typer.echo(f"ERROR [{current_model}]: {e}", err=True)
            raise typer.Exit(1)

    # If experiment directory missing, show only the "directory not found" diagnostic.
    if not summaries and not input_dir.exists():
        typer.echo(
            f"\nNothing judged: expected directory not found — {input_dir}\n"
            f"No experiment '{effective_experiment_id}' exists for {language}. If you "
            "meant a timestamped experiment, re-run with its --experiment-id; "
            "otherwise run 'run-bench' first."
        )
        return

    if skipped_models:
        typer.echo(
            f"\nSkipped {len(skipped_models)} model(s) with no results to judge: "
            f"{', '.join(skipped_models)}"
        )

    # Repeated at the end because the per-model warning above scrolls away behind the
    # judging progress bars of every model that followed it.
    if error_rate_skipped:
        typer.echo(
            f"\n!! NOT JUDGED -- more than {MAX_CALL_ERROR_RATE:.0%} of rows failed "
            f"to call ({len(error_rate_skipped)} model(s)): "
            f"{', '.join(error_rate_skipped)}\n"
            "   These are absent from the summary and detailed CSVs. Re-run them "
            "with 'run-bench --retry-errors', then judge again."
        )

    if not summaries:
        if error_rate_skipped:
            # Don't claim the directory is empty when it isn't: every model that had
            # results was skipped by the error-rate gate above.
            typer.echo(
                "\nNothing judged: every model with results exceeded the "
                f"{MAX_CALL_ERROR_RATE:.0%} call-error limit. No summary written."
            )
            return
        typer.echo(
            f"\nNothing judged: no result files in {input_dir}\n"
            "The directory exists but is empty. Run 'run-bench' for this "
            "experiment-id first."
        )
        return

    results_dir = BENCH_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    log_path = experiment_dir(language, effective_experiment_id) / "log.json"
    log_data: dict[str, dict[str, str | None]] = {}
    if log_path.exists():
        with open(log_path, encoding="utf-8") as _f:
            log_data = json.load(_f)
    log_fields = [
        "responses_timestamp_utc",
        "responses_git_hash",
        "responses_retry_timestamp_utc",
        "responses_retry_git_hash",
        "responses_judged_timestamp_utc",
        "responses_judged_git_hash",
        "short_form_llm_judges",
    ]
    # log.json accumulates across passes, so a standard run's entry can already hold
    # the fallback pass's fields from an earlier --judge-fallback run. Only a fallback
    # summary reports them, so a standard summary has no fallback column at all.
    if judge_fallback:
        log_fields += [
            "responses_judged_fallback_timestamp_utc",
            "responses_judged_fallback_git_hash",
            "mcq_llm_judges_fallback",
        ]
    for s in summaries:
        entry = log_data.get(str(s.get("model", "")), {})
        for field in log_fields:
            s[field] = entry.get(field)
    summary_df = pd.DataFrame(summaries)
    fallback_suffix = "_fallback" if judge_fallback else ""
    summary_name = (
        f"summary_linguistic-knowledge_{language}"
        f"_{effective_experiment_id}{fallback_suffix}.csv"
    )
    summary_path = results_dir / summary_name
    summary_df.to_csv(summary_path, index=False)
    typer.echo(f"\nSummary saved to {summary_path}")

    # Detailed breakdown: one row per question, per-model output + correctness.
    detailed_df = build_detailed(judged_by_model, dataset_path)
    detailed_name = (
        f"summary_linguistic-knowledge_{language}"
        f"_{effective_experiment_id}{fallback_suffix}_detailed.csv"
    )
    detailed_path = results_dir / detailed_name
    detailed_df.to_csv(detailed_path, index=False)
    typer.echo(f"Detailed results saved to {detailed_path}")

    # Pairwise judge agreement across every judged row this experiment, only meaningful
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
            typer.echo(
                "\nPairwise judge agreement (across all judged rows this experiment):"
            )
            typer.echo(agreement_matrix.round(3).to_string())
            agreement_name = (
                f"summary_linguistic-knowledge_{language}"
                f"_{effective_experiment_id}{fallback_suffix}_judge_agreement.csv"
            )
            agreement_path = results_dir / agreement_name
            agreement_matrix.to_csv(agreement_path)
            typer.echo(f"Judge agreement matrix saved to {agreement_path}")


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    app()  # pragma: no cover
