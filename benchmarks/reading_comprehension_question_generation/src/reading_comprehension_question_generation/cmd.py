"""Command-line interface for the reading comprehension question generation (rcqg)."""

from pathlib import Path

import typer
from phonics_benchmark.cmd import (
    EXPECTED_CLI_ERRORS,
    _echo_run_output,
    _execute_with_optional_sync,
    _load_local_run_config,
    _parse_cli_models,
    _timestamp_run_id,
    fetch_results_cmd,
    upload_results_cmd,
)
from phonics_benchmark.config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MODEL_WORKERS,
    apply_execution_overrides,
    build_run_config,
)
from phonics_benchmark.runner import download_default_custom_models

from reading_comprehension_question_generation.items import load_items
from reading_comprehension_question_generation.pairs import (
    PAIRS_FILENAME,
    RESPONSE_STATUS_FILENAME,
    SUMMARY_FILENAME,
    ExtractionError,
    extract_run,
)
from reading_comprehension_question_generation.prompts import (
    DEFAULT_PROMPT_NAME,
    prompt_for_item,
)

# The task name is set by the command, never by the config file (the shared harness
# rejects a "task" key in a config).
QUESTION_GENERATION_TASK = "question-generation"

# Where `scripts/build_benchmark_items.py` writes items. This is the benchmark folder's
# data/ directory; the run's --language is the data folder name (such as 'english'), so
# the default items file for a run is data/<language>/benchmark_items.jsonl.
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
ITEMS_FILENAME = "benchmark_items.jsonl"

app = typer.Typer(
    name="reading-comprehension-question-generation",
    help=(
        "Run the reading comprehension question generation benchmark from a config "
        "file or CLI options."
    ),
)


@app.callback(invoke_without_command=True)
def setup(ctx: typer.Context) -> None:
    """Reading comprehension question generation benchmark."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


def run_generation_cmd(
    ctx: typer.Context,
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        readable=True,
        help=(
            "Path to the JSON run config. Its items_path may be local or a gs:// "
            "URI. If omitted, use CLI options."
        ),
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help=(
            "Run identifier used as the output subdirectory. Defaults to 'default'. "
            "Alphanumeric and hyphens only."
        ),
    ),
    timestamped_output: bool = typer.Option(
        False,
        "--timestamped-output",
        help="Generate a UTC timestamp run ID for a non-overwriting run.",
    ),
    language: str | None = typer.Option(
        None,
        "--language",
        "-l",
        help=(
            "Data folder name for the run, for example 'english'. It is recorded in "
            "each result row and picks the default items file; each prompt's language "
            "comes from its item."
        ),
    ),
    models: str | None = typer.Option(
        None,
        "--models",
        help="Model list as comma-separated names or a JSON array string.",
    ),
    model: list[str] | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Model name to evaluate. Repeatable compatibility option.",
    ),
    output_dir: Path = typer.Option(
        Path("results"),
        "--output-dir",
        help="Directory where run artifacts should be written.",
    ),
    items_path: Path | None = typer.Option(
        None,
        "--items-path",
        exists=True,
        dir_okay=False,
        readable=True,
        help=(
            "JSONL of benchmark items. Defaults to "
            "data/<language>/benchmark_items.jsonl in the benchmark folder."
        ),
    ),
    custom_models_path: Path | None = typer.Option(
        None,
        "--custom-models-path",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the FDLLM custom models YAML file.",
    ),
    skip_models_download: bool = typer.Option(
        False,
        "--skip-models-download",
        help="Do not refresh the shared repo-root custom_models.yaml before the run.",
    ),
    env_path: Path | None = typer.Option(
        None,
        "--env-path",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to a .env file with provider API keys.",
    ),
    prompt_name: str | None = typer.Option(
        None,
        "--prompt-name",
        help="Prompt template to render each item with (a key in prompts.yaml).",
    ),
    max_tokens: int | None = typer.Option(
        None,
        "--max-tokens",
        min=1,
        help="Maximum response tokens for each model call. Defaults to 10,000.",
    ),
    chunk_size: int = typer.Option(
        DEFAULT_CHUNK_SIZE,
        "--chunk-size",
        min=1,
        help="Number of pending items to evaluate before checkpointing results.",
    ),
    model_workers: int | None = typer.Option(
        None,
        "--model-workers",
        min=1,
        help=(
            "Maximum number of provider groups to run concurrently. "
            f"Defaults to the config value, or {DEFAULT_MODEL_WORKERS} without "
            "a config."
        ),
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Ignore existing results and rerun all model/item pairs.",
    ),
    retry_errors: bool = typer.Option(
        False,
        "--retry-errors",
        help="Retry only existing result rows with call_status=error.",
    ),
    results_dest: str | None = typer.Option(
        None,
        "--results-dest",
        help="GCS run prefix to mirror, for example gs://bucket/rcqg/runs/id/.",
    ),
) -> None:
    """Run the benchmark over the benchmark items with the chosen models."""

    try:
        if config is not None:
            run_config = _load_local_run_config(
                config,
                task=QUESTION_GENERATION_TASK,
                default_prompt_name=DEFAULT_PROMPT_NAME,
            )
            run_config = apply_execution_overrides(
                run_config,
                force=force,
                retry_errors=retry_errors,
                model_workers=model_workers,
                results_dest=results_dest,
            )
        else:
            effective_run_id = (
                _timestamp_run_id() if timestamped_output and run_id is None else run_id
            )
            model_names = _parse_cli_models(models, model)
            if items_path is None and language:
                # Resolved here because the shared config requires an items path.
                items_path = DEFAULT_DATA_ROOT / language / ITEMS_FILENAME
            run_config = build_run_config(
                run_id=effective_run_id,
                language=language,
                output_dir=output_dir,
                models=model_names,
                base_dir=Path.cwd(),
                task=QUESTION_GENERATION_TASK,
                items_path=items_path,
                prompt_name=prompt_name or DEFAULT_PROMPT_NAME,
                env_path=env_path,
                custom_models_path=custom_models_path,
                max_tokens=max_tokens,
                chunk_size=chunk_size,
                model_workers=(
                    model_workers
                    if model_workers is not None
                    else DEFAULT_MODEL_WORKERS
                ),
                force=force,
                retry_errors=retry_errors,
                results_dest=results_dest,
            )

        if not skip_models_download:
            download_default_custom_models()
        output = _execute_with_optional_sync(
            run_config,
            item_loader=load_items,
            prompt_builder=prompt_for_item,
        )
    except EXPECTED_CLI_ERRORS as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(1) from e

    _echo_run_output(output)
    if output["error_count"] > 0:
        raise typer.Exit(1)


def extract_pairs_cmd(
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Run identifier: the run folder under --output-dir.",
    ),
    output_dir: Path = typer.Option(
        Path("results"),
        "--output-dir",
        help="Directory that holds the run folder (used with --run-id).",
    ),
    run_dir: Path | None = typer.Option(
        None,
        "--run-dir",
        exists=True,
        file_okay=False,
        help="Path to the run folder itself, instead of --run-id and --output-dir.",
    ),
    items_path: Path | None = typer.Option(
        None,
        "--items-path",
        exists=True,
        dir_okay=False,
        readable=True,
        help=(
            "Items file for the hash check. Defaults to the path recorded in the "
            "run's config.json. Only the check uses it."
        ),
    ),
) -> None:
    """Extract Q&A pairs from a run's raw responses and report parse problems."""
    if run_dir is None:
        if run_id is None:
            typer.echo(
                "ERROR: give --run-dir, or --run-id with --output-dir.", err=True
            )
            raise typer.Exit(1)
        run_dir = output_dir / run_id

    try:
        summary = extract_run(run_dir, items_path)
    except (ExtractionError, OSError) as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(1) from e

    typer.echo(
        f"Run: {summary['run_id']} "
        f"(requested pairs per response: {summary['requested_pairs']})"
    )
    for model, counts in summary["models"].items():
        statuses = ", ".join(
            f"{k}={v}" for k, v in sorted(counts["status_counts"].items())
        )
        typer.echo(
            f"  {model}: {counts['responses']} response(s), "
            f"{counts['total_pairs']} pair(s); {statuses}"
        )
    match = summary["items_file"]["match"]
    if match is False:
        typer.echo(
            "WARNING: the items file has changed since the run "
            "(its SHA-256 differs from the one in config.json)."
        )
    typer.echo(
        f"Wrote {PAIRS_FILENAME}, {RESPONSE_STATUS_FILENAME} and {SUMMARY_FILENAME} "
        f"in {run_dir}"
    )


app.command("run-generation")(run_generation_cmd)
app.command("extract-pairs")(extract_pairs_cmd)

# The GCS sync commands are task-agnostic; reuse the phonics ones unchanged.
app.command("upload-results")(upload_results_cmd)
app.command("fetch-results")(fetch_results_cmd)


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()  # pragma: no cover
