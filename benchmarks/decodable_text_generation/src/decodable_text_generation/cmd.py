"""Command-line interface for the decodable text generation benchmarks."""

from functools import partial
from pathlib import Path

import typer
from phonics_benchmark.cmd import (
    DEFAULT_CLOUD_WORKDIR,
    EXPECTED_CLI_ERRORS,
    _echo_run_output,
    _execute_with_optional_sync,
    _load_cloud_run_config,
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

from decodable_text_generation.items import load_generation_items
from decodable_text_generation.prompts import prompt_from_item

GENERATION_CHILD_TASK = "generation-child"
GENERATION_CURRICULUM_TASK = "generation-curriculum"

# The same callbacks are registered under one name per benchmark; the invoked
# command name selects the task recorded on every result row.
_TASK_BY_COMMAND = {
    "run-generation-child": GENERATION_CHILD_TASK,
    "run-generation-curriculum": GENERATION_CURRICULUM_TASK,
    "run-generation-child-cloud": GENERATION_CHILD_TASK,
    "run-generation-curriculum-cloud": GENERATION_CURRICULUM_TASK,
}

# Curriculum items carry a `target` object and are prompted with the
# {target}-bearing template; child items use the plain generation template.
_IS_CURRICULUM = {
    GENERATION_CHILD_TASK: False,
    GENERATION_CURRICULUM_TASK: True,
}
DEFAULT_PROMPT_NAMES = {
    GENERATION_CHILD_TASK: "decodable_text_generation",
    GENERATION_CURRICULUM_TASK: "decodable_text_generation_curriculum",
}

app = typer.Typer(
    name="decodable-text-generation",
    help=(
        "Run decodable text generation benchmarks (child and curriculum) "
        "from a config file or CLI options."
    ),
)


@app.callback(invoke_without_command=True)
def setup(ctx: typer.Context) -> None:
    """Decodable text generation benchmarks."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


def _task_for(ctx: typer.Context) -> str:
    return _TASK_BY_COMMAND[ctx.info_name or ""]


def _item_loader_for(task: str):
    return partial(load_generation_items, curriculum=_IS_CURRICULUM[task])


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
        help="Language registry key for the benchmark run, for example 'sw'.",
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
        help="JSONL containing completed benchmark items.",
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
        help="Prompt template to evaluate each item with.",
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
        help="GCS run prefix to mirror, for example gs://bucket/dtg/runs/id/.",
    ),
) -> None:
    """Run a decodable text generation benchmark from a config file or CLI options."""

    task = _task_for(ctx)
    item_loader = _item_loader_for(task)
    try:
        if config is not None:
            run_config = _load_local_run_config(
                config,
                task=task,
                default_prompt_name=DEFAULT_PROMPT_NAMES[task],
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
            run_config = build_run_config(
                run_id=effective_run_id,
                language=language,
                output_dir=output_dir,
                models=model_names,
                base_dir=Path.cwd(),
                task=task,
                items_path=items_path,
                prompt_name=prompt_name or DEFAULT_PROMPT_NAMES[task],
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
            item_loader=item_loader,
            prompt_builder=prompt_from_item,
        )
    except EXPECTED_CLI_ERRORS as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(1) from e

    _echo_run_output(output)
    if output["error_count"] > 0:
        raise typer.Exit(1)


def run_generation_cloud_cmd(
    ctx: typer.Context,
    config: str = typer.Option(
        ...,
        "--config",
        "-c",
        help="GCS URI of the run config, for example gs://bucket/runs/id/config.json.",
    ),
    workdir: Path = typer.Option(
        DEFAULT_CLOUD_WORKDIR,
        "--workdir",
        file_okay=False,
        help="Writable workspace used for downloaded inputs and run outputs.",
    ),
    skip_models_download: bool = typer.Option(
        False,
        "--skip-models-download",
        help="Do not refresh the shared custom_models.yaml before the run.",
    ),
) -> None:
    """Fetch a GCS config and items, then execute a continuously synced run."""

    task = _task_for(ctx)
    try:
        run_config = _load_cloud_run_config(
            config,
            workdir=workdir,
            task=task,
            default_prompt_name=DEFAULT_PROMPT_NAMES[task],
        )
        if not skip_models_download:
            download_default_custom_models(run_config.custom_models_path)
        output = _execute_with_optional_sync(
            run_config,
            item_loader=_item_loader_for(task),
            prompt_builder=prompt_from_item,
        )
    except EXPECTED_CLI_ERRORS as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(1) from e

    _echo_run_output(output)
    if output["error_count"] > 0:
        raise typer.Exit(1)


for _command_name in (
    "run-generation-child",
    "run-generation-curriculum",
):
    app.command(_command_name)(run_generation_cmd)

for _command_name in (
    "run-generation-child-cloud",
    "run-generation-curriculum-cloud",
):
    app.command(_command_name)(run_generation_cloud_cmd)

# The GCS sync commands are task-agnostic; reuse the phonics ones unchanged.
app.command("upload-results")(upload_results_cmd)
app.command("fetch-results")(fetch_results_cmd)


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()  # pragma: no cover
