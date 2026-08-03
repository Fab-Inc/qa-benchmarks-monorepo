"""Command-line interface for the phonics benchmark."""

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import typer

from phonics_benchmark.config import (
    DECODABILITY_TASK,
    ConfigError,
    apply_execution_overrides,
    build_run_config,
    load_run_config,
)
from phonics_benchmark.runner import (
    RunOutput,
    download_default_custom_models,
    execute_run,
)

app = typer.Typer(
    name="phonics-benchmark",
    help="Run phonics benchmark experiment from a config file or CLI options.",
)


@app.callback(invoke_without_command=True)
def setup(ctx: typer.Context) -> None:
    """Phonics benchmark."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


@app.command("run-decodability")
def run_decodability_cmd(
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the JSON run config. If omitted, use CLI options.",
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
        help="Language label for the benchmark run and prompt.",
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
        help="CSV containing already-composed benchmark items.",
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
    max_tokens: int | None = typer.Option(
        None,
        "--max-tokens",
        min=1,
        help="Maximum response tokens for each model call. Defaults to 10,000.",
    ),
    chunk_size: int = typer.Option(
        100,
        "--chunk-size",
        min=1,
        help="Number of pending items to evaluate before checkpointing results.",
    ),
    model_workers: int | None = typer.Option(
        None,
        "--model-workers",
        min=1,
        help=(
            "Maximum number of models to run concurrently. "
            "Defaults to the config value, or 1 without a config."
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
) -> None:
    """Run the decodability benchmark from a config file or CLI options."""

    try:
        if config is not None:
            run_config = load_run_config(
                config,
                task=DECODABILITY_TASK,
            )
            run_config = apply_execution_overrides(
                run_config,
                force=force,
                retry_errors=retry_errors,
                model_workers=model_workers,
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
                task=DECODABILITY_TASK,
                items_path=items_path,
                env_path=env_path,
                custom_models_path=custom_models_path,
                max_tokens=max_tokens,
                chunk_size=chunk_size,
                model_workers=model_workers if model_workers is not None else 1,
                force=force,
                retry_errors=retry_errors,
            )

        if not skip_models_download:
            download_default_custom_models()
        output = execute_run(run_config)
    except (ConfigError, FileNotFoundError) as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(1) from e

    _echo_run_output(output)
    if output["error_count"] > 0:
        raise typer.Exit(1)


def _timestamp_run_id() -> str:
    """Return a filesystem-safe UTC timestamp for non-overwriting runs."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{uuid4().hex[:8]}"


def _parse_cli_models(
    models: str | None,
    repeated_models: Sequence[str] | None,
) -> list[Any]:
    """Translate supported CLI model syntax without applying config semantics."""
    parsed: list[Any] = []
    if models is not None:
        parsed.extend(_parse_models_value(models))
    if repeated_models:
        parsed.extend(repeated_models)
    return parsed


def _parse_models_value(value: str) -> list[Any]:
    stripped = value.strip()
    if not stripped:
        return []
    if stripped.startswith("["):
        try:
            raw_models = json.loads(stripped)
        except json.JSONDecodeError as e:
            raise ConfigError(f"Invalid JSON for --models: {e}") from e
        if not isinstance(raw_models, list):
            raise ConfigError("--models JSON value must be an array")
        return raw_models
    return stripped.split(",")


def _echo_run_output(output: RunOutput) -> None:
    typer.echo(f"Run ID: {output['run_id']}")
    typer.echo(f"Config ID: {output['config_id']}")
    typer.echo(f"Results: {output['results_dir']}")
    typer.echo(f"Executed: {output['executed_count']}")
    typer.echo(f"Skipped: {output['skipped_count']}")
    typer.echo(f"Errors: {output['error_count']}")


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()  # pragma: no cover
