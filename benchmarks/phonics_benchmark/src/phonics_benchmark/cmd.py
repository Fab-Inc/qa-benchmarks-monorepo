"""Command-line interface for the phonics benchmark."""

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import typer
from fab_benchmark_publisher import (
    RunSync,
    RunSyncConflictError,
    download_file,
    parse_gs_uri,
)
from google.api_core.exceptions import GoogleAPIError
from google.auth.exceptions import GoogleAuthError

from phonics_benchmark.config import (
    DECODABILITY_TASK,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MODEL_WORKERS,
    BenchmarkRunConfig,
    ConfigError,
    apply_execution_overrides,
    build_run_config,
    config_identity_sha256,
    load_run_config,
)
from phonics_benchmark.items import load_benchmark_items
from phonics_benchmark.runner import (
    CUSTOM_MODELS_FILE_NAME,
    RunOutput,
    download_default_custom_models,
    execute_run,
    model_results_filename,
)

DEFAULT_CLOUD_WORKDIR = Path("/workspace")
EXPECTED_CLI_ERRORS = (
    ConfigError,
    FileNotFoundError,
    GoogleAPIError,
    GoogleAuthError,
    RunSyncConflictError,
    ValueError,
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
        help="GCS run prefix to mirror, for example gs://bucket/phonics/runs/id/.",
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
                task=DECODABILITY_TASK,
                items_path=items_path,
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
        output = _execute_with_optional_sync(run_config)
    except EXPECTED_CLI_ERRORS as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(1) from e

    _echo_run_output(output)
    if output["error_count"] > 0:
        raise typer.Exit(1)


@app.command("upload-results")
def upload_results_cmd(
    run_dir: Path = typer.Option(
        ...,
        "--run-dir",
        exists=True,
        file_okay=False,
        readable=True,
        help="Local run directory to upload.",
    ),
    dest: str = typer.Option(
        ...,
        "--dest",
        help="Destination GCS run prefix.",
    ),
) -> None:
    """Backfill local run artifacts into an empty GCS prefix."""

    try:
        pushed_count = RunSync(run_dir, dest).push_missing_or_stale()
    except EXPECTED_CLI_ERRORS as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"Uploaded: {pushed_count}")


@app.command("fetch-results")
def fetch_results_cmd(
    src: str = typer.Option(..., "--src", help="Source GCS run prefix."),
    run_dir: Path = typer.Option(
        ...,
        "--run-dir",
        file_okay=False,
        help="Local directory where run artifacts should be downloaded.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Download only this model's response file.",
    ),
) -> None:
    """Fetch all run artifacts, or one model response file, from GCS."""

    relative_paths = None
    if model is not None:
        relative_paths = {f"responses/{model_results_filename(model)}"}
    try:
        downloaded_count = RunSync(run_dir, src).pull(relative_paths=relative_paths)
    except EXPECTED_CLI_ERRORS as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"Downloaded: {downloaded_count}")


@app.command("run-cloud")
def run_cloud_cmd(
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

    try:
        run_config = _load_cloud_run_config(config, workdir=workdir)
        if not skip_models_download:
            download_default_custom_models(run_config.custom_models_path)
        output = _execute_with_optional_sync(run_config)
    except EXPECTED_CLI_ERRORS as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(1) from e

    _echo_run_output(output)
    if output["error_count"] > 0:
        raise typer.Exit(1)


def _execute_with_optional_sync(run_config: BenchmarkRunConfig) -> RunOutput:
    """Execute locally, or wrap execution in the generation-safe sync lifecycle."""

    if run_config.results_dest is None:
        return execute_run(run_config)

    run_dir = run_config.output_dir / run_config.run_id
    sync = RunSync(run_dir, run_config.results_dest)
    sync.pull()
    _validate_resume_snapshot(run_config, run_dir / "config.json")

    def push_with_warning(local_path: Path) -> None:
        try:
            sync.push_file(local_path)
        except RunSyncConflictError:
            raise
        except Exception as e:
            typer.echo(
                f"WARNING: could not upload {local_path}: {e}",
                err=True,
            )

    try:
        output = execute_run(
            run_config,
            on_artifact_write=push_with_warning,
        )
    except BaseException:
        try:
            _finalize_sync(sync, run_dir)
        except Exception as cleanup_error:
            typer.echo(
                "WARNING: sync cleanup failed while preserving the original "
                f"error: {cleanup_error}",
                err=True,
            )
        raise
    _finalize_sync(sync, run_dir)
    return output


def _finalize_sync(sync: RunSync, run_dir: Path) -> None:
    """Push the final log and reconcile all artifacts after execution."""

    log_path = run_dir / "log.json"
    if log_path.exists():
        try:
            sync.push_file(log_path)
        except RunSyncConflictError:
            raise
        except Exception as e:
            typer.echo(
                f"WARNING: could not upload final log {log_path}: {e}",
                err=True,
            )
    sync.push_missing_or_stale()


def _validate_resume_snapshot(
    run_config: BenchmarkRunConfig,
    snapshot_path: Path,
) -> None:
    """Reject a reused remote run prefix whose semantic inputs have changed."""

    if not snapshot_path.exists():
        return
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ConfigError(
            f"Could not read pulled config snapshot {snapshot_path}: {e}"
        ) from e
    if not isinstance(snapshot, dict):
        raise ConfigError(f"Pulled config snapshot {snapshot_path} must be an object")

    # Before the first run, run-cloud's source config occupies this object. The first
    # config-snapshot checkpoint replaces it and all later resumes take this branch.
    if "config_id" not in snapshot and "resolved_config" not in snapshot:
        return
    remote_config_id = snapshot.get("config_id")
    expected_config_id = config_identity_sha256(run_config)
    if remote_config_id != expected_config_id:
        raise ConfigError(
            "Remote run config_id does not match this run config: "
            f"{remote_config_id!r} != {expected_config_id!r}"
        )
    resolved_config = snapshot.get("resolved_config")
    if not isinstance(resolved_config, dict):
        raise ConfigError("Remote run config snapshot lacks resolved_config")
    remote_items_sha256 = resolved_config.get("items_sha256")
    _items, local_items_sha256 = load_benchmark_items(run_config.items_path)
    if remote_items_sha256 != local_items_sha256:
        raise ConfigError(
            "Remote run items_sha256 does not match the local items artifact: "
            f"{remote_items_sha256!r} != {local_items_sha256!r}"
        )


def _load_cloud_run_config(
    config_uri: str,
    *,
    workdir: Path,
) -> BenchmarkRunConfig:
    """Download and pre-process a cloud config into the normal local contract."""

    try:
        bucket_name, object_name = parse_gs_uri(config_uri)
    except ValueError as e:
        raise ConfigError(f"Invalid cloud config URI: {e}") from e
    if "/" not in object_name:
        raise ConfigError("Cloud config URI must be inside a run prefix")

    workspace = workdir.expanduser().resolve()
    inputs_dir = workspace / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    downloaded_config_path = inputs_dir / "run-config-source.json"
    download_file(
        bucket_name=bucket_name,
        file_name=object_name,
        write_path=downloaded_config_path,
    )
    try:
        payload = json.loads(downloaded_config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ConfigError(f"Invalid downloaded cloud config: {e}") from e
    if not isinstance(payload, dict):
        raise ConfigError("Downloaded cloud config must be a JSON object")

    is_snapshot = "resolved_config" in payload and "raw_config" in payload
    if is_snapshot:
        raw_config = payload.get("raw_config")
        if not isinstance(raw_config, dict):
            raise ConfigError("Remote run snapshot raw_config must be an object")
        raw_config = dict(raw_config)
    else:
        raw_config = dict(payload)

    metadata_value = raw_config.get("metadata", {})
    if not isinstance(metadata_value, dict):
        raise ConfigError("'metadata' must be an object when provided")
    metadata = dict(metadata_value)
    configured_items_path = raw_config.get("items_path")
    items_source_uri = (
        configured_items_path
        if isinstance(configured_items_path, str)
        and configured_items_path.startswith("gs://")
        else metadata.get("items_source_uri")
    )
    if isinstance(items_source_uri, str) and items_source_uri.startswith("gs://"):
        try:
            items_bucket, items_object = parse_gs_uri(items_source_uri)
        except ValueError as e:
            raise ConfigError(f"Invalid cloud items_path: {e}") from e
        local_items_path = inputs_dir / "benchmark-items.jsonl"
        download_file(
            bucket_name=items_bucket,
            file_name=items_object,
            write_path=local_items_path,
        )
        raw_config["items_path"] = str(local_items_path)
        metadata["items_source_uri"] = items_source_uri
        raw_config["metadata"] = metadata

    run_prefix = object_name.rsplit("/", 1)[0] + "/"
    raw_config.setdefault("results_dest", f"gs://{bucket_name}/{run_prefix}")
    if is_snapshot:
        raw_config["output_dir"] = str(workspace / "outputs")
    else:
        raw_config.setdefault("output_dir", str(workspace / "outputs"))
    raw_config["custom_models_path"] = str(inputs_dir / CUSTOM_MODELS_FILE_NAME)
    processed_config_path = inputs_dir / "run-config.json"
    processed_config_path.write_text(
        json.dumps(raw_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return load_run_config(processed_config_path, task=DECODABILITY_TASK)


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
