"""Command-line interface for the phonics benchmark."""

import json
import os
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from uuid import uuid4

import typer
from fab_benchmark_publisher import (
    RunSync,
    RunSyncConflictError,
    download_file,
    download_file_with_generation,
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
    _parse_run_id,
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
            run_config = _load_local_run_config(config)
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
    typer.echo(f"Uploaded: {pushed_count} to {dest}")


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
    typer.echo(f"Downloaded: {downloaded_count} from {src}")


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

    run_dir = run_config.output_dir / run_config.run_id
    if run_config.results_dest is None:
        _validate_resume_snapshot(run_config, run_dir / "config.json")
        return execute_run(run_config)

    sync = RunSync(run_dir, run_config.results_dest)
    typer.echo(f"Reading existing results from {run_config.results_dest}")
    sync.pull()
    _validate_resume_snapshot(run_config, run_dir / "config.json")
    typer.echo(f"Writing results to {run_config.results_dest}")

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
    typer.echo(f"Wrote results to {run_config.results_dest}")
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
    """Reject a reused run whose semantic inputs have changed."""

    if not snapshot_path.exists():
        return
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ConfigError(
            f"Could not read existing config snapshot {snapshot_path}: {e}"
        ) from e
    if not isinstance(snapshot, dict):
        raise ConfigError(f"Existing config snapshot {snapshot_path} must be an object")

    # Before the first run, run-cloud's source config occupies this object. The first
    # config-snapshot checkpoint replaces it and all later resumes take this branch.
    if "config_id" not in snapshot and "resolved_config" not in snapshot:
        return
    remote_config_id = snapshot.get("config_id")
    expected_config_id = config_identity_sha256(run_config)
    if remote_config_id != expected_config_id:
        raise ConfigError(
            "Existing run config_id does not match this run config: "
            f"{remote_config_id!r} != {expected_config_id!r}"
        )
    resolved_config = snapshot.get("resolved_config")
    if not isinstance(resolved_config, dict):
        raise ConfigError("Existing run config snapshot lacks resolved_config")
    remote_items_sha256 = resolved_config.get("items_sha256")
    _items, local_items_sha256 = load_benchmark_items(run_config.items_path)
    if remote_items_sha256 != local_items_sha256:
        raise ConfigError(
            "Existing run items_sha256 does not match the local items artifact: "
            f"{remote_items_sha256!r} != {local_items_sha256!r}"
        )


def _load_local_run_config(config_path: Path) -> BenchmarkRunConfig:
    """Load a local config, materializing a GCS items source when configured."""

    source_config_path = config_path.expanduser().resolve()
    raw_config = _read_config_object(source_config_path, description="config file")
    items_source_uri = _items_source_uri(raw_config, use_metadata_fallback=False)
    if items_source_uri is None:
        return load_run_config(source_config_path, task=DECODABILITY_TASK)

    preliminary_config = load_run_config(
        source_config_path,
        task=DECODABILITY_TASK,
    )
    inputs_root = (
        preliminary_config.inputs_dir
        or (source_config_path.parent / "inputs").resolve()
    )
    return _materialize_run_config(
        raw_config,
        source_config_path=source_config_path,
        inputs_dir=inputs_root / preliminary_config.run_id,
        use_metadata_fallback=False,
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
    typer.echo(f"Reading config from {config_uri}")
    download_file(
        bucket_name=bucket_name,
        file_name=object_name,
        write_path=downloaded_config_path,
    )
    payload = _read_config_object(
        downloaded_config_path,
        description="downloaded cloud config",
    )

    is_snapshot = "resolved_config" in payload and "raw_config" in payload
    if is_snapshot:
        raw_config = payload.get("raw_config")
        if not isinstance(raw_config, dict):
            raise ConfigError("Remote run snapshot raw_config must be an object")
        raw_config = dict(raw_config)
    else:
        raw_config = dict(payload)

    run_prefix = object_name.rsplit("/", 1)[0] + "/"
    raw_config.setdefault("results_dest", f"gs://{bucket_name}/{run_prefix}")
    if is_snapshot:
        raw_config["output_dir"] = str(workspace / "outputs")
    else:
        raw_config.setdefault("output_dir", str(workspace / "outputs"))
    raw_config["inputs_dir"] = str(inputs_dir)
    raw_config["custom_models_path"] = str(inputs_dir / CUSTOM_MODELS_FILE_NAME)
    return _materialize_run_config(
        raw_config,
        source_config_path=downloaded_config_path,
        inputs_dir=inputs_dir / _parse_run_id(raw_config.get("run_id")),
        use_metadata_fallback=True,
    )


def _materialize_run_config(
    raw_config: dict[str, Any],
    *,
    source_config_path: Path,
    inputs_dir: Path,
    use_metadata_fallback: bool,
) -> BenchmarkRunConfig:
    """Turn a local or GCS items source into the runner's local-file contract."""

    processed_config = dict(raw_config)
    snapshot_config = dict(raw_config)
    _resolve_materialized_config_paths(
        processed_config,
        base_dir=source_config_path.parent,
    )
    processed_config["inputs_dir"] = str(inputs_dir.parent)
    items_source_uri = _items_source_uri(
        raw_config,
        use_metadata_fallback=use_metadata_fallback,
    )
    if items_source_uri is not None:
        metadata_value = raw_config.get("metadata", {})
        if not isinstance(metadata_value, dict):
            raise ConfigError("'metadata' must be an object when provided")
        metadata = dict(metadata_value)
        local_items_path = inputs_dir / "benchmark-items.jsonl"
        generation = _download_items_file(items_source_uri, local_items_path)
        metadata["items_source_uri"] = items_source_uri
        metadata["items_source_generation"] = generation
        processed_config["items_path"] = str(local_items_path)
        processed_config["metadata"] = metadata
        snapshot_config["items_path"] = items_source_uri
        snapshot_config["metadata"] = metadata

    inputs_dir.mkdir(parents=True, exist_ok=True)
    processed_config_path = inputs_dir / "run-config.json"
    processed_config_path.write_text(
        json.dumps(processed_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    config = load_run_config(processed_config_path, task=DECODABILITY_TASK)
    return replace(
        config,
        raw_config=snapshot_config,
        config_path=source_config_path,
    )


def _resolve_materialized_config_paths(
    processed_config: dict[str, Any],
    *,
    base_dir: Path,
) -> None:
    """Keep relative paths anchored to the source config after materialization."""

    path_values = {
        "output_dir": processed_config.get("output_dir", "results"),
        "env_path": processed_config.get("env_path"),
        "custom_models_path": processed_config.get("custom_models_path"),
    }
    for field_name, value in path_values.items():
        if not isinstance(value, str) or not value.strip():
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        processed_config[field_name] = str(path.resolve())


def _items_source_uri(
    raw_config: dict[str, Any],
    *,
    use_metadata_fallback: bool,
) -> str | None:
    configured_items_path = raw_config.get("items_path")
    if isinstance(configured_items_path, str) and configured_items_path.startswith(
        "gs://"
    ):
        return configured_items_path
    if not use_metadata_fallback:
        return None
    metadata = raw_config.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ConfigError("'metadata' must be an object when provided")
    items_source_uri = metadata.get("items_source_uri")
    if isinstance(items_source_uri, str) and items_source_uri.startswith("gs://"):
        return items_source_uri
    return None


def _download_items_file(items_source_uri: str, destination: Path) -> int:
    try:
        bucket_name, object_name = parse_gs_uri(items_source_uri)
    except ValueError as e:
        raise ConfigError(f"Invalid cloud items_path: {e}") from e

    typer.echo(f"Reading items from {items_source_uri}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        generation = download_file_with_generation(
            bucket_name=bucket_name,
            file_name=object_name,
            write_path=temporary_path,
        )
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return generation


def _read_config_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON in {description}: {e}") from e
    if not isinstance(payload, dict):
        raise ConfigError(f"{description.capitalize()} must be a JSON object")
    return payload


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
