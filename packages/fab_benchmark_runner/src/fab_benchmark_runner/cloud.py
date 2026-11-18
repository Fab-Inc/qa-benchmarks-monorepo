"""Loading configs and items from GCS, and mirroring run results to GCS."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from typing import Any

import typer

from fab_benchmark_runner import gcs
from fab_benchmark_runner.benchmark import Benchmark
from fab_benchmark_runner.config import (
    RunConfig,
    parse_run_config,
    parse_run_id,
    read_config_object,
)
from fab_benchmark_runner.errors import ConfigError
from fab_benchmark_runner.run_sync import RunSync, RunSyncConflictError
from fab_benchmark_runner.runner import RunOutput, RunPlan, execute_run, plan_run

ITEMS_FILE_STEM = "benchmark-items"


def load_local_run_config(config_path: Path, benchmark: Benchmark) -> RunConfig:
    """Load a local config, downloading its items first when they are on GCS."""

    source_config_path = config_path.expanduser().resolve()
    raw_config = read_config_object(source_config_path, description="config file")
    inputs_dir = raw_config.get("inputs_dir")
    if isinstance(inputs_dir, str) and inputs_dir.strip():
        inputs_root = Path(inputs_dir).expanduser()
        if not inputs_root.is_absolute():
            inputs_root = source_config_path.parent / inputs_root
    else:
        inputs_root = source_config_path.parent / "inputs"
    return materialize_run_config(
        raw_config,
        benchmark,
        source_config_path=source_config_path,
        inputs_dir=inputs_root.resolve() / parse_run_id(raw_config.get("run_id")),
    )


def load_cloud_run_config(
    config_uri: str,
    benchmark: Benchmark,
    *,
    workdir: Path,
) -> RunConfig:
    """Download a run config and its items from GCS.

    The config object's prefix is the run's results prefix unless the config sets
    ``results_dest``. To resume a run, run the same config again. A run folder's
    ``config.json`` is its run record, not a config, and is refused here.
    """

    try:
        bucket_name, object_name = gcs.parse_gs_uri(config_uri)
    except ValueError as e:
        raise ConfigError(f"Invalid cloud config URI: {e}") from e
    if "/" not in object_name:
        raise ConfigError("Cloud config URI must be inside a run prefix")

    workspace = workdir.expanduser().resolve()
    inputs_dir = workspace / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    downloaded_config_path = inputs_dir / "run-config-source.json"
    typer.echo(f"Reading config from {config_uri}")
    gcs.download_file(
        bucket_name=bucket_name,
        file_name=object_name,
        write_path=downloaded_config_path,
    )
    payload = read_config_object(
        downloaded_config_path, description="downloaded cloud config"
    )

    if "config_id" in payload and "models" not in payload:
        raise ConfigError(
            f"{config_uri} is a run folder's record, not a run config. Run the "
            "config the run was launched with, for example run-configs/<run_id>.json."
        )
    raw_config = dict(payload)

    if raw_config.pop("custom_models_path", None) is not None:
        typer.echo(
            "Ignoring custom_models_path in a cloud config; set custom_models_uri "
            "and custom_models_sha256 to pin a registry.",
            err=True,
        )
    run_prefix = object_name.rsplit("/", 1)[0] + "/"
    raw_config.setdefault("results_dest", f"gs://{bucket_name}/{run_prefix}")
    _reject_config_inside_its_record(config_uri, raw_config["results_dest"])
    raw_config.setdefault("output_dir", str(workspace / "outputs"))
    raw_config["inputs_dir"] = str(inputs_dir)
    return materialize_run_config(
        raw_config,
        benchmark,
        source_config_path=downloaded_config_path,
        inputs_dir=inputs_dir / parse_run_id(raw_config.get("run_id")),
    )


def materialize_run_config(
    raw_config: dict[str, Any],
    benchmark: Benchmark,
    *,
    source_config_path: Path,
    inputs_dir: Path,
) -> RunConfig:
    """Download a GCS items source, then parse the config against local files.

    The returned config keeps the GCS items URI in ``raw_config``, so the model
    records still point at the source, while ``items_path`` points at the local copy.
    """

    processed_config = dict(raw_config)
    snapshot_config = dict(raw_config)
    items_source_uri = raw_config.get("items_path")
    if isinstance(items_source_uri, str) and items_source_uri.startswith("gs://"):
        metadata_value = raw_config.get("metadata", {})
        if not isinstance(metadata_value, dict):
            raise ConfigError("'metadata' must be an object when provided")
        metadata = dict(metadata_value)
        suffix = PurePosixPath(gcs.parse_gs_uri(items_source_uri)[1]).suffix
        local_items_path = inputs_dir / f"{ITEMS_FILE_STEM}{suffix}"
        metadata["items_source_uri"] = items_source_uri
        metadata["items_source_generation"] = download_items_file(
            items_source_uri, local_items_path
        )
        processed_config["items_path"] = str(local_items_path)
        processed_config["metadata"] = metadata
        snapshot_config["items_path"] = items_source_uri
        snapshot_config["metadata"] = metadata

    config = parse_run_config(
        processed_config,
        benchmark,
        base_dir=source_config_path.parent,
        config_path=source_config_path,
    )
    return replace(config, raw_config=snapshot_config)


def download_items_file(items_source_uri: str, destination: Path) -> int:
    """Atomically download an items object and return its GCS generation."""

    try:
        bucket_name, object_name = gcs.parse_gs_uri(items_source_uri)
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
        generation = gcs.download_file_with_generation(
            bucket_name=bucket_name,
            file_name=object_name,
            write_path=temporary_path,
        )
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return generation


def execute_with_optional_sync(
    config: RunConfig,
    *,
    execute: Callable[..., RunOutput] = execute_run,
) -> RunOutput:
    """Execute locally, or inside the generation-safe GCS sync lifecycle.

    The run's GCS folder is downloaded first, so existing results are reused and the
    runner can check the folder's run record. Uploads then cover only the files
    this job writes, so jobs running other models in the same folder are unaffected.
    """

    if config.results_dest is None:
        return execute(config)

    sync = RunSync(config.run_dir, config.results_dest)
    typer.echo(f"Reading existing results from {config.results_dest}")
    sync.pull()
    typer.echo(f"Writing results to {config.results_dest}")

    def push_with_warning(local_path: Path) -> None:
        try:
            sync.push_file(local_path)
        except RunSyncConflictError:
            raise
        except Exception as e:
            typer.echo(f"WARNING: could not upload {local_path}: {e}", err=True)

    try:
        output = execute(config, on_artifact_write=push_with_warning)
    except BaseException:
        try:
            _finalize_sync(sync)
        except Exception as cleanup_error:
            typer.echo(
                "WARNING: sync cleanup failed while preserving the original "
                f"error: {cleanup_error}",
                err=True,
            )
        raise
    _finalize_sync(sync)
    typer.echo(f"Wrote results to {config.results_dest}")
    return output


def plan_with_optional_sync(
    config: RunConfig,
    *,
    plan: Callable[[RunConfig], RunPlan] = plan_run,
) -> RunPlan:
    """Plan a run against its existing results, pulling them from GCS if synced.

    Nothing is pushed: a dry run never changes the remote run.
    """

    if config.results_dest is not None:
        typer.echo(f"Reading existing results from {config.results_dest}")
        RunSync(config.run_dir, config.results_dest).pull()
    return plan(config)


def _finalize_sync(sync: RunSync) -> None:
    """Retry this job's own uploads that haven't been confirmed."""

    sync.push_pending()


def _reject_config_inside_its_record(config_uri: str, results_dest: Any) -> None:
    """A config stored as the run folder's config.json would be replaced by the
    folder's run record, so it could never be run again."""

    if not isinstance(results_dest, str):
        return
    try:
        bucket_name, prefix = gcs.parse_gs_uri(results_dest)
    except ValueError:
        return  # parse_run_config reports the invalid results_dest
    record_uri = f"gs://{bucket_name}/{prefix.rstrip('/')}/config.json"
    if config_uri == record_uri:
        raise ConfigError(
            f"{config_uri} is where the run folder keeps its run record. Store the "
            "run config elsewhere, for example run-configs/<run_id>.json."
        )
