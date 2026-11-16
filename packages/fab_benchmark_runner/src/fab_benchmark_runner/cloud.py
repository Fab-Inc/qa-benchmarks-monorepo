"""Loading configs and items from GCS, and mirroring run results to GCS."""

from __future__ import annotations

import json
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
    config_identity_sha256,
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
        use_metadata_fallback=False,
    )


def load_cloud_run_config(
    config_uri: str,
    benchmark: Benchmark,
    *,
    workdir: Path,
) -> RunConfig:
    """Download a run config (or a prior run's snapshot) and its items from GCS.

    The config object's prefix is the run's results prefix unless the config sets
    ``results_dest``. Resuming from the ``config.json`` snapshot a run wrote keeps
    the original raw config, so pinned registries and items carry over.
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

    is_snapshot = "resolved_config" in payload and "raw_config" in payload
    if is_snapshot:
        raw_config = payload.get("raw_config")
        if not isinstance(raw_config, dict):
            raise ConfigError("Remote run snapshot raw_config must be an object")
        raw_config = dict(raw_config)
    else:
        raw_config = dict(payload)

    if raw_config.pop("custom_models_path", None) is not None:
        typer.echo(
            "Ignoring custom_models_path in a cloud config; set custom_models_uri "
            "and custom_models_sha256 to pin a registry.",
            err=True,
        )
    run_prefix = object_name.rsplit("/", 1)[0] + "/"
    raw_config.setdefault("results_dest", f"gs://{bucket_name}/{run_prefix}")
    if is_snapshot:
        raw_config["output_dir"] = str(workspace / "outputs")
    else:
        raw_config.setdefault("output_dir", str(workspace / "outputs"))
    raw_config["inputs_dir"] = str(inputs_dir)
    return materialize_run_config(
        raw_config,
        benchmark,
        source_config_path=downloaded_config_path,
        inputs_dir=inputs_dir / parse_run_id(raw_config.get("run_id")),
        use_metadata_fallback=True,
    )


def materialize_run_config(
    raw_config: dict[str, Any],
    benchmark: Benchmark,
    *,
    source_config_path: Path,
    inputs_dir: Path,
    use_metadata_fallback: bool,
) -> RunConfig:
    """Download a GCS items source, then parse the config against local files.

    The returned config keeps the GCS items URI in ``raw_config`` so its snapshot
    still points at the source, while ``items_path`` points at the local copy.
    """

    processed_config = dict(raw_config)
    snapshot_config = dict(raw_config)
    items_source_uri = _items_source_uri(
        raw_config, use_metadata_fallback=use_metadata_fallback
    )
    if items_source_uri is not None:
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


def validate_resume_snapshot(config: RunConfig, snapshot_path: Path) -> None:
    """Reject reusing a run whose semantic inputs have changed."""

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

    # Before a run's first checkpoint, the uploaded source config occupies this
    # object. The first snapshot write replaces it; every later resume checks it.
    if "config_id" not in snapshot and "resolved_config" not in snapshot:
        return
    remote_config_id = snapshot.get("config_id")
    expected_config_id = config_identity_sha256(config)
    if remote_config_id != expected_config_id:
        raise ConfigError(
            "Existing run config_id does not match this run config: "
            f"{remote_config_id!r} != {expected_config_id!r}"
        )
    resolved_config = snapshot.get("resolved_config")
    if not isinstance(resolved_config, dict):
        raise ConfigError("Existing run config snapshot lacks resolved_config")
    remote_items_sha256 = resolved_config.get("items_sha256")
    _items, local_items_sha256 = config.benchmark.load_items(config.items_path)
    if remote_items_sha256 != local_items_sha256:
        raise ConfigError(
            "Existing run items_sha256 does not match the local items artifact: "
            f"{remote_items_sha256!r} != {local_items_sha256!r}"
        )


def execute_with_optional_sync(
    config: RunConfig,
    *,
    execute: Callable[..., RunOutput] = execute_run,
) -> RunOutput:
    """Execute locally, or inside the generation-safe GCS sync lifecycle."""

    snapshot_path = config.run_dir / "config.json"
    if config.results_dest is None:
        validate_resume_snapshot(config, snapshot_path)
        return execute(config)

    sync = RunSync(config.run_dir, config.results_dest)
    typer.echo(f"Reading existing results from {config.results_dest}")
    sync.pull()
    validate_resume_snapshot(config, snapshot_path)
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
            _finalize_sync(sync, config.run_dir)
        except Exception as cleanup_error:
            typer.echo(
                "WARNING: sync cleanup failed while preserving the original "
                f"error: {cleanup_error}",
                err=True,
            )
        raise
    _finalize_sync(sync, config.run_dir)
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
    validate_resume_snapshot(config, config.run_dir / "config.json")
    return plan(config)


def _finalize_sync(sync: RunSync, run_dir: Path) -> None:
    """Push the final log, then reconcile every artifact."""

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
