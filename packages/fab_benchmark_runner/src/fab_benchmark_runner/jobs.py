"""Jobs: run or plan a run config on its experiment, local or on GCS.

A job on a local experiment works in its folder. A job on an experiment on GCS
works on a local mirror in its workdir. It downloads the experiment's own files,
every model's record, and the results of the models it runs, never another model's
results. It uploads only the files it writes, each with a GCS generation check, so
jobs running other models on the same experiment never conflict, and a second job
on the same model stops instead of overwriting the first one's results.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import typer

from fab_benchmark_runner import gcs
from fab_benchmark_runner.config import RunConfig
from fab_benchmark_runner.errors import ConfigError
from fab_benchmark_runner.experiment import (
    EXPERIMENT_RECORD_NAME,
    IMAGES_DIR,
    INPUTS_FILE_NAME,
    read_experiment,
)
from fab_benchmark_runner.llm import call_model_with_fdllm
from fab_benchmark_runner.records import MODEL_RECORDS_DIR, RESPONSES_DIR
from fab_benchmark_runner.registry import (
    ModelRegistry,
    check_provider_groups,
    prepare_fdllm,
    prepare_model_registry,
)
from fab_benchmark_runner.results import model_results_filename
from fab_benchmark_runner.run_sync import RunSync, RunSyncConflictError
from fab_benchmark_runner.runner import (
    CallModel,
    RunPlan,
    RunResult,
    execute_run,
    plan_run,
)

MIRRORS_DIR = "experiments"


def plan_job(
    config: RunConfig,
    *,
    workdir: Path,
    download_public_registry: bool = True,
    public_registry_path: Path | None = None,
    call_model: CallModel | None = None,
    storage_client: Any | None = None,
) -> RunPlan:
    """Report what a job would do. An experiment on GCS is read, never written."""

    _registry, model_configs, _call = _prepare_models(
        config,
        workdir=workdir,
        download_public_registry=download_public_registry,
        public_registry_path=public_registry_path,
        call_model=call_model,
    )
    if config.experiment_is_gcs:
        experiment_dir = experiment_mirror(workdir, config.experiment)
        _pull_job_files(
            RunSync(experiment_dir, config.experiment, storage_client=storage_client),
            config,
        )
    else:
        experiment_dir = Path(config.experiment)
    return plan_run(config, experiment_dir, model_configs=model_configs)


def run_job(
    config: RunConfig,
    *,
    workdir: Path,
    download_public_registry: bool = True,
    public_registry_path: Path | None = None,
    call_model: CallModel | None = None,
    storage_client: Any | None = None,
) -> RunResult:
    """Run a config's models on its experiment.

    Without ``call_model`` the models are called through FDLLM, after registering
    the run's model registry; a supplied ``call_model`` replaces FDLLM, and the
    model configs in call IDs are then empty.
    """

    registry, model_configs, call = _prepare_models(
        config,
        workdir=workdir,
        download_public_registry=download_public_registry,
        public_registry_path=public_registry_path,
        call_model=call_model,
    )
    if not config.experiment_is_gcs:
        experiment_dir = Path(config.experiment)
        _check_runnable(experiment_dir, uses_fdllm=call_model is None)
        return execute_run(
            config,
            experiment_dir,
            model_configs=model_configs,
            call_model=call,
            registry=registry.record(),
        )

    experiment_dir = experiment_mirror(workdir, config.experiment)
    sync = RunSync(experiment_dir, config.experiment, storage_client=storage_client)
    _pull_job_files(sync, config)
    _check_runnable(experiment_dir, uses_fdllm=call_model is None)
    typer.echo(f"Writing results to {config.experiment}")

    def push_with_warning(local_path: Path) -> None:
        try:
            sync.push_file(local_path)
        except RunSyncConflictError:
            raise
        except Exception as e:
            typer.echo(f"WARNING: could not upload {local_path}: {e}", err=True)

    try:
        result = execute_run(
            config,
            experiment_dir,
            model_configs=model_configs,
            call_model=call,
            registry=registry.record(),
            on_artifact_write=push_with_warning,
        )
    except BaseException:
        try:
            sync.push_pending()
        except Exception as cleanup_error:
            typer.echo(
                "WARNING: retrying uploads failed while preserving the original "
                f"error: {cleanup_error}",
                err=True,
            )
        raise
    sync.push_pending()
    return result


def fetch_results(
    experiment_uri: str,
    dest: Path,
    *,
    model: str | None = None,
    storage_client: Any | None = None,
) -> int:
    """Download an experiment's inputs, records and results, or one model's.

    Images are not downloaded. Returns the number of files downloaded.
    """

    def wanted(name: str) -> bool:
        if model is None:
            return not name.startswith(f"{IMAGES_DIR}/")
        return _is_job_file(name, [model])

    sync = RunSync(dest, _gcs_experiment(experiment_uri), storage_client=storage_client)
    count = sync.pull(select=wanted)
    if not (dest / EXPERIMENT_RECORD_NAME).exists():
        raise ConfigError(f"{experiment_uri} holds no experiment")
    return count


def upload_experiment(
    source: Path, dest_uri: str, *, storage_client: Any | None = None
) -> int:
    """Copy a local experiment, with any results, to GCS; return the files sent.

    Files are uploaded in the order a creation writes them, record last, and never
    overwrite a different file. Images referenced by a local path are refused:
    Cloud Run can't read them, and copying them would change the stored inputs.
    """

    experiment = read_experiment(source)
    local_images = sorted(
        {
            image.uri
            for model_input in experiment.inputs
            for image in model_input.images
            if not image.is_gcs and Path(image.uri).is_absolute()
        }
    )
    if local_images:
        raise ConfigError(
            f"{source} references images at local paths, such as "
            f"{local_images[0]}; create the experiment with --copy-images to store "
            "them in its folder"
        )
    sync = RunSync(source, _gcs_experiment(dest_uri), storage_client=storage_client)
    folder = source.expanduser().resolve()
    ordered = [
        *sorted((folder / IMAGES_DIR).glob("*")),
        folder / INPUTS_FILE_NAME,
        folder / EXPERIMENT_RECORD_NAME,
        *sorted((folder / RESPONSES_DIR).glob("*.jsonl")),
        *sorted((folder / MODEL_RECORDS_DIR).glob("*.json")),
    ]
    for path in ordered:
        if path.is_file():
            sync.push_file(path)
    return sum(path.is_file() for path in ordered)


def experiment_mirror(workdir: Path, experiment_uri: str) -> Path:
    """Return where a job keeps its local copy of an experiment on GCS."""

    bucket_name, prefix = gcs.parse_gs_uri(_gcs_experiment(experiment_uri) + "/")
    return (
        workdir.expanduser().resolve()
        / MIRRORS_DIR
        / bucket_name
        / Path(*prefix.strip("/").split("/"))
    )


def _prepare_models(
    config: RunConfig,
    *,
    workdir: Path,
    download_public_registry: bool,
    public_registry_path: Path | None,
    call_model: CallModel | None,
) -> tuple[ModelRegistry, dict[str, Any], CallModel]:
    registry = prepare_model_registry(
        config,
        workdir=workdir,
        download_public=download_public_registry,
        public_path=public_registry_path,
    )
    check_provider_groups(registry.path, list(config.models))
    if call_model is not None:
        return registry, {}, call_model
    model_configs = prepare_fdllm(registry, config.models, env_path=config.env_path)
    return registry, model_configs, call_model_with_fdllm


def _pull_job_files(sync: RunSync, config: RunConfig) -> None:
    typer.echo(f"Reading the experiment from {config.experiment}")
    sync.pull(select=lambda name: _is_job_file(name, config.models))
    if not (sync.run_dir / EXPERIMENT_RECORD_NAME).exists():
        raise ConfigError(f"{config.experiment} holds no experiment")


def _is_job_file(name: str, models: Iterable[str]) -> bool:
    if name in {EXPERIMENT_RECORD_NAME, INPUTS_FILE_NAME}:
        return True
    if name.startswith(f"{MODEL_RECORDS_DIR}/"):
        return True
    return name in {
        f"{RESPONSES_DIR}/{model_results_filename(model)}" for model in models
    }


def _check_runnable(experiment_dir: Path, *, uses_fdllm: bool) -> None:
    """Refuse, before any call, what FDLLM calls can't send yet."""

    if not uses_fdllm:
        return
    experiment = read_experiment(experiment_dir)
    if any(model_input.images for model_input in experiment.inputs):
        raise ConfigError(
            f"Experiment {experiment.experiment_id!r} has images, which runs don't "
            "send yet"
        )
    if experiment.max_tokens is None:
        raise ConfigError(
            f"Experiment {experiment.experiment_id!r} has no token limit, which runs "
            "don't support yet"
        )


def _gcs_experiment(uri: str) -> str:
    if not isinstance(uri, str) or not uri.startswith("gs://"):
        raise ConfigError(f"{uri!r} is not a gs:// URI")
    try:
        gcs.parse_gs_uri(uri.rstrip("/") + "/")
    except ValueError as e:
        raise ConfigError(f"Invalid experiment URI {uri}: {e}") from e
    return uri.rstrip("/")
