"""The ``fab-bench`` command line: create experiments and run models on them."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import typer
from google.api_core.exceptions import GoogleAPIError as GoogleCloudAPIError
from google.auth.exceptions import GoogleAuthError

from fab_benchmark_runner.config import (
    RunConfig,
    apply_execution_overrides,
    load_run_config,
)
from fab_benchmark_runner.errors import ConfigError
from fab_benchmark_runner.experiment import (
    ExperimentCreation,
    create_experiment,
    raise_max_tokens,
)
from fab_benchmark_runner.ids import canonical_json
from fab_benchmark_runner.jobs import (
    fetch_results,
    plan_job,
    run_job,
    upload_experiment,
)
from fab_benchmark_runner.registry import CUSTOM_MODELS_FILE_NAME
from fab_benchmark_runner.run_sync import RunSyncConflictError
from fab_benchmark_runner.runner import RunPlan, RunResult

DEFAULT_LOCAL_WORKDIR = Path(".fab-bench")
DEFAULT_CLOUD_WORKDIR = Path("/workspace")
EXPECTED_CLI_ERRORS = (
    ConfigError,
    FileNotFoundError,
    GoogleCloudAPIError,
    GoogleAuthError,
    RunSyncConflictError,
    ValueError,
)


SAMPLE_ROWS = 3
SAMPLE_ROW_CHARACTERS = 300

app = typer.Typer(
    name="fab-bench",
    help="Create benchmark experiments and run models on them.",
)


@app.callback(invoke_without_command=True)
def _fab_bench(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


@app.command("create-experiment")
def create_experiment_cmd(
    inputs: str = typer.Option(
        ...,
        "--inputs",
        help=(
            "Inputs file, a local path or a gs:// URI: one rendered input per item, "
            "as JSON lines."
        ),
    ),
    dest: str = typer.Option(
        ..., "--dest", help="Experiment folder: a local path or a gs:// URI."
    ),
    benchmark: str = typer.Option(..., "--benchmark", help="The benchmark's name."),
    max_tokens: str = typer.Option(
        ...,
        "--max-tokens",
        help="Response token limit for every call, or 'none' for no limit.",
    ),
    experiment_id: str | None = typer.Option(
        None,
        "--experiment-id",
        help="The experiment's name. Defaults to a short hash of its contents.",
    ),
    label: list[str] | None = typer.Option(
        None,
        "--label",
        help="A key=value note for analysis, such as subset=level1. Repeatable.",
    ),
    copy_images: bool = typer.Option(
        False,
        "--copy-images",
        help="Copy images into the experiment folder instead of referencing them.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Check everything and show a few inputs, without writing anything.",
    ),
) -> None:
    """Store an inputs file as an experiment that any number of models can run on."""

    try:
        creation = create_experiment(
            inputs,
            dest=dest,
            benchmark=benchmark,
            max_tokens=_parse_max_tokens(max_tokens),
            experiment_id=experiment_id,
            labels=_parse_labels(label or []),
            copy_images=copy_images,
            dry_run=dry_run,
        )
    except EXPECTED_CLI_ERRORS as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(1) from e
    echo_creation(creation)


def echo_creation(creation: ExperimentCreation) -> None:
    experiment = creation.experiment
    record = experiment.record
    limit = "none" if record["max_tokens"] is None else record["max_tokens"]
    typer.echo(f"Experiment: {record['experiment_id']}")
    typer.echo(f"Folder: {experiment.location}")
    typer.echo(f"Benchmark: {record['benchmark']}")
    typer.echo(f"Items: {record['item_count']}")
    typer.echo(f"Images: {record['image_count']}")
    typer.echo(f"Max tokens: {limit}")
    typer.echo(f"Inputs SHA-256: {record['inputs_sha256']}")
    if creation.status == "created":
        typer.echo("Created.")
    elif creation.status == "exists":
        typer.echo("This experiment already exists; nothing was written.")
    else:
        typer.echo("Dry run: nothing was written. First inputs:")
        for model_input in experiment.inputs[:SAMPLE_ROWS]:
            row = canonical_json(model_input.to_row())
            if len(row) > SAMPLE_ROW_CHARACTERS:
                row = row[:SAMPLE_ROW_CHARACTERS] + "…"
            typer.echo(f"  {row}")


def _parse_max_tokens(value: str) -> int | None:
    if value.strip().lower() == "none":
        return None
    try:
        max_tokens = int(value)
    except ValueError:
        max_tokens = 0
    if max_tokens < 1:
        raise typer.BadParameter(
            "must be a positive whole number or 'none'", param_hint="--max-tokens"
        )
    return max_tokens


def _parse_labels(values: list[str]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for value in values:
        key, separator, label = value.partition("=")
        key = key.strip()
        if not separator or not key:
            raise typer.BadParameter(
                f"{value!r} is not key=value", param_hint="--label"
            )
        if key in labels:
            raise typer.BadParameter(f"{key!r} is given twice", param_hint="--label")
        labels[key] = label
    return labels


@app.command("raise-max-tokens")
def raise_max_tokens_cmd(
    experiment: str = typer.Option(
        ..., "--experiment", help="Experiment folder: a local path or a gs:// URI."
    ),
    max_tokens: str = typer.Option(
        ...,
        "--max-tokens",
        help="The new, higher token limit, or 'none' for no limit.",
    ),
) -> None:
    """Raise an experiment's token limit; results that finished stay valid.

    Then run with --retry-errors to rerun the calls that failed, including those
    the old limit left empty.
    """

    try:
        change = raise_max_tokens(experiment, _parse_max_tokens(max_tokens))
    except EXPECTED_CLI_ERRORS as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(1) from e
    if change.status == "unchanged":
        typer.echo(f"max_tokens is already {_limit_text(change.max_tokens)}.")
        return
    typer.echo(
        f"Raised max_tokens from {_limit_text(change.previous)} to "
        f"{_limit_text(change.max_tokens)}. Run with --retry-errors to rerun the "
        "calls that failed."
    )


@app.command("run")
def run_cmd(
    config: str = typer.Option(
        ..., "--config", "-c", help="Run config: a local JSON file or a gs:// URI."
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Ignore existing results and rerun all calls."
    ),
    retry_errors: bool = typer.Option(
        False, "--retry-errors", help="Rerun only the calls that failed."
    ),
    model_workers: int | None = typer.Option(
        None, "--model-workers", min=1, help="How many models to run at once."
    ),
    workdir: Path = typer.Option(
        DEFAULT_LOCAL_WORKDIR,
        "--workdir",
        file_okay=False,
        help="Where to keep copies of experiments on GCS and pinned registries.",
    ),
    skip_models_download: bool = typer.Option(
        False,
        "--skip-models-download",
        help="Don't refresh the shared public custom_models.yaml first.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Report the calls a run would make or reuse, without running them.",
    ),
) -> None:
    """Run a config's models on its experiment, local or on GCS."""

    def load() -> RunConfig:
        return apply_execution_overrides(
            load_run_config(config),
            force=force,
            retry_errors=retry_errors,
            model_workers=model_workers,
        )

    _run_or_plan(
        load,
        workdir=workdir,
        download_public=not skip_models_download,
        public_registry_path=None,
        dry_run=dry_run,
    )


@app.command("run-cloud")
def run_cloud_cmd(
    config: str = typer.Option(
        ...,
        "--config",
        "-c",
        help="GCS URI of the run config, e.g. gs://bucket/run-configs/<hash>.json.",
    ),
    workdir: Path = typer.Option(
        DEFAULT_CLOUD_WORKDIR,
        "--workdir",
        file_okay=False,
        help="Writable workspace for the experiment's copy and the registry.",
    ),
    skip_models_download: bool = typer.Option(
        False,
        "--skip-models-download",
        help="Don't refresh the shared public custom_models.yaml first.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Report the calls a run would make or reuse, without running them.",
    ),
) -> None:
    """Run a config from GCS, as a Cloud Run job does."""

    def load() -> RunConfig:
        if not config.startswith("gs://"):
            raise ConfigError("run-cloud reads its run config from a gs:// URI")
        return load_run_config(config)

    _run_or_plan(
        load,
        workdir=workdir,
        download_public=not skip_models_download,
        public_registry_path=workdir.expanduser().resolve() / CUSTOM_MODELS_FILE_NAME,
        dry_run=dry_run,
    )


@app.command("fetch-results")
def fetch_results_cmd(
    experiment: str = typer.Option(
        ..., "--experiment", help="The experiment's gs:// URI."
    ),
    dest: Path = typer.Option(
        ..., "--dest", file_okay=False, help="Local folder to download into."
    ),
    model: str | None = typer.Option(
        None, "--model", help="Download only this model's results and record."
    ),
) -> None:
    """Download an experiment's inputs, records and results, without images."""

    try:
        count = fetch_results(experiment, dest, model=model)
    except EXPECTED_CLI_ERRORS as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"Downloaded {count} files from {experiment} to {dest}")


@app.command("upload-experiment")
def upload_experiment_cmd(
    source: Path = typer.Argument(
        ..., file_okay=False, help="Local experiment folder."
    ),
    dest: str = typer.Argument(..., help="Destination gs:// URI."),
) -> None:
    """Copy a local experiment, and any results, to GCS so Cloud Run can use it."""

    try:
        count = upload_experiment(source, dest)
    except EXPECTED_CLI_ERRORS as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"Uploaded {count} files to {dest}")


def _run_or_plan(
    load_config: Callable[[], RunConfig],
    *,
    workdir: Path,
    download_public: bool,
    public_registry_path: Path | None,
    dry_run: bool,
) -> None:
    try:
        run_config = load_config()
        if dry_run:
            plan = plan_job(
                run_config,
                workdir=workdir,
                download_public_registry=download_public,
                public_registry_path=public_registry_path,
            )
        else:
            result = run_job(
                run_config,
                workdir=workdir,
                download_public_registry=download_public,
                public_registry_path=public_registry_path,
            )
    except EXPECTED_CLI_ERRORS as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(1) from e

    if dry_run:
        echo_plan(plan)
        return
    echo_run_result(result)
    if result.errors > 0:
        raise typer.Exit(1)


def echo_version_warnings(warnings: tuple[str, ...]) -> None:
    if not warnings:
        return
    typer.echo(
        "WARNING: this experiment already has results from other versions:", err=True
    )
    for warning in warnings:
        typer.echo(f"  {warning}", err=True)


def echo_run_result(result: RunResult) -> None:
    echo_version_warnings(result.version_warnings)
    typer.echo(f"Experiment: {result.experiment_id}")
    typer.echo(f"Folder: {result.experiment_dir}")
    typer.echo(f"Executed: {result.executed}")
    typer.echo(f"Reused: {result.reused}")
    typer.echo(f"Errors: {result.errors}")


def echo_plan(plan: RunPlan) -> None:
    echo_version_warnings(plan.version_warnings)
    typer.echo(f"Experiment: {plan.experiment_id}")
    typer.echo(f"Max tokens: {_limit_text(plan.max_tokens)}")
    typer.echo("Dry run: no model was called and nothing was written.")
    for model in plan.models:
        retries = f" ({model.retries} error retries)" if model.retries else ""
        stale = (
            f", {model.stale} stale rows dropped on next write" if model.stale else ""
        )
        typer.echo(
            f"{model.model}: {model.planned} planned, {model.reused} reused, "
            f"{model.to_run} to run{retries}{stale}"
        )
    typer.echo(
        f"Total: {plan.planned} planned, {plan.reused} reused, {plan.to_run} to run"
    )


def _limit_text(max_tokens: int | None) -> str:
    return "none" if max_tokens is None else str(max_tokens)
