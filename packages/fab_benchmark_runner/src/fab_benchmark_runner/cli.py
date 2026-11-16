"""The command-line interface a benchmark gets from its adapter."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import typer
from google.api_core.exceptions import GoogleAPIError as GoogleCloudAPIError
from google.auth.exceptions import GoogleAuthError

from fab_benchmark_runner.benchmark import Benchmark
from fab_benchmark_runner.cloud import (
    execute_with_optional_sync,
    load_cloud_run_config,
    load_local_run_config,
    plan_with_optional_sync,
)
from fab_benchmark_runner.config import RunConfig, apply_execution_overrides
from fab_benchmark_runner.errors import ConfigError
from fab_benchmark_runner.registry import (
    CUSTOM_MODELS_FILE_NAME,
    prepare_model_registry,
)
from fab_benchmark_runner.results import model_results_filename
from fab_benchmark_runner.run_sync import RunSync, RunSyncConflictError
from fab_benchmark_runner.runner import RunOutput, RunPlan

DEFAULT_CLOUD_WORKDIR = Path("/workspace")
EXPECTED_CLI_ERRORS = (
    ConfigError,
    FileNotFoundError,
    GoogleCloudAPIError,
    GoogleAuthError,
    RunSyncConflictError,
    ValueError,
)


def build_app(benchmark: Benchmark) -> typer.Typer:
    """Return a Typer app with run, run-cloud and results sync commands.

    A benchmark can expose it as its console script directly, or add its own
    commands to it.
    """

    app = typer.Typer(
        name=benchmark.name,
        help=f"Run the {benchmark.name} benchmark.",
    )

    @app.callback(invoke_without_command=True)
    def setup(ctx: typer.Context) -> None:
        if ctx.invoked_subcommand is None:
            typer.echo(ctx.get_help())
            raise typer.Exit()

    @app.command("run")
    def run_cmd(
        config: Path = typer.Option(
            ...,
            "--config",
            "-c",
            exists=True,
            dir_okay=False,
            readable=True,
            help="JSON run config. Its items_path may be local or a gs:// URI.",
        ),
        force: bool = typer.Option(
            False, "--force", "-f", help="Ignore existing results and rerun all calls."
        ),
        retry_errors: bool = typer.Option(
            False,
            "--retry-errors",
            help="Rerun only existing results whose call failed.",
        ),
        model_workers: int | None = typer.Option(
            None,
            "--model-workers",
            min=1,
            help="Maximum number of provider groups to run concurrently.",
        ),
        results_dest: str | None = typer.Option(
            None,
            "--results-dest",
            help="GCS run prefix to mirror results to, e.g. gs://bucket/runs/id/.",
        ),
        skip_models_download: bool = typer.Option(
            False,
            "--skip-models-download",
            help="Do not refresh the shared public custom_models.yaml first.",
        ),
        dry_run: bool = typer.Option(
            False,
            "--dry-run",
            help="Report the calls a run would make or reuse, without running them.",
        ),
    ) -> None:
        """Run the benchmark from a local config file."""

        def load() -> RunConfig:
            run_config = load_local_run_config(config, benchmark)
            run_config = apply_execution_overrides(
                run_config,
                force=force,
                retry_errors=retry_errors,
                model_workers=model_workers,
                results_dest=results_dest,
            )
            return prepare_model_registry(
                run_config, download_public=not skip_models_download
            )

        _run_or_plan(load, dry_run=dry_run)

    @app.command("run-cloud")
    def run_cloud_cmd(
        config: str = typer.Option(
            ...,
            "--config",
            "-c",
            help="GCS URI of the run config, e.g. gs://bucket/run-configs/id.json.",
        ),
        workdir: Path = typer.Option(
            DEFAULT_CLOUD_WORKDIR,
            "--workdir",
            file_okay=False,
            help="Writable workspace for downloaded inputs and run outputs.",
        ),
        skip_models_download: bool = typer.Option(
            False,
            "--skip-models-download",
            help="Do not refresh the shared public custom_models.yaml first.",
        ),
        dry_run: bool = typer.Option(
            False,
            "--dry-run",
            help="Report the calls a run would make or reuse, without running them.",
        ),
    ) -> None:
        """Fetch a GCS config and its items, then run with results synced to GCS."""

        def load() -> RunConfig:
            run_config = load_cloud_run_config(config, benchmark, workdir=workdir)
            return prepare_model_registry(
                run_config,
                download_public=not skip_models_download,
                public_path=workdir.expanduser().resolve()
                / "inputs"
                / CUSTOM_MODELS_FILE_NAME,
            )

        _run_or_plan(load, dry_run=dry_run)

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
        dest: str = typer.Option(..., "--dest", help="Destination GCS run prefix."),
    ) -> None:
        """Upload local run files that are new or changed to a GCS run prefix."""

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
            help="Local directory to download the run files into.",
        ),
        model: str | None = typer.Option(
            None, "--model", help="Download only this model's response file."
        ),
    ) -> None:
        """Download all run files, or one model's response file, from GCS."""

        relative_paths = None
        if model is not None:
            relative_paths = {f"responses/{model_results_filename(model)}"}
        try:
            downloaded_count = RunSync(run_dir, src).pull(relative_paths=relative_paths)
        except EXPECTED_CLI_ERRORS as e:
            typer.echo(f"ERROR: {e}", err=True)
            raise typer.Exit(1) from e
        typer.echo(f"Downloaded: {downloaded_count} from {src}")

    return app


def _run_or_plan(load_config: Callable[[], RunConfig], *, dry_run: bool) -> None:
    try:
        run_config = load_config()
        if dry_run:
            plan = plan_with_optional_sync(run_config)
        else:
            output = execute_with_optional_sync(run_config)
    except EXPECTED_CLI_ERRORS as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(1) from e

    if dry_run:
        echo_plan(plan)
        return
    echo_run_output(output)
    if output["error_count"] > 0:
        raise typer.Exit(1)


def echo_run_output(output: RunOutput) -> None:
    typer.echo(f"Run ID: {output['run_id']}")
    typer.echo(f"Config ID: {output['config_id']}")
    typer.echo(f"Results: {output['results_dir']}")
    typer.echo(f"Executed: {output['executed_count']}")
    typer.echo(f"Skipped: {output['skipped_count']}")
    typer.echo(f"Errors: {output['error_count']}")


def echo_plan(plan: RunPlan) -> None:
    typer.echo(f"Run ID: {plan.run_id}")
    typer.echo(f"Config ID: {plan.config_id}")
    typer.echo("Dry run: no model was called and no result was written.")
    for model in plan.models:
        retries = f" ({model.retries} error retries)" if model.retries else ""
        stale = (
            f", {model.stale} stale rows dropped on next write" if model.stale else ""
        )
        typer.echo(
            f"{model.model}: {model.planned} planned, {model.skipped} reused, "
            f"{model.to_run} to run{retries}{stale}"
        )
    typer.echo(
        f"Total: {plan.planned} planned, {plan.skipped} reused, {plan.to_run} to run"
    )
