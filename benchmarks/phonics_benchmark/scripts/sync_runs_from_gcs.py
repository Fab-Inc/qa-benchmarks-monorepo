"""Refresh local run directories from GCS before reporting on them.

A cloud run writes its artifacts to ``gs://<bucket>/phonics/runs/<run_id>/`` while
it executes, so the local copy under ``outputs/`` goes stale as soon as more
responses land. This pulls the current remote state down for one or more runs, so
that report_on_benchmark_run.py and plot_pooled_f2_across_runs.py score what the
models have actually returned so far.

This is ``phonics-benchmark fetch-results`` over a list of runs, with the source
prefix taken from the run itself instead of typed out, and with two differences
from a plain ``gcloud storage cp --recursive``:

- By default only the run's own artifacts are pulled — ``config.json`` and
  ``responses/*``. A ``report/`` directory is written locally by
  report_on_benchmark_run.py, and a partial or older remote copy of it would
  otherwise overwrite the scoring you just generated. ``--all-artifacts`` pulls
  everything under the run prefix.
- Each file is downloaded to a temporary name and moved into place, so a run
  directory is never left holding a half-written responses file if the transfer
  dies partway.

Downloads are unconditional overwrites: whatever is in GCS wins. Any local edit
to a synced file is lost, which is the point — the remote copy is the run.

The bucket and the GCP project come from the same ``.env`` files the rest of the
benchmark uses: ``BUCKET_NAME`` in ``benchmarks/phonics_benchmark/.env`` and
``GOOGLE_CLOUD_PROJECT`` in the repo root's. Nothing needs exporting first.

Run from the repository root, e.g.:

    uv run --frozen --package phonics-benchmark python \\
        benchmarks/phonics_benchmark/scripts/sync_runs_from_gcs.py \\
        --run-id en-stratified-hf-parallel-20260917 \\
        --run-id en-ufli-schwa-stratified-20260915

or refresh every cloud run already present locally:

    uv run --frozen --package phonics-benchmark python \\
        benchmarks/phonics_benchmark/scripts/sync_runs_from_gcs.py --all-local

``--all-local`` passes over runs executed on this machine: a run directory names
its own remote prefix in ``config.json`` (``results_dest``) when it was synced
to a bucket, and a local run has none, so the two never have to be kept in
separate folders.

The benchmark items a cloud run used are not under the run prefix; they live at
``gs://<bucket>/phonics/datasets/<run_id>.jsonl`` and are fetched separately for
``report_on_benchmark_run.py --items-path``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import typer
from dotenv import load_dotenv
from fab_benchmark_publisher import RunSync, list_gcs_objects, parse_gs_uri

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BENCHMARK_ROOT.parents[1]
DEFAULT_OUTPUT_ROOT = BENCHMARK_ROOT / "outputs"

# Same two files, in the same order, as tests/conftest.py: the repo root holds
# GOOGLE_CLOUD_PROJECT (which the storage client needs) and the benchmark's own
# holds BUCKET_NAME. Loaded at import because the CLI option defaults below read
# the environment. Neither overrides a variable already exported in the shell.
load_dotenv(REPO_ROOT / ".env")
load_dotenv(BENCHMARK_ROOT / ".env")

# Where launch-run.sh puts a cloud run: "${RESULTS_BUCKET}/phonics/runs/${RUN_ID}/".
# RESULTS_BUCKET is a full gs:// URI and wins when a shell has it set for a
# launch; otherwise the bucket comes from BUCKET_NAME in .env.
RUNS_PREFIX_SUFFIX = "phonics/runs"

# The run's own output, as opposed to anything a local reporting script wrote
# into the run directory afterwards. A path is kept if it is one of these or
# sits under one of them.
RUN_ARTIFACT_PATHS = ("config.json", "responses")


def default_runs_prefix() -> str:
    """Return the GCS prefix holding run directories, or "" if none is configured."""
    bucket = os.environ.get("RESULTS_BUCKET")
    if not bucket:
        bucket_name = os.environ.get("BUCKET_NAME")
        bucket = f"gs://{bucket_name}" if bucket_name else ""
    return f"{bucket.rstrip('/')}/{RUNS_PREFIX_SUFFIX}" if bucket else ""


def recorded_results_dest(run_dir: Path) -> str | None:
    """Return the GCS prefix a local run recorded writing to, if it had one.

    This is what distinguishes a cloud run from one executed on this machine:
    the runner records ``results_dest`` in the run's ``config.json`` snapshot
    only when it was syncing to a bucket. It is also the authoritative source
    prefix for that run — more reliable than rebuilding one from the run ID,
    which silently assumes the run went to the bucket currently configured.
    """
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        return None
    try:
        snapshot = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return snapshot.get("resolved_config", {}).get("results_dest") or None


def resolve_run_ids(
    run_ids: list[str],
    all_local: bool,
    output_root: Path,
) -> list[str]:
    """Return the run IDs to sync, from the flags or from what is already local.

    ``--all-local`` refreshes the cloud runs already sitting in the output root,
    and only those: it cannot pull down every run the bucket has ever held, and
    it passes over runs executed on this machine, which have no remote copy to
    be stale against.
    """
    if run_ids and all_local:
        raise typer.BadParameter("Pass --run-id or --all-local, not both.")
    if run_ids:
        return list(dict.fromkeys(run_ids))
    if not all_local:
        raise typer.BadParameter("Pass at least one --run-id, or --all-local.")

    if not output_root.is_dir():
        raise typer.BadParameter(f"Output root does not exist: {output_root}")
    # outputs/ also holds analysis/ and loose plot files, so a run directory is
    # identified by having responses/ — and a cloud one by its results_dest.
    cloud_run_ids = sorted(
        path.name
        for path in output_root.iterdir()
        if path.is_dir()
        and (path / "responses").is_dir()
        and recorded_results_dest(path) is not None
    )
    if not cloud_run_ids:
        raise typer.BadParameter(
            f"No cloud run directories found under {output_root}. Name the run "
            "with --run-id if it has not been downloaded here yet."
        )
    return cloud_run_ids


def is_run_artifact(relative_path: str) -> bool:
    """Say whether a path under the run prefix is the run's own output."""
    return any(
        relative_path == name or relative_path.startswith(f"{name}/")
        for name in RUN_ARTIFACT_PATHS
    )


def list_remote_run_paths(source_uri: str, storage_client: Any) -> list[str]:
    """Return every object beneath a run prefix, as run-relative paths."""
    bucket_name, prefix = parse_gs_uri(source_uri)
    prefix = prefix.rstrip("/") + "/"
    return [
        relative_path
        for remote_object in list_gcs_objects(
            bucket_name=bucket_name,
            prefix=prefix,
            storage_client=storage_client,
        )
        # A prefix "directory" placeholder ends in "/" and has no content.
        if (relative_path := remote_object.name.removeprefix(prefix))
        and not relative_path.endswith("/")
    ]


def resolve_source_uri(run_id: str, runs_prefix: str, output_root: Path) -> str:
    """Return the GCS prefix to pull a run from.

    A run already downloaded here says where it came from, so that is used in
    preference to the configured bucket: re-syncing a run must go back to the
    bucket it actually ran against. A run being fetched for the first time has
    nothing to say, so its prefix is built from the run ID.
    """
    recorded = recorded_results_dest(output_root / run_id)
    if recorded:
        return recorded.rstrip("/") + "/"
    if not runs_prefix:
        raise typer.BadParameter(
            f"{run_id} has not been downloaded here, so its source prefix has "
            f"to be configured: set BUCKET_NAME in {BENCHMARK_ROOT / '.env'} "
            "(see .env.example), export RESULTS_BUCKET, or pass --runs-prefix."
        )
    return f"{runs_prefix.rstrip('/')}/{run_id}/"


def sync_run(
    run_id: str,
    runs_prefix: str,
    output_root: Path,
    all_artifacts: bool,
    storage_client: Any,
) -> int | None:
    """Pull one run into ``output_root/run_id``; return files downloaded, or None.

    None means the run prefix holds nothing to pull, which is reported as a
    failure by the caller rather than raised: with several runs given, one
    mistyped or not-yet-started run should not abandon the others.
    """
    source_uri = resolve_source_uri(run_id, runs_prefix, output_root)
    remote_paths = list_remote_run_paths(source_uri, storage_client)
    if not remote_paths:
        typer.secho(
            f"WARNING: nothing found at {source_uri} — skipping {run_id}.",
            fg=typer.colors.YELLOW,
            err=True,
        )
        return None

    selected = None if all_artifacts else set(filter(is_run_artifact, remote_paths))
    if selected is not None and not selected:
        typer.secho(
            f"WARNING: {source_uri} holds no config.json or responses/ — "
            f"skipping {run_id}. Pass --all-artifacts to pull it anyway.",
            fg=typer.colors.YELLOW,
            err=True,
        )
        return None

    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    sync = RunSync(run_dir, source_uri, storage_client=storage_client)
    return sync.pull(relative_paths=selected)


def sync_runs(
    run_id: list[str] = typer.Option(
        [],
        "--run-id",
        help="Run to sync. Repeat the flag for several runs.",
    ),
    all_local: bool = typer.Option(
        False,
        "--all-local",
        help=(
            "Sync every run directory already present under the output root, "
            "instead of naming them with --run-id."
        ),
    ),
    output_root: Path = typer.Option(
        DEFAULT_OUTPUT_ROOT,
        "--output-root",
        file_okay=False,
        help=(
            "Where run directories are written. Defaults to the package's "
            "outputs/ folder, which is where the reporting scripts look."
        ),
    ),
    runs_prefix: str = typer.Option(
        default_runs_prefix(),
        "--runs-prefix",
        help=(
            "GCS prefix holding run directories. Defaults to "
            "$RESULTS_BUCKET/phonics/runs, or gs://$BUCKET_NAME/phonics/runs "
            "from .env — the layout launch-run.sh writes."
        ),
    ),
    all_artifacts: bool = typer.Option(
        False,
        "--all-artifacts",
        help=(
            "Pull everything under the run prefix, not just config.json and "
            "responses/. This can overwrite a local report/ directory with an "
            "older remote copy."
        ),
    ),
) -> None:
    """Download the current remote state of one or more benchmark runs."""
    run_ids = resolve_run_ids(run_id, all_local, output_root)

    # One client for the whole batch: each RunSync would otherwise build its own,
    # re-doing credential discovery per run. The project comes from
    # GOOGLE_CLOUD_PROJECT in the repo-root .env.
    from google.cloud.storage import Client

    storage_client = Client()

    typer.echo(f"Syncing {len(run_ids)} run(s) into {output_root}")
    skipped: list[str] = []
    for current_run_id in run_ids:
        downloaded = sync_run(
            current_run_id,
            runs_prefix,
            output_root,
            all_artifacts,
            storage_client,
        )
        if downloaded is None:
            skipped.append(current_run_id)
            continue
        typer.echo(f"  {current_run_id}: {downloaded} file(s)")

    if skipped:
        typer.secho(
            f"\nSkipped {len(skipped)} run(s) with nothing to pull: "
            f"{', '.join(skipped)}",
            fg=typer.colors.YELLOW,
            err=True,
        )
        # Non-zero so that `sync && report` does not go on to score stale data.
        raise typer.Exit(1)


def main() -> None:
    """Entry point."""
    typer.run(sync_runs)


if __name__ == "__main__":
    main()
