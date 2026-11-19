"""Experiments on GCS: many jobs, one experiment, no overwrites.

Decisions this file describes:

- An experiment can live on GCS or in a local folder; Cloud Run jobs need one on
  GCS. A job works on a local copy in its own workdir, the way a fresh container
  does.
- A job downloads the experiment's inputs, every model's record, and the results of
  the models it runs, never another model's results, and uploads only the files it
  writes.
- Every upload is checked against the GCS generation the job last saw. Jobs running
  different models never touch the same file; a second job on the same model stops
  instead of overwriting the first one's results.
- A job that crashed or timed out is resumed by launching the same run config
  again: what it uploaded is reused.
- A local experiment can be uploaded and continued on GCS.
"""

from __future__ import annotations

import pytest

from fab_benchmark_runner import ConfigError, parse_run_config, upload_experiment
from fab_benchmark_runner.run_sync import RunSyncConflictError

from .conftest import Bench

EXPERIMENT = "gs://bucket/lk/experiments/sw-v1"


def test_jobs_running_different_models_share_one_experiment(bench: Bench) -> None:
    bench.create(dest=EXPERIMENT)

    bench.run("model-a", experiment=EXPERIMENT, job="job-1")
    bench.run("model-b", experiment=EXPERIMENT, job="job-2")

    assert bench.gcs_files() == {
        "lk/experiments/sw-v1/experiment.json",
        "lk/experiments/sw-v1/inputs.jsonl",
        "lk/experiments/sw-v1/models/model-a.json",
        "lk/experiments/sw-v1/models/model-b.json",
        "lk/experiments/sw-v1/responses/responses_model-a.jsonl",
        "lk/experiments/sw-v1/responses/responses_model-b.jsonl",
    }


def test_a_job_downloads_only_what_it_needs(bench: Bench) -> None:
    """Other models' results stay on GCS; their records come along, so a dry run can
    compare the versions they used."""

    bench.create(dest=EXPERIMENT)
    bench.run("model-a", "model-b", experiment=EXPERIMENT, job="job-1")

    bench.run("model-c", experiment=EXPERIMENT, job="job-2")

    mirror = bench.root / "job-2" / "experiments" / "bucket" / "lk" / "experiments"
    downloaded = {
        path.relative_to(mirror / "sw-v1").as_posix()
        for path in (mirror / "sw-v1").rglob("*")
        if path.is_file()
    }
    assert downloaded == {
        "experiment.json",
        "inputs.jsonl",
        "models/model-a.json",
        "models/model-b.json",
        "models/model-c.json",
        "responses/responses_model-c.jsonl",
    }


def test_a_second_job_on_the_same_model_stops_instead_of_overwriting(
    bench: Bench,
) -> None:
    """While job 2 is running, job 1 finishes the same model and uploads first."""

    bench.create(dest=EXPERIMENT)
    first_job_done = []

    def finish_job_1_first(_model, _input) -> None:
        if not first_job_done:
            first_job_done.append(True)
            bench.models.before_call = None
            bench.run("model-a", experiment=EXPERIMENT, job="job-1", chunk_size=None)
            bench.models.before_call = finish_job_1_first
            bench.models.fail("open-1")  # job 2's results now differ from job 1's

    bench.models.before_call = finish_job_1_first

    with pytest.raises(RunSyncConflictError, match="changed since it was pulled"):
        bench.run("model-a", experiment=EXPERIMENT, job="job-2", chunk_size=None)


def test_a_crashed_job_is_resumed_by_launching_it_again(bench: Bench) -> None:
    """A new container starts from what the crashed one uploaded."""

    bench.create(dest=EXPERIMENT)
    bench.models.crash("open-1")
    with pytest.raises(RuntimeError):
        bench.run("model-a", experiment=EXPERIMENT, job="crashed", chunk_size=1)

    bench.models.recover("open-1")
    bench.models.forget_calls()
    bench.run("model-a", experiment=EXPERIMENT, job="relaunched", chunk_size=1)

    assert bench.models.called() == ["open-1"]


def test_a_dry_run_never_writes_to_gcs(bench: Bench) -> None:
    bench.create(dest=EXPERIMENT)
    uploads_before = list(bench.storage.upload_preconditions)

    plan = bench.plan("model-a", experiment=EXPERIMENT)

    assert plan.to_run == 3
    assert bench.storage.upload_preconditions == uploads_before


def test_a_local_experiment_can_be_continued_on_gcs(bench: Bench) -> None:
    """Results already made locally come along and are reused on GCS."""

    bench.create()
    bench.run("model-a")
    upload_experiment(bench.folder(), EXPERIMENT, storage_client=bench.storage)
    bench.models.forget_calls()

    bench.run("model-a", "model-b", experiment=EXPERIMENT, job="cloud")

    assert {model for model, _, _ in bench.models.calls} == {"model-b"}


def test_local_images_must_be_copied_into_an_experiment_on_gcs(bench: Bench) -> None:
    """Cloud Run jobs can't read a local disk."""

    (bench.root / "figure.png").write_bytes(b"pixels")
    inputs = [
        {
            "benchmark_item_id": "visual-1",
            "prompt": "Which shape is shown?",
            "images": [{"uri": "figure.png", "detail": "high"}],
        }
    ]

    with pytest.raises(ConfigError, match="--copy-images"):
        bench.create(inputs, dest=EXPERIMENT)
    bench.create(inputs, dest=EXPERIMENT, copy_images=True)


def test_a_run_config_on_gcs_runs_an_experiment_on_gcs(bench: Bench) -> None:
    """A config read from GCS is meant for Cloud Run, which can't read local paths."""

    with pytest.raises(ConfigError, match="upload it with fab-bench"):
        parse_run_config(
            {"experiment": "experiments/sw-v1", "models": ["model-a"]},
            base_dir=None,
            source="gs://bucket/run-configs/abc.json",
        )
