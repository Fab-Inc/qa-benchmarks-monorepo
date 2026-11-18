"""Generation-safe mirroring of a run directory to GCS (copied from the publisher)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from fab_benchmark_runner.gcs import parse_gs_uri
from fab_benchmark_runner.run_sync import RunSync, RunSyncConflictError


@pytest.mark.parametrize(
    "uri",
    ["https://bucket/path", "gs://bucket", "gs:///path", "gs://bucket/"],
)
def test_parse_gs_uri_rejects_invalid_values(uri: str) -> None:
    with pytest.raises(ValueError):
        parse_gs_uri(uri)


def test_parse_gs_uri_returns_bucket_and_object() -> None:
    assert parse_gs_uri("gs://results/phonics/runs/run-a/") == (
        "results",
        "phonics/runs/run-a/",
    )


def test_pull_and_push_preserve_paths_and_use_seen_generation(
    tmp_path: Path, storage
) -> None:
    client = storage
    client.objects["results"] = {
        "phonics/runs/run-a/config.json": (7, b'{"config_id":"old"}\n'),
        "phonics/runs/run-a/responses/responses_model-a.jsonl": (8, b"old\n"),
    }
    run_dir = tmp_path / "run-a"
    sync = RunSync(
        run_dir,
        "gs://results/phonics/runs/run-a/",
        storage_client=client,
    )

    assert sync.pull() == 2
    assert (run_dir / "config.json").read_bytes() == b'{"config_id":"old"}\n'
    response_path = run_dir / "responses" / "responses_model-a.jsonl"
    response_path.write_bytes(b"complete\n")
    sync.push_file(response_path)

    assert client.upload_preconditions[-1] == (
        "phonics/runs/run-a/responses/responses_model-a.jsonl",
        8,
    )
    assert (
        client.objects["results"][
            "phonics/runs/run-a/responses/responses_model-a.jsonl"
        ][1]
        == b"complete\n"
    )
    assert sync.push_missing_or_stale() == 0


def test_pull_ignores_directory_placeholder_objects(tmp_path: Path, storage) -> None:
    client = storage
    client.objects["results"] = {
        "phonics/runs/run-a/responses/": (7, b""),
        "phonics/runs/run-a/responses/responses_model-a.jsonl": (8, b"row\n"),
    }
    run_dir = tmp_path / "run-a"
    sync = RunSync(
        run_dir,
        "gs://results/phonics/runs/run-a/",
        storage_client=client,
    )

    assert sync.pull() == 1
    assert (run_dir / "responses").is_dir()
    assert (run_dir / "responses" / "responses_model-a.jsonl").read_bytes() == (
        b"row\n"
    )


def test_run_sync_constructs_one_storage_client(tmp_path: Path, storage) -> None:
    client = storage
    artifact = tmp_path / "run-a" / "config.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n", encoding="utf-8")

    with patch("google.cloud.storage.Client", return_value=client) as client_type:
        sync = RunSync(
            artifact.parent,
            "gs://results/phonics/runs/run-a/",
        )
        assert sync.pull() == 0
        sync.push_file(artifact)
        assert sync.push_missing_or_stale() == 0

    client_type.assert_called_once_with()


def test_new_objects_use_does_not_exist_precondition(tmp_path: Path, storage) -> None:
    client = storage
    artifact = tmp_path / "run-a" / "config.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n", encoding="utf-8")
    sync = RunSync(
        artifact.parent,
        "gs://results/phonics/runs/run-a/",
        storage_client=client,
    )

    sync.push_file(artifact)

    assert client.upload_preconditions == [("phonics/runs/run-a/config.json", 0)]


def test_generation_change_raises_conflict(tmp_path: Path, storage) -> None:
    client = storage
    object_name = "phonics/runs/run-a/config.json"
    client.objects["results"] = {object_name: (7, b"old\n")}
    sync = RunSync(
        tmp_path / "run-a",
        "gs://results/phonics/runs/run-a/",
        storage_client=client,
    )
    sync.pull()
    local_config = tmp_path / "run-a" / "config.json"
    local_config.write_text("new\n", encoding="utf-8")
    client.objects["results"][object_name] = (8, b"winner\n")

    with pytest.raises(RunSyncConflictError, match="config.json"):
        sync.push_file(local_config)

    assert client.objects["results"][object_name] == (8, b"winner\n")


def test_final_reconciliation_retries_failed_checkpoint(
    tmp_path: Path, storage
) -> None:
    client = storage
    response_path = tmp_path / "run-a" / "responses" / "responses_model-a.jsonl"
    response_path.parent.mkdir(parents=True)
    response_path.write_bytes(b"complete\n")
    object_name = "phonics/runs/run-a/responses/responses_model-a.jsonl"
    client.fail_next_upload.add(object_name)
    sync = RunSync(
        tmp_path / "run-a",
        "gs://results/phonics/runs/run-a/",
        storage_client=client,
    )

    with pytest.raises(RuntimeError, match="transient"):
        sync.push_file(response_path)

    assert sync.push_missing_or_stale() == 1
    assert client.upload_preconditions == [(object_name, 0), (object_name, 0)]
    assert client.objects["results"][object_name][1] == b"complete\n"


def test_next_checkpoint_recovers_committed_upload_with_lost_ack(
    tmp_path: Path, storage
) -> None:
    client = storage
    response_path = tmp_path / "run-a" / "responses" / "responses_model-a.jsonl"
    response_path.parent.mkdir(parents=True)
    response_path.write_bytes(b"initial\n")
    object_name = "phonics/runs/run-a/responses/responses_model-a.jsonl"
    client.objects["results"] = {object_name: (7, b"initial\n")}
    sync = RunSync(
        tmp_path / "run-a",
        "gs://results/phonics/runs/run-a/",
        storage_client=client,
    )
    sync.pull()
    response_path.write_bytes(b"first checkpoint\n")
    client.commit_then_fail_next_upload.add(object_name)

    with pytest.raises(RuntimeError, match="lost upload acknowledgement"):
        sync.push_file(response_path)

    response_path.write_bytes(b"second checkpoint\n")
    sync.push_file(response_path)

    assert client.upload_preconditions == [
        (object_name, 7),
        (object_name, 7),
        (object_name, 100),
    ]
    assert client.objects["results"][object_name] == (101, b"second checkpoint\n")


def test_push_rejects_artifacts_outside_run_directory(tmp_path: Path, storage) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    sync = RunSync(
        tmp_path / "run-a",
        "gs://results/phonics/runs/run-a/",
        storage_client=storage,
    )

    with pytest.raises(ValueError, match="inside run directory"):
        sync.push_file(outside)


def test_push_pending_retries_only_this_jobs_unconfirmed_files(
    tmp_path: Path, storage
) -> None:
    prefix = "gs://results/phonics/runs/run-a/"
    shared = "phonics/runs/run-a/responses/responses_model-a.jsonl"
    storage.objects["results"] = {shared: (7, b"a v1\n")}
    job_a = RunSync(tmp_path / "a", prefix, storage_client=storage)
    job_b = RunSync(tmp_path / "b", prefix, storage_client=storage)
    job_a.pull()
    job_b.pull()

    a_file = job_a.run_dir / "responses" / "responses_model-a.jsonl"
    a_file.write_bytes(b"a v2\n")
    job_a.push_file(a_file)
    b_file = job_b.run_dir / "responses" / "responses_model-b.jsonl"
    b_file.write_bytes(b"b v1\n")
    storage.fail_next_upload.add("phonics/runs/run-a/responses/responses_model-b.jsonl")
    with pytest.raises(RuntimeError, match="transient"):
        job_b.push_file(b_file)

    assert job_b.push_pending() == 1
    assert job_b.push_pending() == 0
    assert storage.objects["results"][shared] == (100, b"a v2\n")
    assert storage.objects["results"][
        "phonics/runs/run-a/responses/responses_model-b.jsonl"
    ][1] == (b"b v1\n")


def test_two_jobs_creating_the_same_file_with_equal_bytes_agree(
    tmp_path: Path, storage
) -> None:
    prefix = "gs://results/phonics/runs/run-a/"
    jobs = [RunSync(tmp_path / name, prefix, storage_client=storage) for name in "ab"]
    for job in jobs:
        job.pull()
        record = job.run_dir / "config.json"
        record.parent.mkdir(parents=True)
        record.write_bytes(b'{"config_id": "x"}\n')

    for job in jobs:
        job.push_file(job.run_dir / "config.json")
        assert job.push_pending() == 0

    assert storage.upload_preconditions == [
        ("phonics/runs/run-a/config.json", 0),
        ("phonics/runs/run-a/config.json", 0),
    ]


def test_two_jobs_creating_the_same_file_with_different_bytes_conflict(
    tmp_path: Path, storage
) -> None:
    prefix = "gs://results/phonics/runs/run-a/"
    jobs = [RunSync(tmp_path / name, prefix, storage_client=storage) for name in "ab"]
    for job, content in zip(jobs, (b'{"config_id": "x"}\n', b'{"config_id": "y"}\n')):
        job.pull()
        record = job.run_dir / "config.json"
        record.parent.mkdir(parents=True)
        record.write_bytes(content)

    jobs[0].push_file(jobs[0].run_dir / "config.json")
    with pytest.raises(RunSyncConflictError, match="config.json"):
        jobs[1].push_file(jobs[1].run_dir / "config.json")
