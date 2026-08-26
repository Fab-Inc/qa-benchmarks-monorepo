from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import patch

import google_crc32c
import pytest
from fab_benchmark_publisher import RunSync, RunSyncConflictError, parse_gs_uri
from google.api_core.exceptions import NotFound, PreconditionFailed


class FakeListedBlob:
    def __init__(self, name: str, generation: int, content: bytes) -> None:
        self.name = name
        self.generation = generation
        checksum = google_crc32c.Checksum(content)
        self.crc32c = base64.b64encode(checksum.digest()).decode("ascii")


class FakeBlob:
    def __init__(self, client: FakeStorageClient, bucket_name: str, name: str) -> None:
        self.client = client
        self.bucket_name = bucket_name
        self.name = name
        self.generation: int | None = None

    def reload(self) -> None:
        try:
            self.generation = self.client.objects[self.bucket_name][self.name][0]
        except KeyError as e:
            raise NotFound(self.name) from e

    def download_to_filename(
        self,
        filename: str,
        *,
        if_generation_match: int,
    ) -> None:
        generation, content = self.client.objects[self.bucket_name][self.name]
        if generation != if_generation_match:
            raise PreconditionFailed(self.name)
        Path(filename).write_bytes(content)

    def upload_from_filename(
        self,
        filename: str,
        *,
        if_generation_match: int,
    ) -> None:
        self.client.upload_preconditions.append((self.name, if_generation_match))
        if self.name in self.client.fail_next_upload:
            self.client.fail_next_upload.remove(self.name)
            raise RuntimeError("simulated transient upload failure")
        bucket = self.client.objects.setdefault(self.bucket_name, {})
        current_generation = bucket.get(self.name, (0, b""))[0]
        if current_generation != if_generation_match:
            raise PreconditionFailed(self.name)
        self.generation = self.client.next_generation
        self.client.next_generation += 1
        bucket[self.name] = (self.generation, Path(filename).read_bytes())
        if self.name in self.client.commit_then_fail_next_upload:
            self.client.commit_then_fail_next_upload.remove(self.name)
            raise RuntimeError("simulated lost upload acknowledgement")


class FakeBucket:
    def __init__(self, client: FakeStorageClient, name: str) -> None:
        self.client = client
        self.name = name

    def blob(self, name: str) -> FakeBlob:
        return FakeBlob(self.client, self.name, name)


class FakeStorageClient:
    def __init__(self) -> None:
        self.objects: dict[str, dict[str, tuple[int, bytes]]] = {}
        self.next_generation = 100
        self.upload_preconditions: list[tuple[str, int]] = []
        self.fail_next_upload: set[str] = set()
        self.commit_then_fail_next_upload: set[str] = set()

    def bucket(self, name: str) -> FakeBucket:
        return FakeBucket(self, name)

    def list_blobs(self, bucket_name: str, *, prefix: str) -> list[FakeListedBlob]:
        return [
            FakeListedBlob(name, generation, content)
            for name, (generation, content) in self.objects.get(bucket_name, {}).items()
            if name.startswith(prefix)
        ]


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


def test_pull_and_push_preserve_paths_and_use_seen_generation(tmp_path: Path) -> None:
    client = FakeStorageClient()
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


def test_pull_ignores_directory_placeholder_objects(tmp_path: Path) -> None:
    client = FakeStorageClient()
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


def test_run_sync_constructs_one_storage_client(tmp_path: Path) -> None:
    client = FakeStorageClient()
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


def test_new_objects_use_does_not_exist_precondition(tmp_path: Path) -> None:
    client = FakeStorageClient()
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


def test_generation_change_raises_conflict(tmp_path: Path) -> None:
    client = FakeStorageClient()
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


def test_final_reconciliation_retries_failed_checkpoint(tmp_path: Path) -> None:
    client = FakeStorageClient()
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
    tmp_path: Path,
) -> None:
    client = FakeStorageClient()
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


def test_push_rejects_artifacts_outside_run_directory(tmp_path: Path) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    sync = RunSync(
        tmp_path / "run-a",
        "gs://results/phonics/runs/run-a/",
        storage_client=FakeStorageClient(),
    )

    with pytest.raises(ValueError, match="inside run directory"):
        sync.push_file(outside)
