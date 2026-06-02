import json
from pathlib import Path

from fab_benchmark_publisher import LeaderboardRecord, publisher


class FakeBlob:
    def __init__(self) -> None:
        self.uploaded_string: str | None = None
        self.uploaded_filename: str | None = None
        self.content_type: str | None = None

    def upload_from_string(self, payload: str, *, content_type: str) -> None:
        self.uploaded_string = payload
        self.content_type = content_type

    def upload_from_filename(self, filename: str) -> None:
        self.uploaded_filename = filename


class FakeBucket:
    def __init__(self) -> None:
        self.blobs: dict[str, FakeBlob] = {}

    def blob(self, file_name: str) -> FakeBlob:
        blob = FakeBlob()
        self.blobs[file_name] = blob
        return blob


class FakeStorageClient:
    def __init__(self) -> None:
        self.buckets: dict[str, FakeBucket] = {}

    def bucket(self, bucket_name: str) -> FakeBucket:
        bucket = FakeBucket()
        self.buckets[bucket_name] = bucket
        return bucket


def test_upload_json_serializes_payload_and_uploads_to_selected_bucket(
    monkeypatch,
) -> None:
    import google.cloud.storage

    client = FakeStorageClient()
    monkeypatch.setattr(google.cloud.storage, "Client", lambda: client)

    file_name = publisher.upload_json(
        {"model": "example", "accuracy": 91.5},
        bucket_name="results-bucket",
        file_name="dev/PHONICS/leaderboard.json",
    )

    blob = client.buckets["results-bucket"].blobs["dev/PHONICS/leaderboard.json"]
    assert file_name == "dev/PHONICS/leaderboard.json"
    assert json.loads(blob.uploaded_string) == {
        "model": "example",
        "accuracy": 91.5,
    }
    assert blob.content_type == "application/json"


def test_upload_json_uses_bucket_environment_and_preserves_string_payload(
    monkeypatch,
) -> None:
    import google.cloud.storage

    client = FakeStorageClient()
    monkeypatch.setattr(google.cloud.storage, "Client", lambda: client)
    monkeypatch.setenv("BUCKET_NAME", "env-bucket")

    publisher.upload_json(
        "[1, 2, 3]",
        file_name="dev/PHONICS/leaderboard.json",
    )

    blob = client.buckets["env-bucket"].blobs["dev/PHONICS/leaderboard.json"]
    assert blob.uploaded_string == "[1, 2, 3]"


def test_upload_file_uploads_source_path(monkeypatch, tmp_path: Path) -> None:
    import google.cloud.storage

    client = FakeStorageClient()
    source_path = tmp_path / "leaderboard.json"
    source_path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(google.cloud.storage, "Client", lambda: client)

    file_name = publisher.upload_file(
        source_path=source_path,
        bucket_name="results-bucket",
        file_name="dev/PHONICS/leaderboard.json",
    )

    blob = client.buckets["results-bucket"].blobs["dev/PHONICS/leaderboard.json"]
    assert file_name == "dev/PHONICS/leaderboard.json"
    assert blob.uploaded_filename == str(source_path)


def test_publish_records_builds_blob_name_and_uploads_payload(monkeypatch) -> None:
    calls = []

    def fake_upload_json(data: str, *, bucket_name: str, file_name: str) -> None:
        calls.append((json.loads(data), bucket_name, file_name))

    monkeypatch.setattr(publisher, "upload_json", fake_upload_json)

    blob_name = publisher.publish_records(
        benchmark_domain="PHONICS",
        records=[
            LeaderboardRecord(
                benchmark_domain="PHONICS",
                model="example-model",
                accuracy=88.0,
            )
        ],
        artifact_name="leaderboard.json",
        bucket_name="results-bucket",
        file_prefix="dev",
    )

    assert blob_name == "dev/PHONICS/leaderboard.json"
    assert calls == [
        (
            [
                {
                    "benchmark_domain": "PHONICS",
                    "model": "example-model",
                    "accuracy": 88.0,
                }
            ],
            "results-bucket",
            "dev/PHONICS/leaderboard.json",
        )
    ]


def test_write_and_publish_records_writes_file_before_uploading(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls = []

    def fake_upload_file(
        *,
        source_path: Path,
        bucket_name: str,
        file_name: str,
    ) -> None:
        calls.append((source_path, bucket_name, file_name))

    monkeypatch.setattr(publisher, "upload_file", fake_upload_file)

    blob_name = publisher.write_and_publish_records(
        benchmark_domain="PHONICS",
        records=[
            LeaderboardRecord(
                benchmark_domain="PHONICS",
                model="example-model",
                accuracy=88.0,
            )
        ],
        artifact_name="leaderboard.json",
        output_dir=tmp_path,
        bucket_name="results-bucket",
        file_prefix="dev",
    )

    output_path = tmp_path / "leaderboard.json"
    assert blob_name == "dev/PHONICS/leaderboard.json"
    assert json.loads(output_path.read_text(encoding="utf-8")) == [
        {
            "benchmark_domain": "PHONICS",
            "model": "example-model",
            "accuracy": 88.0,
        }
    ]
    assert calls == [(output_path, "results-bucket", "dev/PHONICS/leaderboard.json")]
