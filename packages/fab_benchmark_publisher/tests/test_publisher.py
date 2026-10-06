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


# --------------------------------------------------------------------------- #
# fetch_providermeta
# --------------------------------------------------------------------------- #
def fake_azure_blob(monkeypatch, payload: bytes):
    """Stand in for the Azure container, recording how the client was built."""
    import azure.storage.blob

    calls = {}

    class FakeDownload:
        def readall(self):
            return payload

    class FakeBlobClient:
        @staticmethod
        def from_blob_url(url, credential=None):
            calls["url"] = url
            calls["credential"] = credential
            return FakeBlobClient()

        def download_blob(self):
            return FakeDownload()

    monkeypatch.setattr(azure.storage.blob, "BlobClient", FakeBlobClient)
    return calls


def test_fetch_providermeta_reads_the_public_container_without_a_credential(
    monkeypatch,
) -> None:
    """The container is public, so an unset AZURE_SAS_TOKEN must mean 'anonymous',
    not an empty-string credential the SDK would try to use."""
    monkeypatch.delenv("AZURE_SAS_TOKEN", raising=False)
    calls = fake_azure_blob(monkeypatch, b'{"0": {"metadata": {}, "models": {}}}')

    assert publisher.fetch_providermeta() == {"0": {"metadata": {}, "models": {}}}
    assert calls["url"] == publisher.AZURE_PROVIDERMETA_URL
    assert calls["credential"] is None


def test_fetch_providermeta_passes_a_sas_token_when_one_is_set(monkeypatch) -> None:
    """For the day the container stops being public -- no code change needed then."""
    monkeypatch.setenv("AZURE_SAS_TOKEN", "sv=2026&sig=abc")
    calls = fake_azure_blob(monkeypatch, b"{}")

    publisher.fetch_providermeta()

    assert calls["credential"] == "sv=2026&sig=abc"


def test_fetch_providermeta_treats_an_empty_sas_token_as_unset(monkeypatch) -> None:
    """An AZURE_SAS_TOKEN= line left blank in .env must not become a credential."""
    monkeypatch.setenv("AZURE_SAS_TOKEN", "")
    calls = fake_azure_blob(monkeypatch, b"{}")

    publisher.fetch_providermeta()

    assert calls["credential"] is None


# --------------------------------------------------------------------------- #
# ensure_shared_config
# --------------------------------------------------------------------------- #
def fake_bucket_holding(crc32c, monkeypatch, *, raises=None):
    """Stand in for the shared-config bucket, reporting `crc32c` for any blob."""
    import google.cloud.storage

    class FakeConfigBlob:
        def __init__(self) -> None:
            self.crc32c = crc32c

        def reload(self) -> None:
            if raises is not None:
                raise raises

    class FakeClient:
        @staticmethod
        def create_anonymous_client():
            return FakeClient()

        def bucket(self, _name):
            return self

        def blob(self, _name):
            return FakeConfigBlob()

    monkeypatch.setattr(google.cloud.storage, "Client", FakeClient)


def test_ensure_shared_config_keeps_a_local_file_matching_the_bucket(
    tmp_path, monkeypatch
) -> None:
    """The common case: one metadata call, no download."""
    path = tmp_path / "models.csv"
    path.write_text("model_id,provider\nm1,OpenCo\n", encoding="utf-8")
    from fab_benchmark_publisher.run_sync import _file_crc32c

    fake_bucket_holding(_file_crc32c(path), monkeypatch)

    def fail(**kwargs):  # pragma: no cover - asserted never to run
        raise AssertionError("must not download a file that matches the bucket")

    monkeypatch.setattr(publisher, "download_file", fail)

    assert publisher.ensure_shared_config(path) == path


def test_ensure_shared_config_refetches_a_local_file_that_differs(
    tmp_path, monkeypatch, capsys
) -> None:
    """A stale copy must be replaced: these files gain models over time, and keeping
    an old one drops the new models from a bundle with no visible failure."""
    path = tmp_path / "models.csv"
    path.write_text("model_id,provider\nold,OpenCo\n", encoding="utf-8")
    fake_bucket_holding("a-different-checksum", monkeypatch)

    def fake_download(*, bucket_name, file_name, write_path, require_auth):
        Path(write_path).write_text("model_id,provider\nnew,OpenCo\n", encoding="utf-8")

    monkeypatch.setattr(publisher, "download_file", fake_download)

    assert publisher.ensure_shared_config(path) == path
    assert "new,OpenCo" in path.read_text(encoding="utf-8")
    assert "Refreshing models.csv" in capsys.readouterr().out


def test_ensure_shared_config_keeps_the_local_file_when_the_bucket_is_unreachable(
    tmp_path, monkeypatch, capsys
) -> None:
    """Offline must still work: not being able to confirm freshness is a reason to say
    so, not a reason to refuse to build anything."""
    path = tmp_path / "models.csv"
    path.write_text("model_id,provider\nlocal,OpenCo\n", encoding="utf-8")
    fake_bucket_holding(None, monkeypatch, raises=OSError("no network"))

    def fail(**kwargs):  # pragma: no cover - asserted never to run
        raise AssertionError("must not download when the bucket is unreachable")

    monkeypatch.setattr(publisher, "download_file", fail)

    assert publisher.ensure_shared_config(path) == path
    assert "local,OpenCo" in path.read_text(encoding="utf-8")
    out = capsys.readouterr().out
    assert "cannot check the bucket" in out
    assert "may be out of date" in out


def test_ensure_shared_config_keeps_a_local_file_the_bucket_cannot_checksum(
    tmp_path, monkeypatch
) -> None:
    """No reported checksum means "cannot tell", which must not be read as "differs"
    and trigger a pointless re-download."""
    path = tmp_path / "models.csv"
    path.write_text("model_id,provider\nm1,OpenCo\n", encoding="utf-8")
    fake_bucket_holding(None, monkeypatch)

    def fail(**kwargs):  # pragma: no cover - asserted never to run
        raise AssertionError("must not download when the bucket reports no checksum")

    monkeypatch.setattr(publisher, "download_file", fail)

    assert publisher.ensure_shared_config(path) == path


def test_ensure_shared_config_returns_a_present_non_shared_path_untouched(
    tmp_path, monkeypatch
) -> None:
    """A deliberately-local --models-csv must never be checked against the bucket."""
    path = tmp_path / "my-own.csv"
    path.write_text("model_id\nm1\n", encoding="utf-8")
    fake_bucket_holding(None, monkeypatch, raises=AssertionError("must not be reached"))

    assert publisher.ensure_shared_config(path) == path


def test_ensure_shared_config_fetches_a_missing_file(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_download(*, bucket_name, file_name, write_path, require_auth):
        calls.append((bucket_name, file_name, require_auth))
        Path(write_path).write_text("model_id,provider\nm1,OpenCo\n", encoding="utf-8")

    monkeypatch.setattr(publisher, "download_file", fake_download)
    path = tmp_path / "nested" / "models.csv"

    assert publisher.ensure_shared_config(path) == path
    assert path.is_file()
    # Anonymous: the shared config bucket is public, so no credentials are needed.
    assert calls == [(publisher.SHARED_CONFIG_BUCKET, "models.csv", False)]


def test_ensure_shared_config_refuses_a_path_that_is_not_a_shared_config(
    tmp_path, monkeypatch
) -> None:
    """A mistyped --models-csv must not be sent to the bucket as an object name, and a
    deliberately-local path must never trigger a network call."""

    def fail(**kwargs):  # pragma: no cover - asserted never to run
        raise AssertionError("must not download an arbitrary path")

    monkeypatch.setattr(publisher, "download_file", fail)

    try:
        publisher.ensure_shared_config(tmp_path / "typo.csv")
    except FileNotFoundError as e:
        assert "typo.csv" in str(e)
    else:  # pragma: no cover - the call above must raise
        raise AssertionError("expected FileNotFoundError")


def test_ensure_shared_config_covers_every_declared_shared_name(
    tmp_path, monkeypatch
) -> None:
    """Each name in SHARED_CONFIG_NAMES must actually be fetchable, or the guard and
    the bucket have drifted apart."""
    fetched = []

    def fake_download(*, bucket_name, file_name, write_path, require_auth):
        fetched.append(file_name)
        Path(write_path).write_text("x", encoding="utf-8")

    monkeypatch.setattr(publisher, "download_file", fake_download)
    for name in sorted(publisher.SHARED_CONFIG_NAMES):
        publisher.ensure_shared_config(tmp_path / name)

    assert fetched == sorted(publisher.SHARED_CONFIG_NAMES)
