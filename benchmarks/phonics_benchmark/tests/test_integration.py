"""
Integration tests — require real GCS credentials.
Run with: uv run pytest -m integration
"""

import os
from pathlib import Path
from uuid import uuid4

import pytest
from fab_benchmark_publisher import RunSync, RunSyncConflictError
from google.cloud.storage import Client


@pytest.mark.integration
def test_gcs_bucket_accessible() -> None:
    if not os.environ.get("BUCKET_NAME"):
        pytest.skip("BUCKET_NAME not set — fill in benchmarks/phonics_benchmark/.env")
    client = Client()
    bucket = client.bucket(os.environ["BUCKET_NAME"])
    blobs = list(bucket.list_blobs(max_results=1))
    assert isinstance(blobs, list)


@pytest.mark.integration
def test_run_sync_round_trip_and_generation_conflict(tmp_path: Path) -> None:
    bucket_name = os.environ.get("BUCKET_NAME")
    if not bucket_name:
        pytest.skip("BUCKET_NAME not set — fill in benchmarks/phonics_benchmark/.env")

    client = Client()
    prefix = f"integration/phonics-run-sync/{uuid4().hex}/"
    dest = f"gs://{bucket_name}/{prefix}"
    first_dir = tmp_path / "first"
    first_dir.mkdir()
    first_config = first_dir / "config.json"
    first_config.write_text('{"attempt":1}\n', encoding="utf-8")
    second_dir = tmp_path / "second"

    try:
        first_sync = RunSync(first_dir, dest, storage_client=client)
        first_sync.push_file(first_config)

        second_sync = RunSync(second_dir, dest, storage_client=client)
        assert second_sync.pull() == 1
        assert (second_dir / "config.json").read_text(encoding="utf-8") == (
            '{"attempt":1}\n'
        )

        first_config.write_text('{"attempt":2}\n', encoding="utf-8")
        first_sync.push_file(first_config)
        second_config = second_dir / "config.json"
        second_config.write_text('{"attempt":"stale"}\n', encoding="utf-8")
        with pytest.raises(RunSyncConflictError):
            second_sync.push_file(second_config)
    finally:
        for blob in client.list_blobs(bucket_name, prefix=prefix):
            blob.delete()
