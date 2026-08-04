"""
Integration tests — require real GCS credentials.
Run with: uv run pytest -m integration
"""

import os

import pytest
from google.cloud.storage import Client


@pytest.mark.integration
def test_gcs_bucket_accessible() -> None:
    if not os.environ.get("BUCKET_NAME"):
        pytest.skip("BUCKET_NAME not set — fill in benchmarks/phonics_benchmark/.env")
    client = Client()
    bucket = client.bucket(os.environ["BUCKET_NAME"])
    blobs = list(bucket.list_blobs(max_results=1))
    assert isinstance(blobs, list)
