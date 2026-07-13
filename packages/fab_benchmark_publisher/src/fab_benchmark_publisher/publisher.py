import json
import os
from pathlib import Path

from fab_benchmark_publisher.contracts import LeaderboardRecord
from fab_benchmark_publisher.results import results_to_json, write_results_json


def download_file(
    *,
    bucket_name: str | None = None,
    file_name: str,
    write_path: str | Path,
    require_auth: bool = True,
):
    """
    Downloads the file 'file_name'
    from the bucket 'bucket_name'
    to the local path 'write_path'
    """

    from google.cloud.storage import Client

    selected_bucket = bucket_name or os.environ["BUCKET_NAME"]
    if require_auth:
        client = Client()
    else:
        client = Client().create_anonymous_client()
    bucket = client.bucket(selected_bucket)
    blob = bucket.blob(file_name)
    blob.download_to_filename(write_path)


def upload_json(
    data: object,
    *,
    bucket_name: str | None = None,
    file_name: str,
) -> str:
    """Upload JSON data to GCS using the workflow-provided credentials."""

    from google.cloud.storage import Client

    selected_bucket = bucket_name or os.environ["BUCKET_NAME"]
    client = Client()
    bucket = client.bucket(selected_bucket)
    blob = bucket.blob(file_name)
    payload = (
        data
        if isinstance(data, str)
        else json.dumps(data, ensure_ascii=False, indent=2)
    )
    blob.upload_from_string(
        payload,
        content_type="application/json",
    )
    return file_name


def upload_file(
    *,
    source_path: str | Path,
    bucket_name: str | None = None,
    file_name: str,
) -> str:
    """Upload an existing file to GCS using the workflow-provided credentials."""

    from google.cloud.storage import Client

    selected_bucket = bucket_name or os.environ["BUCKET_NAME"]
    client = Client()
    bucket = client.bucket(selected_bucket)
    blob = bucket.blob(file_name)
    blob.upload_from_filename(str(source_path))
    return file_name


def results_blob_name(
    *,
    benchmark_domain: str,
    artifact_name: str,
    file_prefix: str | None = None,
) -> str:
    """Build the GCS object name used for benchmark result payloads."""

    selected_prefix = file_prefix or os.environ["FILE_PREFIX"]
    return f"{selected_prefix}/{benchmark_domain}/{artifact_name}"


def publish_records(
    *,
    benchmark_domain: str,
    records: list[LeaderboardRecord],
    artifact_name: str,
    bucket_name: str | None = None,
    file_prefix: str | None = None,
) -> str:
    """Serialize benchmark records and upload them to GCS."""

    blob_name = results_blob_name(
        benchmark_domain=benchmark_domain,
        artifact_name=artifact_name,
        file_prefix=file_prefix,
    )
    upload_json(results_to_json(records), bucket_name=bucket_name, file_name=blob_name)
    return blob_name


def write_and_publish_records(
    *,
    benchmark_domain: str,
    records: list[LeaderboardRecord],
    artifact_name: str,
    output_dir: str | Path = "data/web",
    bucket_name: str | None = None,
    file_prefix: str | None = None,
) -> str:
    """Write result JSON locally, then upload that file to GCS."""

    output_path = write_results_json(records, Path(output_dir) / artifact_name)
    blob_name = results_blob_name(
        benchmark_domain=benchmark_domain,
        artifact_name=artifact_name,
        file_prefix=file_prefix,
    )
    upload_file(source_path=output_path, bucket_name=bucket_name, file_name=blob_name)
    return blob_name
