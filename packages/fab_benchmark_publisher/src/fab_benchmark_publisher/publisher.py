import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fab_benchmark_publisher.contracts import LeaderboardRecord
from fab_benchmark_publisher.results import results_to_json, write_results_json


@dataclass(frozen=True)
class GCSObject:
    """The object metadata needed for generation-aware synchronization."""

    name: str
    generation: int
    crc32c: str | None = None


def parse_gs_uri(uri: str) -> tuple[str, str]:
    """Split a ``gs://bucket/object`` URI into its bucket and object name."""

    if not isinstance(uri, str) or not uri.startswith("gs://"):
        raise ValueError("GCS URI must start with 'gs://'")
    remainder = uri.removeprefix("gs://")
    bucket_name, separator, object_name = remainder.partition("/")
    if (
        not separator
        or not bucket_name
        or not object_name
        or bucket_name.strip() != bucket_name
        or object_name.strip() != object_name
        or "?" in bucket_name
        or "#" in bucket_name
        or "?" in object_name
        or "#" in object_name
    ):
        raise ValueError("GCS URI must have the form 'gs://bucket/path'")
    return bucket_name, object_name


def list_gcs_objects(
    *,
    bucket_name: str,
    prefix: str,
    storage_client: Any | None = None,
) -> list[GCSObject]:
    """List object names and generations beneath a GCS prefix."""

    if storage_client is None:
        from google.cloud.storage import Client

        storage_client = Client()
    return sorted(
        (
            GCSObject(
                name=blob.name,
                generation=int(blob.generation),
                crc32c=getattr(blob, "crc32c", None),
            )
            for blob in storage_client.list_blobs(bucket_name, prefix=prefix)
        ),
        key=lambda item: item.name,
    )


def download_file_with_generation(
    *,
    bucket_name: str,
    file_name: str,
    write_path: str | Path,
    storage_client: Any | None = None,
) -> int:
    """Download one authenticated object and return its current generation."""

    if storage_client is None:
        from google.cloud.storage import Client

        storage_client = Client()
    blob = storage_client.bucket(bucket_name).blob(file_name)
    blob.reload()
    generation = int(blob.generation)
    destination = Path(write_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(
        str(destination),
        if_generation_match=generation,
    )
    return generation


def upload_file_with_generation(
    *,
    source_path: str | Path,
    bucket_name: str,
    file_name: str,
    if_generation_match: int,
    storage_client: Any | None = None,
) -> int:
    """Conditionally upload one file and return the resulting generation."""

    if storage_client is None:
        from google.cloud.storage import Client

        storage_client = Client()
    blob = storage_client.bucket(bucket_name).blob(file_name)
    blob.upload_from_filename(
        str(source_path),
        if_generation_match=if_generation_match,
    )
    if blob.generation is None:  # pragma: no cover - defensive client fallback
        blob.reload()
    return int(blob.generation)


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
        client = Client.create_anonymous_client()
    bucket = client.bucket(selected_bucket)
    blob = bucket.blob(file_name)
    blob.download_to_filename(write_path)


# Config every benchmark shares, published to a public bucket rather than committed
# here so the bucket is the single source of truth and no checkout can drift from it.
# The consequence is that a fresh checkout has none of them.
SHARED_CONFIG_BUCKET = "fab-open-configs"

# Every object ensure_shared_config is allowed to fetch, by its basename. Without this
# guard a mistyped --models-csv would send its typo to the bucket as an object name,
# and a test passing a deliberately-absent path would make a real network call.
SHARED_CONFIG_NAMES = frozenset({"custom_models.yaml", "models.csv", "providers.csv"})


def ensure_shared_config(path: str | Path) -> Path:
    """Return `path`, fetching it from the shared-config bucket if missing or stale.

    Raises FileNotFoundError for a path that is not a shared config -- that is a typo
    or an intentionally-local file, and neither can be downloaded.
    """
    destination = Path(path)
    if destination.name not in SHARED_CONFIG_NAMES:
        if destination.is_file():
            return destination
        raise FileNotFoundError(
            f"{destination} not found, and is not one of the shared config files "
            f"({', '.join(sorted(SHARED_CONFIG_NAMES))}) that can be fetched"
        )
    present = destination.is_file()
    if present:
        # Existence alone is not enough: these files gain models and providers over
        # time, and a stale copy makes new models silently vanish from a published
        # bundle rather than fail visibly. Compare checksums -- one metadata call, no
        # download -- and re-fetch only on a mismatch. If the bucket is unreachable,
        # carry on with the local copy: being unable to confirm freshness is a reason
        # to say so, not a reason to refuse to work offline.
        from google.cloud.storage import Client

        from fab_benchmark_publisher.run_sync import _file_crc32c

        try:
            blob = (
                Client.create_anonymous_client()
                .bucket(SHARED_CONFIG_BUCKET)
                .blob(destination.name)
            )
            blob.reload()
            remote = blob.crc32c
            # None means the bucket reported no checksum: "cannot tell", not
            # "differs" -- re-downloading on that would be pure waste.
            if remote is None or remote == _file_crc32c(destination):
                return destination
        except Exception as e:
            print(f"  ({destination.name}: cannot check the bucket -- {e})")
            print("   using the local copy, which may be out of date")
            return destination
    verb = "Refreshing" if present else "Fetching"
    print(f"{verb} {destination.name} from gs://{SHARED_CONFIG_BUCKET}/ ...")
    destination.parent.mkdir(parents=True, exist_ok=True)
    download_file(
        bucket_name=SHARED_CONFIG_BUCKET,
        file_name=destination.name,
        write_path=destination,
        require_auth=False,
    )
    return destination


# The leaderboard's global model and provider catalogue: every provider's colour and
# logo, and every model's display name, price, size and release date, keyed by the same
# positional ids the bundle's `data` rows carry. It is shared by every benchmark cluster
# on the site and published from an Azure container we do not own, so nothing here
# builds or uploads it -- we only read it, to check that the models we publish can
# actually be resolved against it.
AZURE_PROVIDERMETA_URL = (
    "https://fabdatastorage.blob.core.windows.net/ai-for-ed-public/providermeta.json"
)


def fetch_providermeta(url: str = AZURE_PROVIDERMETA_URL) -> dict:
    """Download and parse the leaderboard's providermeta.json from Azure.

    The container is public, so no credential is needed. AZURE_SAS_TOKEN is used when
    set, for the day it stops being public -- no code change needed then.
    """
    from azure.storage.blob import BlobClient

    blob = BlobClient.from_blob_url(
        url, credential=os.environ.get("AZURE_SAS_TOKEN") or None
    )
    return json.loads(blob.download_blob().readall())


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
