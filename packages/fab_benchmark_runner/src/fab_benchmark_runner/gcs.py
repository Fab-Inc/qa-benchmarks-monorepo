"""Google Cloud Storage helpers used by the run lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


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


def create_storage_client(*, anonymous: bool = False) -> Any:
    """Create a Cloud Storage client, anonymous for public buckets."""

    from google.cloud.storage import Client

    return Client.create_anonymous_client() if anonymous else Client()


def download_file(
    *,
    bucket_name: str,
    file_name: str,
    write_path: str | Path,
    require_auth: bool = True,
) -> None:
    """Download one object to a local path."""

    client = create_storage_client(anonymous=not require_auth)
    client.bucket(bucket_name).blob(file_name).download_to_filename(str(write_path))


def list_gcs_objects(
    *,
    bucket_name: str,
    prefix: str,
    storage_client: Any | None = None,
) -> list[GCSObject]:
    """List object names and generations beneath a GCS prefix."""

    client = storage_client or _default_client()
    return sorted(
        (
            GCSObject(
                name=blob.name,
                generation=int(blob.generation),
                crc32c=getattr(blob, "crc32c", None),
            )
            for blob in client.list_blobs(bucket_name, prefix=prefix)
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

    client = storage_client or _default_client()
    blob = client.bucket(bucket_name).blob(file_name)
    blob.reload()
    generation = int(blob.generation)
    destination = Path(write_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(destination), if_generation_match=generation)
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

    client = storage_client or _default_client()
    blob = client.bucket(bucket_name).blob(file_name)
    blob.upload_from_filename(str(source_path), if_generation_match=if_generation_match)
    if blob.generation is None:  # pragma: no cover - defensive client fallback
        blob.reload()
    return int(blob.generation)


def _default_client() -> Any:
    return create_storage_client()
