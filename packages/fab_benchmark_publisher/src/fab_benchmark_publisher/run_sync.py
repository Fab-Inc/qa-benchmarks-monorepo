"""Generation-safe mirroring between a local run directory and GCS."""

from __future__ import annotations

import base64
import os
import threading
from collections.abc import Collection
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from typing import Any

import google_crc32c
from google.api_core.exceptions import PreconditionFailed

from fab_benchmark_publisher.publisher import (
    download_file_with_generation,
    list_gcs_objects,
    parse_gs_uri,
    upload_file_with_generation,
)


class RunSyncConflictError(RuntimeError):
    """Remote state changed generation, so this process must abort."""


class RunSync:
    """Mirror durable run artifacts while guarding every remote overwrite."""

    def __init__(
        self,
        run_dir: Path,
        dest: str,
        *,
        storage_client: Any | None = None,
    ) -> None:
        self.run_dir = run_dir.expanduser().resolve()
        self.bucket_name, prefix = parse_gs_uri(dest)
        self.prefix = prefix.rstrip("/") + "/"
        if storage_client is None:
            from google.cloud.storage import Client

            storage_client = Client()
        self._storage_client = storage_client
        self._generations: dict[str, int] = {}
        self._pending_crc32c: dict[str, str] = {}
        self._dirty: set[str] = set()
        self._lock = threading.Lock()

    def pull(self, *, relative_paths: Collection[str] | None = None) -> int:
        """Atomically download matching remote run artifacts and record generations."""

        selected_paths = set(relative_paths) if relative_paths is not None else None
        downloaded_count = 0
        for remote_object in list_gcs_objects(
            bucket_name=self.bucket_name,
            prefix=self.prefix,
            storage_client=self._storage_client,
        ):
            relative_name = remote_object.name.removeprefix(self.prefix)
            if (
                not relative_name
                or relative_name.endswith("/")
                or (selected_paths is not None and relative_name not in selected_paths)
            ):
                continue
            local_path = self._local_path(relative_name)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(
                dir=local_path.parent,
                prefix=f".{local_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
            try:
                generation = download_file_with_generation(
                    bucket_name=self.bucket_name,
                    file_name=remote_object.name,
                    write_path=temporary_path,
                    storage_client=self._storage_client,
                )
                os.replace(temporary_path, local_path)
            finally:
                temporary_path.unlink(missing_ok=True)
            with self._lock:
                self._generations[remote_object.name] = generation
                self._pending_crc32c.pop(remote_object.name, None)
                self._dirty.discard(remote_object.name)
            downloaded_count += 1
        return downloaded_count

    def push_file(self, local_path: Path) -> None:
        """Conditionally upload one artifact relative to the local run directory."""

        resolved_path = local_path.expanduser().resolve()
        try:
            relative_name = resolved_path.relative_to(self.run_dir).as_posix()
        except ValueError as e:
            raise ValueError(
                f"Artifact path must be inside run directory {self.run_dir}: "
                f"{resolved_path}"
            ) from e
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Run artifact does not exist: {resolved_path}")

        object_name = self.prefix + relative_name
        local_crc32c = _file_crc32c(resolved_path)
        with self._lock:
            expected_generation = self._generations.get(object_name, 0)
            self._dirty.add(object_name)

        for retry_index in range(2):
            try:
                new_generation = upload_file_with_generation(
                    source_path=resolved_path,
                    bucket_name=self.bucket_name,
                    file_name=object_name,
                    if_generation_match=expected_generation,
                    storage_client=self._storage_client,
                )
            except PreconditionFailed as e:
                remote_matches_current = self._adopt_matching_remote_generation(
                    object_name,
                    local_crc32c=local_crc32c,
                    cause=e,
                )
                if remote_matches_current:
                    return
                if retry_index == 1:  # pragma: no cover - defensive retry bound
                    self._raise_conflict(object_name, e)
                with self._lock:
                    expected_generation = self._generations[object_name]
                continue
            except Exception:
                with self._lock:
                    self._pending_crc32c[object_name] = local_crc32c
                raise
            with self._lock:
                self._generations[object_name] = new_generation
                self._pending_crc32c.pop(object_name, None)
                self._dirty.discard(object_name)
            return

    def push_missing_or_stale(self) -> int:
        """Reconcile new, failed, or remotely changed local run artifacts."""

        remote_generations = {
            remote_object.name: remote_object.generation
            for remote_object in list_gcs_objects(
                bucket_name=self.bucket_name,
                prefix=self.prefix,
                storage_client=self._storage_client,
            )
        }
        pushed_count = 0
        for local_path in sorted(
            path for path in self.run_dir.rglob("*") if path.is_file()
        ):
            object_name = self.prefix + local_path.relative_to(self.run_dir).as_posix()
            with self._lock:
                expected_generation = self._generations.get(object_name, 0)
                is_dirty = object_name in self._dirty
            remote_generation = remote_generations.get(object_name, 0)
            if (
                not is_dirty
                and expected_generation != 0
                and expected_generation == remote_generation
            ):
                continue
            self.push_file(local_path)
            pushed_count += 1
        return pushed_count

    def _adopt_matching_remote_generation(
        self,
        object_name: str,
        *,
        local_crc32c: str,
        cause: PreconditionFailed,
    ) -> bool:
        """Recover a committed upload whose success acknowledgement was lost."""

        with self._lock:
            pending_crc32c = self._pending_crc32c.get(object_name)
        remote_object = next(
            (
                item
                for item in list_gcs_objects(
                    bucket_name=self.bucket_name,
                    prefix=object_name,
                    storage_client=self._storage_client,
                )
                if item.name == object_name
            ),
            None,
        )
        if remote_object is None or remote_object.crc32c not in {
            local_crc32c,
            pending_crc32c,
        }:
            self._raise_conflict(object_name, cause)

        remote_matches_current = remote_object.crc32c == local_crc32c
        with self._lock:
            self._generations[object_name] = remote_object.generation
            self._pending_crc32c.pop(object_name, None)
            if remote_matches_current:
                self._dirty.discard(object_name)
        return remote_matches_current

    def _raise_conflict(
        self,
        object_name: str,
        cause: PreconditionFailed,
    ) -> None:
        raise RunSyncConflictError(
            "GCS object changed since it was pulled: "
            f"gs://{self.bucket_name}/{object_name}"
        ) from cause

    def _local_path(self, relative_name: str) -> Path:
        relative_path = PurePosixPath(relative_name)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe object path beneath run prefix: {relative_name}")
        local_path = self.run_dir.joinpath(*relative_path.parts).resolve()
        if not local_path.is_relative_to(self.run_dir):  # pragma: no cover - defensive
            raise ValueError(f"Unsafe object path beneath run prefix: {relative_name}")
        return local_path


def _file_crc32c(path: Path) -> str:
    """Return the base64-encoded CRC32C representation used by GCS."""

    checksum = google_crc32c.Checksum()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            checksum.update(chunk)
    return base64.b64encode(checksum.digest()).decode("ascii")
