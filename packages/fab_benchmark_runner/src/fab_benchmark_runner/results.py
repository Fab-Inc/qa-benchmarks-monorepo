"""Per-model JSONL result files."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from fab_benchmark_runner.errors import ConfigError
from fab_benchmark_runner.ids import canonical_json

CALL_STATUS_ERROR = "error"
CALL_STATUS_SUCCESS = "success"


def load_results(path: str | Path, *, force: bool = False) -> list[dict[str, Any]]:
    """Load existing result rows, or none when forced or the file is absent."""

    results_path = Path(path)
    if force or not results_path.exists():
        return []
    return list(iter_results(results_path))


def iter_results(path: Path) -> Iterator[dict[str, Any]]:
    """Yield a results file's rows, reading one line at a time.

    Lines end at a newline only: JSON text keeps other line separators, such as
    U+2028, as they are. Raises ConfigError naming the line of invalid JSON or of
    a row without a ``result_id``.
    """

    with path.open("rb") as lines:
        for line_number, raw_line in enumerate(lines, start=1):
            line = raw_line.decode("utf-8")
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ConfigError(
                    f"{path}:{line_number}: invalid JSON in results: {e}"
                ) from e
            if not isinstance(row, dict) or "result_id" not in row:
                raise ConfigError(
                    f"{path}:{line_number}: existing result lacks result_id"
                )
            yield row


def read_result_statuses(path: Path) -> dict[str, Any]:
    """Map each result ID in a results file to its ``call_status``, and nothing more.

    Planning needs only these; a row's prompt and output stay on disk. As when
    loading rows, the last row with an ID wins.
    """

    if not path.exists():
        return {}
    return {str(row["result_id"]): row.get("call_status") for row in iter_results(path)}


def save_results(rows: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    """Atomically write result rows as JSONL with sorted keys, one row at a time."""

    results_path = Path(path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    _write_chunks_atomic(
        results_path,
        ((canonical_json(dict(row)) + "\n").encode("utf-8") for row in rows),
    )
    return results_path


def error_count(rows: Iterable[Mapping[str, Any]]) -> int:
    """Count rows whose model call failed."""

    return sum(row.get("call_status") == CALL_STATUS_ERROR for row in rows)


def model_results_filename(model_name: str) -> str:
    """Return the stable per-model response file name."""

    return f"responses_{model_file_key(model_name)}.jsonl"


def model_record_filename(model_name: str) -> str:
    """Return the file name of a model's record in the run folder."""

    return f"{model_file_key(model_name)}.json"


def model_file_key(model_name: str) -> str:
    """Return a file-safe, collision-free key for a model name."""

    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("._-")
    if not safe_name:
        safe_name = "model"
    if safe_name != model_name:
        safe_name += "-" + sha256(model_name.encode("utf-8")).hexdigest()[:8]
    return safe_name


def write_text_atomic(path: Path, content: str) -> None:
    """Write UTF-8 text through a temporary file and a rename."""

    write_bytes_atomic(path, content.encode("utf-8"))


def write_bytes_atomic(path: Path, content: bytes) -> None:
    """Write through a temporary file and a rename, so readers never see half."""

    _write_chunks_atomic(path, [content])


def _write_chunks_atomic(path: Path, chunks: Iterable[bytes]) -> None:
    """Write chunks as they come through a temporary file, then rename it."""

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as temporary_file:
            for chunk in chunks:
                temporary_file.write(chunk)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def utc_now() -> str:
    """Return the current UTC time as an ISO 8601 string ending in Z."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 of a file's bytes."""

    digest = sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
