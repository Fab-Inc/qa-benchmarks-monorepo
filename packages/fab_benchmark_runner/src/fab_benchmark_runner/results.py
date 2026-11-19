"""Per-model JSONL result files."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping
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

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        results_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise ConfigError(
                f"{results_path}:{line_number}: invalid JSON in results: {e}"
            ) from e
        if not isinstance(row, dict) or "result_id" not in row:
            raise ConfigError(
                f"{results_path}:{line_number}: existing result lacks result_id"
            )
        rows.append(row)
    return rows


def save_results(rows: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    """Atomically write result rows as JSONL with sorted keys."""

    results_path = Path(path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(canonical_json(dict(row)) + "\n" for row in rows)
    write_text_atomic(results_path, content)
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

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as temporary_file:
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
