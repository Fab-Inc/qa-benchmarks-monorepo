"""Loading benchmark items."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

from fab_benchmark_runner.errors import ConfigError

ITEM_ID_KEY = "benchmark_item_id"


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 of a file's exact bytes."""

    try:
        return sha256(Path(path).read_bytes()).hexdigest()
    except OSError as e:
        raise ConfigError(f"Could not read items file {path}: {e}") from e


def load_jsonl_items(items_path: str | Path) -> tuple[list[dict[str, Any]], str]:
    """Load JSONL items and hash the exact bytes read.

    Every record must be an object with a ``benchmark_item_id``. All other fields
    are kept for the benchmark's prompt builder.
    """

    path = Path(items_path)
    try:
        artifact_bytes = path.read_bytes()
    except OSError as e:
        raise ConfigError(f"Could not read benchmark items JSONL {path}: {e}") from e
    try:
        artifact_text = artifact_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ConfigError(
            f"Benchmark items JSONL {path} is not valid UTF-8: {e}"
        ) from e

    items: list[dict[str, Any]] = []
    for line_number, line in enumerate(artifact_text.splitlines(), start=1):
        if not line.strip():
            raise ConfigError(
                f"{path}:{line_number}: blank JSONL lines are not allowed"
            )
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            raise ConfigError(f"{path}:{line_number}: invalid JSON: {e}") from e
        if not isinstance(record, dict):
            raise ConfigError(f"{path}:{line_number}: benchmark item must be an object")
        items.append(record)

    validate_items(items, source=path)
    return items, sha256(artifact_bytes).hexdigest()


def validate_items(items: Iterable[Mapping[str, Any]], *, source: str | Path) -> None:
    """Require non-empty, trimmed and unique benchmark item IDs."""

    seen_ids: set[str] = set()
    count = 0
    for index, item in enumerate(items, start=1):
        count += 1
        if not isinstance(item, Mapping):
            raise ConfigError(f"{source}: item {index} must be an object")
        item_id = item.get(ITEM_ID_KEY)
        if not isinstance(item_id, str) or not item_id:
            raise ConfigError(
                f"{source}: item {index}: {ITEM_ID_KEY} must be a non-empty string"
            )
        if item_id != item_id.strip():
            raise ConfigError(
                f"{source}: item {index}: {ITEM_ID_KEY} must not have surrounding "
                "whitespace"
            )
        if item_id in seen_ids:
            raise ConfigError(f"{source}: duplicate {ITEM_ID_KEY} '{item_id}'")
        seen_ids.add(item_id)
    if count == 0:
        raise ConfigError(f"No benchmark items were loaded from {source}")
