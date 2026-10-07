"""Loading of completed benchmark items for phonics runs."""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

from phonics_benchmark.config import ConfigError

# benchmark_item_id is always required and gets its own checks (trimmed,
# unique); the rest map each required field to the type its value must have.
DEFAULT_REQUIRED_FIELDS: dict[str, type] = {
    "benchmark_item_id": str,
    "content": str,
    "student_knowledge": dict,
}


def load_benchmark_items(
    items_path: str | Path,
    *,
    required_fields: Mapping[str, type] | None = None,
    extra_fields: Collection[str] = (),
) -> tuple[list[dict[str, Any]], str]:
    """Load normalized execution items and hash the exact JSONL bytes consumed.

    ``required_fields`` maps each field the task needs to the type its value
    must have (``str`` = non-empty, non-whitespace; ``dict`` = object).
    ``benchmark_item_id`` is always required and must be a trimmed, unique,
    non-empty string. Items keep the required fields plus any of
    ``extra_fields`` present on the record; dataset-owned fields beyond those
    are dropped.
    """

    path = Path(items_path)
    try:
        artifact_bytes = path.read_bytes()
    except OSError as e:
        raise ConfigError(f"Could not read benchmark items JSONL {path}: {e}") from e

    items_sha256 = sha256(artifact_bytes).hexdigest()
    try:
        artifact_text = artifact_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ConfigError(
            f"Benchmark items JSONL {path} is not valid UTF-8: {e}"
        ) from e

    if not artifact_text:
        raise ConfigError(f"No benchmark items were loaded from {path}")

    required = dict(required_fields or DEFAULT_REQUIRED_FIELDS)
    required["benchmark_item_id"] = str

    items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
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

        benchmark_item_id = record.get("benchmark_item_id")
        if not isinstance(benchmark_item_id, str) or not benchmark_item_id:
            raise ConfigError(
                f"{path}:{line_number}: benchmark_item_id must be a non-empty string"
            )
        if benchmark_item_id != benchmark_item_id.strip():
            raise ConfigError(
                f"{path}:{line_number}: benchmark_item_id must not have surrounding "
                "whitespace"
            )
        if benchmark_item_id in seen_ids:
            raise ConfigError(
                f"{path}:{line_number}: duplicate benchmark_item_id "
                f"'{benchmark_item_id}'"
            )

        for field_name, field_type in required.items():
            if field_name == "benchmark_item_id":
                continue
            value = record.get(field_name)
            if field_type is str:
                if not isinstance(value, str) or not value.strip():
                    raise ConfigError(
                        f"{path}:{line_number}: {field_name} must be a non-empty, "
                        "non-whitespace string"
                    )
            elif not isinstance(value, field_type):
                raise ConfigError(
                    f"{path}:{line_number}: {field_name} must be an object"
                    if field_type is dict
                    else f"{path}:{line_number}: {field_name} has the wrong type"
                )

        seen_ids.add(benchmark_item_id)
        kept_fields = set(required) | set(extra_fields)
        items.append(
            {
                field_name: record[field_name]
                for field_name in record
                if field_name in kept_fields
            }
        )

    if not items:
        raise ConfigError(f"No benchmark items were loaded from {path}")
    return items, items_sha256
