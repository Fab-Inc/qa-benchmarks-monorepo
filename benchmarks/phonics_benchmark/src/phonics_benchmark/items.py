"""Loading of completed benchmark items for phonics runs."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from phonics_benchmark.config import ConfigError


def load_benchmark_items(
    items_path: str | Path,
) -> tuple[list[dict[str, Any]], str]:
    """Load normalized execution items and hash the exact JSONL bytes consumed."""

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

        content = record.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ConfigError(
                f"{path}:{line_number}: content must be a non-empty, "
                "non-whitespace string"
            )

        student_knowledge = record.get("student_knowledge")
        if not isinstance(student_knowledge, dict):
            raise ConfigError(
                f"{path}:{line_number}: student_knowledge must be an object"
            )

        seen_ids.add(benchmark_item_id)
        items.append(
            {
                "benchmark_item_id": benchmark_item_id,
                "content": content,
                "student_knowledge": student_knowledge,
            }
        )

    if not items:
        raise ConfigError(f"No benchmark items were loaded from {path}")
    return items, items_sha256
