"""Benchmark item loading for phonics runs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from phonics_benchmark.config import ConfigError

REQUIRED_ITEM_COLUMNS = {"item_id", "text"}


def load_benchmark_items(language: str, items_path: str | Path) -> pd.DataFrame:
    """Load benchmark items for a language from an opaque source path.

    This is the placeholder abstraction for future language-specific item
    routing. The runner deliberately calls this function instead of assuming
    a dataset layout.
    """

    path = Path(items_path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark items file not found: {path}")

    if path.suffix == ".jsonl":
        raw_items = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        raw_data = json.loads(path.read_text(encoding="utf-8"))
        raw_items = raw_data.get("items") if isinstance(raw_data, dict) else raw_data

    if not isinstance(raw_items, list) or not raw_items:
        raise ConfigError(
            f"No benchmark items were loaded for language {language!r} from {path}"
        )

    items = pd.DataFrame(raw_items)
    return validate_benchmark_items(items, language=language, source=path)


def validate_benchmark_items(
    items: pd.DataFrame,
    *,
    language: str,
    source: str | Path | None = None,
) -> pd.DataFrame:
    """Validate and normalize loaded benchmark items."""

    if items.empty:
        raise ConfigError(f"No benchmark items were loaded for language {language!r}")

    normalized = items.copy()
    if "item_id" not in normalized.columns and "id" in normalized.columns:
        normalized = normalized.rename(columns={"id": "item_id"})

    missing_columns = REQUIRED_ITEM_COLUMNS - set(normalized.columns)
    if missing_columns:
        location = f" from {source}" if source else ""
        raise ConfigError(
            f"Benchmark items{location} are missing columns: {sorted(missing_columns)}"
        )

    _ensure_non_empty_string_column(normalized, "item_id")
    _ensure_non_empty_string_column(normalized, "text")

    if normalized["item_id"].duplicated().any():
        duplicates = sorted(
            normalized.loc[normalized["item_id"].duplicated(), "item_id"].unique()
        )
        raise ConfigError(f"Duplicate benchmark item IDs: {duplicates}")

    if "metadata" in normalized.columns:
        invalid_metadata = normalized["metadata"].apply(
            lambda value: not (_is_missing(value) or isinstance(value, dict))
        )
        if invalid_metadata.any():
            raise ConfigError("Benchmark item metadata must be objects when provided")

    if "student_knowledge" not in normalized.columns:
        normalized["student_knowledge"] = None
    if "metadata" not in normalized.columns:
        normalized["metadata"] = [{} for _ in range(len(normalized))]

    return normalized.astype(object).where(pd.notna(normalized), None)


def item_to_dict(item: dict[str, Any] | pd.Series) -> dict[str, Any]:
    """Return a JSON-safe item dictionary."""

    values = item.to_dict() if isinstance(item, pd.Series) else dict(item)
    return {key: value for key, value in values.items() if not _is_missing(value)}


def _ensure_non_empty_string_column(items: pd.DataFrame, column: str) -> None:
    invalid = items[column].apply(
        lambda value: not isinstance(value, str) or not value.strip()
    )
    if invalid.any():
        raise ConfigError(f"Benchmark item column {column!r} must be non-empty strings")


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False
