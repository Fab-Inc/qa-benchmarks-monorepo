"""Loading of already-composed benchmark items for phonics runs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from phonics_benchmark.config import ConfigError

COMPOSED_CSV_REQUIRED_COLUMNS = {
    "benchmark_item_id",
    "content",
    "student_knowledge_id",
    "graphemes",
    "whole_words",
}


def load_benchmark_items(
    items_path: str | Path,
) -> pd.DataFrame:
    """Load already-composed benchmark items from a CSV file."""

    frame = _load_composed_items_csv(items_path)
    if frame.empty:
        raise ConfigError(f"No benchmark items were loaded from {items_path}")
    return frame.astype(object).where(pd.notna(frame), None)


def _load_composed_items_csv(items_path: str | Path) -> pd.DataFrame:
    """Load QFB-style composed benchmark items from a CSV file."""

    path = Path(items_path)
    try:
        source = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        raise ConfigError(f"Could not read composed items CSV {path}: {e}") from e

    missing_columns = COMPOSED_CSV_REQUIRED_COLUMNS.difference(source.columns)
    if missing_columns:
        raise ConfigError(
            f"Composed items CSV {path} is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    duplicate_ids = source.loc[
        source["benchmark_item_id"].duplicated(), "benchmark_item_id"
    ].tolist()
    if duplicate_ids:
        raise ConfigError(
            f"Composed items CSV {path} contains duplicate benchmark_item_id values: "
            + ", ".join(duplicate_ids)
        )

    records = []
    for row_index, row in source.iterrows():
        line_number = row_index + 2
        item_id = row["benchmark_item_id"].strip()
        text = row["content"].strip()
        student_knowledge_id = row["student_knowledge_id"].strip()
        if not item_id or not text or not student_knowledge_id:
            raise ConfigError(
                f"{path}:{line_number}: benchmark_item_id, content, and "
                "student_knowledge_id must be non-empty"
            )

        student_knowledge = {
            "graphemes": _parse_string_list(
                row["graphemes"],
                path=path,
                line_number=line_number,
                field="graphemes",
            ),
            "whole_words": _parse_string_list(
                row["whole_words"],
                path=path,
                line_number=line_number,
                field="whole_words",
            ),
        }
        metadata = {
            key: value
            for key, value in row.to_dict().items()
            if key not in COMPOSED_CSV_REQUIRED_COLUMNS and value != ""
        }
        if "non_decodable_words" in metadata:
            metadata["non_decodable_words"] = _parse_string_list(
                metadata["non_decodable_words"],
                path=path,
                line_number=line_number,
                field="non_decodable_words",
            )

        records.append(
            {
                "item_id": item_id,
                "benchmark_item_id": item_id,
                "student_knowledge_id": student_knowledge_id,
                "text": text,
                "student_knowledge": student_knowledge,
                "metadata": metadata,
            }
        )

    return pd.DataFrame(records)


def _parse_string_list(
    value: str,
    *,
    path: Path,
    line_number: int,
    field: str,
) -> list[str]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as e:
        raise ConfigError(f"{path}:{line_number}: invalid JSON in {field}: {e}") from e
    if not isinstance(parsed, list) or not all(
        isinstance(item, str) for item in parsed
    ):
        raise ConfigError(
            f"{path}:{line_number}: {field} must be a JSON array of strings"
        )
    return parsed


def item_to_dict(item: dict[str, Any] | pd.Series) -> dict[str, Any]:
    """Return a JSON-safe item dictionary."""

    values = item.to_dict() if isinstance(item, pd.Series) else dict(item)
    return {key: value for key, value in values.items() if not _is_missing(value)}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False
