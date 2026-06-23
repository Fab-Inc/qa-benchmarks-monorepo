"""Benchmark item assembly for phonics runs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from phonics_benchmark.config import ConfigError
from phonics_benchmark.data_io import (
    read_benchmark_items,
    read_decodable_texts,
    read_student_knowledge_profiles,
)


def load_benchmark_items(language: str, data_root: str | Path) -> pd.DataFrame:
    """Load joined benchmark items for a language."""

    root = Path(data_root)
    try:
        texts = {text.text_id: text for text in read_decodable_texts(language, root)}
        students = {
            student.student_knowledge_id: student
            for student in read_student_knowledge_profiles(language, root)
        }
        items = read_benchmark_items(language, root)
    except ValueError as e:
        raise ConfigError(str(e)) from e

    if not items:
        raise ConfigError(f"No benchmark items were loaded for language {language!r}")

    records: list[dict[str, Any]] = []
    for item in items:
        text = texts.get(item.text_id)
        if text is None:
            raise ConfigError(
                f"benchmark_items references unknown text_id '{item.text_id}'"
            )
        student = students.get(item.student_knowledge_id)
        if student is None:
            raise ConfigError(
                "benchmark_items references unknown "
                f"student_knowledge_id '{item.student_knowledge_id}'"
            )

        records.append(
            {
                "item_id": item.benchmark_item_id,
                "benchmark_item_id": item.benchmark_item_id,
                "text_id": item.text_id,
                "student_knowledge_id": item.student_knowledge_id,
                "text": text.content,
                "student_knowledge": student.taught_knowledge,
                "metadata": {
                    "decodable_text": text.model_dump(),
                    "student_knowledge_profile": student.model_dump(),
                },
            }
        )

    frame = pd.DataFrame(records)
    return frame.astype(object).where(pd.notna(frame), None)


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
