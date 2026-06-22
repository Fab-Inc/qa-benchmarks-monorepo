"""Utilities for reading phonics_benchmark data files."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from phonics_benchmark.models import (
    BenchmarkItem,
    DecodableText,
    StudentKnowledgeProfile,
)

# Anchored to the benchmark project root (3 levels up from this file).
DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"


def get_data_dir(language: str) -> Path:
    """Return the Path to data/<language>/. Raises FileNotFoundError if absent."""
    path = DATA_ROOT / language
    if not path.is_dir():
        raise FileNotFoundError(
            f"No data directory for language '{language}': expected {path}"
        )
    return path


def _open_jsonl(path: Path, language: str, description: str):
    """Open a JSONL file, re-raising FileNotFoundError with a descriptive message."""
    try:
        return path.open(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Missing {description} for language '{language}': expected {path}"
        ) from None


def read_decodable_texts(language: str) -> list[DecodableText]:
    """Parse decodable_texts.jsonl for the given language.

    Returns a list of DecodableText objects.
    """
    path = get_data_dir(language) / "decodable_texts.jsonl"
    texts: list[DecodableText] = []
    seen_ids: set[str] = set()
    with _open_jsonl(path, language, "decodable_texts.jsonl") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                text = DecodableText.model_validate(record)
            except (json.JSONDecodeError, ValidationError) as exc:
                raise ValueError(
                    f"{path}:{line_no}: failed to parse record — {exc}"
                ) from exc
            if text.text_id in seen_ids:
                raise ValueError(
                    f"{path}:{line_no}: duplicate text_id '{text.text_id}'"
                )
            seen_ids.add(text.text_id)
            texts.append(text)
    return texts


def read_students(language: str) -> list[StudentKnowledgeProfile]:
    """Parse student_knowledge_profiles.jsonl for the given language.

    Returns a list of StudentKnowledgeProfile objects.
    """
    path = get_data_dir(language) / "student_knowledge_profiles.jsonl"
    profiles: list[StudentKnowledgeProfile] = []
    seen_ids: set[str] = set()
    with _open_jsonl(path, language, "student_knowledge_profiles.jsonl") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                profile = StudentKnowledgeProfile.model_validate(record)
            except (json.JSONDecodeError, ValidationError) as exc:
                raise ValueError(
                    f"{path}:{line_no}: failed to parse record — {exc}"
                ) from exc
            dup_id = profile.student_knowledge_id
            if dup_id in seen_ids:
                raise ValueError(
                    f"{path}:{line_no}: duplicate student_knowledge_id '{dup_id}'"
                )
            seen_ids.add(profile.student_knowledge_id)
            profiles.append(profile)
    return profiles


def read_benchmark_items(language: str) -> list[BenchmarkItem]:
    """Parse benchmark_items.jsonl for the given language.

    Returns a list of BenchmarkItem objects. Cross-reference validation
    (checking that text_id and student_knowledge_id exist in the other
    data files) is not performed here — call validate_benchmark_dataset()
    separately if referential integrity is required.
    """
    path = get_data_dir(language) / "benchmark_items.jsonl"
    items: list[BenchmarkItem] = []
    seen_ids: set[str] = set()
    with _open_jsonl(path, language, "benchmark_items.jsonl") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                item = BenchmarkItem.model_validate(record)
            except (json.JSONDecodeError, ValidationError) as exc:
                raise ValueError(
                    f"{path}:{line_no}: failed to parse record — {exc}"
                ) from exc
            dup_id = item.benchmark_item_id
            if dup_id in seen_ids:
                raise ValueError(
                    f"{path}:{line_no}: duplicate benchmark_item_id '{dup_id}'"
                )
            seen_ids.add(item.benchmark_item_id)
            items.append(item)
    return items


def validate_benchmark_dataset(language: str) -> None:
    """Check referential integrity across all three data files for a language.

    Raises ValueError if any benchmark_item references a text_id or
    student_knowledge_id that does not exist in the corresponding data file.
    """
    texts = read_decodable_texts(language)
    students = read_students(language)
    items = read_benchmark_items(language)

    text_ids = {t.text_id for t in texts}
    student_ids = {s.student_knowledge_id for s in students}

    for item in items:
        if item.text_id not in text_ids:
            raise ValueError(
                f"benchmark_items references unknown text_id '{item.text_id}'"
            )
        if item.student_knowledge_id not in student_ids:
            raise ValueError(
                f"benchmark_items references unknown"
                f" student_knowledge_id '{item.student_knowledge_id}'"
            )
