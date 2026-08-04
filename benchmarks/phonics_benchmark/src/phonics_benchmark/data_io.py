"""Utilities for reading phonics_benchmark data files."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

from decodability import DECODABILITY_REGISTRY
from pydantic import ValidationError

from phonics_benchmark.models import (
    BenchmarkItem,
    DecodableText,
    StudentKnowledge,
)


def get_data_dir(language: str, data_root: Path) -> Path:
    """Return the Path to data_root/<language>/"""
    return data_root / language


def _open_jsonl(path: Path, language: str, description: str):
    """Open a JSONL file, FileNotFoundError with a descriptive message."""
    try:
        return path.open(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Missing {description} for language '{language}': expected {path}"
        ) from None


def read_decodable_texts(language: str, data_root: Path) -> list[DecodableText]:
    """Parse decodable_texts.jsonl for the given language.

    Returns a list of DecodableText objects.
    """
    path = get_data_dir(language, data_root) / "decodable_texts.jsonl"
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


def read_student_knowledge_profiles(
    language: str, data_root: Path
) -> list[StudentKnowledge]:
    """Parse student_knowledge_profiles.jsonl for the given language.

    Returns a list of StudentKnowledge objects.
    """
    path = get_data_dir(language, data_root) / "student_knowledge_profiles.jsonl"
    profiles: list[StudentKnowledge] = []
    seen_ids: set[str] = set()
    with _open_jsonl(path, language, "student_knowledge_profiles.jsonl") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                profile = StudentKnowledge.model_validate(record)
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


def read_benchmark_items(language: str, data_root: Path) -> list[BenchmarkItem]:
    """Parse benchmark_items.jsonl for the given language.

    Returns a list of BenchmarkItem objects. Cross-reference validation
    (checking that text_id and student_knowledge_id exist in the other
    data files) is not performed here — call validate_benchmark_dataset()
    separately if referential integrity is required.
    """
    path = get_data_dir(language, data_root) / "benchmark_items.jsonl"
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


def validate_student_knowledge(
    profiles: list[StudentKnowledge], *, strict: bool = False
) -> None:
    """Check that each profile's student_knowledge is valid for its language.

    Delegates to the language-specific model in `decodability` (e.g.
    KiswahiliStudentKnowledge for "sw"). Profiles in a language with no
    registered validator emit a warning, since their student_knowledge goes
    unchecked — pass strict=True to raise instead (e.g. once a language is
    expected to always have a registered validator).
    """
    for profile in profiles:
        language_integration = DECODABILITY_REGISTRY.get(profile.language_iso_code)
        if language_integration is None:
            message = (
                f"No known student_knowledge validator registered for language "
                f"'{profile.language_iso_code}' (student_knowledge_id "
                f"'{profile.student_knowledge_id}')"
            )
            if strict:
                raise ValueError(message)
            warnings.warn(f"{message}; skipping validation.", stacklevel=2)
            continue
        student_knowledge_model = language_integration["student_knowledge_model"]
        try:
            student_knowledge_model.model_validate(profile.student_knowledge)
        except ValidationError as exc:
            raise ValueError(
                f"student_knowledge_id '{profile.student_knowledge_id}': {exc}"
            ) from exc


def validate_benchmark_dataset(language: str, data_root: Path) -> None:
    """Check referential integrity across all three data files for a language.

    Raises ValueError if any benchmark_item references a text_id or
    student_knowledge_id that does not exist in the corresponding data file.
    """
    texts = read_decodable_texts(language, data_root)
    student_knowledge_profiles = read_student_knowledge_profiles(language, data_root)
    items = read_benchmark_items(language, data_root)

    text_ids = {t.text_id for t in texts}
    student_ids = {s.student_knowledge_id for s in student_knowledge_profiles}

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
