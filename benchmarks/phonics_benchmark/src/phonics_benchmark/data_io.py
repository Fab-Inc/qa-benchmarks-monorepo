"""Utilities for reading phonics_benchmark data files."""

from __future__ import annotations

import json
from pathlib import Path

from decodability.kiswahili.models import KiswahiliStudentKnowledge
from pydantic import ValidationError

from phonics_benchmark.models import (
    BenchmarkItem,
    DecodableText,
    StudentKnowledgeProfile,
)

# Per-language known-grapheme/whole-word validators. A language with no entry
# here has no grapheme validation defined yet and is silently skipped.
_GRAPHEME_VALIDATORS = {
    "sw": KiswahiliStudentKnowledge,
}


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
) -> list[StudentKnowledgeProfile]:
    """Parse student_knowledge_profiles.jsonl for the given language.

    Returns a list of StudentKnowledgeProfile objects.
    """
    path = get_data_dir(language, data_root) / "student_knowledge_profiles.jsonl"
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


def _as_str_list(
    value: list[str] | int | None, field: str, student_knowledge_id: str
) -> list[str]:
    """Return value as a list[str], or raise if taught_knowledge[field] is malformed."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(
            f"student_knowledge_id '{student_knowledge_id}': expected a list for "
            f"taught_knowledge['{field}'], got {type(value).__name__}"
        )
    return value


def validate_known_graphemes(profiles: list[StudentKnowledgeProfile]) -> None:
    """Check that each profile's graphemes/whole_words are valid for its language.

    Delegates to the language-specific model in `decodability` (e.g.
    KiswahiliStudentKnowledge for "sw"). Profiles in a language with no
    registered validator are skipped, not rejected.
    """
    for profile in profiles:
        validator = _GRAPHEME_VALIDATORS.get(profile.language_iso_code)
        if validator is None:
            continue
        graphemes = _as_str_list(
            profile.taught_knowledge.get("graphemes"),
            "graphemes",
            profile.student_knowledge_id,
        )
        whole_words = _as_str_list(
            profile.taught_knowledge.get("whole_words"),
            "whole_words",
            profile.student_knowledge_id,
        )
        try:
            validator(graphemes=set(graphemes), whole_words=set(whole_words))
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
