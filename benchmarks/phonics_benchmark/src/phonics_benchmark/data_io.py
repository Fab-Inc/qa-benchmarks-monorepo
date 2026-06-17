"""Utilities for reading phonics_benchmark data files."""

from __future__ import annotations

import json
from pathlib import Path

from phonics_benchmark.models import AssessmentTextItem, StudentProfile

# Anchored to the benchmark project root (3 levels up from this file).
DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"


def get_data_dir(language: str) -> Path:
    """
    Return the Path to data/<language>/
    Raises FileNotFoundError if absent.
    """
    path = DATA_ROOT / language
    if not path.is_dir():
        raise FileNotFoundError(
            f"No data directory for language '{language}': expected {path}"
        )
    return path


def read_items(language: str) -> list[AssessmentTextItem]:
    """
    Parse items.jsonl for the given language and return AssessmentTextItem objects.
    """
    path = get_data_dir(language) / "items.jsonl"
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            items.append(
                AssessmentTextItem(
                    item_id=record["item_id"],
                    language=record["language"],
                    student_id=record["student_id"],
                    text=record["text"],
                    undecodable_words=record["undecodable_words"],
                )
            )
    return items


def read_students(language: str) -> list[StudentProfile]:
    """Parse students.jsonl for the given language and return StudentProfile objects."""
    path = get_data_dir(language) / "students.jsonl"
    profiles = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            profiles.append(
                StudentProfile(
                    student_id=record["student_id"],
                    language=record["language"],
                    grade_level=record["grade_level"],
                    description=record["description"],
                    known_graphemes=record["known_graphemes"],
                    sight_words=record["sight_words"],
                )
            )
    return profiles


def read_language_spec(language: str) -> dict:
    """Load language_spec.json for the given language and return the raw dict."""
    path = get_data_dir(language) / "language_spec.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)
