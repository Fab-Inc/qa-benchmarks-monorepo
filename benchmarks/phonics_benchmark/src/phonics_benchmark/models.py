"""Data models for phonics_benchmark."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StudentProfile:
    student_id: str
    language: str
    grade_level: str
    description: str
    known_graphemes: dict[str, list[str]]
    sight_words: list[str]


@dataclass(frozen=True)
class AssessmentTextItem:
    item_id: str
    language: str
    student_id: str
    text: str
    undecodable_words: list[dict]
