"""Tests for the data_io module."""

import pytest
from phonics_benchmark.data_io import (
    get_data_dir,
    read_items,
    read_language_spec,
    read_students,
)
from phonics_benchmark.models import AssessmentTextItem, StudentProfile


def test_get_data_dir_kiswahili() -> None:
    path = get_data_dir("kiswahili")
    assert path.is_dir()


def test_get_data_dir_missing_language() -> None:
    with pytest.raises(FileNotFoundError, match="nonexistent_language"):
        get_data_dir("nonexistent_language")


def test_read_students_returns_profiles() -> None:
    students = read_students("kiswahili")
    assert len(students) == 2
    assert all(isinstance(s, StudentProfile) for s in students)


def test_read_students_first_profile() -> None:
    students = read_students("kiswahili")
    first = students[0]
    assert first.student_id == "stu-grade1-early"
    assert "vowels" in first.known_graphemes
    assert first.sight_words == []


def test_read_items_returns_items() -> None:
    items = read_items("kiswahili")
    assert len(items) >= 1
    assert all(isinstance(item, AssessmentTextItem) for item in items)


def test_read_items_first_item() -> None:
    items = read_items("kiswahili")
    first = items[0]
    assert first.item_id == "sw-001"
    assert first.language == "sw"
    assert isinstance(first.undecodable_words, list)


def test_read_language_spec_returns_dict() -> None:
    spec = read_language_spec("kiswahili")
    assert isinstance(spec, dict)
    assert "decodability_rule" in spec
    assert "unit_categories" in spec


def test_read_items_missing_file() -> None:
    # english/ directory exists but has no items.jsonl yet
    with pytest.raises(FileNotFoundError):
        read_items("english")
