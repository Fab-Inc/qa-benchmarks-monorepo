"""Tests for the data_io module."""

import json

import pytest
from phonics_benchmark.data_io import (
    get_data_dir,
    read_benchmark_items,
    read_decodable_texts,
    read_students,
    validate_benchmark_dataset,
)
from phonics_benchmark.models import (
    BenchmarkItem,
    DecodableText,
    StudentKnowledgeProfile,
)


def test_get_data_dir_kiswahili() -> None:
    """Resolves the kiswahili data directory to an existing path."""
    path = get_data_dir("kiswahili")
    assert path.is_dir()


# --- read_students ---


def test_read_students_returns_profiles() -> None:
    """Returns the correct number of StudentKnowledgeProfile objects."""
    students = read_students("kiswahili")
    assert len(students) == 2
    assert all(isinstance(s, StudentKnowledgeProfile) for s in students)


def test_read_students_first_profile() -> None:
    """First profile has the expected ID, vowels key, and empty sight_words."""
    students = read_students("kiswahili")
    first = students[0]
    assert first.student_knowledge_id == "stu-sw-001"
    assert "known_syllables" in first.taught_knowledge
    assert first.taught_knowledge["max_word_length"] == 4


def test_read_students_missing_file() -> None:
    """Raises FileNotFoundError naming the file when no profiles JSONL exists."""
    with pytest.raises(FileNotFoundError, match="student_knowledge_profiles.jsonl"):
        read_students("english")


def test_read_students_duplicate_id(tmp_path) -> None:
    """Raises ValueError when two records share the same student_knowledge_id."""
    lang_dir = tmp_path / "testlang"
    lang_dir.mkdir()
    (lang_dir / "student_knowledge_profiles.jsonl").write_text(
        json.dumps(
            {
                "student_knowledge_id": "stu-001",
                "language_iso_code": "en",
                "taught_knowledge": {"vowels": ["a"]},
            }
        )
        + "\n"
        + json.dumps(
            {
                "student_knowledge_id": "stu-001",
                "language_iso_code": "en",
                "taught_knowledge": {"vowels": ["a"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    import phonics_benchmark.data_io as data_io

    original = data_io.DATA_ROOT
    data_io.DATA_ROOT = tmp_path
    try:
        with pytest.raises(ValueError, match="duplicate student_knowledge_id"):
            read_students("testlang")
    finally:
        data_io.DATA_ROOT = original


def test_read_students_bad_line_reports_line_number(tmp_path) -> None:
    """Error message includes the 1-based line number of the invalid record."""
    lang_dir = tmp_path / "testlang"
    lang_dir.mkdir()
    (lang_dir / "student_knowledge_profiles.jsonl").write_text(
        json.dumps(
            {
                "student_knowledge_id": "stu-001",
                "language_iso_code": "en",
                "taught_knowledge": {"vowels": ["a"]},
            }
        )
        + "\n"
        + "not valid json\n",
        encoding="utf-8",
    )
    import phonics_benchmark.data_io as data_io

    original = data_io.DATA_ROOT
    data_io.DATA_ROOT = tmp_path
    try:
        with pytest.raises(ValueError, match=":2:"):
            read_students("testlang")
    finally:
        data_io.DATA_ROOT = original


# --- read_decodable_texts ---


def test_read_decodable_texts_returns_items() -> None:
    """Returns at least one DecodableText object for kiswahili."""
    items = read_decodable_texts("kiswahili")
    assert len(items) >= 1
    assert all(isinstance(item, DecodableText) for item in items)


def test_read_decodable_texts_first_item() -> None:
    """First item has the expected ID, language code, and non-empty content."""
    items = read_decodable_texts("kiswahili")
    first = items[0]
    assert first.text_id == "sw-001"
    assert first.language_iso_code == "sw"
    assert isinstance(first.content, str) and len(first.content) > 0


def test_read_decodable_texts_missing_file() -> None:
    """Raises FileNotFoundError naming the file when no texts JSONL exists."""
    with pytest.raises(FileNotFoundError, match="decodable_texts.jsonl"):
        read_decodable_texts("english")


# --- read_benchmark_items ---


def test_read_benchmark_items_returns_items() -> None:
    """Returns all four BenchmarkItem objects for kiswahili."""
    items = read_benchmark_items("kiswahili")
    assert len(items) == 4
    assert all(isinstance(item, BenchmarkItem) for item in items)


def test_read_benchmark_items_first_item() -> None:
    """First item links the expected text and student knowledge profile."""
    items = read_benchmark_items("kiswahili")
    first = items[0]
    assert first.benchmark_item_id == "bi-sw-001-stu-sw-001"
    assert first.text_id == "sw-001"
    assert first.student_knowledge_id == "stu-sw-001"


def test_read_benchmark_items_missing_file() -> None:
    """Raises FileNotFoundError naming the file when no benchmark items JSONL exists."""
    with pytest.raises(FileNotFoundError, match="benchmark_items.jsonl"):
        read_benchmark_items("english")


# --- validate_benchmark_dataset ---


def test_validate_benchmark_dataset_passes_for_valid_data() -> None:
    """Passes without error when all benchmark item references are valid."""
    validate_benchmark_dataset("kiswahili")


def test_validate_benchmark_dataset_unknown_text_id(tmp_path) -> None:
    """Raises ValueError naming the text_id when it is absent from decodable_texts."""
    lang_dir = tmp_path / "testlang"
    lang_dir.mkdir()
    (lang_dir / "decodable_texts.jsonl").write_text(
        json.dumps({"text_id": "sw-001", "language_iso_code": "sw", "content": "Tatu."})
        + "\n",
        encoding="utf-8",
    )
    (lang_dir / "student_knowledge_profiles.jsonl").write_text(
        json.dumps(
            {
                "student_knowledge_id": "stu-sw-001",
                "language_iso_code": "sw",
                "taught_knowledge": {"vowels": ["a"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (lang_dir / "benchmark_items.jsonl").write_text(
        json.dumps(
            {
                "benchmark_item_id": "bi-001",
                "text_id": "sw-999",
                "student_knowledge_id": "stu-sw-001",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    import phonics_benchmark.data_io as data_io

    original = data_io.DATA_ROOT
    data_io.DATA_ROOT = tmp_path
    try:
        with pytest.raises(ValueError, match="sw-999"):
            validate_benchmark_dataset("testlang")
    finally:
        data_io.DATA_ROOT = original


def test_validate_benchmark_dataset_unknown_student_knowledge_id(tmp_path) -> None:
    """Raises ValueError naming the student_knowledge_id when absent from profiles."""
    lang_dir = tmp_path / "testlang"
    lang_dir.mkdir()
    (lang_dir / "decodable_texts.jsonl").write_text(
        json.dumps({"text_id": "sw-001", "language_iso_code": "sw", "content": "Tatu."})
        + "\n",
        encoding="utf-8",
    )
    (lang_dir / "student_knowledge_profiles.jsonl").write_text(
        json.dumps(
            {
                "student_knowledge_id": "stu-sw-001",
                "language_iso_code": "sw",
                "taught_knowledge": {"vowels": ["a"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (lang_dir / "benchmark_items.jsonl").write_text(
        json.dumps(
            {
                "benchmark_item_id": "bi-001",
                "text_id": "sw-001",
                "student_knowledge_id": "stu-sw-999",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    import phonics_benchmark.data_io as data_io

    original = data_io.DATA_ROOT
    data_io.DATA_ROOT = tmp_path
    try:
        with pytest.raises(ValueError, match="stu-sw-999"):
            validate_benchmark_dataset("testlang")
    finally:
        data_io.DATA_ROOT = original
