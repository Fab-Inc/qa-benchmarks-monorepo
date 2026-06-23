"""Tests for the data_io module."""

import json

import phonics_benchmark.data_io as data_io
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


@pytest.fixture
def student_record():
    """Minimal valid student knowledge profile record."""
    return {
        "student_knowledge_id": "stu-001",
        "language_iso_code": "sw",
        "taught_knowledge": {
            "known_syllables": ["a"],
            "known_syllable_patterns": ["CV"],
        },
    }


@pytest.fixture
def text_record():
    """Minimal valid decodable text record."""
    return {"text_id": "sw-001", "language_iso_code": "sw", "content": "Tatu."}


@pytest.fixture
def item_record():
    """Minimal valid benchmark item record."""
    return {
        "benchmark_item_id": "bi-001",
        "text_id": "sw-001",
        "student_knowledge_id": "stu-001",
    }


@pytest.fixture
def patched_data_root(tmp_path, monkeypatch):
    """Redirect DATA_ROOT to tmp_path for the duration of the test."""
    monkeypatch.setattr(data_io, "DATA_ROOT", tmp_path)
    return tmp_path


@pytest.fixture
def testlang_dir(patched_data_root, text_record, student_record, item_record):
    """A testlang/ directory pre-populated with one valid record per data file."""
    d = patched_data_root / "testlang"
    d.mkdir()
    (d / "decodable_texts.jsonl").write_text(
        json.dumps(text_record) + "\n", encoding="utf-8"
    )
    (d / "student_knowledge_profiles.jsonl").write_text(
        json.dumps(student_record) + "\n", encoding="utf-8"
    )
    (d / "benchmark_items.jsonl").write_text(
        json.dumps(item_record) + "\n", encoding="utf-8"
    )
    return d


class TestGetDataDir:
    """Test for get_data_dir() function."""

    def test_resolves_existing_language(self) -> None:
        """Resolves the kiswahili data directory to an existing path."""
        path = get_data_dir("kiswahili")
        assert path.is_dir()


class TestReadStudents:
    """Test for read_students() function."""

    def test_returns_profiles(self) -> None:
        """Returns the correct number of StudentKnowledgeProfile objects."""
        students = read_students("kiswahili")
        assert len(students) == 2
        assert all(isinstance(s, StudentKnowledgeProfile) for s in students)

    def test_first_profile(self) -> None:
        """First profile has the expected ID, known_syllables, and max_word_length."""
        students = read_students("kiswahili")
        first = students[0]
        assert first.student_knowledge_id == "stu-sw-001"
        assert "known_syllables" in first.taught_knowledge
        assert first.taught_knowledge["max_word_length"] == 4

    def test_missing_file_raises(self) -> None:
        """Raises FileNotFoundError naming the file when no profiles JSONL exists."""
        with pytest.raises(FileNotFoundError, match="student_knowledge_profiles.jsonl"):
            read_students("english")

    def test_duplicate_id_raises(self, patched_data_root, student_record) -> None:
        """Raises ValueError when two records share the same student_knowledge_id."""
        d = patched_data_root / "testlang"
        d.mkdir()
        (d / "student_knowledge_profiles.jsonl").write_text(
            json.dumps(student_record) + "\n" + json.dumps(student_record) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="duplicate student_knowledge_id"):
            read_students("testlang")

    def test_bad_line_reports_line_number(
        self, patched_data_root, student_record
    ) -> None:
        """Error message includes the 1-based line number of the invalid record."""
        d = patched_data_root / "testlang"
        d.mkdir()
        (d / "student_knowledge_profiles.jsonl").write_text(
            json.dumps(student_record) + "\n" + "not valid json\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=":2:"):
            read_students("testlang")

    def test_rejects_extra_field(self, patched_data_root, student_record) -> None:
        """Raises ValueError when a record contains an unexpected field."""
        d = patched_data_root / "testlang"
        d.mkdir()
        (d / "student_knowledge_profiles.jsonl").write_text(
            json.dumps({**student_record, "unexpected_field": "value"}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=":1:"):
            read_students("testlang")

    def test_rejects_wrong_field_type(self, patched_data_root, student_record) -> None:
        """Raises ValueError when taught_knowledge is not a dict."""
        d = patched_data_root / "testlang"
        d.mkdir()
        (d / "student_knowledge_profiles.jsonl").write_text(
            json.dumps({**student_record, "taught_knowledge": "not-a-dict"}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=":1:"):
            read_students("testlang")

    def test_rejects_empty_id(self, patched_data_root, student_record) -> None:
        """Raises ValueError when student_knowledge_id is an empty string."""
        d = patched_data_root / "testlang"
        d.mkdir()
        (d / "student_knowledge_profiles.jsonl").write_text(
            json.dumps({**student_record, "student_knowledge_id": ""}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=":1:"):
            read_students("testlang")


class TestReadDecodableTexts:
    """Test for read_decodable_texts() function."""

    def test_returns_items(self) -> None:
        """Returns at least one DecodableText object for kiswahili."""
        items = read_decodable_texts("kiswahili")
        assert len(items) >= 1
        assert all(isinstance(item, DecodableText) for item in items)

    def test_first_item(self) -> None:
        """First item has the expected ID, language code, and non-empty content."""
        items = read_decodable_texts("kiswahili")
        first = items[0]
        assert first.text_id == "sw-001"
        assert first.language_iso_code == "sw"
        assert isinstance(first.content, str) and len(first.content) > 0

    def test_missing_file_raises(self) -> None:
        """Raises FileNotFoundError naming the file when no texts JSONL exists."""
        with pytest.raises(FileNotFoundError, match="decodable_texts.jsonl"):
            read_decodable_texts("english")

    def test_duplicate_id_raises(self, patched_data_root, text_record) -> None:
        """Raises ValueError when two records share the same text_id."""
        d = patched_data_root / "testlang"
        d.mkdir()
        (d / "decodable_texts.jsonl").write_text(
            json.dumps(text_record) + "\n" + json.dumps(text_record) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="duplicate text_id"):
            read_decodable_texts("testlang")

    def test_bad_line_reports_line_number(self, patched_data_root, text_record) -> None:
        """Error message includes the 1-based line number of the invalid record."""
        d = patched_data_root / "testlang"
        d.mkdir()
        (d / "decodable_texts.jsonl").write_text(
            json.dumps(text_record) + "\n" + "not valid json\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=":2:"):
            read_decodable_texts("testlang")

    def test_rejects_extra_field(self, patched_data_root, text_record) -> None:
        """Raises ValueError when a record contains an unexpected field."""
        d = patched_data_root / "testlang"
        d.mkdir()
        (d / "decodable_texts.jsonl").write_text(
            json.dumps({**text_record, "unexpected_field": "value"}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=":1:"):
            read_decodable_texts("testlang")

    def test_rejects_wrong_field_type(self, patched_data_root, text_record) -> None:
        """Raises ValueError when content is not a string."""
        d = patched_data_root / "testlang"
        d.mkdir()
        (d / "decodable_texts.jsonl").write_text(
            json.dumps({**text_record, "content": ["not", "a", "string"]}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=":1:"):
            read_decodable_texts("testlang")

    def test_rejects_empty_text_id(self, patched_data_root, text_record) -> None:
        """Raises ValueError when text_id is an empty string."""
        d = patched_data_root / "testlang"
        d.mkdir()
        (d / "decodable_texts.jsonl").write_text(
            json.dumps({**text_record, "text_id": ""}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=":1:"):
            read_decodable_texts("testlang")


class TestReadBenchmarkItems:
    """Test for read_benchmark_items() function."""

    def test_returns_items(self) -> None:
        """Returns all four BenchmarkItem objects for kiswahili."""
        items = read_benchmark_items("kiswahili")
        assert len(items) == 4
        assert all(isinstance(item, BenchmarkItem) for item in items)

    def test_first_item(self) -> None:
        """First item links the expected text and student knowledge profile."""
        items = read_benchmark_items("kiswahili")
        first = items[0]
        assert first.benchmark_item_id == "bi-sw-001-stu-sw-001"
        assert first.text_id == "sw-001"
        assert first.student_knowledge_id == "stu-sw-001"

    def test_missing_file_raises(self) -> None:
        """Raises FileNotFoundError naming the file when no benchmark items JSONL."""
        with pytest.raises(FileNotFoundError, match="benchmark_items.jsonl"):
            read_benchmark_items("english")

    def test_duplicate_id_raises(self, patched_data_root, item_record) -> None:
        """Raises ValueError when two records share the same benchmark_item_id."""
        d = patched_data_root / "testlang"
        d.mkdir()
        (d / "benchmark_items.jsonl").write_text(
            json.dumps(item_record) + "\n" + json.dumps(item_record) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="duplicate benchmark_item_id"):
            read_benchmark_items("testlang")

    def test_bad_line_reports_line_number(self, patched_data_root, item_record) -> None:
        """Error message includes the 1-based line number of the invalid record."""
        d = patched_data_root / "testlang"
        d.mkdir()
        (d / "benchmark_items.jsonl").write_text(
            json.dumps(item_record) + "\n" + "not valid json\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=":2:"):
            read_benchmark_items("testlang")

    def test_rejects_extra_field(self, patched_data_root, item_record) -> None:
        """Raises ValueError when a record contains an unexpected field."""
        d = patched_data_root / "testlang"
        d.mkdir()
        (d / "benchmark_items.jsonl").write_text(
            json.dumps({**item_record, "unexpected_field": "value"}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=":1:"):
            read_benchmark_items("testlang")

    def test_rejects_wrong_field_type(self, patched_data_root, item_record) -> None:
        """Raises ValueError when benchmark_item_id is not a string."""
        d = patched_data_root / "testlang"
        d.mkdir()
        (d / "benchmark_items.jsonl").write_text(
            json.dumps({**item_record, "benchmark_item_id": ["not", "a", "string"]})
            + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=":1:"):
            read_benchmark_items("testlang")

    def test_rejects_empty_id(self, patched_data_root, item_record) -> None:
        """Raises ValueError when benchmark_item_id is an empty string."""
        d = patched_data_root / "testlang"
        d.mkdir()
        (d / "benchmark_items.jsonl").write_text(
            json.dumps({**item_record, "benchmark_item_id": ""}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=":1:"):
            read_benchmark_items("testlang")


class TestValidateBenchmarkDataset:
    """Test for validate_benchmark_dataset() function."""

    def test_passes_for_valid_data(self) -> None:
        """Passes without error when all benchmark item references are valid."""
        validate_benchmark_dataset("kiswahili")

    def test_unknown_text_id_raises(self, testlang_dir, item_record) -> None:
        """Raises ValueError naming the text_id when absent from decodable_texts."""
        (testlang_dir / "benchmark_items.jsonl").write_text(
            json.dumps({**item_record, "text_id": "sw-999"}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="sw-999"):
            validate_benchmark_dataset("testlang")

    def test_unknown_student_knowledge_id_raises(
        self, testlang_dir, item_record
    ) -> None:
        """Raises ValueError naming the student_knowledge_id when absent from data."""
        (testlang_dir / "benchmark_items.jsonl").write_text(
            json.dumps({**item_record, "student_knowledge_id": "stu-sw-999"}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="stu-sw-999"):
            validate_benchmark_dataset("testlang")
