"""Tests for the data_io module."""

import json
from pathlib import Path

import pytest
from phonics_benchmark.data_io import (
    get_data_dir,
    read_benchmark_items,
    read_decodable_texts,
    read_student_knowledge_profiles,
    validate_benchmark_dataset,
    validate_known_graphemes,
)
from phonics_benchmark.models import (
    BenchmarkItem,
    DecodableText,
    StudentKnowledgeProfile,
)


@pytest.fixture
def student_profile_record():
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
def real_data_root():
    """Path to the actual benchmark data directory."""
    return Path(__file__).resolve().parent.parent / "data"


@pytest.fixture
def patched_data_root(tmp_path):
    """Isolated temporary directory used as a data root in tests."""
    return tmp_path


@pytest.fixture
def testlang_dir(patched_data_root, text_record, student_profile_record, item_record):
    """A testlang/ directory pre-populated with one valid record per data file."""
    d = patched_data_root / "testlang"
    d.mkdir()
    (d / "decodable_texts.jsonl").write_text(
        json.dumps(text_record) + "\n", encoding="utf-8"
    )
    (d / "student_knowledge_profiles.jsonl").write_text(
        json.dumps(student_profile_record) + "\n", encoding="utf-8"
    )
    (d / "benchmark_items.jsonl").write_text(
        json.dumps(item_record) + "\n", encoding="utf-8"
    )
    return d


class TestGetDataDir:
    """Test for get_data_dir() function."""

    def test_resolves_existing_language(self, real_data_root) -> None:
        """Resolves the kiswahili data directory to an existing path."""
        path = get_data_dir("kiswahili", real_data_root)
        assert path.is_dir()


class TestReadStudentKnowledgeProfiles:
    """Test for read_student_knowledge_profiles() function."""

    def test_returns_profiles(self, real_data_root) -> None:
        """Returns the correct number of StudentKnowledgeProfile objects."""
        profiles = read_student_knowledge_profiles("kiswahili", real_data_root)
        assert len(profiles) == 2
        assert all(isinstance(p, StudentKnowledgeProfile) for p in profiles)

    def test_first_profile(self, real_data_root) -> None:
        """First profile has the expected ID, known_syllables, and max_word_length."""
        profiles = read_student_knowledge_profiles("kiswahili", real_data_root)
        first = profiles[0]
        assert first.student_knowledge_id == "stu-sw-001"
        assert "known_syllables" in first.taught_knowledge
        assert first.taught_knowledge["max_word_length"] == 4

    def test_missing_file_raises(self, real_data_root) -> None:
        """Raises FileNotFoundError naming the file when no profiles JSONL exists."""
        with pytest.raises(FileNotFoundError, match="student_knowledge_profiles.jsonl"):
            read_student_knowledge_profiles("english", real_data_root)

    def test_duplicate_id_raises(
        self, patched_data_root, student_profile_record
    ) -> None:
        """Raises ValueError when two records share the same student_knowledge_id."""
        d = patched_data_root / "testlang"
        d.mkdir()
        (d / "student_knowledge_profiles.jsonl").write_text(
            json.dumps(student_profile_record)
            + "\n"
            + json.dumps(student_profile_record)
            + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="duplicate student_knowledge_id"):
            read_student_knowledge_profiles("testlang", patched_data_root)

    def test_bad_line_reports_line_number(
        self, patched_data_root, student_profile_record
    ) -> None:
        """Error message includes the 1-based line number of the invalid record."""
        d = patched_data_root / "testlang"
        d.mkdir()
        (d / "student_knowledge_profiles.jsonl").write_text(
            json.dumps(student_profile_record) + "\n" + "not valid json\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=":2:"):
            read_student_knowledge_profiles("testlang", patched_data_root)

    def test_rejects_extra_field(
        self, patched_data_root, student_profile_record
    ) -> None:
        """Raises ValueError when a record contains an unexpected field."""
        d = patched_data_root / "testlang"
        d.mkdir()
        (d / "student_knowledge_profiles.jsonl").write_text(
            json.dumps({**student_profile_record, "unexpected_field": "value"}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=":1:"):
            read_student_knowledge_profiles("testlang", patched_data_root)

    def test_rejects_wrong_field_type(
        self, patched_data_root, student_profile_record
    ) -> None:
        """Raises ValueError when taught_knowledge is not a dict."""
        d = patched_data_root / "testlang"
        d.mkdir()
        (d / "student_knowledge_profiles.jsonl").write_text(
            json.dumps({**student_profile_record, "taught_knowledge": "not-a-dict"})
            + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=":1:"):
            read_student_knowledge_profiles("testlang", patched_data_root)

    def test_rejects_empty_id(self, patched_data_root, student_profile_record) -> None:
        """Raises ValueError when student_knowledge_id is an empty string."""
        d = patched_data_root / "testlang"
        d.mkdir()
        (d / "student_knowledge_profiles.jsonl").write_text(
            json.dumps({**student_profile_record, "student_knowledge_id": ""}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=":1:"):
            read_student_knowledge_profiles("testlang", patched_data_root)


class TestReadDecodableTexts:
    """Test for read_decodable_texts() function."""

    def test_returns_items(self, real_data_root) -> None:
        """Returns at least one DecodableText object for kiswahili."""
        items = read_decodable_texts("kiswahili", real_data_root)
        assert len(items) >= 1
        assert all(isinstance(item, DecodableText) for item in items)

    def test_first_item(self, real_data_root) -> None:
        """First item has the expected ID, language code, and non-empty content."""
        items = read_decodable_texts("kiswahili", real_data_root)
        first = items[0]
        assert first.text_id == "sw-001"
        assert first.language_iso_code == "sw"
        assert isinstance(first.content, str) and len(first.content) > 0

    def test_missing_file_raises(self, real_data_root) -> None:
        """Raises FileNotFoundError naming the file when no texts JSONL exists."""
        with pytest.raises(FileNotFoundError, match="decodable_texts.jsonl"):
            read_decodable_texts("english", real_data_root)

    def test_duplicate_id_raises(self, patched_data_root, text_record) -> None:
        """Raises ValueError when two records share the same text_id."""
        d = patched_data_root / "testlang"
        d.mkdir()
        (d / "decodable_texts.jsonl").write_text(
            json.dumps(text_record) + "\n" + json.dumps(text_record) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="duplicate text_id"):
            read_decodable_texts("testlang", patched_data_root)

    def test_bad_line_reports_line_number(self, patched_data_root, text_record) -> None:
        """Error message includes the 1-based line number of the invalid record."""
        d = patched_data_root / "testlang"
        d.mkdir()
        (d / "decodable_texts.jsonl").write_text(
            json.dumps(text_record) + "\n" + "not valid json\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=":2:"):
            read_decodable_texts("testlang", patched_data_root)

    def test_rejects_extra_field(self, patched_data_root, text_record) -> None:
        """Raises ValueError when a record contains an unexpected field."""
        d = patched_data_root / "testlang"
        d.mkdir()
        (d / "decodable_texts.jsonl").write_text(
            json.dumps({**text_record, "unexpected_field": "value"}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=":1:"):
            read_decodable_texts("testlang", patched_data_root)

    def test_rejects_wrong_field_type(self, patched_data_root, text_record) -> None:
        """Raises ValueError when content is not a string."""
        d = patched_data_root / "testlang"
        d.mkdir()
        (d / "decodable_texts.jsonl").write_text(
            json.dumps({**text_record, "content": ["not", "a", "string"]}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=":1:"):
            read_decodable_texts("testlang", patched_data_root)

    def test_rejects_empty_text_id(self, patched_data_root, text_record) -> None:
        """Raises ValueError when text_id is an empty string."""
        d = patched_data_root / "testlang"
        d.mkdir()
        (d / "decodable_texts.jsonl").write_text(
            json.dumps({**text_record, "text_id": ""}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=":1:"):
            read_decodable_texts("testlang", patched_data_root)


class TestReadBenchmarkItems:
    """Test for read_benchmark_items() function."""

    def test_returns_items(self, real_data_root) -> None:
        """Returns all four BenchmarkItem objects for kiswahili."""
        items = read_benchmark_items("kiswahili", real_data_root)
        assert len(items) == 4
        assert all(isinstance(item, BenchmarkItem) for item in items)

    def test_first_item(self, real_data_root) -> None:
        """First item links the expected text and student knowledge profile."""
        items = read_benchmark_items("kiswahili", real_data_root)
        first = items[0]
        assert first.benchmark_item_id == "bi-sw-001-stu-sw-001"
        assert first.text_id == "sw-001"
        assert first.student_knowledge_id == "stu-sw-001"

    def test_missing_file_raises(self, real_data_root) -> None:
        """Raises FileNotFoundError naming the file when no benchmark items JSONL."""
        with pytest.raises(FileNotFoundError, match="benchmark_items.jsonl"):
            read_benchmark_items("english", real_data_root)

    def test_duplicate_id_raises(self, patched_data_root, item_record) -> None:
        """Raises ValueError when two records share the same benchmark_item_id."""
        d = patched_data_root / "testlang"
        d.mkdir()
        (d / "benchmark_items.jsonl").write_text(
            json.dumps(item_record) + "\n" + json.dumps(item_record) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="duplicate benchmark_item_id"):
            read_benchmark_items("testlang", patched_data_root)

    def test_bad_line_reports_line_number(self, patched_data_root, item_record) -> None:
        """Error message includes the 1-based line number of the invalid record."""
        d = patched_data_root / "testlang"
        d.mkdir()
        (d / "benchmark_items.jsonl").write_text(
            json.dumps(item_record) + "\n" + "not valid json\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=":2:"):
            read_benchmark_items("testlang", patched_data_root)

    def test_rejects_extra_field(self, patched_data_root, item_record) -> None:
        """Raises ValueError when a record contains an unexpected field."""
        d = patched_data_root / "testlang"
        d.mkdir()
        (d / "benchmark_items.jsonl").write_text(
            json.dumps({**item_record, "unexpected_field": "value"}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=":1:"):
            read_benchmark_items("testlang", patched_data_root)

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
            read_benchmark_items("testlang", patched_data_root)

    def test_rejects_empty_id(self, patched_data_root, item_record) -> None:
        """Raises ValueError when benchmark_item_id is an empty string."""
        d = patched_data_root / "testlang"
        d.mkdir()
        (d / "benchmark_items.jsonl").write_text(
            json.dumps({**item_record, "benchmark_item_id": ""}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=":1:"):
            read_benchmark_items("testlang", patched_data_root)


class TestValidateBenchmarkDataset:
    """Test for validate_benchmark_dataset() function."""

    def test_passes_for_valid_data(self, real_data_root) -> None:
        """Passes without error when all benchmark item references are valid."""
        validate_benchmark_dataset("kiswahili", real_data_root)

    def test_unknown_text_id_raises(
        self, testlang_dir, item_record, patched_data_root
    ) -> None:
        """Raises ValueError naming the text_id when absent from decodable_texts."""
        (testlang_dir / "benchmark_items.jsonl").write_text(
            json.dumps({**item_record, "text_id": "sw-999"}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="sw-999"):
            validate_benchmark_dataset("testlang", patched_data_root)

    def test_unknown_student_knowledge_id_raises(
        self, testlang_dir, item_record, patched_data_root
    ) -> None:
        """Raises ValueError naming the student_knowledge_id when absent from data."""
        (testlang_dir / "benchmark_items.jsonl").write_text(
            json.dumps({**item_record, "student_knowledge_id": "stu-sw-999"}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="stu-sw-999"):
            validate_benchmark_dataset("testlang", patched_data_root)


class TestValidateKnownGraphemes:
    """Test for validate_known_graphemes() function."""

    def test_passes_for_valid_kiswahili_graphemes(self) -> None:
        """Passes without error for a profile with only valid Kiswahili graphemes."""
        profile = StudentKnowledgeProfile(
            student_knowledge_id="stu-sw-001",
            language_iso_code="sw",
            taught_knowledge={"graphemes": ["a", "m"], "whole_words": ["mama"]},
        )
        validate_known_graphemes([profile])

    def test_invalid_kiswahili_grapheme_raises(self) -> None:
        """Raises ValueError naming the profile when a grapheme isn't valid."""
        profile = StudentKnowledgeProfile(
            student_knowledge_id="stu-bad",
            language_iso_code="sw",
            taught_knowledge={"graphemes": ["a", "xyz"], "whole_words": []},
        )
        with pytest.raises(ValueError, match="stu-bad"):
            validate_known_graphemes([profile])

    def test_unregistered_language_is_skipped(self) -> None:
        """Does not raise for a language with no registered grapheme validator."""
        profile = StudentKnowledgeProfile(
            student_knowledge_id="stu-en",
            language_iso_code="en",
            taught_knowledge={"graphemes": ["not-a-real-grapheme"], "whole_words": []},
        )
        validate_known_graphemes([profile])

    def test_non_list_graphemes_raises(self) -> None:
        """Raises ValueError naming the profile when graphemes isn't a list."""
        profile = StudentKnowledgeProfile(
            student_knowledge_id="stu-malformed",
            language_iso_code="sw",
            taught_knowledge={"graphemes": 7, "whole_words": []},
        )
        with pytest.raises(ValueError, match="stu-malformed"):
            validate_known_graphemes([profile])
