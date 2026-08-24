"""Tests for the data_io module.

Every test builds its own dataset under ``tmp_path``. Nothing here reads the repo's
``data/`` directory: these tests cover the readers' parsing and validation logic, not
the contents of any particular dataset, and we expect to keep adding datasets (new
student knowledge profiles, new text sets) that would otherwise keep breaking them.
"""

import json
import re
from pathlib import Path

import pandas as pd
import pytest
from phonics_benchmark.data_io import (
    BENCHMARK_CANDIDATES_TO_INSPECT_FILENAME,
    BENCHMARK_ITEMS_FILENAME,
    STUDENT_PROFILES_FILENAME,
    TEXTS_FILENAME,
    convert_dataframe_to_records,
    convert_records_to_dataframe,
    get_data_dir,
    read_benchmark_item_candidates,
    read_benchmark_items,
    read_decodable_texts,
    read_student_knowledge_profiles,
    validate_student_knowledge,
    write_dataset,
)
from phonics_benchmark.models import (
    DecodableText,
    LabelledBenchmarkItem,
    StudentKnowledge,
)

LANGUAGE = "testlang"

TEXT_RECORD = {"text_id": "t-001", "language_iso_code": "sw", "content": "Tatu."}
PROFILE_RECORD = {
    "student_knowledge_id": "stu-001",
    "language_iso_code": "sw",
    "student_knowledge": {"graphemes": ["a"], "whole_words": ["mama", "paka"]},
}
# The bare pairing, as written to benchmark_items_full_crossproduct.jsonl.
ITEM_RECORD = {
    "benchmark_item_id": "bi-001",
    "text_id": "t-001",
    "student_knowledge_id": "stu-001",
}
# The same pairing with text, knowledge and analysis inlined — the shape of both
# benchmark_item_candidates.jsonl and benchmark_items.jsonl.
CANDIDATE_RECORD = {
    **ITEM_RECORD,
    "language_iso_code": "sw",
    "content": "Tatu.",
    "student_knowledge": {"graphemes": ["a"], "whole_words": ["mama", "paka"]},
    "nondecodable_words": ["tatu"],
    "num_nondecodable_words": 1,
    "num_total_words": 4,
    "nondecodable_ratio": 0.25,
}


def write_data_file(data_root: Path, filename: str, *lines: str) -> Path:
    """Write raw JSONL lines to <data_root>/<LANGUAGE>/<filename>.

    Unlike the ``phonics_benchmark.data_io.write_dataset``, this takes
    already-serialised strings so that malformed records can be written for
    negative tests.
    """
    data_dir = get_data_dir(LANGUAGE, data_root)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / filename
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return path


def at_line(filename: str, line_no: int) -> str:
    """Regex matching an error that points at a specific line of a data file."""
    return rf"{re.escape(filename)}:{line_no}:"


@pytest.fixture
def valid_dataset(tmp_path) -> Path:
    """A data root whose cross-product references texts and profiles that exist."""
    write_data_file(tmp_path, TEXTS_FILENAME, json.dumps(TEXT_RECORD))
    write_data_file(tmp_path, STUDENT_PROFILES_FILENAME, json.dumps(PROFILE_RECORD))
    write_data_file(
        tmp_path, BENCHMARK_CANDIDATES_TO_INSPECT_FILENAME, json.dumps(ITEM_RECORD)
    )
    return tmp_path


# The readers are structurally identical, so one spec drives one set of tests across
# all of them. Adding a reader means adding a row here, not copying tests.
READER_ARGS = "reader, filename, record, model, id_field, wrong_typed_field"
READER_CASES = [
    pytest.param(
        read_decodable_texts,
        TEXTS_FILENAME,
        TEXT_RECORD,
        DecodableText,
        "text_id",
        {"content": ["not", "a", "string"]},
        id="texts",
    ),
    pytest.param(
        read_student_knowledge_profiles,
        STUDENT_PROFILES_FILENAME,
        PROFILE_RECORD,
        StudentKnowledge,
        "student_knowledge_id",
        {"student_knowledge": "not-a-dict"},
        id="profiles",
    ),
    pytest.param(
        read_benchmark_item_candidates,
        BENCHMARK_CANDIDATES_TO_INSPECT_FILENAME,
        CANDIDATE_RECORD,
        LabelledBenchmarkItem,
        "benchmark_item_id",
        {"num_total_words": "not-an-int"},
        id="candidates",
    ),
    pytest.param(
        read_benchmark_items,
        BENCHMARK_ITEMS_FILENAME,
        CANDIDATE_RECORD,
        LabelledBenchmarkItem,
        "benchmark_item_id",
        {"nondecodable_words": "not-a-list"},
        id="items",
    ),
]


class TestGetDataDir:
    """Tests for get_data_dir()."""

    def test_joins_language_onto_data_root(self) -> None:
        """Resolves to <data_root>/<language>, without touching the filesystem."""
        assert get_data_dir("kiswahili", Path("/data")) == Path("/data/kiswahili")

    def test_languages_resolve_to_distinct_dirs(self) -> None:
        """Two languages under one root never collide."""
        root = Path("/data")
        assert get_data_dir("kiswahili", root) != get_data_dir("english", root)


@pytest.mark.parametrize(READER_ARGS, READER_CASES)
class TestReaders:
    """Behaviour shared by all three JSONL readers."""

    def test_returns_parsed_records(
        self, tmp_path, reader, filename, record, model, id_field, wrong_typed_field
    ) -> None:
        """Parses every record into the reader's model, preserving order and IDs."""
        second = {**record, id_field: "second-id"}
        write_data_file(tmp_path, filename, json.dumps(record), json.dumps(second))

        parsed = reader(LANGUAGE, tmp_path)

        assert [type(p) for p in parsed] == [model, model]
        assert [getattr(p, id_field) for p in parsed] == [record[id_field], "second-id"]

    def test_missing_file_raises(
        self, tmp_path, reader, filename, record, model, id_field, wrong_typed_field
    ) -> None:
        """Raises FileNotFoundError naming the file when it doesn't exist."""
        with pytest.raises(FileNotFoundError, match=re.escape(filename)):
            reader(LANGUAGE, tmp_path)

    def test_duplicate_id_raises(
        self, tmp_path, reader, filename, record, model, id_field, wrong_typed_field
    ) -> None:
        """Raises ValueError when two records share the same ID."""
        write_data_file(tmp_path, filename, json.dumps(record), json.dumps(record))
        with pytest.raises(ValueError, match=f"duplicate {id_field}"):
            reader(LANGUAGE, tmp_path)

    def test_skips_blank_lines(
        self, tmp_path, reader, filename, record, model, id_field, wrong_typed_field
    ) -> None:
        """Blank and whitespace-only lines are ignored rather than parsed."""
        second = {**record, id_field: "second-id"}
        write_data_file(
            tmp_path, filename, json.dumps(record), "", "   ", json.dumps(second), ""
        )
        assert len(reader(LANGUAGE, tmp_path)) == 2

    def test_bad_line_reports_line_number(
        self, tmp_path, reader, filename, record, model, id_field, wrong_typed_field
    ) -> None:
        """Error names the file and the 1-based line number of the bad record.

        Blank lines are skipped but still counted, so the number points at the
        offending line as it appears in the file.
        """
        write_data_file(tmp_path, filename, json.dumps(record), "", "not valid json")
        with pytest.raises(ValueError, match=at_line(filename, 3)):
            reader(LANGUAGE, tmp_path)

    def test_rejects_extra_field(
        self, tmp_path, reader, filename, record, model, id_field, wrong_typed_field
    ) -> None:
        """Raises ValueError when a record carries an unexpected field."""
        write_data_file(
            tmp_path, filename, json.dumps({**record, "unexpected_field": "value"})
        )
        with pytest.raises(ValueError, match=at_line(filename, 1)):
            reader(LANGUAGE, tmp_path)

    def test_rejects_wrong_field_type(
        self, tmp_path, reader, filename, record, model, id_field, wrong_typed_field
    ) -> None:
        """Raises ValueError when a field holds a value of the wrong type."""
        write_data_file(tmp_path, filename, json.dumps({**record, **wrong_typed_field}))
        with pytest.raises(ValueError, match=at_line(filename, 1)):
            reader(LANGUAGE, tmp_path)

    def test_rejects_empty_id(
        self, tmp_path, reader, filename, record, model, id_field, wrong_typed_field
    ) -> None:
        """Raises ValueError when the ID field is an empty string."""
        write_data_file(tmp_path, filename, json.dumps({**record, id_field: ""}))
        with pytest.raises(ValueError, match=at_line(filename, 1)):
            reader(LANGUAGE, tmp_path)


class TestWriteDataset:
    """Tests for write_dataset()."""

    def test_round_trips_through_the_readers(self, tmp_path) -> None:
        """Everything written is read back identically by the matching reader.

        This pins the writers and the readers to each other: whatever a pipeline
        stage writes must always load again in the next one.
        """
        texts = [DecodableText(**TEXT_RECORD)]
        profiles = [StudentKnowledge(**PROFILE_RECORD)]
        candidates = [LabelledBenchmarkItem(**CANDIDATE_RECORD)]

        write_dataset(
            LANGUAGE,
            tmp_path,
            texts=texts,
            profiles=profiles,
            benchmark_candidates=candidates,
            benchmark_items=candidates,
        )

        assert read_decodable_texts(LANGUAGE, tmp_path) == texts
        assert read_student_knowledge_profiles(LANGUAGE, tmp_path) == profiles
        assert read_benchmark_item_candidates(LANGUAGE, tmp_path) == candidates
        assert read_benchmark_items(LANGUAGE, tmp_path) == candidates

    def test_writes_multiple_records_as_separate_lines(self, tmp_path) -> None:
        """Consecutive records stay separate — no missing newline gluing them."""
        texts = [
            DecodableText(**TEXT_RECORD),
            DecodableText(**{**TEXT_RECORD, "text_id": "t-002"}),
        ]
        write_dataset(LANGUAGE, tmp_path, texts=texts)
        assert read_decodable_texts(LANGUAGE, tmp_path) == texts

    def test_creates_missing_data_dir(self, tmp_path) -> None:
        """Creates the language directory when it doesn't exist yet."""
        data_root = tmp_path / "brand" / "new" / "root"

        data_dir = write_dataset(
            LANGUAGE, data_root, texts=[DecodableText(**TEXT_RECORD)]
        )

        assert data_dir == get_data_dir(LANGUAGE, data_root)
        assert (data_dir / TEXTS_FILENAME).is_file()


class TestDataFrameBoundary:
    """Tests for the DataFrame<->models converters."""

    def test_round_trips_records_through_a_dataframe(self) -> None:
        """models -> DataFrame -> models returns the originals unchanged."""
        candidates = [LabelledBenchmarkItem(**CANDIDATE_RECORD)]
        frame = convert_records_to_dataframe(candidates)
        assert convert_dataframe_to_records(frame, LabelledBenchmarkItem) == candidates

    def test_rejects_a_bad_row_naming_its_position_and_id(self) -> None:
        """A row failing validation names both its index and its item id."""
        frame = pd.DataFrame(
            [
                CANDIDATE_RECORD,
                {
                    **CANDIDATE_RECORD,
                    "benchmark_item_id": "bi-bad",
                    "nondecodable_ratio": 42.0,
                },
            ]
        )
        with pytest.raises(ValueError, match=r"row 1 \(bi-bad\)"):
            convert_dataframe_to_records(frame, LabelledBenchmarkItem)

    def test_rejects_a_stray_intermediate_column(self) -> None:
        """extra="forbid" catches an in-memory-only column being persisted.

        `vocab_set` holds Python sets and is derivable from `content`; it must not
        reach a data file. This is the check that stops it.
        """
        frame = pd.DataFrame([{**CANDIDATE_RECORD, "vocab_set": {"tatu"}}])
        with pytest.raises(ValueError, match="vocab_set"):
            convert_dataframe_to_records(frame, LabelledBenchmarkItem)


class TestValidateStudentKnowledge:
    """Tests for validate_student_knowledge()."""

    @pytest.fixture
    def unregistered_language_profile(self) -> StudentKnowledge:
        """A profile for a language with no registered knowledge validator."""
        return StudentKnowledge(
            student_knowledge_id="stu-en",
            language_iso_code="en",
            student_knowledge={"graphemes": ["not-a-real-grapheme"], "whole_words": []},
        )

    def test_passes_for_valid_kiswahili_student_knowledge(self) -> None:
        """Passes without error for a profile with valid Kiswahili student_knowledge."""
        profile = StudentKnowledge(
            student_knowledge_id="stu-sw-001",
            language_iso_code="sw",
            student_knowledge={"graphemes": ["a", "m"], "whole_words": ["mama"]},
        )
        validate_student_knowledge([profile])

    def test_invalid_kiswahili_student_knowledge_raises(self) -> None:
        """Raises ValueError naming the profile when a grapheme isn't valid."""
        profile = StudentKnowledge(
            student_knowledge_id="stu-bad",
            language_iso_code="sw",
            student_knowledge={"graphemes": ["a", "xyz"], "whole_words": []},
        )
        with pytest.raises(ValueError, match="stu-bad"):
            validate_student_knowledge([profile])

    def test_unregistered_language_warns_but_does_not_raise(
        self, unregistered_language_profile
    ) -> None:
        """Warns, but does not raise, for a language with no registered validator."""
        with pytest.warns(UserWarning, match="stu-en"):
            validate_student_knowledge([unregistered_language_profile])

    def test_unregistered_language_raises_when_strict(
        self, unregistered_language_profile
    ) -> None:
        """Raises ValueError naming the profile when strict=True and no validator."""
        with pytest.raises(ValueError, match="stu-en"):
            validate_student_knowledge([unregistered_language_profile], strict=True)
