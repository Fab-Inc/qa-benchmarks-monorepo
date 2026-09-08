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
    ITEM_SET_FILENAMES,
    STUDENT_PROFILES_FILENAME,
    convert_dataframe_to_records,
    convert_records_to_dataframe,
    get_data_dir,
    read_benchmark_items,
    read_student_knowledge_profiles,
    resolve_language_iso_code,
    validate_student_knowledge,
    write_dataset,
)
from phonics_benchmark.models import (
    GenerationBenchmarkItem,
    StudentKnowledge,
)

LANGUAGE = "testlang"

PROFILE_RECORD = {
    "student_knowledge_id": "stu-001",
    "language_iso_code": "sw",
    "student_knowledge": {
        "graphemes": ["a"],
        "clusters": [],
        "cluster_patterns": [],
        "whole_words": ["mama", "paka"],
    },
}
# One (scenario x profile) pairing — the shape of both generation item files.
ITEM_RECORD = {
    "benchmark_item_id": "story_stu-001",
    "scenario_id": "story",
    "scenario": "Write a short story of about 50 words.",
    "student_knowledge_id": "stu-001",
    "language_iso_code": "sw",
    "student_knowledge": {
        "graphemes": ["a"],
        "clusters": [],
        "cluster_patterns": [],
        "whole_words": ["mama", "paka"],
    },
    "curriculum_phase": "very-early",
    "n_graphemes_known": 1,
    "n_clusters_known": 0,
    "n_cluster_patterns_known": 0,
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


# The readers are structurally identical, so one spec drives one set of tests across
# all of them. Adding a reader means adding a row here, not copying tests.
READER_ARGS = "reader, filename, record, model, id_field, wrong_typed_field"
READER_CASES = [
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
        read_benchmark_items,
        ITEM_SET_FILENAMES["full"],
        ITEM_RECORD,
        GenerationBenchmarkItem,
        "benchmark_item_id",
        {"n_graphemes_known": "not-an-int"},
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
        profiles = [StudentKnowledge(**PROFILE_RECORD)]
        items = [GenerationBenchmarkItem(**ITEM_RECORD)]

        write_dataset(
            LANGUAGE,
            tmp_path,
            profiles=profiles,
            item_sets={"full": items, "dev": items, "smoke": items},
        )

        assert read_student_knowledge_profiles(LANGUAGE, tmp_path) == profiles
        for name in ITEM_SET_FILENAMES:
            assert read_benchmark_items(LANGUAGE, tmp_path, item_set=name) == items

    def test_each_item_set_is_a_separate_file(self, tmp_path) -> None:
        """The sets never overwrite each other."""
        by_set = {
            name: [
                GenerationBenchmarkItem(
                    **{**ITEM_RECORD, "benchmark_item_id": f"story_{name}"}
                )
            ]
            for name in ITEM_SET_FILENAMES
        }

        write_dataset(LANGUAGE, tmp_path, item_sets=by_set)

        data_dir = get_data_dir(LANGUAGE, tmp_path)
        for name, filename in ITEM_SET_FILENAMES.items():
            assert (data_dir / filename).is_file()
            assert (
                read_benchmark_items(LANGUAGE, tmp_path, item_set=name) == by_set[name]
            )

    def test_writes_multiple_records_as_separate_lines(self, tmp_path) -> None:
        """Consecutive records stay separate — no missing newline gluing them."""
        profiles = [
            StudentKnowledge(**PROFILE_RECORD),
            StudentKnowledge(**{**PROFILE_RECORD, "student_knowledge_id": "stu-002"}),
        ]
        write_dataset(LANGUAGE, tmp_path, profiles=profiles)
        assert read_student_knowledge_profiles(LANGUAGE, tmp_path) == profiles

    def test_creates_missing_data_dir(self, tmp_path) -> None:
        """Creates the language directory when it doesn't exist yet."""
        data_root = tmp_path / "brand" / "new" / "root"

        data_dir = write_dataset(
            LANGUAGE, data_root, profiles=[StudentKnowledge(**PROFILE_RECORD)]
        )

        assert data_dir == get_data_dir(LANGUAGE, data_root)
        assert (data_dir / STUDENT_PROFILES_FILENAME).is_file()

    def test_only_writes_what_it_is_given(self, tmp_path) -> None:
        """Omitted collections leave no file behind."""
        data_dir = write_dataset(
            LANGUAGE, tmp_path, profiles=[StudentKnowledge(**PROFILE_RECORD)]
        )
        for filename in ITEM_SET_FILENAMES.values():
            assert not (data_dir / filename).exists()

    def test_unknown_item_set_name_raises(self, tmp_path) -> None:
        """A typo'd set name fails rather than writing an unexpected file."""
        items = [GenerationBenchmarkItem(**ITEM_RECORD)]
        with pytest.raises(ValueError, match="Unknown item_set 'nope'"):
            write_dataset(LANGUAGE, tmp_path, item_sets={"nope": items})


class TestReadBenchmarkItems:
    """Selecting between the item sets by name."""

    def test_unknown_item_set_name_raises(self, tmp_path) -> None:
        """Reading a set that does not exist names the valid ones."""
        with pytest.raises(ValueError, match="Unknown item_set 'nope'"):
            read_benchmark_items(LANGUAGE, tmp_path, item_set="nope")


class TestDataFrameBoundary:
    """Tests for the DataFrame<->models converters."""

    def test_round_trips_records_through_a_dataframe(self) -> None:
        """models -> DataFrame -> models returns the originals unchanged."""
        items = [GenerationBenchmarkItem(**ITEM_RECORD)]
        frame = convert_records_to_dataframe(items)
        assert convert_dataframe_to_records(frame, GenerationBenchmarkItem) == items

    def test_rejects_a_bad_row_naming_its_position_and_id(self) -> None:
        """A row failing validation names both its index and its item id."""
        frame = pd.DataFrame(
            [
                ITEM_RECORD,
                {
                    **ITEM_RECORD,
                    "benchmark_item_id": "bi-bad",
                    "curriculum_phase": "not-a-phase",
                },
            ]
        )
        with pytest.raises(ValueError, match=r"row 1 \(bi-bad\)"):
            convert_dataframe_to_records(frame, GenerationBenchmarkItem)

    def test_rejects_a_stray_intermediate_column(self) -> None:
        """extra="forbid" catches an in-memory-only column being persisted.

        The item builder carries working columns that must not reach a data
        file. This is the check that stops them.
        """
        frame = pd.DataFrame([{**ITEM_RECORD, "vocab_set": {"tatu"}}])
        with pytest.raises(ValueError, match="vocab_set"):
            convert_dataframe_to_records(frame, GenerationBenchmarkItem)


class TestResolveLanguageIsoCode:
    """Tests for resolve_language_iso_code()."""

    def test_returns_the_shared_code(self) -> None:
        """One language per data folder is the pipeline's rule."""
        profiles = [
            StudentKnowledge(**PROFILE_RECORD),
            StudentKnowledge(**{**PROFILE_RECORD, "student_knowledge_id": "stu-002"}),
        ]
        assert resolve_language_iso_code(profiles) == "sw"

    def test_no_profiles_raises(self) -> None:
        """There is no code to resolve from nothing."""
        with pytest.raises(ValueError, match="no student profiles"):
            resolve_language_iso_code([])

    def test_mixed_codes_raise(self) -> None:
        """A folder mixing languages is a data error, named explicitly."""
        profiles = [
            StudentKnowledge(**PROFILE_RECORD),
            StudentKnowledge(
                **{
                    **PROFILE_RECORD,
                    "student_knowledge_id": "stu-en",
                    "language_iso_code": "en",
                }
            ),
        ]
        with pytest.raises(ValueError, match="found: en, sw"):
            resolve_language_iso_code(profiles)


class TestValidateStudentKnowledge:
    """Tests for validate_student_knowledge()."""

    @pytest.fixture
    def unregistered_language_profile(self) -> StudentKnowledge:
        """A profile for a language with no registered knowledge validator.

        Uses ``xx`` — the ISO 639 code reserved for "no linguistic content" —
        so this stays unregistered no matter what languages are added.
        """
        return StudentKnowledge(
            student_knowledge_id="stu-xx",
            language_iso_code="xx",
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
        with pytest.warns(UserWarning, match="stu-xx"):
            validate_student_knowledge([unregistered_language_profile])

    def test_unregistered_language_raises_when_strict(
        self, unregistered_language_profile
    ) -> None:
        """Raises ValueError naming the profile when strict=True and no validator."""
        with pytest.raises(ValueError, match="stu-xx"):
            validate_student_knowledge([unregistered_language_profile], strict=True)
