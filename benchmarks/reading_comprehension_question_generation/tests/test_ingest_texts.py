import json
from pathlib import Path
from types import ModuleType

import pytest
import typer
from typer.testing import CliRunner

EXPECTED_KEYS = [
    "text_id",
    "text_title",
    "text_content",
    "text_type",
    "country_code",
    "language_iso_code",
    "grade",
    "source",
    "location",
    "source_grade_label",
    "is_target",
    "is_adaptation",
    "adaptation_source",
    "has_images",
]
NOT_CARRIED = {"existing_comprehension_questions", "notes"}


def _cli(script: ModuleType) -> typer.Typer:
    """Wrap the script's command in an app so CliRunner can invoke it."""
    app = typer.Typer()
    app.command()(script.ingest_texts)
    return app


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_columns_in_any_order_and_extra_columns_are_ignored(
    tmp_path, ingest_script, text_row, write_workbook, workbook_columns
) -> None:
    rows = [text_row(), text_row(text_id="ENG-002", text_title="The Dog")]
    ordered = write_workbook(tmp_path / "ordered.xlsx", rows)

    shuffled_columns = [*reversed(workbook_columns), "unexpected_extra"]
    changed = [
        {
            **row,
            "existing_comprehension_questions": "different questions",
            "notes": "different notes",
            "unexpected_extra": "ignored",
        }
        for row in rows
    ]
    shuffled = write_workbook(
        tmp_path / "shuffled.xlsx", changed, columns=shuffled_columns
    )

    from_ordered, _ = ingest_script.ingest_workbook(ordered, "a", tmp_path / "out_a")
    from_shuffled, _ = ingest_script.ingest_workbook(shuffled, "a", tmp_path / "out_b")

    records_a = _read_jsonl(from_ordered)
    records_b = _read_jsonl(from_shuffled)
    assert records_a == records_b
    assert [r["text_id"] for r in records_a] == ["ENG-001", "ENG-002"]
    for record in records_a:
        assert list(record) == EXPECTED_KEYS
        assert not NOT_CARRIED & set(record)


def test_values_are_kept_exactly_as_typed(
    tmp_path, ingest_script, text_row, write_workbook
) -> None:
    row = text_row(
        text_id=" ENG-001 ",
        text_title="  A Title  ",
        text_content="\n  First line.\nSecond line.  \n",
        text_type="None",
        country_code="XX",
        grade="zz",
        source="maybe",
        location="NA",
        source_grade_label="   ",
        is_target="TRUE",
        is_adaptation="yes",
        adaptation_source=None,
        has_images="0",
    )
    path = write_workbook(tmp_path / "w.xlsx", [row])

    (record,) = ingest_script.read_texts_workbook(path)

    assert record.model_dump() == {
        "text_id": "ENG-001",
        "text_title": "A Title",
        "text_content": "First line.\nSecond line.",
        "text_type": "None",
        "country_code": "XX",
        "language_iso_code": "en",
        "grade": "zz",
        "source": "maybe",
        "location": "NA",
        "source_grade_label": None,
        "is_target": "TRUE",
        "is_adaptation": "yes",
        "adaptation_source": None,
        "has_images": "0",
    }


def test_duplicate_text_id_is_an_error_naming_the_rows(
    tmp_path, ingest_script, text_row, write_workbook
) -> None:
    rows = [text_row(text_id="A"), text_row(text_id="B"), text_row(text_id="A")]
    path = write_workbook(tmp_path / "w.xlsx", rows)

    with pytest.raises(ingest_script.TextIngestError) as error:
        ingest_script.read_texts_workbook(path)

    assert "duplicate text_id 'A' in rows 2, 4" in str(error.value)


@pytest.mark.parametrize("field", ["text_id", "text_content", "language_iso_code"])
def test_populated_row_missing_a_required_field_is_an_error(
    tmp_path, ingest_script, text_row, write_workbook, field
) -> None:
    second = text_row(text_id="B")
    second[field] = None
    path = write_workbook(tmp_path / "w.xlsx", [text_row(), second])

    with pytest.raises(ingest_script.TextIngestError) as error:
        ingest_script.read_texts_workbook(path)

    assert f"row 3: missing {field}" in str(error.value)


def test_more_than_one_language_code_is_an_error(
    tmp_path, ingest_script, text_row, write_workbook
) -> None:
    rows = [
        text_row(text_id="A", language_iso_code="en"),
        text_row(text_id="B", language_iso_code="sw"),
    ]
    path = write_workbook(tmp_path / "w.xlsx", rows)

    with pytest.raises(ingest_script.TextIngestError) as error:
        ingest_script.read_texts_workbook(path)

    assert "Expected one language_iso_code" in str(error.value)
    assert "en, sw" in str(error.value)


def test_workbook_with_no_populated_rows_is_an_error(
    tmp_path, ingest_script, write_workbook
) -> None:
    path = write_workbook(tmp_path / "w.xlsx", [])

    with pytest.raises(ingest_script.TextIngestError, match="no populated rows"):
        ingest_script.read_texts_workbook(path)


def test_unsafe_language_argument_is_rejected_and_writes_nothing(
    tmp_path, ingest_script, text_row, write_workbook
) -> None:
    workbook = write_workbook(tmp_path / "w.xlsx", [text_row()])
    data_root = tmp_path / "data"

    with pytest.raises(ingest_script.TextIngestError, match="Invalid --language"):
        ingest_script.ingest_workbook(workbook, "../escape", data_root)

    assert not (tmp_path / "escape").exists()
    assert not data_root.exists()


def test_cli_reports_problems_without_a_traceback_and_writes_no_file(
    tmp_path, ingest_script, text_row, write_workbook
) -> None:
    workbook = write_workbook(
        tmp_path / "w.xlsx", [text_row(text_id="A"), text_row(text_id="A")]
    )
    data_root = tmp_path / "data"

    result = CliRunner().invoke(
        _cli(ingest_script),
        [
            "--language",
            "english",
            "--workbook",
            str(workbook),
            "--data-root",
            str(data_root),
        ],
    )

    assert result.exit_code == 1
    assert "ERROR:" in result.output
    assert "duplicate text_id 'A'" in result.output
    assert "Traceback" not in result.output
    assert not (data_root / "english" / "texts.jsonl").exists()


def test_missing_required_column_is_an_error(
    tmp_path, ingest_script, text_row, write_workbook, workbook_columns
) -> None:
    columns = [name for name in workbook_columns if name != "language_iso_code"]
    path = write_workbook(tmp_path / "w.xlsx", [text_row()], columns=columns)

    with pytest.raises(
        ingest_script.TextIngestError,
        match=r"missing required column\(s\): language_iso_code",
    ):
        ingest_script.read_texts_workbook(path)
