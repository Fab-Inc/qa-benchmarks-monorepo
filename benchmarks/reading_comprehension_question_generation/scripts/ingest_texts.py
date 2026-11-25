r"""Ingest a single-language rcqg workbook into ``data/<language>/texts.jsonl``.

Run from the repository root with (as an example):

    uv run --package reading-comprehension-question-generation python \\
        benchmarks/reading_comprehension_question_generation/scripts/ingest_texts.py \\
        --language english \\
        --workbook path/to/english_workbook.xlsx

There is one workbook per language. ``--language`` only selects the data folder; it is
not compared with the workbook's ``language_iso_code``. The ``Texts`` sheet is read by
column header, so column order does not matter and extra columns are ignored.

Ingestion is strict about structure and relaxed about content. It fails, naming the
rows or values, on a duplicate ``text_id``, a populated row missing ``text_id``,
``text_content``, ``language_iso_code`` or ``grade``, or more than one language code.
"""

import re
from collections.abc import Hashable
from pathlib import Path

import pandas as pd
import typer
from phonics_benchmark.data_io import write_jsonl
from phonics_benchmark.utils import require_single_language_iso_code
from reading_comprehension_question_generation.models import TextRecord

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"

TEXTS_SHEET = "Texts"
TEXTS_FILENAME = "texts.jsonl"

# The four fields the pipeline depends on; a populated row must have all of them.
REQUIRED_FIELDS = ("text_id", "text_content", "language_iso_code", "grade")
FIELD_NAMES = tuple(TextRecord.model_fields)


# Plain digits with an optional fractional part of zeros only: 3, 03, 3.0. Anything else
# (3.5, +3, 1e1, 3,5, Grade 3) is rejected. [0-9] is explicit so that digits from other
# scripts are not accepted.
_GRADE_PATTERN = re.compile(r"([0-9]+)(\.0+)?")


class TextIngestError(ValueError):
    """Raised when a workbook cannot be ingested; the message names the problems."""


def validate_language_folder(language: str) -> str:
    """Return ``language`` if it is safe to use as a single data-folder name.

    The argument only selects ``data/<language>/``. It is not looked up in a list and
    is not compared with the workbook's language code.
    """
    if (
        not language.strip()
        or language in {".", ".."}
        or language != Path(language).name
        or "\\" in language
    ):
        raise TextIngestError(
            f"Invalid --language '{language}': use a single folder name such as "
            "'english', without path separators."
        )
    return language


def parse_grade(value: str) -> int | None:
    """Return ``value`` as a positive whole number, or None if it is not one.

    ``value`` is a trimmed, non-blank workbook cell.
    """
    match = _GRADE_PATTERN.fullmatch(value)
    if match is None:
        return None
    grade = int(match.group(1))
    return grade if grade >= 1 else None


def read_texts_workbook(workbook: Path) -> list[TextRecord]:
    """Read the ``Texts`` sheet of a single-language workbook into text records.

    All problems found are reported together in one ``TextIngestError``.
    """
    try:
        # dtype=str keeps every cell as typed; keep_default_na=False stops pandas
        # turning values such as "None" or "NA" into missing values.
        sheet = pd.read_excel(
            workbook, sheet_name=TEXTS_SHEET, dtype=str, keep_default_na=False
        )
    except ValueError as exc:
        # pandas raises ValueError when the sheet does not exist.
        raise TextIngestError(f"Cannot read sheet '{TEXTS_SHEET}': {exc}") from exc

    missing_columns = [name for name in REQUIRED_FIELDS if name not in sheet.columns]
    if missing_columns:
        raise TextIngestError(
            f"Sheet '{TEXTS_SHEET}' is missing required column(s): "
            f"{', '.join(missing_columns)}"
        )

    # Trim every cell (including columns we ignore) so that a row of spaces counts as
    # blank, then drop all-blank rows as spacing. The DataFrame index is not
    # reset, so it still identifies each row: Excel row = index + 2 (the header is
    # row 1).
    sheet = sheet.apply(lambda column: column.str.strip())
    rows = sheet[(sheet != "").any(axis=1)]

    problems: list[str] = []
    if rows.empty:
        raise TextIngestError(
            f"Cannot ingest {workbook.name}:\n  - no populated rows in sheet "
            f"'{TEXTS_SHEET}'"
        )

    for field in REQUIRED_FIELDS:
        for index in rows.index[rows[field] == ""]:
            problems.append(f"row {index + 2}: missing {field}")

    # Grade is parsed once, here. A blank grade is already reported as missing above.
    grades: dict[Hashable, int] = {}  # keyed by the DataFrame row index
    for index in rows.index[rows["grade"] != ""]:
        value = str(rows.at[index, "grade"])  # cells are strings (dtype=str)
        grade = parse_grade(value)
        if grade is None:
            problems.append(
                f"row {index + 2}: grade '{value}' is not a positive whole number"
            )
        else:
            grades[index] = grade

    with_id = rows[rows["text_id"] != ""]
    repeated = with_id[with_id["text_id"].duplicated(keep=False)]
    for text_id, group in repeated.groupby("text_id", sort=False):
        excel_rows = ", ".join(str(index + 2) for index in group.index)
        problems.append(f"duplicate text_id '{text_id}' in rows {excel_rows}")

    language_codes = rows.loc[rows["language_iso_code"] != "", "language_iso_code"]
    if not language_codes.empty:
        # Reuses the phonics check: one workbook holds one language.
        try:
            require_single_language_iso_code(
                language_codes, source=f"sheet '{TEXTS_SHEET}'"
            )
        except ValueError as exc:
            problems.append(str(exc))

    if problems:
        raise TextIngestError(
            f"Cannot ingest {workbook.name}:\n  - " + "\n  - ".join(problems)
        )

    # Keep only the fields in the model; blank optional values become None. The four
    # required fields are valid here (checked above), and grade is the parsed integer.
    # model_validate takes a plain dict, so the type checker does not see "None" going
    # into a required argument.
    records = []
    for index, row in rows.to_dict(orient="index").items():
        values = {name: row.get(name) or None for name in FIELD_NAMES}
        values["grade"] = grades[index]
        records.append(TextRecord.model_validate(values))
    return records


def ingest_workbook(workbook: Path, language: str, data_root: Path) -> tuple[Path, int]:
    """Ingest ``workbook`` and write ``data_root/<language>/texts.jsonl``.

    Returns the written path and the number of text records.
    """
    folder = validate_language_folder(language)
    records = read_texts_workbook(workbook)

    output_path = data_root / folder / TEXTS_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(records, output_path)
    return output_path, len(records)


def ingest_texts(
    language: str = typer.Option(
        ...,
        "--language",
        "-l",
        help="Data folder name under the data root, for example 'english'.",
    ),
    workbook: Path = typer.Option(
        ...,
        "--workbook",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the filled-in single-language .xlsx workbook.",
    ),
    data_root: Path = typer.Option(
        DEFAULT_DATA_ROOT,
        "--data-root",
        file_okay=False,
        help="Root data directory. Defaults to the package's data/ folder.",
    ),
) -> None:
    """Parse a workbook's Texts sheet into text records for one language."""
    try:
        output_path, count = ingest_workbook(workbook, language, data_root)
    except TextIngestError as exc:
        # Print the problems without a traceback; they are input errors, not bugs.
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo(f"Wrote {count} text record(s) to {output_path}")


def main() -> None:
    """Entry point."""
    typer.run(ingest_texts)


if __name__ == "__main__":
    main()
