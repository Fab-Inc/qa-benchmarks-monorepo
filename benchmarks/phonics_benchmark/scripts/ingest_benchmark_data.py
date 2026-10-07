r"""Ingest a language benchmark data-entry workbook into JSONL data files.

Run from the repository root with (as an example):

    uv run --package phonics-benchmark python \\
        benchmarks/phonics_benchmark/scripts/ingest_benchmark_data.py \\
        --language kiswahili \\
        --workbook path/to/benchmark_template.xlsx

The workbook is a template filled in by language experts, with a "Student
Profiles" tab and a "Texts" tab. Machine-read columns are identified by exact
header name (case-sensitive), independent of column order, so the template
can carry additional expert-context columns without breaking this script.
Rows with a blank ID column are treated as spacing/notes and skipped.

The cell parsers and sheet helpers live in ``phonics_benchmark.ingest`` so
other benchmarks can share them.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from phonics_benchmark.data_io import validate_student_knowledge, write_dataset
from phonics_benchmark.ingest import (
    PROFILE_TYPE_COLUMN,
    STUDENT_KNOWLEDGE_FIELDS,
    get_language_iso_code,
    is_jagged_profile,
    preprocess_sheet_data,
)
from phonics_benchmark.models import DecodableText, StudentKnowledge

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"

STUDENT_PROFILES_SHEET = "Student Profiles"
TEXTS_SHEET = "Texts"


def extract_student_profiles(
    df: pd.DataFrame,
    fields: dict[str, Callable[[str], Any]],
) -> list[StudentKnowledge]:
    return [
        StudentKnowledge(
            student_knowledge_id=str(row.student_knowledge_id),
            language_iso_code=str(row.language_iso_code),
            jagged=is_jagged_profile(str(getattr(row, PROFILE_TYPE_COLUMN))),
            student_knowledge={
                field: parser(str(getattr(row, field)))
                for field, parser in fields.items()
            },
        )
        for row in df.itertuples(index=False)
    ]


def extract_texts(df: pd.DataFrame) -> list[DecodableText]:
    return [
        DecodableText(
            text_id=str(row.text_id),
            language_iso_code=str(row.language_iso_code),
            content=str(row.content),
        )
        for row in df.itertuples(index=False)
    ]


def ingest_benchmark_data(
    language: str = typer.Option(
        ...,
        "--language",
        "-l",
        help="Language folder name under the data root, for example 'kiswahili'.",
    ),
    workbook: Path = typer.Option(
        ...,
        "--workbook",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the filled-in .xlsx data-entry template.",
    ),
    data_root: Path = typer.Option(
        DEFAULT_DATA_ROOT,
        "--data-root",
        file_okay=False,
        help="Root data directory. Defaults to the package's data/ folder.",
    ),
) -> None:
    """Parse a data-entry workbook into decodable texts and student profiles."""
    student_df = pd.read_excel(workbook, sheet_name=STUDENT_PROFILES_SHEET)
    texts_df = pd.read_excel(workbook, sheet_name=TEXTS_SHEET)

    student_iso = get_language_iso_code(student_df, f"'{STUDENT_PROFILES_SHEET}' sheet")

    student_fields = STUDENT_KNOWLEDGE_FIELDS.get(student_iso)
    if student_fields is None:
        raise ValueError(
            f"No student_knowledge field registry for language '{student_iso}'. "
            f"Add an entry to STUDENT_KNOWLEDGE_FIELDS."
        )

    student_df = preprocess_sheet_data(
        student_df,
        {"student_knowledge_id", "language_iso_code", PROFILE_TYPE_COLUMN}
        | set(student_fields.keys()),
        "student_knowledge_id",
        f"'{STUDENT_PROFILES_SHEET}' sheet",
    )
    texts_df = preprocess_sheet_data(
        texts_df,
        {"text_id", "language_iso_code", "content"},
        "text_id",
        f"'{TEXTS_SHEET}' sheet",
    )

    profiles = extract_student_profiles(student_df, student_fields)
    validate_student_knowledge(profiles, strict=True)
    texts = extract_texts(texts_df)

    data_dir = write_dataset(language, data_root, texts=texts, profiles=profiles)

    typer.echo(
        f"Wrote {len(profiles)} student knowledge profiles and "
        f"{len(texts)} decodable texts to {data_dir}"
    )
    typer.echo(
        "Next: build_benchmark_item_candidates.py to pair them and add analysis "
        "columns."
    )


def main() -> None:
    """Entry point."""
    typer.run(ingest_benchmark_data)


if __name__ == "__main__":
    main()
