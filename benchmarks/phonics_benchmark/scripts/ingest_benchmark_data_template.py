"""Ingest a language benchmark data-entry workbook into JSONL data files.

Run from the repository root with:

    uv run --package phonics-benchmark python \\
        benchmarks/phonics_benchmark/scripts/ingest_benchmark_data_template.py \\
        --language kiswahili \\
        --workbook path/to/benchmark_template.xlsx

The workbook is a template filled in by language experts, with a "Student
Profiles" tab and a "Texts" tab. Machine-read columns are identified by exact
header name (case-sensitive), independent of column order, so the template
can carry additional expert-context columns without breaking this script.
Rows with a blank ID column are treated as spacing/notes and skipped.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from phonics_benchmark.data_io import get_data_dir
from phonics_benchmark.models import DecodableText, StudentKnowledge

STUDENT_PROFILES_SHEET = "Student Profiles"
TEXTS_SHEET = "Texts"


def split_list(value: str) -> list[str]:
    """Split a comma-separated cell value into a stripped, non-empty list."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


# Registry: maps language_iso_code to the fields that belong in student_knowledge
# and how to parse each one from a workbook cell string.
# Add a new entry here when adding support for a new language.
STUDENT_KNOWLEDGE_FIELDS: dict[str, dict[str, Callable[[str], list[str] | int]]] = {
    "sw": {
        "graphemes": split_list,
        "whole_words": split_list,
    },
}


def get_language_iso_code(df: pd.DataFrame, description: str) -> str:
    """Read and validate the language_iso_code from a sheet DataFrame.

    Raises ValueError if the column is missing, empty, or contains more than
    one distinct value — all rows in a sheet must belong to the same language.
    """
    col = "language_iso_code"
    if col not in df.columns:
        raise ValueError(f"{description} is missing required column '{col}'")
    values = df[col].dropna().astype(str).str.strip()
    values = values[values != ""]
    if values.empty:
        raise ValueError(f"No '{col}' values found in {description}")
    unique = values.unique()
    if len(unique) > 1:
        raise ValueError(
            f"Multiple language_iso_code values in {description}: "
            f"{', '.join(sorted(unique))}"
        )
    return unique[0]


def preprocess_sheet_data(
    sheet_data: pd.DataFrame,
    required_columns: set[str],
    id_col: str,
    description: str,
) -> pd.DataFrame:
    """Validate, normalize, and filter a sheet DataFrame.

    Raises ValueError if required columns are missing, no data rows remain
    after filtering blank IDs, or any ID value is duplicated.
    """
    missing_columns = required_columns - set(sheet_data.columns)
    if missing_columns:
        raise ValueError(
            f"{description} is missing required column(s): "
            f"{', '.join(sorted(missing_columns))}"
        )

    for col in required_columns:
        sheet_data[col] = sheet_data[col].fillna("").astype(str).str.strip()

    filtered_data = sheet_data[sheet_data[id_col] != ""]

    if filtered_data.empty:
        raise ValueError(f"No records extracted from {description}")

    dupes = filtered_data[filtered_data[id_col].duplicated(keep=False)][id_col].unique()
    if len(dupes):
        raise ValueError(
            f"Duplicate {id_col} in {description}: {', '.join(sorted(dupes))}"
        )

    return filtered_data


def extract_student_profiles(
    df: pd.DataFrame,
    fields: dict[str, Callable[[str], list[str] | int]],
) -> list[StudentKnowledge]:
    return [
        StudentKnowledge(
            student_knowledge_id=str(row.student_knowledge_id),
            language_iso_code=str(row.language_iso_code),
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


def write_jsonl(records: list, path: Path) -> None:
    """Write a list of Pydantic model instances to a JSONL file."""
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json())
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--language", required=True, help="Language folder name, e.g. kiswahili"
    )
    parser.add_argument(
        "--workbook", required=True, type=Path, help="Path to the .xlsx template"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Root data directory (defaults to the package's data/ folder)",
    )
    args = parser.parse_args()

    student_df = pd.read_excel(args.workbook, sheet_name=STUDENT_PROFILES_SHEET)
    texts_df = pd.read_excel(args.workbook, sheet_name=TEXTS_SHEET)

    student_iso = get_language_iso_code(student_df, f"'{STUDENT_PROFILES_SHEET}' sheet")
    student_fields = STUDENT_KNOWLEDGE_FIELDS.get(student_iso)
    if student_fields is None:
        raise ValueError(
            f"No student_knowledge field registry for language '{student_iso}'. "
            f"Add an entry to STUDENT_KNOWLEDGE_FIELDS."
        )

    student_df = preprocess_sheet_data(
        student_df,
        {"student_knowledge_id", "language_iso_code"} | set(student_fields.keys()),
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
    texts = extract_texts(texts_df)

    data_dir = get_data_dir(args.language, args.data_root)
    profiles_path = data_dir / "student_knowledge_profiles.jsonl"
    texts_path = data_dir / "decodable_texts.jsonl"
    write_jsonl(profiles, profiles_path)
    write_jsonl(texts, texts_path)

    print(
        f"Wrote {len(profiles)} student knowledge profiles and "
        f"{len(texts)} decodable texts to {data_dir}"
    )


if __name__ == "__main__":
    main()
