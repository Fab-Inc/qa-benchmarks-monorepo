"""Ingest a language benchmark data-entry workbook into JSONL data files.

Run from the repository root with:

    uv run --package phonics-benchmark python \\
        benchmarks/phonics_benchmark/scripts/ingest_benchmark_data_template.py \\
        --language kiswahili \\
        --workbook path/to/benchmark_template.xlsx

The workbook is a template filled in by language experts, withP a "Student
Profiles" tab and a "Texts" tab. Machine-read columns are identified by exact
header name (case-sensitive), independent of column order, so the template
can carry additional expert-context columns without breaking this script.
Rows with a blank ID column are treated as spacing/notes and skipped.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import openpyxl
from openpyxl.worksheet.worksheet import Worksheet
from phonics_benchmark.data_io import get_data_dir
from phonics_benchmark.models import DecodableText, StudentKnowledgeProfile

STUDENT_PROFILES_SHEET = "Student Profiles"
TEXTS_SHEET = "Texts"


def cell_str(value: object) -> str:
    """Coerce a worksheet cell value to a stripped string.

    openpyxl types cell values as str | int | float | datetime | ... —
    coercing explicitly keeps this robust to a stray non-text value (e.g. a
    number or date typed into a text column) instead of crashing on .strip().
    """
    if value is None:
        return ""
    return str(value).strip()


def split_list(value: object) -> list[str]:
    """Split a comma-separated cell value into a stripped, non-empty list."""
    text = cell_str(value)
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def iter_rows(worksheet: Worksheet):
    """Yield each data row as a dict keyed by its column header.

    Columns are matched by header name (row 1), not position.
    """
    header = [cell.value for cell in worksheet[1]]
    for row in worksheet.iter_rows(min_row=2, values_only=True):
        yield dict(zip(header, row, strict=True))


def require_columns(worksheet: Worksheet, required: set[str], description: str) -> None:
    """Raise ValueError if the worksheet's header row is missing any required column.

    Without this, a renamed/typo'd header silently yields None for every row
    via row.get(...), so every row gets skipped with no indication why.
    """
    header = {cell_str(cell.value) for cell in worksheet[1]}
    missing = required - header
    if missing:
        raise ValueError(
            f"{description} is missing required column(s): {', '.join(sorted(missing))}"
        )


def extract_student_profiles(worksheet: Worksheet) -> list[StudentKnowledgeProfile]:
    profiles = []
    for row in iter_rows(worksheet):
        student_knowledge_id = cell_str(row.get("student_knowledge_id"))
        if not student_knowledge_id:
            continue
        profiles.append(
            StudentKnowledgeProfile(
                student_knowledge_id=student_knowledge_id,
                language_iso_code=cell_str(row.get("language_iso_code")),
                taught_knowledge={
                    "graphemes": split_list(row.get("graphemes")),
                    "whole_words": split_list(row.get("whole_words")),
                },
            )
        )
    return profiles


def extract_texts(worksheet: Worksheet) -> list[DecodableText]:
    texts = []
    for row in iter_rows(worksheet):
        text_id = cell_str(row.get("text_id"))
        if not text_id:
            continue
        texts.append(
            DecodableText(
                text_id=text_id,
                language_iso_code=cell_str(row.get("language_iso_code")),
                content=cell_str(row.get("content")),
            )
        )
    return texts


def check_unique_ids(records: list, id_field: str, description: str) -> None:
    """Raise ValueError if any records share the same id_field value.

    Runs before anything is written, so a duplicate in the workbook never
    overwrites the existing good JSONL files with bad data.
    """
    seen: set[str] = set()
    for record in records:
        record_id = getattr(record, id_field)
        if record_id in seen:
            raise ValueError(f"duplicate {id_field} '{record_id}' in {description}")
        seen.add(record_id)


def check_non_empty(records: list, description: str) -> None:
    """Raise ValueError if no records were extracted.

    Without this, an empty/misconfigured sheet would silently overwrite
    good existing JSONL files with zero records.
    """
    if not records:
        raise ValueError(f"No records extracted from {description}")


def write_jsonl(records: list, path: Path) -> None:
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

    workbook = openpyxl.load_workbook(args.workbook, data_only=True, read_only=True)
    student_profiles_ws = workbook[STUDENT_PROFILES_SHEET]
    texts_ws = workbook[TEXTS_SHEET]

    require_columns(
        student_profiles_ws,
        {"student_knowledge_id", "language_iso_code", "graphemes", "whole_words"},
        f"'{STUDENT_PROFILES_SHEET}' sheet",
    )
    require_columns(
        texts_ws,
        {"text_id", "language_iso_code", "content"},
        f"'{TEXTS_SHEET}' sheet",
    )

    profiles = extract_student_profiles(student_profiles_ws)
    texts = extract_texts(texts_ws)

    # Validate before writing anything, so a bad workbook never clobbers
    # the existing good JSONL files with duplicate-riddled or empty data.
    check_non_empty(profiles, f"'{STUDENT_PROFILES_SHEET}' sheet")
    check_non_empty(texts, f"'{TEXTS_SHEET}' sheet")
    check_unique_ids(profiles, "student_knowledge_id", "Student Profiles sheet")
    check_unique_ids(texts, "text_id", "Texts sheet")

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
