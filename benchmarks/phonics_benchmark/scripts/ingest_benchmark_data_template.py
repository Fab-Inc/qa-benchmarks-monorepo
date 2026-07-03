"""Ingest a language benchmark data-entry workbook into JSONL data files.

Run from the repository root with:

    uv run --package phonics-benchmark python \\
        benchmarks/phonics_benchmark/scripts/ingest_benchmark_template.py \\
        --language kiswahili \\
        --workbook path/to/benchmark_data_tusome_ke.xlsx

The workbook is a template filled in by language experts, with a "Student
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
from phonics_benchmark.data_io import (
    get_data_dir,
    read_decodable_texts,
    read_student_knowledge_profiles,
)
from phonics_benchmark.models import DecodableText, StudentKnowledgeProfile

STUDENT_PROFILES_SHEET = "Student Profiles"
TEXTS_SHEET = "Texts"


def split_list(value: str | None) -> list[str]:
    """Split a comma-separated cell value into a stripped, non-empty list."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def iter_rows(worksheet: Worksheet):
    """Yield each data row as a dict keyed by its column header.

    Columns are matched by header name (row 1), not position.
    """
    header = [cell.value for cell in worksheet[1]]
    for row in worksheet.iter_rows(min_row=2, values_only=True):
        yield dict(zip(header, row, strict=True))


def extract_student_profiles(worksheet: Worksheet) -> list[StudentKnowledgeProfile]:
    profiles = []
    for row in iter_rows(worksheet):
        student_knowledge_id = (row.get("student_knowledge_id") or "").strip()
        if not student_knowledge_id:
            continue
        profiles.append(
            StudentKnowledgeProfile(
                student_knowledge_id=student_knowledge_id,
                language_iso_code=(row.get("language_iso_code") or "").strip(),
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
        text_id = (row.get("text_id") or "").strip()
        if not text_id:
            continue
        texts.append(
            DecodableText(
                text_id=text_id,
                language_iso_code=(row.get("language_iso_code") or "").strip(),
                content=(row.get("content") or "").strip(),
            )
        )
    return texts


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
    profiles = extract_student_profiles(workbook[STUDENT_PROFILES_SHEET])
    texts = extract_texts(workbook[TEXTS_SHEET])

    data_dir = get_data_dir(args.language, args.data_root)
    profiles_path = data_dir / "student_knowledge_profiles.jsonl"
    texts_path = data_dir / "decodable_texts.jsonl"
    write_jsonl(profiles, profiles_path)
    write_jsonl(texts, texts_path)

    # Fail fast on duplicate IDs or schema issues using the already-tested readers.
    read_student_knowledge_profiles(args.language, args.data_root)
    read_decodable_texts(args.language, args.data_root)

    print(
        f"Wrote {len(profiles)} student knowledge profiles and "
        f"{len(texts)} decodable texts to {data_dir}"
    )


if __name__ == "__main__":
    main()
