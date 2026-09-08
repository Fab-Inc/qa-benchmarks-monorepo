"""Utilities for reading and writing phonics_benchmark data files."""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from decodability import DECODABILITY_REGISTRY
from pydantic import BaseModel, ValidationError

from phonics_benchmark.models import (
    GenerationBenchmarkItem,
    StudentKnowledge,
)

STUDENT_PROFILES_FILENAME = "student_knowledge_profiles.jsonl"
BENCHMARK_ITEMS_FILENAME = "benchmark_items.jsonl"
BENCHMARK_ITEMS_DEV_FILENAME = "benchmark_items_dev.jsonl"

STUDENT_PROFILES_ID_NAME = "student_knowledge_id"
BENCHMARK_ITEM_ID_NAME = "benchmark_item_id"


def get_data_dir(language: str, data_root: Path) -> Path:
    """Return the Path to data_root/<language>/"""
    return data_root / language


def _read_records[ModelT: BaseModel](
    language: str,
    data_root: Path,
    filename: str,
    model: type[ModelT],
    id_field: str,
) -> list[ModelT]:
    """Parse data_root/<language>/<filename> into a list of `model` instances.

    Blank lines are skipped but still counted, so reported line numbers match the
    file. Errors name the file and the 1-based line number of the offending record.

    Raises FileNotFoundError if the file is absent, or ValueError if a record fails
    to parse or repeats an `id_field` value already seen.
    """
    path = get_data_dir(language, data_root) / filename
    try:
        handle = path.open(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Missing {filename} for language '{language}': expected {path}"
        ) from None

    records: list[ModelT] = []
    seen_ids: set[str] = set()
    with handle as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                parsed = model.model_validate(json.loads(line))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise ValueError(
                    f"{path}:{line_no}: failed to parse record — {exc}"
                ) from exc
            record_id = getattr(parsed, id_field)
            if record_id in seen_ids:
                raise ValueError(
                    f"{path}:{line_no}: duplicate {id_field} '{record_id}'"
                )
            seen_ids.add(record_id)
            records.append(parsed)
    return records


def read_student_knowledge_profiles(
    language: str, data_root: Path
) -> list[StudentKnowledge]:
    """Parse student_knowledge_profiles.jsonl for the given language.

    Only checks the generic StudentKnowledge shape. The contents of each profile's
    `student_knowledge` are language-specific — call validate_student_knowledge()
    to check those against the language's decodability model.
    """
    return _read_records(
        language,
        data_root,
        STUDENT_PROFILES_FILENAME,
        StudentKnowledge,
        STUDENT_PROFILES_ID_NAME,
    )


def read_benchmark_items(
    language: str, data_root: Path, *, dev: bool = False
) -> list[GenerationBenchmarkItem]:
    """Parse the generation item set the benchmark runs.

    ``dev=True`` reads the smaller stratified dev set instead of the full one.
    Both files share a schema: the dev set selects rows, it doesn't reshape them.
    """
    return _read_records(
        language,
        data_root,
        BENCHMARK_ITEMS_DEV_FILENAME if dev else BENCHMARK_ITEMS_FILENAME,
        GenerationBenchmarkItem,
        BENCHMARK_ITEM_ID_NAME,
    )


def write_jsonl(records: Iterable[BaseModel], path: Path) -> None:
    """Write Pydantic model instances to a JSONL file, one record per line."""
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json())
            f.write("\n")


def write_json(data: dict, path: Path) -> None:
    """Write a plain dict to a JSON file."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_dataset(
    language: str,
    data_root: Path,
    *,
    profiles: Iterable[StudentKnowledge] | None = None,
    benchmark_items: Iterable[GenerationBenchmarkItem] | None = None,
    benchmark_items_dev: Iterable[GenerationBenchmarkItem] | None = None,
) -> Path:
    """Write the given records into data_root/<language>/, creating it if needed.

    Only the collections passed are written.

    Returns the data directory that was written to.
    """
    data_dir = get_data_dir(language, data_root)
    data_dir.mkdir(parents=True, exist_ok=True)

    for records, filename in (
        (profiles, STUDENT_PROFILES_FILENAME),
        (benchmark_items, BENCHMARK_ITEMS_FILENAME),
        (benchmark_items_dev, BENCHMARK_ITEMS_DEV_FILENAME),
    ):
        if records is not None:
            write_jsonl(records, data_dir / filename)

    return data_dir


def convert_records_to_dataframe(records: Iterable[BaseModel]) -> pd.DataFrame:
    """Convert model instances into a DataFrame."""
    return pd.DataFrame([record.model_dump() for record in records])


def convert_dataframe_to_records[ModelT: BaseModel](
    frame: pd.DataFrame, model: type[ModelT]
) -> list[ModelT]:
    """Validate each row of `frame` against `model`"""
    records: list[ModelT] = []

    for position, row in enumerate(frame.to_dict(orient="records")):
        try:
            records.append(model.model_validate(row))
        except ValidationError as exc:
            label = row.get("benchmark_item_id", "<no benchmark_item_id>")
            raise ValueError(
                f"row {position} ({label}) is not a valid {model.__name__} — {exc}"
            ) from exc

    return records


def validate_student_knowledge(
    profiles: list[StudentKnowledge], *, strict: bool = False
) -> None:
    """Check that each profile's student_knowledge is valid for its language.

    Delegates to the language-specific model in `decodability` (e.g.
    KiswahiliStudentKnowledge for "sw"). Profiles in a language with no
    registered validator emit a warning, since their student_knowledge goes
    unchecked — pass strict=True to raise instead (e.g. once a language is
    expected to always have a registered validator).
    """
    for profile in profiles:
        language_integration = DECODABILITY_REGISTRY.get(profile.language_iso_code)
        if language_integration is None:
            message = (
                f"No known student_knowledge validator registered for language "
                f"'{profile.language_iso_code}' (student_knowledge_id "
                f"'{profile.student_knowledge_id}')"
            )
            if strict:
                raise ValueError(message)
            warnings.warn(f"{message}; skipping validation.", stacklevel=2)
            continue
        student_knowledge_model = language_integration["student_knowledge_model"]
        try:
            student_knowledge_model.model_validate(profile.student_knowledge)
        except ValidationError as exc:
            raise ValueError(
                f"student_knowledge_id '{profile.student_knowledge_id}': {exc}"
            ) from exc


def resolve_language_iso_code(profiles: list[StudentKnowledge]) -> str:
    """Return the single ISO code shared by every student profile.

    Raises ValueError if the records are empty, or if more than one language code
    appears across them — the pipeline only handles one language per data folder.
    """
    codes = {profile.language_iso_code for profile in profiles}

    if not codes:
        raise ValueError("Cannot resolve a language_iso_code: no student profiles")
    if len(codes) > 1:
        raise ValueError(
            "Expected one language_iso_code across the dataset, found: "
            f"{', '.join(sorted(codes))}"
        )
    return codes.pop()
