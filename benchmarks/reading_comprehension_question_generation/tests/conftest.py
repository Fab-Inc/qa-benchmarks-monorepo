"""Shared fixtures for the rcqg tests.

Workbooks, text files, and run folders are generated into ``tmp_path``. Nothing here
reads the user's local data, calls a provider, or touches the network.
"""

import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from openpyxl import Workbook

BENCHMARK_ROOT = Path(__file__).parents[1]
SCRIPTS_DIR = BENCHMARK_ROOT / "scripts"

# The 16 columns of the workbook's ``Texts`` sheet, in workbook order. Transcribed by
# hand from the workbook, not derived from the code under test.
WORKBOOK_COLUMNS = [
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
    "existing_comprehension_questions",
    "notes",
]

# The 14 fields a text record carries: the workbook columns minus the two that are
# deliberately not carried.
CARRIED_FIELDS = [
    name
    for name in WORKBOOK_COLUMNS
    if name not in {"existing_comprehension_questions", "notes"}
]


def _load_script(filename: str) -> ModuleType:
    """Import a script by file path, as the scripts are not an importable package."""
    spec = importlib.util.spec_from_file_location(
        f"rcqg_{filename.removesuffix('.py')}_script", SCRIPTS_DIR / filename
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def ingest_script() -> ModuleType:
    """The ``scripts/ingest_texts.py`` module."""
    return _load_script("ingest_texts.py")


@pytest.fixture(scope="session")
def build_script() -> ModuleType:
    """The ``scripts/build_benchmark_items.py`` module."""
    return _load_script("build_benchmark_items.py")


@pytest.fixture()
def workbook_columns() -> list[str]:
    """The 16 workbook columns in order (a copy, so a test may reorder it)."""
    return list(WORKBOOK_COLUMNS)


@pytest.fixture()
def text_row() -> Callable[..., dict[str, Any]]:
    """Return a factory for one synthetic workbook row (all 16 columns)."""

    def make(**overrides: Any) -> dict[str, Any]:
        row: dict[str, Any] = {
            "text_id": "ENG-001",
            "text_title": "The Hen",
            "text_content": "A hen sat on a mat.\nIt laid an egg.",
            "text_type": "Narrative",
            "country_code": "KE",
            "language_iso_code": "en",
            "grade": "2",
            "source": "Synthetic source",
            "location": None,
            "source_grade_label": "Grade 2",
            "is_target": "TRUE",
            "is_adaptation": "FALSE",
            "adaptation_source": None,
            "has_images": "TRUE",
            "existing_comprehension_questions": "Who sat on the mat?",
            "notes": "A synthetic note.",
        }
        row.update(overrides)
        return row

    return make


@pytest.fixture()
def text_record(
    text_row: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    """Return a factory for one synthetic ``texts.jsonl`` record (the 14 fields)."""

    def make(**overrides: Any) -> dict[str, Any]:
        row = text_row(**overrides)
        return {name: row[name] for name in CARRIED_FIELDS}

    return make


@pytest.fixture()
def write_workbook() -> Callable[..., Path]:
    """Return a writer for a two-sheet workbook shaped like the rcqg template."""

    def write(
        path: Path,
        rows: list[dict[str, Any]],
        *,
        columns: list[str] | None = None,
        sheet_name: str = "Texts",
    ) -> Path:
        workbook = Workbook()
        definitions = workbook.create_sheet("Field Definitions")
        del workbook["Sheet"]
        definitions.append(["field", "definition"])
        sheet = workbook.create_sheet(sheet_name)
        header = columns if columns is not None else WORKBOOK_COLUMNS
        sheet.append(header)
        for row in rows:
            sheet.append([row.get(name) for name in header])
        workbook.save(path)
        return path

    return write


@pytest.fixture()
def write_run_dir() -> Callable[..., Path]:
    """Return a writer for a run folder holding raw response rows per model.

    The rows have the fields extraction reads (``run_id``, ``model``,
    ``benchmark_item_id``, ``call_status``, ``output``). When ``items_path`` is given,
    ``config.json`` records its path and SHA-256 in the shape the harness uses.
    """

    def write(
        parent: Path,
        rows_by_model: dict[str, list[dict[str, Any]]],
        *,
        run_id: str = "run-1",
        items_path: Path | None = None,
        items_sha256: str | None = None,
    ) -> Path:
        run_dir = parent / run_id
        (run_dir / "responses").mkdir(parents=True)
        for model, rows in rows_by_model.items():
            lines = [
                json.dumps(
                    {
                        "run_id": run_id,
                        "model": model,
                        "call_status": "success",
                        **row,
                    }
                )
                for row in rows
            ]
            (run_dir / "responses" / f"responses_{model}.jsonl").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
        if items_path is not None:
            resolved = {"items_path": str(items_path), "items_sha256": items_sha256}
            (run_dir / "config.json").write_text(
                json.dumps({"resolved_config": resolved}), encoding="utf-8"
            )
        return run_dir

    return write
