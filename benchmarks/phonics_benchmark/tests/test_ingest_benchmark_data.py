"""End-to-end test of scripts/ingest_benchmark_data.py on the committed template.

The Kiswahili data-entry template under data/kiswahili/templates/ is checked into
git and is the reference example handed to language experts, so it doubles as a
fixture here. This test pins two things:

1. the script runs to completion on that workbook, and
2. every machine-read column lands in the output with the right value —
   including that the template's expert-context columns (Curriculum stage,
   Description / notes, Grade, ...) are *not* picked up.

Expected values below are transcribed from the template by hand rather than
re-derived from it, so that editing a cell in the workbook without updating this
test is a visible failure rather than a silently-passing tautology.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import typer
from phonics_benchmark.data_io import (
    read_decodable_texts,
    read_student_knowledge_profiles,
)
from typer.testing import CliRunner

BENCHMARK_ROOT = Path(__file__).parents[1]
SCRIPT_PATH = BENCHMARK_ROOT / "scripts" / "ingest_benchmark_data.py"
TEMPLATE_DIR = BENCHMARK_ROOT / "data" / "kiswahili" / "templates"
KISWAHILI_TEMPLATE_PATH = TEMPLATE_DIR / "benchmark_data_template_kiswahili.xlsx"

# Same data that makes upfixtures/completed_benchmark_items.jsonl,
# but keyed by benchmark_item_id for easy lookup.
SW_EXPECTED_PROFILES = {
    "tusome-ke-stage-01": {
        "graphemes": ["a", "u", "i", "m", "k", "t"],
        "clusters": [],
        "cluster_patterns": [],
        "whole_words": ["anaona", "na", "paka"],
    },
    "tusome-ke-stage-02": {
        "graphemes": [
            "a",
            "u",
            "i",
            "o",
            "e",
            "m",
            "k",
            "t",
            "n",
            "w",
            "l",
            "h",
            "s",
            "b",
            "y",
            "z",
            "g",
            "d",
            "j",
            "r",
            "p",
            "f",
            "v",
            "sh",
            "ny",
            "dh",
            "th",
            "ch",
            "gh",
            "ng",
            "ng'",
        ],
        "clusters": ["nd", "nz", "nj", "njw", "st", "bl", "kt"],
        "cluster_patterns": [
            "prenasalised_m",
            "labialised_glide",
            "palatalised_glide",
            "m_labialised_glide",
            "m_palatalised_glide",
        ],
        "whole_words": ["anaona", "na", "paka"],
    },
}

SW_EXPECTED_TEXTS = {
    "SW-KE-S2": (
        "Mama anaona kuku na paka. Kaka akaamua kukaa. Kuku akamuuma kaka. "
        "Mama akaamua kukaa na kuku. Kaka anaona soko kuu."
    ),
}


def _load_script() -> ModuleType:
    """Import the ingest script by path — scripts/ is not an importable package."""
    spec = importlib.util.spec_from_file_location(
        "ingest_benchmark_data_script", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="class")
def sw_ingested(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the script on the committed template into a throwaway data root.

    Returns the data root, so assertions read the JSONL the script actually
    wrote rather than inspecting in-process objects. Writing into tmp keeps the
    repo's real data/kiswahili/*.jsonl untouched.
    """
    assert KISWAHILI_TEMPLATE_PATH.exists(), (
        f"Missing committed template: {KISWAHILI_TEMPLATE_PATH}"
    )

    script = _load_script()
    app = typer.Typer()
    app.command()(script.ingest_benchmark_data)

    data_root = tmp_path_factory.mktemp("data_root")
    result = CliRunner().invoke(
        app,
        [
            "--language",
            "kiswahili",
            "--workbook",
            str(KISWAHILI_TEMPLATE_PATH),
            "--data-root",
            str(data_root),
        ],
    )

    assert result.exit_code == 0, (
        f"ingest failed (exit {result.exit_code}): {result.output}\n"
        f"{result.exception!r}"
    )
    return data_root


class TestKiswahiliTemplateIngest:
    def test_script_runs_and_writes_both_files(self, sw_ingested: Path) -> None:
        """It completes on the committed template and writes both output files."""
        written = {path.name for path in (sw_ingested / "kiswahili").iterdir()}
        assert written == {"student_knowledge_profiles.jsonl", "decodable_texts.jsonl"}

    def test_student_profile_ids_and_language_are_ingested(
        self, sw_ingested: Path
    ) -> None:
        """student_knowledge_id and language_iso_code come through for every row."""
        profiles = read_student_knowledge_profiles("kiswahili", sw_ingested)

        assert [p.student_knowledge_id for p in profiles] == list(SW_EXPECTED_PROFILES)
        assert {p.language_iso_code for p in profiles} == {"sw"}

    @pytest.mark.parametrize("profile_id", sorted(SW_EXPECTED_PROFILES))
    def test_student_knowledge_fields_are_ingested(
        self, sw_ingested: Path, profile_id: str
    ) -> None:
        """graphemes, clusters, cluster_patterns and whole_words parse correctly.

        Comparing the whole student_knowledge dict (not field by field) also asserts
        that no extra template column leaks in and no expected field is dropped.
        """
        profiles = {
            p.student_knowledge_id: p
            for p in read_student_knowledge_profiles("kiswahili", sw_ingested)
        }

        assert (
            profiles[profile_id].student_knowledge == SW_EXPECTED_PROFILES[profile_id]
        )

    def test_texts_are_ingested(self, sw_ingested: Path) -> None:
        """text_id, language_iso_code and content come through verbatim."""
        texts = read_decodable_texts("kiswahili", sw_ingested)

        assert {t.text_id: t.content for t in texts} == SW_EXPECTED_TEXTS
        assert {t.language_iso_code for t in texts} == {"sw"}
