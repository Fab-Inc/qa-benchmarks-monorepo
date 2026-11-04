"""Tests for scripts/build_generation_items.py."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
import typer
from decodable_text_generation.models import GenerationStudentProfile
from phonics_benchmark.data_io import write_jsonl
from typer.testing import CliRunner

BENCHMARK_ROOT = Path(__file__).parents[1]
SCRIPT_PATH = BENCHMARK_ROOT / "scripts" / "build_generation_items.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_generation_items_script", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EN_KNOWLEDGE = {"gpcs": [["a", "æ"], ["t", "t"]], "sight_words": ["the"]}

CHILD_PROFILES = [
    {
        "student_knowledge_id": "en-aligned",
        "language_iso_code": "en",
        "is_jagged": False,
        "student_knowledge": EN_KNOWLEDGE,
    },
    {
        "student_knowledge_id": "en-jagged",
        "language_iso_code": "en",
        "is_jagged": True,
        "student_knowledge": EN_KNOWLEDGE,
    },
]

CURRICULUM_PROFILES = [
    {
        "student_knowledge_id": "en-targeted",
        "language_iso_code": "en",
        "student_knowledge": EN_KNOWLEDGE,
        "target_knowledge": {"gpcs": [["n", "n"]], "sight_words": []},
    },
    {
        "student_knowledge_id": "en-untargeted",
        "language_iso_code": "en",
        "student_knowledge": EN_KNOWLEDGE,
        "target_knowledge": {},
    },
]


@pytest.fixture()
def data_root(tmp_path: Path) -> Path:
    """Write child_adapted and curriculum_aligned profile fixtures."""
    for benchmark_dir, profiles in (
        ("child_adapted", CHILD_PROFILES),
        ("curriculum_aligned", CURRICULUM_PROFILES),
    ):
        data_dir = tmp_path / "data" / benchmark_dir / "english"
        data_dir.mkdir(parents=True)
        write_jsonl(
            [GenerationStudentProfile(**p) for p in profiles],
            data_dir / "student_knowledge_profiles.jsonl",
        )
    return tmp_path / "data"


def _run_build(data_root: Path, benchmark: str, *extra: str):
    script = _load_script()
    app = typer.Typer()
    app.command()(script.build_generation_items)
    return CliRunner().invoke(
        app,
        [
            "--language",
            "english",
            "--benchmark",
            benchmark,
            "--data-root",
            str(data_root),
            *extra,
        ],
    )


def _read_items(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_child_items_default_to_story_scenario(data_root: Path) -> None:
    result = _run_build(data_root, "child")

    assert result.exit_code == 0, result.output
    items = _read_items(
        data_root / "child_adapted" / "english" / "benchmark_items.jsonl"
    )
    assert len(items) == len(CHILD_PROFILES)
    assert {item["scenario_id"] for item in items} == {"story"}
    assert {item["benchmark_item_id"] for item in items if item["is_jagged"]} == {
        "en-jagged_story"
    }
    first = items[0]
    assert "DECODING RULES" in first["prompt"]
    assert "TARGET" not in first["prompt"]
    assert "short story" in first["prompt"]


def test_scenarios_option_can_build_all(data_root: Path) -> None:
    result = _run_build(data_root, "child", "-s", "words,sentences,story")

    assert result.exit_code == 0, result.output
    items = _read_items(
        data_root / "child_adapted" / "english" / "benchmark_items.jsonl"
    )
    assert len(items) == len(CHILD_PROFILES) * 3
    words_item = next(i for i in items if i["scenario_id"] == "words")
    assert "Generate a list of 20 words" in words_item["prompt"]


def test_curriculum_items_keep_only_targeted_profiles(data_root: Path) -> None:
    result = _run_build(data_root, "curriculum")

    assert result.exit_code == 0, result.output
    items = _read_items(
        data_root / "curriculum_aligned" / "english" / "benchmark_items.jsonl"
    )
    assert {item["student_knowledge_id"] for item in items} == {"en-targeted"}
    assert len(items) == 1
    assert all(
        item["target_knowledge"] == {"gpcs": [["n", "n"]], "sight_words": []}
        for item in items
    )
    assert all("TARGET" in item["prompt"] for item in items)
    assert all('- Target GPCs: "n" → "n"' in item["prompt"] for item in items)


def test_curriculum_fails_without_targeted_profiles(tmp_path: Path) -> None:
    data_dir = tmp_path / "data" / "curriculum_aligned" / "english"
    data_dir.mkdir(parents=True)
    write_jsonl(
        [
            GenerationStudentProfile(
                student_knowledge_id="en-jagged",
                language_iso_code="en",
                student_knowledge=EN_KNOWLEDGE,
            )
        ],
        data_dir / "student_knowledge_profiles.jsonl",
    )

    script = _load_script()
    with pytest.raises(ValueError, match="curriculum"):
        script.build_generation_items(
            language="english",
            benchmark=script.Benchmark.CURRICULUM,
            data_root=tmp_path / "data",
        )


def test_build_fails_on_invalid_student_knowledge(tmp_path: Path) -> None:
    data_dir = tmp_path / "data" / "child_adapted" / "english"
    data_dir.mkdir(parents=True)
    write_jsonl(
        [
            GenerationStudentProfile(
                student_knowledge_id="en-bad",
                language_iso_code="en",
                student_knowledge={"gpcs": [["zz", "zz"]], "sight_words": []},
            )
        ],
        data_dir / "student_knowledge_profiles.jsonl",
    )

    script = _load_script()
    with pytest.raises(ValueError, match="Invalid student_knowledge"):
        script.build_generation_items(
            language="english",
            benchmark=script.Benchmark.CHILD,
            data_root=tmp_path / "data",
        )


def test_unknown_scenario_rejected(data_root: Path) -> None:
    result = _run_build(data_root, "child", "-s", "words,poem")
    # exits non-zero via CliRunner or raises when called directly
    assert result.exit_code != 0 or "Unknown scenario" in str(result.exception)


def test_scenarios_option_filters_to_story(data_root: Path) -> None:
    result = _run_build(data_root, "child", "-s", "story")

    assert result.exit_code == 0, result.output
    items = _read_items(
        data_root / "child_adapted" / "english" / "benchmark_items.jsonl"
    )
    assert len(items) == len(CHILD_PROFILES)
    assert {item["scenario_id"] for item in items} == {"story"}
    assert all("short story" in item["prompt"] for item in items)
