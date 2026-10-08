"""Tests for generation benchmark item loading."""

import json
from pathlib import Path

import pytest
from decodable_text_generation.items import load_generation_items
from phonics_benchmark.config import ConfigError

FIXTURE = Path(__file__).parent / "fixtures" / "generation_benchmark_items.jsonl"


def _write_items(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "items.jsonl"
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def test_load_generation_items_keeps_generation_fields() -> None:
    items, _sha = load_generation_items(FIXTURE)

    assert [item["benchmark_item_id"] for item in items] == [
        "ufli-en-l008_words",
        "child-jagged-01_story",
    ]
    first = items[0]
    assert first["scenario_id"] == "words"
    assert first["target_knowledge"] == {"gpcs": [["i", "ɪ"]], "sight_words": []}
    assert first["is_jagged"] is False
    assert first["language_iso_code"] == "en"
    assert "TARGET" in first["prompt"]  # curriculum template was baked in
    assert "TARGET" not in items[1]["prompt"]


def test_required_fields(tmp_path: Path) -> None:
    for missing in ("benchmark_item_id", "scenario_id", "student_knowledge", "prompt"):
        record = {
            "benchmark_item_id": "i1",
            "scenario_id": "words",
            "student_knowledge": {},
            "prompt": "CONTEXT ...",
        }
        del record[missing]
        items_path = _write_items(tmp_path, [record])
        with pytest.raises(ConfigError, match=missing):
            load_generation_items(items_path)


def test_curriculum_requires_target_knowledge(tmp_path: Path) -> None:
    items_path = _write_items(
        tmp_path,
        [
            {
                "benchmark_item_id": "i1",
                "scenario_id": "words",
                "student_knowledge": {},
                "prompt": "CONTEXT ...",
            }
        ],
    )
    with pytest.raises(ConfigError, match="target_knowledge"):
        load_generation_items(items_path, curriculum=True)


def test_curriculum_rejects_empty_target(tmp_path: Path) -> None:
    items_path = _write_items(
        tmp_path,
        [
            {
                "benchmark_item_id": "i1",
                "scenario_id": "words",
                "student_knowledge": {},
                "prompt": "CONTEXT ...",
                "target_knowledge": {},
            }
        ],
    )
    with pytest.raises(ConfigError, match="target_knowledge"):
        load_generation_items(items_path, curriculum=True)


def test_curriculum_accepts_items_with_target(tmp_path: Path) -> None:
    items_path = _write_items(
        tmp_path,
        [
            {
                "benchmark_item_id": "i1",
                "scenario_id": "words",
                "student_knowledge": {},
                "prompt": "CONTEXT ...",
                "target_knowledge": {"sight_words": ["was"]},
            }
        ],
    )
    items, _ = load_generation_items(items_path, curriculum=True)
    assert items[0]["target_knowledge"] == {"sight_words": ["was"]}
