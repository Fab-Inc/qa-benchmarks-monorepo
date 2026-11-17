"""Tests for generation benchmark item loading."""

import json
from pathlib import Path

import pytest
from decodable_text_generation.items import load_generation_items
from fab_benchmark_runner import ConfigError

FIXTURE = Path(__file__).parent / "fixtures" / "generation_benchmark_items.jsonl"


def _write_items(
    tmp_path: Path, records: list[dict], name: str = "items.jsonl"
) -> Path:
    path = tmp_path / name
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def _item(**overrides) -> dict:
    record = {
        "benchmark_item_id": "i1",
        "scenario_id": "words",
        "student_knowledge": {},
        "prompt": "CONTEXT ...",
    }
    record.update(overrides)
    return record


def test_load_generation_items_keeps_generation_fields(tmp_path: Path) -> None:
    # The fixture mixes a curriculum and a child item; load each variant from
    # its own homogeneous file.
    records = [
        json.loads(line)
        for line in FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    curriculum_path = _write_items(tmp_path, [records[0]], name="curr.jsonl")
    child_path = _write_items(tmp_path, [records[1]], name="child.jsonl")

    curriculum_items, _ = load_generation_items(curriculum_path)
    child_items, _ = load_generation_items(child_path)

    first = curriculum_items[0]
    assert first["benchmark_item_id"] == "ufli-en-l008_words"
    assert first["scenario_id"] == "words"
    assert first["target_knowledge"] == {"gpcs": [["i", "ɪ"]], "sight_words": []}
    assert first["is_jagged"] is False
    assert first["language_iso_code"] == "en"
    assert "TARGET" in first["prompt"]  # curriculum template was baked in

    second = child_items[0]
    assert second["benchmark_item_id"] == "child-jagged-01_story"
    assert "target_knowledge" not in second
    assert "TARGET" not in second["prompt"]


def test_required_fields(tmp_path: Path) -> None:
    for missing in ("benchmark_item_id", "scenario_id", "student_knowledge", "prompt"):
        record = _item()
        del record[missing]
        items_path = _write_items(tmp_path, [record])
        with pytest.raises(ConfigError, match=missing):
            load_generation_items(items_path)


def test_items_with_target_knowledge_load(tmp_path: Path) -> None:
    items_path = _write_items(
        tmp_path, [_item(target_knowledge={"sight_words": ["was"]})]
    )
    items, _ = load_generation_items(items_path)
    assert items[0]["target_knowledge"] == {"sight_words": ["was"]}


def test_items_without_target_knowledge_load(tmp_path: Path) -> None:
    items_path = _write_items(tmp_path, [_item()])
    items, _ = load_generation_items(items_path)
    assert "target_knowledge" not in items[0]


def test_mixed_target_knowledge_rejected(tmp_path: Path) -> None:
    items_path = _write_items(
        tmp_path,
        [
            _item(benchmark_item_id="i1", target_knowledge={"sight_words": ["was"]}),
            _item(benchmark_item_id="i2"),
        ],
    )
    with pytest.raises(ConfigError, match="target_knowledge"):
        load_generation_items(items_path)


def test_empty_target_knowledge_counts_as_absent(tmp_path: Path) -> None:
    items_path = _write_items(tmp_path, [_item(target_knowledge={})])
    items, _ = load_generation_items(items_path)
    assert items[0]["target_knowledge"] == {}
