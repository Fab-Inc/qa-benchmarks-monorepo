"""Loading JSONL items and validating item IDs."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from fab_benchmark_runner import ConfigError, file_sha256, load_jsonl_items
from fab_benchmark_runner.items import validate_items


def test_load_jsonl_items_keeps_every_field_and_hashes_exact_bytes(
    tmp_path: Path,
) -> None:
    content = (
        b'{"benchmark_item_id": "item-1", "content": "A", "extra": {"k": 1}}\n'
        b'{"benchmark_item_id": "item-2", "content": "B"}\n'
    )
    items_path = tmp_path / "items.jsonl"
    items_path.write_bytes(content)

    items, items_sha256 = load_jsonl_items(items_path)

    assert items == [
        {"benchmark_item_id": "item-1", "content": "A", "extra": {"k": 1}},
        {"benchmark_item_id": "item-2", "content": "B"},
    ]
    assert items_sha256 == sha256(content).hexdigest() == file_sha256(items_path)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"", "No benchmark items"),
        (b'{"benchmark_item_id": "a"}\n\n', "blank JSONL lines"),
        (b"{not json}\n", "invalid JSON"),
        (b"[]\n", "must be an object"),
        (b"\xff\n", "not valid UTF-8"),
        (b'{"content": "x"}\n', "benchmark_item_id must be a non-empty string"),
        (b'{"benchmark_item_id": 3}\n', "benchmark_item_id must be a non-empty"),
        (b'{"benchmark_item_id": " a"}\n', "surrounding whitespace"),
        (
            b'{"benchmark_item_id": "a"}\n{"benchmark_item_id": "a"}\n',
            "duplicate benchmark_item_id 'a'",
        ),
    ],
)
def test_load_jsonl_items_rejects_invalid_files(
    tmp_path: Path, content: bytes, message: str
) -> None:
    items_path = tmp_path / "items.jsonl"
    items_path.write_bytes(content)

    with pytest.raises(ConfigError, match=message):
        load_jsonl_items(items_path)


def test_missing_items_file_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Could not read"):
        load_jsonl_items(tmp_path / "missing.jsonl")
    with pytest.raises(ConfigError, match="Could not read"):
        file_sha256(tmp_path / "missing.jsonl")


def test_validate_items_checks_custom_loader_output() -> None:
    with pytest.raises(ConfigError, match="item 2 must be an object"):
        validate_items([{"benchmark_item_id": "a"}, "b"], source="loader")
