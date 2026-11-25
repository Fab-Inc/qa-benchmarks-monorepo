"""Results files: read one line at a time, written one row at a time."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from fab_benchmark_runner import ConfigError, load_results
from fab_benchmark_runner.results import read_result_statuses, save_results


def test_each_row_is_one_canonical_json_line(tmp_path: Path) -> None:
    path = tmp_path / "responses.jsonl"

    save_results(
        [{"result_id": "b", "output": "é"}, {"result_id": "a", "n": {"z": 1, "a": 2}}],
        path,
    )

    assert (
        path.read_bytes()
        == (
            '{"output": "é", "result_id": "b"}\n'
            '{"n": {"a": 2, "z": 1}, "result_id": "a"}\n'
        ).encode()
    )


def test_text_with_unicode_line_separators_reads_back_as_written(
    tmp_path: Path,
) -> None:
    """Rows are split at newlines only; U+2028, U+2029 and U+0085 stay in the text."""

    path = tmp_path / "responses.jsonl"
    rows = [
        {"result_id": "a", "output": "one two three\x85four"},
        {"result_id": "b", "output": "five"},
    ]

    save_results(rows, path)

    assert load_results(path) == rows


def test_statuses_keep_each_id_and_status_and_the_last_row_wins(
    tmp_path: Path,
) -> None:
    path = tmp_path / "responses.jsonl"
    save_results(
        [
            {"result_id": "a", "call_status": "error", "output": ""},
            {"result_id": "b", "call_status": "success", "output": "x" * 1000},
            {"result_id": "a", "call_status": "success", "output": "y"},
            {"result_id": "c", "output": "no status"},
        ],
        path,
    )

    assert read_result_statuses(path) == {"a": "success", "b": "success", "c": None}
    assert read_result_statuses(tmp_path / "missing.jsonl") == {}


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b'{"result_id": "a"}\n{not json\n', r"responses\.jsonl:2: invalid JSON"),
        (
            b'{"result_id": "a"}\n\n{"output": "x"}\n',
            r"responses\.jsonl:3: existing result lacks result_id",
        ),
    ],
    ids=["invalid JSON", "no result_id"],
)
def test_a_bad_line_is_refused_by_its_number(
    tmp_path: Path, content: bytes, message: str
) -> None:
    path = tmp_path / "responses.jsonl"
    path.write_bytes(content)

    with pytest.raises(ConfigError, match=message):
        load_results(path)
    with pytest.raises(ConfigError, match=message):
        read_result_statuses(path)


def test_a_failed_write_keeps_the_previous_file(tmp_path: Path) -> None:
    path = tmp_path / "responses.jsonl"
    save_results([{"result_id": "a"}], path)

    def rows() -> Iterator[dict[str, Any]]:
        yield {"result_id": "b"}
        raise RuntimeError("broken row")

    with pytest.raises(RuntimeError, match="broken row"):
        save_results(rows(), path)

    assert load_results(path) == [{"result_id": "a"}]
    assert list(tmp_path.iterdir()) == [path]
