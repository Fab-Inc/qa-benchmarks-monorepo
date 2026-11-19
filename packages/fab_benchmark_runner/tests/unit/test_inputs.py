"""Reading and writing the inputs file a benchmark hands to the runner."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from fab_benchmark_runner import (
    ConfigError,
    ImageRef,
    InputsError,
    ModelInput,
    check_inputs_file,
    read_inputs,
    write_inputs,
)
from fab_benchmark_runner.inputs import ITEM_ID_KEY, load_inputs_file, parse_input_row


def _write(path: Path, rows: list[Any]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_reads_text_and_image_inputs(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "inputs.jsonl",
        [
            {
                "benchmark_item_id": "sw-1",
                "prompt": "Evaluate.",
                "system_prompt": "You are a teacher.",
            },
            {
                "benchmark_item_id": "1042",
                "prompt": "Which shape?",
                "images": [
                    {"uri": "gs://bucket/images/1042.jpg", "detail": "high"},
                    {"uri": "images/1042-b.png", "detail": "low"},
                ],
            },
        ],
    )

    first, second = read_inputs(path)

    assert first == ModelInput("sw-1", "Evaluate.", "You are a teacher.")
    assert second.system_prompt is None
    assert second.images == (
        ImageRef("gs://bucket/images/1042.jpg", "high"),
        ImageRef(str((tmp_path / "images/1042-b.png").resolve()), "low"),
    )


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ([1], "must be a JSON object"),
        ({"benchmark_item_id": "a", "prompt": "p", "system": "s"}, "unknown keys"),
        ({"benchmark_item_id": "a"}, "prompt must be"),
        ({"benchmark_item_id": "a", "prompt": ""}, "prompt must be"),
        ({"benchmark_item_id": "a", "prompt": " \n"}, "more than whitespace"),
        ({"benchmark_item_id": 7, "prompt": "p"}, 'write the number 7 as "7"'),
        ({"benchmark_item_id": " a", "prompt": "p"}, "benchmark_item_id must be"),
        ({"benchmark_item_id": "a", "prompt": "p", "system_prompt": ""}, "system_"),
        ({"benchmark_item_id": "a", "prompt": "p", "system_prompt": " "}, "system_"),
        ({"benchmark_item_id": "a", "prompt": "p", "images": {}}, "must be a list"),
        (
            {"benchmark_item_id": "a", "prompt": "p", "images": [{"uri": "x.png"}]},
            "detail must be",
        ),
        (
            {
                "benchmark_item_id": "a",
                "prompt": "p",
                "images": [{"uri": "x.png", "detail": "high", "path": "x"}],
            },
            "unknown keys",
        ),
        (
            {
                "benchmark_item_id": "a",
                "prompt": "p",
                "images": [{"uri": "x.png", "detail": "high", "sha256": "abc"}],
            },
            "sha256 must be",
        ),
    ],
)
def test_malformed_rows_are_refused(tmp_path: Path, row: Any, message: str) -> None:
    path = _write(tmp_path / "inputs.jsonl", [row])

    with pytest.raises(ConfigError, match=message) as raised:
        read_inputs(path)
    assert f"{path}:1" in str(raised.value)


def test_duplicate_ids_blank_lines_and_empty_files_are_refused(tmp_path: Path) -> None:
    row = {"benchmark_item_id": "a", "prompt": "p"}
    duplicate = _write(tmp_path / "duplicate.jsonl", [row, row])
    blank = tmp_path / "blank.jsonl"
    blank.write_text(json.dumps(row) + "\n\n", encoding="utf-8")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")

    with pytest.raises(ConfigError, match="duplicate benchmark_item_id 'a'"):
        read_inputs(duplicate)
    with pytest.raises(ConfigError, match="blank lines"):
        read_inputs(blank)
    with pytest.raises(ConfigError, match="no inputs"):
        read_inputs(empty)


def test_stored_rows_need_hashes_and_keep_relative_paths() -> None:
    digest = sha256(b"image").hexdigest()
    row = {
        "benchmark_item_id": "a",
        "prompt": "p",
        "system_prompt": None,
        "images": [
            {
                "detail": "high",
                "sha256": digest,
                "source_uri": "gs://bucket/a.jpg",
                "uri": f"images/{digest}.jpg",
            }
        ],
    }

    stored = parse_input_row(row, where="stored", stored=True)

    assert stored.images[0].uri == f"images/{digest}.jpg"
    assert stored.to_row() == row
    with pytest.raises(ConfigError, match="must have its sha256"):
        parse_input_row(
            {**row, "images": [{"detail": "high", "uri": "a.jpg"}]},
            where="stored",
            stored=True,
        )
    with pytest.raises(ConfigError, match="unknown keys"):
        parse_input_row(row, where="handed over")


def test_write_inputs_round_trips(tmp_path: Path) -> None:
    inputs = [
        ModelInput("a", "First?", "System."),
        ModelInput("b", "Second?", images=(ImageRef("gs://bucket/b.png", "low"),)),
    ]

    path = write_inputs(inputs, tmp_path / "out" / "inputs.jsonl")

    assert read_inputs(path) == inputs
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0]) == {
        "benchmark_item_id": "a",
        "prompt": "First?",
        "system_prompt": "System.",
    }


def test_reads_an_inputs_file_on_gcs(storage: Any) -> None:
    row = {
        "benchmark_item_id": "1042",
        "prompt": "Which shape?",
        "images": [
            {"uri": "images/1042.jpg", "detail": "high"},
            {"uri": "../shared/grid.png", "detail": "low"},
            {"uri": "gs://other/1042.jpg", "detail": "low"},
        ],
    }
    storage.objects["data"] = {
        "mathv/inputs.jsonl": (7, (json.dumps(row) + "\n").encode("utf-8"))
    }

    inputs, generation = load_inputs_file(
        "gs://data/mathv/inputs.jsonl", storage_client=storage
    )

    assert generation == 7
    assert [image.uri for image in inputs[0].images] == [
        "gs://data/mathv/images/1042.jpg",
        "gs://data/shared/grid.png",
        "gs://other/1042.jpg",
    ]
    assert read_inputs("gs://data/mathv/inputs.jsonl", storage_client=storage) == (
        inputs
    )


@pytest.mark.parametrize(
    ("uri", "message"),
    [
        ("/Users/me/1042.jpg", "not at the local path"),
        ("~/1042.jpg", "not at the local path"),
        ("../../1042.jpg", "leaves the bucket"),
    ],
)
def test_an_inputs_file_on_gcs_only_points_at_gcs(
    storage: Any, uri: str, message: str
) -> None:
    row = {
        "benchmark_item_id": "a",
        "prompt": "p",
        "images": [{"uri": uri, "detail": "high"}],
    }
    storage.objects["data"] = {
        "mathv/inputs.jsonl": (1, (json.dumps(row) + "\n").encode("utf-8"))
    }

    with pytest.raises(ConfigError, match=message) as raised:
        read_inputs("gs://data/mathv/inputs.jsonl", storage_client=storage)
    assert "gs://data/mathv/inputs.jsonl:1" in str(raised.value)


def test_a_missing_inputs_file_on_gcs_is_refused(storage: Any) -> None:
    storage.objects["data"] = {}

    with pytest.raises(ConfigError, match="gs://data/no.jsonl does not exist"):
        read_inputs("gs://data/no.jsonl", storage_client=storage)


def test_a_row_reports_every_problem_it_has() -> None:
    row = {
        "benchmark_item_id": 7,
        "prompt": "",
        "notes": "x",
        "images": [{"uri": " a.png", "detail": "high"}, "b.png"],
    }

    with pytest.raises(InputsError) as raised:
        parse_input_row(row, where="f:3")

    assert raised.value.problems == (
        "f:3: unknown keys ['notes']",
        "f:3: image 1: uri must be a non-empty string without surrounding spaces",
        "f:3: image 2: an image must be a JSON object",
        'f:3: benchmark_item_id must be a string: write the number 7 as "7"',
        "f:3: prompt must be a string with more than whitespace",
    )


def test_python_values_report_every_problem_too() -> None:
    with pytest.raises(ValueError, match='as "7"; prompt must be'):
        ModelInput(7, " ")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Image detail must .*; Image sha256 must"):
        ImageRef("a.png", "medium", sha256="abc")


def test_a_message_lists_the_first_twenty_problems(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "inputs.jsonl",
        [{"benchmark_item_id": number, "prompt": "p"} for number in range(25)],
    )

    with pytest.raises(InputsError) as raised:
        read_inputs(path)

    lines = str(raised.value).splitlines()
    assert len(raised.value.problems) == 25
    assert lines[0] == f"25 problems in {path}:"
    assert (
        lines[1]
        == f'  {path}:1: {ITEM_ID_KEY} must be a string: write the number 0 as "0"'
    )
    assert len(lines) == 22
    assert lines[-1] == (
        f"  … and 5 more; fab-bench check-inputs {path} lists them all"
    )


def test_a_check_raises_only_when_the_file_cant_be_read(
    tmp_path: Path, storage: Any
) -> None:
    storage.objects["data"] = {}

    with pytest.raises(ConfigError, match="Could not read inputs file"):
        check_inputs_file(tmp_path / "missing.jsonl")
    with pytest.raises(ConfigError, match="gs://data/no.jsonl does not exist"):
        check_inputs_file("gs://data/no.jsonl", storage_client=storage)
    assert check_inputs_file(_write(tmp_path / "e.jsonl", [])).problems == (
        f"{tmp_path / 'e.jsonl'}: no inputs",
    )
