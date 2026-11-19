"""The inputs file: the interface between a benchmark and the runner.

INPUTS.md specifies the format; each test here checks one of its rules.

Decisions this file describes:

- A benchmark writes its inputs with its own code, and the runner sends them as they
  are. The runner checks them strictly instead: every rule below is refused at
  creation, naming the line, before anything is stored.
- ``fab-bench check-inputs`` (``check_inputs_file``) checks a file on its own and lists
  every problem at once, each with its line, so one pass finds them all. It reads each
  image to check that it exists and matches a SHA-256 the file gives, but leaves
  checking that it opens to the run that sends it.
- Keys the format doesn't define are refused, so a misspelled field can't silently
  drop part of an input. Item data the model doesn't see, such as answer keys,
  subjects or levels, stays with the benchmark and is joined to results on
  ``benchmark_item_id``: in the inputs it would count in the experiment's hash, so
  fixing an answer key would make a new experiment.
- Prompts and system prompts must hold more than whitespace. No system prompt is
  written as ``null`` or left out.
- Item IDs are strings, so a numeric ID reads the same in every row and language.
- An experiment stores a normalised copy, so files that differ only in key order,
  spacing, line endings or omitted optional keys are the same experiment.
"""

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
    create_experiment,
    read_experiment,
    write_inputs,
)

from .conftest import TEXT_INPUTS, Bench, FakeProviders, png

ITEM = {"benchmark_item_id": "a", "prompt": "Which word is a noun?"}


def _lines(*rows: dict[str, Any]) -> bytes:
    return "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")


def _create(bench: Bench, content: bytes, dest: str = "experiment") -> str:
    """Create an experiment from an inputs file with exactly these bytes."""

    path = bench.root / "inputs.jsonl"
    path.write_bytes(content)
    create_experiment(
        path,
        dest=bench.root / dest,
        benchmark="lk",
        max_tokens=100,
        storage_client=bench.storage,
    )
    return read_experiment(bench.root / dest).experiment_id


# The file


def test_each_line_is_one_item_and_line_order_is_call_order(bench: Bench) -> None:
    inputs = list(reversed(TEXT_INPUTS))
    bench.create(inputs)

    bench.run("model-a")

    order = [row["benchmark_item_id"] for row in inputs]
    assert bench.models.called() == order
    assert [row["benchmark_item_id"] for row in bench.rows()] == order


def test_windows_line_endings_and_a_missing_final_newline_are_the_same_file(
    bench: Bench,
) -> None:
    unix = _lines(ITEM, {**ITEM, "benchmark_item_id": "b"})
    windows = unix.replace(b"\n", b"\r\n").removesuffix(b"\r\n")

    assert _create(bench, unix, "unix") == _create(bench, windows, "windows")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (_lines(ITEM) + b"\n" + _lines({**ITEM, "benchmark_item_id": "b"}), "blank"),
        (b"\xef\xbb\xbf" + _lines(ITEM), "BOM"),
        (b"", "no inputs"),
    ],
    ids=["blank line", "byte-order mark", "empty file"],
)
def test_blank_lines_a_byte_order_mark_and_empty_files_are_refused(
    bench: Bench, content: bytes, message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        _create(bench, content)


def test_errors_name_the_line_and_the_image(bench: Bench) -> None:
    images = [{"uri": "x.png", "detail": "low"}, {"uri": "x.png"}]
    bad_image = {**ITEM, "benchmark_item_id": "b", "images": images}

    with pytest.raises(ConfigError, match=r"inputs\.jsonl:2: image 2: "):
        _create(bench, _lines(ITEM, bad_image))


def test_a_check_lists_every_problem_with_its_line(bench: Bench) -> None:
    (bench.root / "figure.png").write_bytes(png("red"))
    right, wrong = (sha256(png(color)).hexdigest() for color in ("red", "blue"))
    images = [
        {"uri": "missing.png", "detail": "low"},
        {"uri": "figure.png", "detail": "low", "sha256": wrong},
    ]
    path = bench.root / "inputs.jsonl"
    path.write_bytes(
        _lines(
            ITEM,
            {**ITEM, "benchmark_item_id": 1042, "answer": "B"},
            {**ITEM, "prompt": " ", "images": [{"uri": "figure.png", "detail": "mid"}]},
            {**ITEM, "benchmark_item_id": "c", "images": images},
        )
    )

    check = check_inputs_file(path, storage_client=bench.storage)

    figure, missing = (bench.root.resolve() / i["uri"] for i in reversed(images))
    assert check.problems == (
        f"{path}:2: unknown keys ['answer']",
        f"{path}:2: benchmark_item_id must be a string: write the number 1042 as "
        '"1042"',
        f"{path}:3: image 1: detail must be 'low' or 'high', not 'mid'",
        f"{path}:3: prompt must be a string with more than whitespace",
        f"{path}:3: duplicate benchmark_item_id 'a', first on line 1",
        f"{path}:4: image 1: {missing} does not exist",
        f"{path}:4: image 2: {figure} has SHA-256 {right}, not the {wrong} the "
        "file gives",
    )
    with pytest.raises(InputsError) as refused:
        _create(bench, path.read_bytes())
    assert refused.value.problems == check.problems[:5]


def test_a_check_counts_items_and_images_and_writes_nothing(bench: Bench) -> None:
    (bench.root / "figure.png").write_bytes(png())
    twice = [
        {"uri": "figure.png", "detail": "low"},
        {"uri": "figure.png", "detail": "high"},
    ]
    path = bench.root / "inputs.jsonl"
    path.write_bytes(
        _lines({**ITEM, "images": twice}, {**ITEM, "benchmark_item_id": "b"})
    )
    before = sorted(bench.root.rglob("*"))

    check = check_inputs_file(path, storage_client=bench.storage)

    assert (check.problems, check.items) == ((), 2)
    assert (check.image_files, check.image_uses) == (1, 2)
    assert sorted(bench.root.rglob("*")) == before


def test_a_check_reads_a_file_and_its_images_on_gcs(bench: Bench) -> None:
    bench.put("gs://bucket/mathv/images/1.png", png())
    images = [
        {"uri": "images/1.png", "detail": "high"},
        {"uri": "images/2.png", "detail": "high"},
    ]
    bench.put("gs://bucket/mathv/inputs.jsonl", _lines({**ITEM, "images": images}))

    check = check_inputs_file(
        "gs://bucket/mathv/inputs.jsonl", storage_client=bench.storage
    )

    assert check.problems == (
        "gs://bucket/mathv/inputs.jsonl:1: image 2: gs://bucket/mathv/images/2.png "
        "does not exist",
    )


# Items


def test_item_ids_are_unique_strings(bench: Bench) -> None:
    with pytest.raises(ConfigError, match="duplicate benchmark_item_id 'a'"):
        _create(bench, _lines(ITEM, ITEM))
    with pytest.raises(ConfigError, match='write the number 1042 as "1042"'):
        _create(bench, _lines({**ITEM, "benchmark_item_id": 1042}))

    _create(bench, _lines({**ITEM, "benchmark_item_id": "1042"}))


@pytest.mark.parametrize(
    "item",
    [
        {**ITEM, "prompt": ""},
        {**ITEM, "prompt": " \n\t"},
        {**ITEM, "system_prompt": ""},
        {**ITEM, "system_prompt": "  "},
    ],
    ids=["empty prompt", "blank prompt", "empty system", "blank system"],
)
def test_prompts_and_system_prompts_must_hold_more_than_whitespace(
    bench: Bench, item: dict[str, Any]
) -> None:
    with pytest.raises(ConfigError, match="more than whitespace"):
        _create(bench, _lines(item))


def test_no_system_prompt_is_left_out_or_null(
    bench: Bench, providers: FakeProviders
) -> None:
    left_out = _create(bench, _lines(ITEM), "left-out")
    null = _create(bench, _lines({**ITEM, "system_prompt": None}), "null")

    bench.run("model-a", experiment=str(bench.folder("null")), through_fdllm=True)

    assert left_out == null
    [request] = providers.requests
    assert [message.Role for message in request.messages] == ["user"]


def test_keys_the_format_doesnt_define_are_refused(bench: Bench) -> None:
    """An answer key stays with the benchmark, joined on benchmark_item_id."""

    with pytest.raises(ConfigError, match=r"unknown keys \['answer'\]"):
        _create(bench, _lines({**ITEM, "answer": "B"}))
    with pytest.raises(ConfigError, match=r"unknown keys \['systemprompt'\]"):
        _create(bench, _lines({**ITEM, "systemprompt": "Answer briefly."}))


# What a model receives


def test_a_model_receives_the_system_prompt_then_one_user_message(
    bench: Bench, providers: FakeProviders
) -> None:
    (bench.root / "figure.png").write_bytes(png("red"))
    item = {
        **ITEM,
        "system_prompt": "Answer with one word.",
        "images": [{"uri": "figure.png", "detail": "low"}],
    }
    _create(bench, _lines(item))

    bench.run("model-a", through_fdllm=True)

    system, user = providers.request(ITEM["prompt"]).messages
    assert (system.Role, system.Message) == ("system", "Answer with one word.")
    assert (user.Role, user.Message) == ("user", ITEM["prompt"])
    assert [image.Detail for image in user.Images] == ["low"]


# Images


def test_an_image_is_on_gcs_at_an_absolute_path_or_relative_to_the_file(
    bench: Bench, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = bench.root / "home"
    (bench.root / "figures").mkdir()
    home.mkdir()
    for path in (bench.root / "figures/1.png", home / "2.png", bench.root / "3.png"):
        path.write_bytes(png())
    bench.put("gs://bucket/4.png", png())
    monkeypatch.setenv("HOME", str(home))
    uris = ["figures/1.png", "~/2.png", str(bench.root / "3.png"), "gs://bucket/4.png"]
    item = {**ITEM, "images": [{"uri": uri, "detail": "low"} for uri in uris]}

    _create(bench, _lines(item))

    [stored] = read_experiment(bench.folder()).inputs
    assert [image.uri for image in stored.images] == [
        str((bench.root / "figures/1.png").resolve()),
        str((home / "2.png").resolve()),
        str((bench.root / "3.png").resolve()),
        "gs://bucket/4.png",
    ]


@pytest.mark.parametrize(
    ("uri", "outcome"),
    [
        ("../images/1.png", "gs://bucket/mathv/images/1.png"),
        ("../../../1.png", "leaves the bucket"),
        ("/data/1.png", "can only point at images on GCS"),
    ],
)
def test_relative_paths_in_a_file_on_gcs_resolve_within_its_bucket(
    bench: Bench, uri: str, outcome: str
) -> None:
    bench.put("gs://bucket/mathv/images/1.png", png())
    item = {**ITEM, "images": [{"uri": uri, "detail": "high"}]}
    bench.put("gs://bucket/mathv/inputs/inputs.jsonl", _lines(item))

    def create() -> None:
        create_experiment(
            "gs://bucket/mathv/inputs/inputs.jsonl",
            dest=bench.folder(),
            benchmark="mathv",
            max_tokens=100,
            storage_client=bench.storage,
        )

    if outcome.startswith("gs://"):
        create()
        [stored] = read_experiment(bench.folder()).inputs
        assert stored.images[0].uri == outcome
    else:
        with pytest.raises(ConfigError, match=outcome):
            create()


@pytest.mark.parametrize("detail", [None, "medium", "auto"])
def test_an_image_has_a_detail_level_of_low_or_high(
    bench: Bench, detail: str | None
) -> None:
    image = {"uri": "figure.png"} | ({} if detail is None else {"detail": detail})

    with pytest.raises(ConfigError, match="detail must be 'low' or 'high'"):
        _create(bench, _lines({**ITEM, "images": [image]}))


def test_an_images_given_sha256_is_checked_at_creation(bench: Bench) -> None:
    (bench.root / "figure.png").write_bytes(png("red"))
    right = sha256(png("red")).hexdigest()
    wrong = sha256(png("blue")).hexdigest()

    def item(digest: str) -> dict[str, Any]:
        image = {"uri": "figure.png", "detail": "low", "sha256": digest}
        return {**ITEM, "images": [image]}

    with pytest.raises(ConfigError, match=f"not the {wrong} the inputs give"):
        _create(bench, _lines(item(wrong)), "wrong")
    _create(bench, _lines(item(right)), "right")


def test_one_image_can_appear_in_several_items_and_twice_in_one(bench: Bench) -> None:
    (bench.root / "figure.png").write_bytes(png())
    twice = [
        {"uri": "figure.png", "detail": "low"},
        {"uri": "figure.png", "detail": "high"},
    ]
    rows = [
        {**ITEM, "images": twice},
        {**ITEM, "benchmark_item_id": "b", "images": twice[:1]},
    ]

    _create(bench, _lines(*rows))

    assert read_experiment(bench.folder()).record["image_count"] == 1


# How an experiment stores it


def test_files_that_differ_only_in_form_are_the_same_experiment(bench: Bench) -> None:
    compact = json.dumps(ITEM, separators=(",", ":")) + "\n"
    spelled_out = json.dumps(
        {"system_prompt": None, "images": [], **dict(reversed(ITEM.items()))},
        indent=None,
        separators=(" ,  ", " :  "),
    )

    first = _create(bench, compact.encode(), "compact")
    second = _create(bench, (spelled_out + "\n").encode(), "spelled-out")

    assert first == second


def test_a_benchmark_in_python_writes_the_same_file(bench: Bench) -> None:
    from_json = _create(bench, _lines({**ITEM, "system_prompt": "Be brief."}), "json")
    path = bench.root / "written.jsonl"
    write_inputs([ModelInput("a", ITEM["prompt"], "Be brief.")], path)

    from_python = _create(bench, path.read_bytes(), "python")

    assert from_json == from_python


def test_write_inputs_checks_the_same_rules(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="more than whitespace"):
        ModelInput("a", "   ")
    with pytest.raises(ValueError, match="detail must be"):
        ImageRef("figure.png", "medium")
    with pytest.raises(ConfigError, match="duplicate"):
        write_inputs([ModelInput("a", "p"), ModelInput("a", "q")], tmp_path / "x")
