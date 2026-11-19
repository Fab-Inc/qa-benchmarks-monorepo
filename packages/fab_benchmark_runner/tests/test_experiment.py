"""Creating and reading experiments: stored inputs that every model runs on."""

from __future__ import annotations

import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from fab_benchmark_runner import (
    ConfigError,
    ImageRef,
    ModelInput,
    create_experiment,
    default_experiment_id,
    model_call_id,
    read_experiment,
)
from fab_benchmark_runner.experiment import EXPERIMENT_RECORD_NAME, INPUTS_FILE_NAME

TEXT_ROWS = [
    {"benchmark_item_id": "sw-1", "prompt": "Soma: Mama.", "system_prompt": "Teach."},
    {"benchmark_item_id": "sw-2", "prompt": "Soma: Baba.", "system_prompt": "Teach."},
]
IMAGE_BYTES = b"\x89PNG not really, but bytes are all a hash needs"
IMAGE_SHA256 = sha256(IMAGE_BYTES).hexdigest()


def _write_inputs(path: Path, rows: list[dict[str, Any]] = TEXT_ROWS) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _image_rows(uri: str) -> list[dict[str, Any]]:
    return [
        {
            "benchmark_item_id": f"item-{index}",
            "prompt": f"Which shape is in figure {index}?",
            "images": [{"uri": uri, "detail": "high"}],
        }
        for index in (1, 2)
    ]


def test_creates_a_local_text_experiment(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path / "inputs.jsonl")
    dest = tmp_path / "experiments" / "sw"

    creation = create_experiment(
        inputs,
        dest=dest,
        benchmark="phonics",
        max_tokens=30000,
        labels={"prompt_name": "decodability"},
    )

    record = json.loads((dest / EXPERIMENT_RECORD_NAME).read_text(encoding="utf-8"))
    inputs_bytes = (dest / INPUTS_FILE_NAME).read_bytes()
    assert creation.status == "created"
    assert record["inputs_sha256"] == sha256(inputs_bytes).hexdigest()
    assert record["experiment_id"] == default_experiment_id(
        benchmark="phonics",
        inputs_sha256=record["inputs_sha256"],
        max_tokens=30000,
    )
    assert len(record["experiment_id"]) == 16
    assert {key: record[key] for key in ("benchmark", "max_tokens", "labels")} == {
        "benchmark": "phonics",
        "max_tokens": 30000,
        "labels": {"prompt_name": "decodability"},
    }
    assert (record["item_count"], record["image_count"]) == (2, 0)
    assert record["record_format"] == 1
    assert set(record["created"]) == {"git_dirty", "git_hash", "inputs_source", "utc"}
    assert record["created"]["inputs_source"] is None
    assert [json.loads(line) for line in inputs_bytes.splitlines()] == [
        {**row} for row in TEXT_ROWS
    ]
    assert read_experiment(dest).inputs == creation.experiment.inputs
    assert sorted(path.name for path in dest.iterdir()) == [
        EXPERIMENT_RECORD_NAME,
        INPUTS_FILE_NAME,
    ]


def test_a_given_id_and_no_token_limit_are_recorded(tmp_path: Path) -> None:
    creation = create_experiment(
        _write_inputs(tmp_path / "inputs.jsonl"),
        dest=tmp_path / "exp",
        benchmark="linguistic-knowledge",
        max_tokens=None,
        experiment_id="sw-v1",
    )

    assert creation.experiment.experiment_id == "sw-v1"
    assert read_experiment(tmp_path / "exp").max_tokens is None


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"experiment_id": "-v1"}, "Experiment ID"),
        ({"experiment_id": "a/b"}, "Experiment ID"),
        ({"max_tokens": 0}, "max_tokens must be"),
        ({"max_tokens": True}, "max_tokens must be"),
        ({"benchmark": ""}, "benchmark must be"),
        ({"labels": {"subset": 1}}, "Labels must map"),
    ],
)
def test_bad_settings_are_refused(
    tmp_path: Path, settings: dict[str, Any], message: str
) -> None:
    arguments: dict[str, Any] = {"benchmark": "phonics", "max_tokens": 100, **settings}

    with pytest.raises(ConfigError, match=message):
        create_experiment(
            _write_inputs(tmp_path / "inputs.jsonl"), dest=tmp_path / "exp", **arguments
        )
    assert not (tmp_path / "exp").exists()


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    creation = create_experiment(
        _write_inputs(tmp_path / "inputs.jsonl"),
        dest=tmp_path / "exp",
        benchmark="phonics",
        max_tokens=100,
        dry_run=True,
    )

    assert creation.status == "dry-run"
    assert len(creation.experiment.inputs) == 2
    assert not (tmp_path / "exp").exists()


def test_creating_the_same_experiment_again_changes_nothing(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path / "inputs.jsonl")
    create_experiment(
        inputs, dest=tmp_path / "exp", benchmark="phonics", max_tokens=100
    )
    record_bytes = (tmp_path / "exp" / EXPERIMENT_RECORD_NAME).read_bytes()

    again = create_experiment(
        inputs,
        dest=tmp_path / "exp",
        benchmark="phonics",
        max_tokens=100,
        labels={"note": "labels don't define an experiment"},
    )

    assert again.status == "exists"
    assert (tmp_path / "exp" / EXPERIMENT_RECORD_NAME).read_bytes() == record_bytes


@pytest.mark.parametrize(
    "change",
    [{"max_tokens": 200}, {"experiment_id": "other"}, {"benchmark": "dtg"}],
)
def test_a_folder_holding_a_different_experiment_is_refused(
    tmp_path: Path, change: dict[str, Any]
) -> None:
    inputs = _write_inputs(tmp_path / "inputs.jsonl")
    create_experiment(
        inputs, dest=tmp_path / "exp", benchmark="phonics", max_tokens=100
    )

    with pytest.raises(ConfigError, match="already holds experiment"):
        create_experiment(
            inputs,
            dest=tmp_path / "exp",
            **{"benchmark": "phonics", "max_tokens": 100, **change},
        )


def test_an_interrupted_creation_is_finished_but_not_overwritten(
    tmp_path: Path,
) -> None:
    inputs = _write_inputs(tmp_path / "inputs.jsonl")
    dest = tmp_path / "exp"
    create_experiment(inputs, dest=dest, benchmark="phonics", max_tokens=100)
    (dest / EXPERIMENT_RECORD_NAME).unlink()

    finished = create_experiment(inputs, dest=dest, benchmark="phonics", max_tokens=100)

    assert finished.status == "created"
    assert read_experiment(dest).inputs == finished.experiment.inputs
    (dest / EXPERIMENT_RECORD_NAME).unlink()
    other_inputs = _write_inputs(tmp_path / "other.jsonl", TEXT_ROWS[:1])
    with pytest.raises(ConfigError, match="unfinished creation"):
        create_experiment(other_inputs, dest=dest, benchmark="phonics", max_tokens=100)


def test_a_folder_with_other_files_is_refused(tmp_path: Path) -> None:
    dest = tmp_path / "old-run"
    dest.mkdir()
    (dest / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ConfigError, match=r"holds other files \(config.json\)"):
        create_experiment(
            _write_inputs(tmp_path / "inputs.jsonl"),
            dest=dest,
            benchmark="phonics",
            max_tokens=100,
        )


def test_read_experiment_refuses_changed_inputs(tmp_path: Path) -> None:
    dest = tmp_path / "exp"
    create_experiment(
        _write_inputs(tmp_path / "inputs.jsonl"),
        dest=dest,
        benchmark="phonics",
        max_tokens=100,
    )
    with (dest / INPUTS_FILE_NAME).open("a", encoding="utf-8") as inputs_file:
        inputs_file.write(json.dumps({"benchmark_item_id": "x", "prompt": "p"}) + "\n")

    with pytest.raises(ConfigError, match="doesn't match the inputs_sha256"):
        read_experiment(dest)


def test_local_images_are_hashed_and_referenced(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "shape.png").write_bytes(IMAGE_BYTES)
    inputs = _write_inputs(tmp_path / "inputs.jsonl", _image_rows("images/shape.png"))

    creation = create_experiment(
        inputs, dest=tmp_path / "exp", benchmark="mathv", max_tokens=64000
    )

    stored = read_experiment(tmp_path / "exp")
    assert stored.record["image_count"] == 1
    assert stored.inputs[0].images == (
        ImageRef(
            uri=str((tmp_path / "images" / "shape.png").resolve()),
            detail="high",
            sha256=IMAGE_SHA256,
        ),
    )
    assert stored.inputs == creation.experiment.inputs


def test_images_that_are_missing_or_differ_from_their_hash_are_refused(
    tmp_path: Path,
) -> None:
    (tmp_path / "shape.png").write_bytes(IMAGE_BYTES)
    wrong_hash = _image_rows("shape.png")
    wrong_hash[0]["images"][0]["sha256"] = sha256(b"other").hexdigest()

    with pytest.raises(ConfigError, match="missing.png does not exist"):
        create_experiment(
            _write_inputs(tmp_path / "missing.jsonl", _image_rows("missing.png")),
            dest=tmp_path / "exp",
            benchmark="mathv",
            max_tokens=100,
        )
    with pytest.raises(ConfigError, match=f"has SHA-256 {IMAGE_SHA256}"):
        create_experiment(
            _write_inputs(tmp_path / "wrong.jsonl", wrong_hash),
            dest=tmp_path / "exp",
            benchmark="mathv",
            max_tokens=100,
        )
    assert not (tmp_path / "exp").exists()


def test_copied_images_are_stored_once_by_hash(tmp_path: Path) -> None:
    (tmp_path / "shape.PNG").write_bytes(IMAGE_BYTES)
    inputs = _write_inputs(tmp_path / "inputs.jsonl", _image_rows("shape.PNG"))
    dest = tmp_path / "exp"

    create_experiment(
        inputs, dest=dest, benchmark="mathv", max_tokens=100, copy_images=True
    )

    stored = read_experiment(dest)
    copy_name = f"images/{IMAGE_SHA256}.png"
    assert [path.name for path in (dest / "images").iterdir()] == [
        f"{IMAGE_SHA256}.png"
    ]
    assert (dest / copy_name).read_bytes() == IMAGE_BYTES
    assert {item.images[0] for item in stored.inputs} == {
        ImageRef(
            uri=copy_name,
            detail="high",
            sha256=IMAGE_SHA256,
            source_uri=str((tmp_path / "shape.PNG").resolve()),
        )
    }


def test_call_ids_do_not_depend_on_where_images_are(tmp_path: Path) -> None:
    (tmp_path / "shape.png").write_bytes(IMAGE_BYTES)
    inputs = _write_inputs(tmp_path / "inputs.jsonl", _image_rows("shape.png"))
    referenced = create_experiment(
        inputs, dest=tmp_path / "referenced", benchmark="mathv", max_tokens=100
    ).experiment
    copied = create_experiment(
        inputs,
        dest=tmp_path / "copied",
        benchmark="mathv",
        max_tokens=100,
        copy_images=True,
    ).experiment

    def call_id(model_input: ModelInput) -> str:
        return model_call_id(
            benchmark_item_id=model_input.benchmark_item_id,
            model="model-a",
            model_config={},
            prompt=model_input.prompt,
            system_prompt=model_input.system_prompt,
            max_tokens=100,
            images=[asdict(image) for image in model_input.images],
        )

    assert referenced.inputs_sha256 != copied.inputs_sha256
    assert [call_id(item) for item in referenced.inputs] == [
        call_id(item) for item in copied.inputs
    ]


def test_inputs_built_in_python_resolve_paths_from_the_working_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "shape.png").write_bytes(IMAGE_BYTES)
    monkeypatch.chdir(tmp_path)

    creation = create_experiment(
        [ModelInput("1", "Which shape?", images=(ImageRef("shape.png", "low"),))],
        dest=tmp_path / "exp",
        benchmark="mathv",
        max_tokens=100,
    )

    assert creation.experiment.inputs[0].images[0].uri == str(
        (tmp_path / "shape.png").resolve()
    )


def test_a_gcs_experiment_uploads_inputs_then_its_record(
    tmp_path: Path, storage: Any
) -> None:
    storage.objects["data"] = {"images/1.jpg": (1, IMAGE_BYTES)}
    inputs = _write_inputs(
        tmp_path / "inputs.jsonl", _image_rows("gs://data/images/1.jpg")
    )

    creation = create_experiment(
        inputs,
        dest="gs://bucket/mathv/level1",
        benchmark="mathv",
        max_tokens=64000,
        experiment_id="level1-v1",
        storage_client=storage,
    )

    uploads = [name for name, _generation in storage.upload_preconditions]
    assert uploads == [
        "mathv/level1/inputs.jsonl",
        "mathv/level1/experiment.json",
    ]
    record = json.loads(storage.objects["bucket"]["mathv/level1/experiment.json"][1])
    assert record == dict(creation.experiment.record)
    assert creation.experiment.inputs[0].images == (
        ImageRef("gs://data/images/1.jpg", "high", IMAGE_SHA256),
    )

    again = create_experiment(
        inputs,
        dest="gs://bucket/mathv/level1/",
        benchmark="mathv",
        max_tokens=64000,
        experiment_id="level1-v1",
        storage_client=storage,
    )
    assert again.status == "exists"
    assert len(storage.upload_preconditions) == 2


def test_a_gcs_experiment_copies_local_images_first(
    tmp_path: Path, storage: Any
) -> None:
    (tmp_path / "shape.jpg").write_bytes(IMAGE_BYTES)
    inputs = _write_inputs(tmp_path / "inputs.jsonl", _image_rows("shape.jpg"))

    with pytest.raises(ConfigError, match="pass copy_images"):
        create_experiment(
            inputs,
            dest="gs://bucket/exp",
            benchmark="mathv",
            max_tokens=100,
            storage_client=storage,
        )
    create_experiment(
        inputs,
        dest="gs://bucket/exp",
        benchmark="mathv",
        max_tokens=100,
        copy_images=True,
        storage_client=storage,
    )

    assert [name for name, _generation in storage.upload_preconditions] == [
        f"exp/images/{IMAGE_SHA256}.jpg",
        "exp/inputs.jsonl",
        "exp/experiment.json",
    ]


def test_a_missing_gcs_image_is_refused(tmp_path: Path, storage: Any) -> None:
    inputs = _write_inputs(tmp_path / "inputs.jsonl", _image_rows("gs://data/no.jpg"))
    storage.objects["data"] = {}

    with pytest.raises(ConfigError, match="gs://data/no.jpg does not exist"):
        create_experiment(
            inputs,
            dest=tmp_path / "exp",
            benchmark="mathv",
            max_tokens=100,
            storage_client=storage,
        )


def test_an_inputs_file_on_gcs_is_read_and_recorded(
    tmp_path: Path, storage: Any
) -> None:
    rows = _image_rows("images/1.jpg")
    storage.objects["data"] = {
        "mathv/inputs.jsonl": (
            7,
            "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8"),
        ),
        "mathv/images/1.jpg": (8, IMAGE_BYTES),
    }

    creation = create_experiment(
        "gs://data/mathv/inputs.jsonl",
        dest=tmp_path / "exp",
        benchmark="mathv",
        max_tokens=100,
        storage_client=storage,
    )
    assert creation.experiment.record["created"]["inputs_source"] == {
        "generation": 7,
        "uri": "gs://data/mathv/inputs.jsonl",
    }
    assert creation.experiment.inputs[0].images == (
        ImageRef("gs://data/mathv/images/1.jpg", "high", IMAGE_SHA256),
    )


def test_an_image_used_twice_keeps_each_detail_and_given_hash(
    tmp_path: Path,
) -> None:
    (tmp_path / "shape.png").write_bytes(IMAGE_BYTES)
    rows = _image_rows("shape.png")
    rows[1]["images"][0]["detail"] = "low"

    creation = create_experiment(
        _write_inputs(tmp_path / "inputs.jsonl", rows),
        dest=tmp_path / "exp",
        benchmark="mathv",
        max_tokens=100,
    )

    assert creation.experiment.record["image_count"] == 1
    assert [item.images[0].detail for item in creation.experiment.inputs] == [
        "high",
        "low",
    ]
    rows[1]["images"][0]["sha256"] = sha256(b"other").hexdigest()
    with pytest.raises(ConfigError, match=f"has SHA-256 {IMAGE_SHA256}"):
        create_experiment(
            _write_inputs(tmp_path / "wrong.jsonl", rows),
            dest=tmp_path / "wrong",
            benchmark="mathv",
            max_tokens=100,
        )


def test_many_gcs_images_are_hashed_and_the_first_missing_one_reported(
    tmp_path: Path, storage: Any
) -> None:
    contents = {
        f"images/{index}.jpg": f"image {index}".encode() for index in range(100)
    }
    storage.objects["data"] = {
        name: (index + 1, content)
        for index, (name, content) in enumerate(contents.items())
    }
    rows = [
        {
            "benchmark_item_id": f"item-{index}",
            "prompt": "Which shape?",
            "images": [{"uri": f"gs://data/images/{index}.jpg", "detail": "high"}],
        }
        for index in range(100)
    ]

    creation = create_experiment(
        _write_inputs(tmp_path / "inputs.jsonl", rows),
        dest=tmp_path / "exp",
        benchmark="mathv",
        max_tokens=100,
        storage_client=storage,
    )

    assert [item.images[0].sha256 for item in creation.experiment.inputs] == [
        sha256(content).hexdigest() for content in contents.values()
    ]
    del storage.objects["data"]["images/40.jpg"]
    del storage.objects["data"]["images/70.jpg"]
    with pytest.raises(ConfigError, match=r"gs://data/images/40\.jpg does not exist"):
        create_experiment(
            _write_inputs(tmp_path / "inputs.jsonl", rows),
            dest=tmp_path / "other",
            benchmark="mathv",
            max_tokens=100,
            storage_client=storage,
        )
