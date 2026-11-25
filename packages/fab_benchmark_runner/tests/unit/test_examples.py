"""The smoke example that WALKTHROUGH.md uses must stay a valid experiment."""

from __future__ import annotations

import json
from pathlib import Path

from fab_benchmark_runner import (
    ImageFiles,
    check_inputs_file,
    create_experiment,
    read_experiment,
)

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "smoke"


def test_the_smoke_example_is_a_self_contained_experiment(tmp_path: Path) -> None:
    creation = create_experiment(
        EXAMPLE / "inputs.jsonl",
        dest=tmp_path / "smoke-v1",
        benchmark="smoke",
        experiment_id="smoke-v1",
        max_tokens=2048,
        copy_images=True,
    )

    experiment = read_experiment(tmp_path / "smoke-v1")
    assert creation.status == "created"
    assert [item.benchmark_item_id for item in experiment.inputs] == [
        "mcq-1",
        "short-1",
        "count-1",
        "compare-1",
    ]
    assert experiment.record["image_count"] == 2
    images = [image for item in experiment.inputs for image in item.images]
    files = ImageFiles(tmp_path / "copies", experiment_location=tmp_path / "smoke-v1")
    files.fetch(images)


def test_the_answers_stay_outside_the_inputs_and_cover_every_item() -> None:
    def ids(name: str) -> list[str]:
        lines = (EXAMPLE / name).read_text(encoding="utf-8").splitlines()
        return [json.loads(line)["benchmark_item_id"] for line in lines]

    assert ids("answers.jsonl") == ids("inputs.jsonl")
    assert "answer" not in (EXAMPLE / "inputs.jsonl").read_text(encoding="utf-8")


def test_the_smoke_example_passes_a_check() -> None:
    check = check_inputs_file(EXAMPLE / "inputs.jsonl")

    assert check.problems == ()
    assert (check.items, check.image_files, check.image_uses) == (4, 2, 3)
