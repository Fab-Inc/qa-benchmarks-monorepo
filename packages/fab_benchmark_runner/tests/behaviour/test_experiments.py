"""Experiments: a benchmark's inputs, stored once so every model gets the same.

Decisions this file describes:

- A benchmark hands over fully rendered inputs, one row per item; the runner never
  builds prompts. Phonics' prompts were edited in place 8 times in 14 commits, and
  the only signal was old models suddenly rerunning; benchmarks also differ in how
  they pick items, prompts and images.
- Each item carries its own system prompt, because the linguistic knowledge
  benchmark uses different ones per question type.
- An experiment's inputs never change after creation, so a model added months later
  gets exactly what the first model got, however the benchmark's code has changed.
- ``experiment_id`` is a name, chosen at creation or a short hash of the benchmark
  and inputs by default, independent of the folder.
- Images are referenced by URI and identified by their file's SHA-256; copying them
  into the experiment is optional and doesn't change any call ID.
"""

from __future__ import annotations

from hashlib import sha256

import pytest

from fab_benchmark_runner import ConfigError, create_experiment, read_experiment

from .conftest import TEXT_INPUTS, Bench


def test_models_get_exactly_the_stored_inputs_with_each_items_system_prompt(
    bench: Bench,
) -> None:
    """The runner sends what the inputs file says, item by item; it builds nothing."""

    bench.create()

    bench.run("model-a")

    sent = [
        (i.benchmark_item_id, i.system_prompt, i.prompt) for i in bench.models.inputs
    ]
    assert sent == [
        (row["benchmark_item_id"], row.get("system_prompt"), row["prompt"])
        for row in TEXT_INPUTS
    ]


def test_a_model_added_later_gets_the_inputs_as_they_were_at_creation(
    bench: Bench,
) -> None:
    """Changing the benchmark's inputs file afterwards changes no experiment."""

    bench.create()
    bench.run("model-a")
    bench.write_inputs(
        [{**row, "prompt": row["prompt"] + " (reworded)"} for row in TEXT_INPUTS]
    )
    bench.models.forget_calls()

    bench.run("model-b")

    assert [i.prompt for i in bench.models.inputs] == [r["prompt"] for r in TEXT_INPUTS]


def test_a_run_refuses_stored_inputs_that_were_edited(bench: Bench) -> None:
    """The record keeps the inputs' hash; a mismatch stops the run before any call."""

    bench.create()
    inputs = bench.folder() / "inputs.jsonl"
    inputs.write_text(inputs.read_text().replace("cat", "dog"), encoding="utf-8")

    with pytest.raises(ConfigError, match="doesn't match the inputs_sha256"):
        bench.run("model-a")
    assert bench.models.calls == []


def test_creating_the_same_experiment_again_changes_nothing(bench: Bench) -> None:
    """Creation can be repeated safely, for example after an interruption."""

    location = bench.create()
    record_before = (bench.folder() / "experiment.json").read_bytes()

    again = create_experiment(
        bench.write_inputs(), dest=location, benchmark="lk", max_tokens=1000
    )

    assert again.status == "exists"
    assert (bench.folder() / "experiment.json").read_bytes() == record_before


def test_a_folder_holds_one_experiment(bench: Bench) -> None:
    """Different inputs never land in an existing experiment's folder."""

    location = bench.create()

    with pytest.raises(ConfigError, match="already holds experiment"):
        create_experiment(
            bench.write_inputs(TEXT_INPUTS[:1], name="subset.jsonl"),
            dest=location,
            benchmark="lk",
            max_tokens=1000,
        )


def test_the_experiment_id_is_a_name_independent_of_the_folder(bench: Bench) -> None:
    """By default it is a hash of benchmark and inputs, so the same inputs give the
    same ID in any folder; a chosen ID can be anything readable."""

    bench.create(dest="first")
    bench.create(dest="second")
    bench.create(dest="named", experiment_id="lk-sw-v1")

    first, second, named = (
        read_experiment(bench.folder(name)) for name in ("first", "second", "named")
    )
    assert first.experiment_id == second.experiment_id
    assert len(first.experiment_id) == 16
    assert named.experiment_id == "lk-sw-v1"


def test_rows_say_which_experiment_and_benchmark_they_belong_to(bench: Bench) -> None:
    """Combined results stay traceable; merging checks duplicates per experiment_id
    and benchmark."""

    bench.create(experiment_id="lk-sw-v1")

    bench.run("model-a")

    assert {(r["experiment_id"], r["benchmark"]) for r in bench.rows()} == {
        ("lk-sw-v1", "lk")
    }


def test_images_are_identified_by_their_content_not_where_they_are(
    bench: Bench,
) -> None:
    """Referencing an image or copying it into the experiment gives the same call
    IDs, so results don't depend on where images are stored."""

    image = bench.root / "figure.png"
    image.write_bytes(b"pixels")
    inputs = [
        {
            "benchmark_item_id": "visual-1",
            "prompt": "Which shape is shown?",
            "images": [{"uri": "figure.png", "detail": "high"}],
        }
    ]
    bench.create(inputs, dest="referenced")
    bench.create(inputs, dest="copied", copy_images=True)

    bench.run("model-a", experiment=str(bench.folder("referenced")))
    bench.run("model-a", experiment=str(bench.folder("copied")))

    [referenced] = bench.rows(experiment="referenced")
    [copied] = bench.rows(experiment="copied")
    assert referenced["result_id"] == copied["result_id"]
    assert referenced["images"][0]["sha256"] == sha256(b"pixels").hexdigest()
    assert copied["images"][0]["uri"].startswith("images/")
