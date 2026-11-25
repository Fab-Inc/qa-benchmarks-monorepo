"""Running models: results keyed by call IDs, never redone, one set per model.

Decisions this file describes:

- A run config only names the experiment, the models, the registry and how to run.
  Adding a model means writing another run config; nothing shared is rewritten.
- Each model's results and record are its own files, written only by the job that
  runs that model.
- A result is reused whenever its call ID already exists, so a rerun after a crash
  or timeout makes only the missing calls, and a full rerun makes none.
- A provider failure becomes an error row; ``retry_errors`` reruns only error rows
  and never makes new calls. Any other exception stops the run, keeping what was
  already saved.
- Rows repeat the whole input, so each row can be read on its own.
- Memory doesn't grow with the results stored or the number of models. Planning
  reads only each stored result's ID and status. A model's rows are loaded only
  while it runs, because each checkpoint rewrites its file from them, and released
  once its record is written. So a run holds at most ``model_workers`` models'
  rows, and a large experiment with many models fits a Cloud Run job.
"""

from __future__ import annotations

import tracemalloc
from collections.abc import Callable

import pytest

from fab_benchmark_runner import ConfigError, parse_run_config

from .conftest import TEXT_INPUTS, Bench

ITEMS = [row["benchmark_item_id"] for row in TEXT_INPUTS]


def test_each_model_gets_one_results_file_and_one_record(bench: Bench) -> None:
    """The experiment folder gains exactly two files per model, and nothing else."""

    bench.create()

    bench.run("model-a", "model-b")

    assert bench.files() == {
        "experiment.json",
        "inputs.jsonl",
        "models/model-a.json",
        "models/model-b.json",
        "responses/responses_model-a.jsonl",
        "responses/responses_model-b.jsonl",
    }


def test_each_row_repeats_its_input_so_it_reads_on_its_own(bench: Bench) -> None:
    bench.create(experiment_id="lk-sw-v1")

    bench.run("model-a")

    row = bench.row("mcq-1")
    assert {key: row[key] for key in ("prompt", "system_prompt", "images")} == {
        "prompt": TEXT_INPUTS[0]["prompt"],
        "system_prompt": TEXT_INPUTS[0]["system_prompt"],
        "images": [],
    }
    assert row["output"] == "model-a answers mcq-1"
    assert row["call_status"] == "success"
    assert len(row["result_id"]) == 24


def test_running_again_reuses_every_result(bench: Bench) -> None:
    bench.create()
    bench.run("model-a", "model-b")
    bench.models.forget_calls()

    result = bench.run("model-a", "model-b")

    assert bench.models.calls == []
    assert (result.executed, result.reused) == (0, 6)


def test_adding_a_model_later_leaves_the_other_models_files_untouched(
    bench: Bench,
) -> None:
    """A new run config for one new model runs only that model."""

    bench.create()
    bench.run("model-a")
    files = ("models/model-a.json", "responses/responses_model-a.jsonl")
    before = {name: (bench.folder() / name).read_bytes() for name in files}
    bench.models.forget_calls()

    bench.run("model-b")

    assert set(bench.models.called()) == set(ITEMS)
    assert {m for m, _, _ in bench.models.calls} == {"model-b"}
    assert {name: (bench.folder() / name).read_bytes() for name in files} == before


def test_a_run_that_stopped_resumes_where_it_stopped(bench: Bench) -> None:
    """Results are saved every ``chunk_size`` calls; an unexpected exception stops
    the run but keeps them, and the next run makes only the missing calls."""

    bench.create()
    bench.models.crash("short-1")

    with pytest.raises(RuntimeError):
        bench.run("model-a", chunk_size=1)
    assert [r["benchmark_item_id"] for r in bench.rows()] == ["mcq-1"]

    bench.models.recover("short-1")
    bench.models.forget_calls()
    bench.run("model-a", chunk_size=1)

    assert bench.models.called() == ["short-1", "open-1"]


def test_failed_calls_become_error_rows_that_retry_errors_reruns(bench: Bench) -> None:
    bench.create()
    bench.models.fail("short-1")

    first = bench.run("model-a")

    assert first.errors == 1
    failed = bench.row("short-1")
    assert (failed["call_status"], failed["error_type"]) == (
        "error",
        "MistralConnectionException",
    )

    bench.models.recover("short-1")
    bench.models.forget_calls()
    retried = bench.run("model-a", retry_errors=True)

    assert bench.models.called() == ["short-1"]
    assert retried.errors == 0
    assert bench.row("short-1")["call_status"] == "success"


def test_an_empty_response_is_an_error(bench: Bench) -> None:
    """An empty answer can't be scored, so it is retried like any failure."""

    bench.create()
    bench.models.need_tokens("open-1", 5000, empty_below="empty")

    bench.run("model-a")

    assert bench.row("open-1")["error_type"] == "EmptyLLMResponse"


def test_retry_errors_never_makes_new_calls(bench: Bench) -> None:
    """Retrying is for failures already seen; it refuses, before any call, a model
    whose results are missing."""

    bench.create()
    bench.run("model-a")
    bench.models.forget_calls()

    with pytest.raises(ConfigError, match="No results to retry for model-b"):
        bench.run("model-a", "model-b", retry_errors=True)
    assert bench.models.calls == []


def test_force_reruns_every_call(bench: Bench) -> None:
    bench.create()
    bench.run("model-a")
    bench.models.forget_calls()

    bench.run("model-a", force=True)

    assert bench.models.called() == ITEMS


def test_a_dry_run_reports_what_a_run_would_do_and_changes_nothing(
    bench: Bench,
) -> None:
    bench.create()
    bench.run("model-a")
    bench.models.forget_calls()
    files_before = bench.files()

    plan = bench.plan("model-a", "model-b")

    assert [(m.model, m.reused, m.to_run) for m in plan.models] == [
        ("model-a", 3, 0),
        ("model-b", 0, 3),
    ]
    assert bench.models.calls == []
    assert bench.files() == files_before


@pytest.mark.parametrize(
    ("key", "hint"),
    [
        ("max_tokens", "belongs to the experiment"),
        ("items_path", "fab-bench create-experiment"),
        ("run_id", "named by its experiment and models"),
        ("temperature", "Unknown run config keys: temperature"),
    ],
)
def test_a_run_config_names_only_the_experiment_models_and_how_to_run(
    bench: Bench, key: str, hint: str
) -> None:
    """Anything else is refused, so a misspelled or outdated setting can't be
    silently ignored."""

    raw = {"experiment": "experiment", "models": ["model-a"], key: 1}

    with pytest.raises(ConfigError, match=hint):
        parse_run_config(raw, base_dir=bench.root)


MANY_ITEMS = [
    {"benchmark_item_id": f"item-{n}", "prompt": f"Question {n}?"} for n in range(20)
]


def _peak_bytes(action: Callable[[], object]) -> int:
    tracemalloc.start()
    try:
        action()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_planning_holds_none_of_the_stored_answers(bench: Bench) -> None:
    bench.create(MANY_ITEMS)
    bench.models.answer_at_length(100_000)
    models = ("model-a", "model-b", "model-c")
    bench.run(*models)
    one_model = sum(len(row["output"]) for row in bench.rows("model-a"))

    planning = _peak_bytes(lambda: bench.plan(*models))

    assert planning < one_model


def test_a_run_holds_only_the_running_models_answers(bench: Bench) -> None:
    """With one model at a time, as here, a run that redoes every model needs
    memory for one model's answers, not three."""

    bench.create(MANY_ITEMS)
    bench.models.answer_at_length(100_000)
    models = ("model-a", "model-b", "model-c")
    bench.run(*models)
    one_model = sum(len(row["output"]) for row in bench.rows("model-a"))

    running = _peak_bytes(lambda: bench.run(*models, force=True))

    assert running < 2 * one_model
