"""The runner's details: checkpoints, parallel models, stale rows, retry checks.

What runs are meant to do is described in ``tests/behaviour``; these tests pin the
mechanics underneath.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from threading import Barrier, Lock
from time import sleep
from typing import Any

import pytest
from fdllm.errors import EmptyLLMResponse, LLMContentFiltered

from fab_benchmark_runner import (
    ConfigError,
    ModelInput,
    create_experiment,
    execute_run,
    parse_run_config,
    plan_run,
)

ITEMS = [
    {"benchmark_item_id": f"item-{n}", "prompt": f"Prompt {n}?"} for n in (1, 2, 3)
]


@pytest.fixture
def experiment(tmp_path: Path) -> Path:
    inputs = tmp_path / "inputs.jsonl"
    inputs.write_text("".join(json.dumps(row) + "\n" for row in ITEMS))
    create_experiment(inputs, dest=tmp_path / "exp", benchmark="t", max_tokens=100)
    return tmp_path / "exp"


def _config(experiment: Path, *models: str, **raw: Any):
    return parse_run_config(
        {
            "experiment": str(experiment),
            "models": list(models or ("model-a",)),
            "model_workers": 1,
            **raw,
        },
        base_dir=experiment.parent,
    )


def _echo(model: str, model_input: ModelInput, _max_tokens) -> Mapping[str, Any]:
    return {"output": f"{model}:{model_input.benchmark_item_id}"}


def _rows(experiment: Path, model: str = "model-a") -> list[dict[str, Any]]:
    path = experiment / "responses" / f"responses_{model}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_checkpoints_fire_after_each_chunk_then_the_record(experiment: Path) -> None:
    events: list[str] = []

    def call(model, model_input, max_tokens):
        events.append(f"call:{model_input.benchmark_item_id}")
        return {"output": "x"}

    execute_run(
        _config(experiment, chunk_size=2),
        experiment,
        model_configs={},
        call_model=call,
        registry={},
        on_artifact_write=lambda path: events.append(f"write:{path.name}"),
    )

    assert events == [
        "call:item-1",
        "call:item-2",
        "write:responses_model-a.jsonl",
        "call:item-3",
        "write:responses_model-a.jsonl",
        "write:model-a.json",
    ]


def test_a_null_chunk_size_saves_once_per_model(experiment: Path) -> None:
    writes: list[str] = []

    execute_run(
        _config(experiment, chunk_size=None),
        experiment,
        model_configs={},
        call_model=_echo,
        registry={},
        on_artifact_write=lambda path: writes.append(path.name),
    )

    assert writes == ["responses_model-a.jsonl", "model-a.json"]


def test_a_rerun_with_nothing_to_do_writes_nothing(experiment: Path) -> None:
    execute_run(
        _config(experiment), experiment, model_configs={}, call_model=_echo, registry={}
    )
    writes: list[str] = []

    execute_run(
        _config(experiment),
        experiment,
        model_configs={},
        call_model=_echo,
        registry={},
        on_artifact_write=lambda path: writes.append(path.name),
    )

    assert writes == []


@pytest.mark.parametrize("model_workers", [1, 2])
def test_model_workers_caps_the_models_running_at_once(
    experiment: Path, model_workers: int
) -> None:
    lock = Lock()
    active = maximum = 0

    def call(model, model_input, max_tokens):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        sleep(0.02)
        with lock:
            active -= 1
        return {"output": "x"}

    result = execute_run(
        _config(
            experiment, "model-a", "model-b", "model-c", model_workers=model_workers
        ),
        experiment,
        model_configs={},
        call_model=call,
        registry={},
    )

    assert result.executed == 9
    assert maximum == model_workers


def test_parallel_models_each_call_their_items_in_order(experiment: Path) -> None:
    first_calls = Barrier(2, timeout=2)
    calls: dict[str, list[str]] = {"model-a": [], "model-b": []}

    def call(model, model_input, max_tokens):
        calls[model].append(model_input.benchmark_item_id)
        if model_input.benchmark_item_id == "item-1":
            first_calls.wait()
        return {"output": "x"}

    execute_run(
        _config(experiment, "model-a", "model-b", model_workers=2),
        experiment,
        model_configs={},
        call_model=call,
        registry={},
    )

    assert calls == {
        "model-a": ["item-1", "item-2", "item-3"],
        "model-b": ["item-1", "item-2", "item-3"],
    }


def test_a_failing_model_stops_further_models_from_starting(experiment: Path) -> None:
    started: list[str] = []
    lock = Lock()

    def call(model, model_input, max_tokens):
        with lock:
            if model not in started:
                started.append(model)
        if model == "model-a":
            raise RuntimeError("bug in model-a's handling")
        sleep(0.02)
        return {"output": "x"}

    with pytest.raises(RuntimeError, match="model-a"):
        execute_run(
            _config(experiment, "model-a", "model-b", "model-c", model_workers=2),
            experiment,
            model_configs={},
            call_model=call,
            registry={},
        )

    assert started == ["model-a", "model-b"]
    assert len(_rows(experiment, "model-b")) == 3


@pytest.mark.parametrize("error", [KeyError("bug"), ValueError("bug")])
def test_exceptions_other_than_provider_errors_stop_the_run(
    experiment: Path, error: Exception
) -> None:
    def call(*_args):
        raise error

    with pytest.raises(type(error)):
        execute_run(
            _config(experiment),
            experiment,
            model_configs={},
            call_model=call,
            registry={},
        )


def test_fdllm_response_errors_become_error_rows_with_their_stop_reason(
    experiment: Path,
) -> None:
    def call(model, model_input, max_tokens):
        if model_input.benchmark_item_id == "item-1":
            raise EmptyLLMResponse("Empty response", stop_reason="length")
        if model_input.benchmark_item_id == "item-2":
            raise LLMContentFiltered("Refused", stop_reason="content_filter")
        return {"output": "x"}

    result = execute_run(
        _config(experiment), experiment, model_configs={}, call_model=call, registry={}
    )

    assert result.errors == 2
    assert [(r["error_type"], r["stop_reason"]) for r in _rows(experiment)] == [
        ("EmptyLLMResponse", "length"),
        ("LLMContentFiltered", "content_filter"),
        ("", None),
    ]


def test_a_response_cannot_overwrite_the_rows_input_fields(experiment: Path) -> None:
    def call(model, model_input, max_tokens):
        return {
            "output": "x",
            "prompt": "changed",
            "max_tokens": 1,
            "latency_seconds": 2,
        }

    execute_run(
        _config(experiment), experiment, model_configs={}, call_model=call, registry={}
    )

    row = _rows(experiment)[0]
    assert (row["prompt"], row["max_tokens"], row["latency_seconds"]) == (
        "Prompt 1?",
        100,
        2,
    )


def test_model_configs_enter_the_call_ids_so_a_changed_config_reruns_that_model(
    experiment: Path,
) -> None:
    """Old rows of the changed model are dropped at its next write."""

    before = {"model-a": {"Temperature": 0}, "model-b": {"Temperature": 0}}
    after = {"model-a": {"Temperature": 1}, "model-b": {"Temperature": 0}}
    execute_run(
        _config(experiment, "model-a", "model-b"),
        experiment,
        model_configs=before,
        call_model=_echo,
        registry={},
    )

    plan = plan_run(
        _config(experiment, "model-a", "model-b"), experiment, model_configs=after
    )
    execute_run(
        _config(experiment, "model-a", "model-b"),
        experiment,
        model_configs=after,
        call_model=_echo,
        registry={},
    )

    assert [(m.model, m.to_run, m.stale) for m in plan.models] == [
        ("model-a", 3, 3),
        ("model-b", 0, 0),
    ]
    assert len(_rows(experiment, "model-a")) == 3


def test_model_configs_are_recorded_without_secrets(experiment: Path) -> None:
    configs = {"model-a": {"Client_Args": {"api_key": "secret", "base_url": "u"}}}

    execute_run(
        _config(experiment),
        experiment,
        model_configs=configs,
        call_model=_echo,
        registry={},
    )

    record = json.loads((experiment / "models" / "model-a.json").read_text())
    assert record["model_config"] == {"Client_Args": {"base_url": "u"}}


def test_retry_errors_refuses_missing_calls_and_invalid_statuses(
    experiment: Path,
) -> None:
    execute_run(
        _config(experiment), experiment, model_configs={}, call_model=_echo, registry={}
    )
    path = experiment / "responses" / "responses_model-a.jsonl"
    rows = _rows(experiment)

    path.write_text("".join(json.dumps(row) + "\n" for row in rows[:2]))
    with pytest.raises(ConfigError, match=r"missing calls by model: model-a \(1\)"):
        plan_run(_config(experiment, retry_errors=True), experiment, model_configs={})

    rows[2]["call_status"] = "unknown"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ConfigError, match="invalid call_status"):
        plan_run(_config(experiment, retry_errors=True), experiment, model_configs={})


def test_a_model_record_of_another_experiment_is_refused(
    experiment: Path, tmp_path: Path
) -> None:
    execute_run(
        _config(experiment), experiment, model_configs={}, call_model=_echo, registry={}
    )
    record_path = experiment / "models" / "model-a.json"
    record = json.loads(record_path.read_text())
    record["inputs_sha256"] = "other"
    record_path.write_text(json.dumps(record))

    with pytest.raises(ConfigError, match="belongs to experiment"):
        execute_run(
            _config(experiment, force=True),
            experiment,
            model_configs={},
            call_model=_echo,
            registry={},
        )
