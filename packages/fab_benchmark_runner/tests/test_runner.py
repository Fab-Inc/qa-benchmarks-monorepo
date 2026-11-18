"""Executing and planning runs with an injected evaluator."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from threading import Barrier, Lock
from time import sleep
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from anthropic import APIConnectionError as AnthropicAPIConnectionError
from botocore.exceptions import EndpointConnectionError
from google.genai.errors import ServerError as GoogleServerError
from mistralai.exceptions import MistralConnectionException
from openai import APIConnectionError as OpenAIAPIConnectionError
from PIL import Image

from fab_benchmark_runner import (
    Benchmark,
    ConfigError,
    ImageInput,
    ModelRequest,
    RunConfig,
    execute_run,
    load_run_config,
    model_call_id,
    plan_run,
    result_id,
)
from fab_benchmark_runner.results import save_results

Rows = Callable[..., list[dict[str, Any]]]


def _echo(_config: RunConfig, model: str, item: dict, _prompt: str) -> dict:
    return {"output": f"{model}:{item['benchmark_item_id']}"}


@pytest.fixture
def config(benchmark: Benchmark, write_run: Callable[..., Path]) -> RunConfig:
    return load_run_config(write_run(), benchmark)


def test_run_writes_linked_rows_snapshot_and_log(
    tmp_path: Path, config: RunConfig, read_rows: Rows
) -> None:
    calls: list[tuple[str, str, str]] = []

    def evaluator(
        run_config: RunConfig,
        model: str,
        item: dict,
        prompt: str,
    ) -> dict:
        calls.append((run_config.params.language, model, item["benchmark_item_id"]))
        return {
            "output": f"{model}:{item['benchmark_item_id']}",
            "seen": prompt,
        }

    with (
        patch("fab_benchmark_runner.runner.git_hash", return_value="abc123"),
        patch("fab_benchmark_runner.runner.git_dirty", return_value=False),
    ):
        output = execute_run(config, evaluator=evaluator)

    assert calls == [
        ("sw", "model-a", "item-1"),
        ("sw", "model-a", "item-2"),
        ("sw", "model-b", "item-1"),
        ("sw", "model-b", "item-2"),
    ]
    assert (output["executed_count"], output["skipped_count"]) == (4, 0)
    assert output["error_count"] == 0
    assert output["total_expected_results"] == 4
    assert output["results_dir"] == config.run_dir / "responses"
    assert {path.name for path in output["results_paths"].values()} == {
        "responses_model-a.jsonl",
        "responses_model-b.jsonl",
    }

    rows = read_rows(output["results_paths"].values())
    row = rows[0]
    assert row["prompt"] == "[sw/default] A cat sat."
    assert row["seen"] == row["prompt"]
    assert (row["language"], row["prompt_name"]) == ("sw", "default")
    assert row["config_id"] == output["config_id"]
    assert row["config_snapshot_path"] == str(output["config_snapshot_path"])
    assert row["call_status"] == "success"
    assert row["success"] is True
    assert row["model_call_id"] == model_call_id(
        benchmark_item_id="item-1",
        model="model-a",
        model_config={},
        prompt=row["prompt"],
        system_prompt=None,
        max_tokens=10_000,
    )
    assert row["result_id"] == result_id(row["model_call_id"])
    assert len({row["result_id"] for row in rows}) == 4

    snapshot = json.loads(output["config_snapshot_path"].read_text(encoding="utf-8"))
    assert snapshot["benchmark"] == "test-bench"
    assert snapshot["config_id"] == output["config_id"]
    assert (snapshot["code_git_hash"], snapshot["code_git_dirty"]) == ("abc123", False)
    assert snapshot["raw_config"]["language"] == "sw"
    assert snapshot["resolved_config"]["items_sha256"] == (
        sha256((tmp_path / "items.jsonl").read_bytes()).hexdigest()
    )
    assert snapshot["model_configs"] == {}

    log = json.loads((config.run_dir / "log.json").read_text(encoding="utf-8"))
    assert set(log) == {"model-a", "model-b"}
    assert log["model-a"]["responses_git_hash"] == "abc123"
    assert log["model-a"]["responses_git_dirty"] is False


def test_rerun_reuses_every_result(config: RunConfig) -> None:
    execute_run(config, evaluator=_echo)

    def fail(*_args) -> dict:
        pytest.fail("a rerun must reuse every result")

    output = execute_run(config, evaluator=fail)

    assert (output["executed_count"], output["skipped_count"]) == (0, 4)


def test_force_reruns_everything(config: RunConfig) -> None:
    execute_run(config, evaluator=_echo)

    output = execute_run(replace(config, force=True), evaluator=_echo)

    assert (output["executed_count"], output["skipped_count"]) == (4, 0)


def test_adding_a_model_keeps_the_other_models_results(
    benchmark: Benchmark, write_run: Callable[..., Path]
) -> None:
    only_a = load_run_config(write_run(models=["model-a"]), benchmark)
    execute_run(only_a, evaluator=_echo)
    extended = load_run_config(write_run(models=["model-a", "model-b"]), benchmark)
    called: list[str] = []

    def evaluator(_config, model, item, _prompt) -> dict:
        called.append(model)
        return {"output": "x"}

    output = execute_run(extended, evaluator=evaluator)

    assert called == ["model-b", "model-b"]
    assert output["skipped_count"] == 2


def test_changed_prompts_are_recomputed_and_stale_rows_dropped(
    benchmark: Benchmark, write_run: Callable[..., Path], read_rows: Rows
) -> None:
    first = execute_run(load_run_config(write_run(), benchmark), evaluator=_echo)
    changed = load_run_config(write_run(prompt_name="clusters"), benchmark)

    output = execute_run(changed, evaluator=_echo)

    assert output["config_id"] != first["config_id"]
    assert output["executed_count"] == 4
    rows = read_rows(output["results_paths"].values())
    assert len(rows) == 4
    assert {row["prompt_name"] for row in rows} == {"clusters"}


def test_explicit_default_max_tokens_reuses_results(
    benchmark: Benchmark, write_run: Callable[..., Path]
) -> None:
    execute_run(load_run_config(write_run(), benchmark), evaluator=_echo)
    explicit = load_run_config(write_run(max_tokens=10_000), benchmark)

    assert execute_run(explicit, evaluator=_echo)["executed_count"] == 0
    changed = load_run_config(write_run(max_tokens=500), benchmark)
    assert execute_run(changed, evaluator=_echo)["executed_count"] == 4


def test_system_prompt_feeds_the_call_ids(config: RunConfig, read_rows: Rows) -> None:
    with_system = replace(
        config, benchmark=replace(config.benchmark, system_prompt="Be brief.")
    )

    output = execute_run(with_system, evaluator=_echo)

    row = read_rows([output["results_paths"]["model-a"]])[0]
    assert row["model_call_id"] == model_call_id(
        benchmark_item_id="item-1",
        model="model-a",
        model_config={},
        prompt=row["prompt"],
        system_prompt="Be brief.",
        max_tokens=10_000,
    )


@pytest.mark.parametrize(
    "provider_error",
    [
        OpenAIAPIConnectionError(request=httpx.Request("POST", "https://example.test")),
        AnthropicAPIConnectionError(
            request=httpx.Request("POST", "https://example.test")
        ),
        GoogleServerError(
            503, {"error": {"message": "unavailable", "status": "UNAVAILABLE"}}
        ),
        EndpointConnectionError(endpoint_url="https://example.test"),
        MistralConnectionException("connection failed"),
    ],
    ids=["openai", "anthropic", "google", "bedrock", "mistral"],
)
def test_provider_errors_become_error_rows(
    benchmark: Benchmark,
    write_run: Callable[..., Path],
    read_rows: Rows,
    provider_error: Exception,
) -> None:
    config = load_run_config(write_run(models=["model-a"]), benchmark)

    def evaluator(*_args) -> dict:
        raise provider_error

    output = execute_run(config, evaluator=evaluator)

    rows = read_rows(output["results_paths"].values())
    assert output["error_count"] == 2
    assert rows[0]["call_status"] == "error"
    assert rows[0]["success"] is False
    assert rows[0]["output"] == ""
    assert rows[0]["error_type"] == type(provider_error).__name__
    assert rows[0]["error_message"] == str(provider_error)


def test_empty_responses_become_error_rows(config: RunConfig, read_rows: Rows) -> None:
    output = execute_run(config, evaluator=lambda *_args: {"output": ""})

    rows = read_rows(output["results_paths"].values())
    assert {row["error_type"] for row in rows} == {"EmptyLLMResponse"}
    assert output["error_count"] == 4


@pytest.mark.parametrize(
    "error", [RuntimeError("bug"), KeyError("bug"), ValueError("bug")]
)
def test_other_exceptions_stop_the_run(config: RunConfig, error: Exception) -> None:
    def evaluator(*_args) -> dict:
        raise error

    with pytest.raises(type(error)):
        execute_run(config, evaluator=evaluator)


def test_retry_errors_reruns_only_failed_calls(
    config: RunConfig, read_rows: Rows
) -> None:
    def fail_item_2(_config, model, item, _prompt) -> dict:
        if item["benchmark_item_id"] == "item-2":
            raise MistralConnectionException("down")
        return {"output": "ok"}

    first = execute_run(config, evaluator=fail_item_2)
    assert first["error_count"] == 2
    retried: list[tuple[str, str]] = []

    def evaluator(_config, model, item, _prompt) -> dict:
        retried.append((model, item["benchmark_item_id"]))
        return {"output": "fixed"}

    plan = plan_run(replace(config, retry_errors=True), model_configs={})
    output = execute_run(replace(config, retry_errors=True), evaluator=evaluator)

    assert [(m.to_run, m.retries) for m in plan.models] == [(1, 1), (1, 1)]
    assert retried == [("model-a", "item-2"), ("model-b", "item-2")]
    assert (output["executed_count"], output["error_count"]) == (2, 0)
    log = json.loads((config.run_dir / "log.json").read_text(encoding="utf-8"))
    assert "responses_retry_timestamp_utc" in log["model-a"]
    assert all(row["success"] for row in read_rows(output["results_paths"].values()))


def test_retry_errors_requires_existing_results(config: RunConfig) -> None:
    with pytest.raises(FileNotFoundError, match="No cached results found"):
        execute_run(replace(config, retry_errors=True), evaluator=_echo)


def test_retry_errors_rejects_missing_calls_before_any_call(
    benchmark: Benchmark, write_run: Callable[..., Path]
) -> None:
    execute_run(load_run_config(write_run(), benchmark), evaluator=_echo)
    changed = load_run_config(
        write_run(prompt_name="clusters", retry_errors=True), benchmark
    )

    def fail(*_args) -> dict:
        pytest.fail("no call may start")

    with pytest.raises(ConfigError, match="model-a \\(2\\), model-b \\(2\\)"):
        execute_run(changed, evaluator=fail)


def test_retry_errors_rejects_invalid_cached_statuses(config: RunConfig) -> None:
    output = execute_run(config, evaluator=_echo)
    path = output["results_paths"]["model-a"]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    save_results([{**rows[0], "call_status": "pending"}, rows[1]], path)

    with pytest.raises(ConfigError, match="invalid call_status values"):
        execute_run(replace(config, retry_errors=True), evaluator=_echo)


@pytest.mark.parametrize("model_workers", [1, 2, 3])
def test_model_workers_caps_models_running_at_once(
    benchmark: Benchmark, write_run: Callable[..., Path], model_workers: int
) -> None:
    """Models of one provider group run in parallel too; only the cap applies."""

    models = ["model-a", "model-b", "model-c"]
    config = load_run_config(
        write_run(
            models=models,
            groups={"shared": models},
            model_workers=model_workers,
        ),
        benchmark,
    )
    lock = Lock()
    active = maximum = 0

    def evaluator(*_args) -> dict:
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        sleep(0.02)
        with lock:
            active -= 1
        return {"output": "x"}

    assert execute_run(config, evaluator=evaluator)["executed_count"] == 6
    assert maximum == model_workers


def test_models_of_one_provider_run_in_parallel_with_items_in_order(
    benchmark: Benchmark, write_run: Callable[..., Path]
) -> None:
    config = load_run_config(
        write_run(groups={"shared": ["model-a", "model-b"]}, model_workers=2),
        benchmark,
    )
    first_calls = Barrier(2, timeout=2)
    calls: dict[str, list[str]] = {"model-a": [], "model-b": []}

    def evaluator(_config, model, item, _prompt) -> dict:
        calls[model].append(item["benchmark_item_id"])
        if item["benchmark_item_id"] == "item-1":
            first_calls.wait()
        return {"output": "x"}

    execute_run(config, evaluator=evaluator)

    assert calls == {"model-a": ["item-1", "item-2"], "model-b": ["item-1", "item-2"]}


def test_a_failing_model_stops_further_models_from_starting(
    benchmark: Benchmark, write_run: Callable[..., Path]
) -> None:
    models = ["model-a", "model-b", "model-c", "model-d"]
    config = load_run_config(write_run(models=models, model_workers=2), benchmark)
    started: list[str] = []
    lock = Lock()

    def evaluator(_config, model, _item, _prompt) -> dict:
        with lock:
            if model not in started:
                started.append(model)
        if model == "model-a":
            raise RuntimeError("bug in model-a's prompt handling")
        sleep(0.02)
        return {"output": "x"}

    with pytest.raises(RuntimeError, match="model-a"):
        execute_run(config, evaluator=evaluator)

    assert started == ["model-a", "model-b"]
    finished = config.run_dir / "responses" / "responses_model-b.jsonl"
    assert len(finished.read_text(encoding="utf-8").splitlines()) == 2


def test_checkpoints_fire_after_each_chunk_in_run_order(
    benchmark: Benchmark, write_run: Callable[..., Path]
) -> None:
    config = load_run_config(
        write_run(
            models=["model-a"],
            items=[
                {"benchmark_item_id": f"item-{n}", "content": "x"} for n in (1, 2, 3)
            ],
            chunk_size=2,
        ),
        benchmark,
    )
    events: list[str] = []

    def evaluator(_config, _model, item, _prompt) -> dict:
        events.append(f"evaluate:{item['benchmark_item_id']}")
        return {"output": "x"}

    execute_run(
        config,
        evaluator=evaluator,
        on_artifact_write=lambda path: events.append(f"write:{path.name}"),
    )

    assert events == [
        "write:config.json",
        "evaluate:item-1",
        "evaluate:item-2",
        "write:responses_model-a.jsonl",
        "evaluate:item-3",
        "write:responses_model-a.jsonl",
        "write:log.json",
    ]


def test_a_null_chunk_size_saves_once_per_model(
    benchmark: Benchmark, write_run: Callable[..., Path]
) -> None:
    config = load_run_config(write_run(models=["model-a"], chunk_size=None), benchmark)
    writes: list[str] = []

    execute_run(
        config, evaluator=_echo, on_artifact_write=lambda p: writes.append(p.name)
    )

    assert writes.count("responses_model-a.jsonl") == 1


def test_custom_item_loader_and_prompt_builder(
    tmp_path: Path, write_run: Callable[..., Path], read_rows: Rows
) -> None:
    def load_items(path: Path) -> tuple[list[dict[str, Any]], str]:
        return [{"benchmark_item_id": "q-1", "question": "1+1?"}], "loader-hash"

    custom = Benchmark(
        name="custom",
        build_prompt=lambda item, _params: f"Q: {item['question']}",
        load_items=load_items,
    )
    config = load_run_config(write_run(models=["model-a"]), custom)

    output = execute_run(config, evaluator=_echo)

    [row] = read_rows(output["results_paths"].values())
    assert (row["benchmark_item_id"], row["prompt"]) == ("q-1", "Q: 1+1?")
    snapshot = json.loads(output["config_snapshot_path"].read_text(encoding="utf-8"))
    assert snapshot["resolved_config"]["items_sha256"] == "loader-hash"


def test_multimodal_request_is_hashed_stored_and_reused(
    write_run: Callable[..., Path], read_rows: Rows
) -> None:
    image_bytes = _image_bytes()

    def load_items(_path: Path) -> tuple[list[dict[str, Any]], str]:
        return [
            {
                "benchmark_item_id": "visual-1",
                "question": "Which shape is shown?",
                "image_bytes": image_bytes,
            }
        ], "visual-items-hash"

    benchmark = Benchmark(
        name="visual",
        load_items=load_items,
        build_request=lambda item, _params: ModelRequest(
            prompt=item["question"],
            images=(
                ImageInput(
                    data=item["image_bytes"],
                    detail="high",
                    source_name="shape.jpg",
                ),
            ),
        ),
    )
    config = load_run_config(write_run(models=["model-a"]), benchmark)
    seen_prompts: list[str] = []

    def evaluator(_config, _model, _item, prompt: str) -> dict:
        seen_prompts.append(prompt)
        return {"output": "triangle"}

    first = execute_run(config, evaluator=evaluator)
    [row] = read_rows(first["results_paths"].values())
    [image] = row["images"]

    assert row["prompt"] == "Which shape is shown?"
    assert image["source_name"] == "shape.jpg"
    assert image["detail"] == "high"
    assert image["source_sha256"] != image["payload_sha256"]
    assert "image_bytes" not in row
    assert seen_prompts == ["Which shape is shown?"]
    assert row["model_call_id"] == model_call_id(
        benchmark_item_id="visual-1",
        model="model-a",
        model_config={},
        prompt=row["prompt"],
        system_prompt=None,
        max_tokens=10_000,
        images=[
            {
                "sha256": image["payload_sha256"],
                "detail": image["detail"],
                "width": image["payload_width"],
                "height": image["payload_height"],
            }
        ],
    )

    second = execute_run(
        config,
        evaluator=lambda *_args: pytest.fail("existing result should be reused"),
    )
    assert (second["executed_count"], second["skipped_count"]) == (0, 1)


def test_custom_loader_output_is_validated(write_run: Callable[..., Path]) -> None:
    duplicate = Benchmark(
        name="dup",
        build_prompt=lambda *_: "p",
        load_items=lambda _p: ([{"benchmark_item_id": "a"}] * 2, "h"),
    )
    config = load_run_config(write_run(), duplicate)

    with pytest.raises(ConfigError, match="duplicate benchmark_item_id"):
        execute_run(config, evaluator=_echo)


def _image_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (20, 10), color="green").save(buffer, format="JPEG")
    return buffer.getvalue()


def test_prompts_must_be_strings(write_run: Callable[..., Path]) -> None:
    broken = Benchmark(name="broken", build_prompt=lambda *_: ["not", "a", "string"])
    config = load_run_config(write_run(), broken)

    with pytest.raises(ConfigError, match="returned list"):
        execute_run(config, evaluator=_echo)


def test_identity_fields_may_not_clash_with_result_columns(
    write_run: Callable[..., Path],
) -> None:
    clashing = Benchmark(
        name="clash", build_prompt=lambda *_: "p", identity=lambda _: {"model": "x"}
    )
    config = load_run_config(write_run(), clashing)

    with pytest.raises(ConfigError, match="clash with result columns: model"):
        execute_run(config, evaluator=_echo)


def test_plan_reports_counts_without_calling_or_writing(
    config: RunConfig, read_rows: Rows
) -> None:
    fresh = plan_run(config, model_configs={})
    assert (fresh.planned, fresh.skipped, fresh.to_run) == (4, 0, 4)
    assert not config.run_dir.exists()

    output = execute_run(replace(config, models=["model-a"]), evaluator=_echo)
    rows = read_rows([output["results_paths"]["model-a"]])
    stale = {**rows[0], "result_id": "no-longer-planned"}
    save_results([*rows, stale], output["results_paths"]["model-a"])
    files_before = sorted(p.name for p in config.run_dir.rglob("*"))

    plan = plan_run(config, model_configs={})

    assert [
        (m.model, m.planned, m.skipped, m.to_run, m.stale) for m in plan.models
    ] == [
        ("model-a", 2, 2, 0, 1),
        ("model-b", 2, 0, 2, 0),
    ]
    assert plan.config_id == output["config_id"]
    assert sorted(p.name for p in config.run_dir.rglob("*")) == files_before
    assert execute_run(config, evaluator=_echo)["executed_count"] == plan.to_run


def test_plan_uses_fdllm_model_configs_by_default(config: RunConfig) -> None:
    with (
        patch("fab_benchmark_runner.runner.prepare_fdllm_for_run") as prepare,
        patch(
            "fab_benchmark_runner.runner.snapshot_model_configs",
            return_value={"model-a": {"Api_Model_Name": "a"}, "model-b": {}},
        ),
    ):
        plan = plan_run(config)

    prepare.assert_called_once_with(config)
    assert plan.to_run == 4


def test_default_evaluator_uses_fdllm_and_its_model_configs(
    config: RunConfig, read_rows: Rows
) -> None:
    model_configs = {
        "model-a": {"Api_Model_Name": "a", "Client_Args": {"api_key": "secret"}},
        "model-b": {},
    }
    with (
        patch("fab_benchmark_runner.runner.prepare_fdllm_for_run") as prepare,
        patch(
            "fab_benchmark_runner.runner.snapshot_model_configs",
            return_value=model_configs,
        ),
        patch(
            "fab_benchmark_runner.runner.evaluate_item_with_fdllm",
            side_effect=_echo,
        ),
    ):
        output = execute_run(config)

    prepare.assert_called_once_with(config)
    row = read_rows([output["results_paths"]["model-a"]])[0]
    assert row["model_call_id"] == model_call_id(
        benchmark_item_id="item-1",
        model="model-a",
        model_config={"Api_Model_Name": "a", "Client_Args": {}},
        prompt=row["prompt"],
        system_prompt=None,
        max_tokens=10_000,
    )
    snapshot = json.loads(output["config_snapshot_path"].read_text(encoding="utf-8"))
    assert snapshot["model_configs"]["model-a"] == {
        "Api_Model_Name": "a",
        "Client_Args": {},
    }
