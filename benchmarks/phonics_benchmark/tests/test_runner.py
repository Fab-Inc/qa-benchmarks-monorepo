"""Tests for the config-driven phonics benchmark runner."""

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from threading import Barrier, Lock
from time import sleep
from unittest.mock import patch

import httpx
import pytest
from anthropic import APIConnectionError as AnthropicAPIConnectionError
from botocore.exceptions import EndpointConnectionError
from google.genai.errors import ServerError as GoogleServerError
from mistralai.exceptions import MistralConnectionException
from openai import APIConnectionError as OpenAIAPIConnectionError
from phonics_benchmark.config import (
    BenchmarkRunConfig,
    ConfigError,
    config_identity_sha256,
    load_run_config,
)
from phonics_benchmark.items import load_benchmark_items
from phonics_benchmark.llm import DEFAULT_MAX_TOKENS
from phonics_benchmark.prompts import build_decodability_prompt
from phonics_benchmark.results import save_results
from phonics_benchmark.runner import (
    DEFAULT_CUSTOM_MODELS_PATH,
    download_default_custom_models,
    evaluate_item_with_fdllm,
    execute_run,
    prepare_fdllm_for_run,
    run_benchmark,
    run_from_config_file,
    snapshot_model_configs,
)

_CHUNK_SIZE_UNSET = object()
CONTRACT_FIXTURE = (
    Path(__file__).parent / "fixtures" / "completed_benchmark_items.jsonl"
)


def test_load_benchmark_items_reads_exact_jsonl_snapshot(tmp_path: Path) -> None:
    """The loader normalizes call inputs and hashes the bytes it parsed."""

    items_path = tmp_path / "items.jsonl"
    artifact = (
        b'{"benchmark_item_id":"item-1","content":"  Mama anaona kuku.  ",'
        b'"student_knowledge":{"graphemes":["a","m"],"whole_words":["mama"]},'
        b'"source_id":"source-1","expected_answer":["kuku"]}\n'
        b'{"benchmark_item_id":"item-2","content":"Tai ni ndege.",'
        b'"student_knowledge":{"graphemes":["a","i"],"whole_words":[]}}\n'
    )
    items_path.write_bytes(artifact)

    items, items_sha256 = load_benchmark_items(items_path)

    assert items_sha256 == sha256(artifact).hexdigest()
    assert len(items) == 2
    assert items[0]["benchmark_item_id"] == "item-1"
    assert items[0]["content"] == "  Mama anaona kuku.  "
    assert items[0]["student_knowledge"] == {
        "graphemes": ["a", "m"],
        "whole_words": ["mama"],
    }
    assert set(items[0]) == {
        "benchmark_item_id",
        "content",
        "student_knowledge",
    }


def test_contract_fixture_exposes_completed_execution_envelope() -> None:
    items, _items_sha256 = load_benchmark_items(CONTRACT_FIXTURE)

    assert [item["benchmark_item_id"] for item in items] == [
        "bi-SW-KE-S2-tusome-ke-stage-01",
        "bi-SW-KE-S4-tusome-ke-stage-02",
    ]
    assert all(
        set(item) == {"benchmark_item_id", "content", "student_knowledge"}
        for item in items
    )


def test_load_benchmark_items_rejects_padded_ids(
    tmp_path: Path,
) -> None:
    """Identifiers are validated without silently changing them."""

    items_path = _write_items_jsonl(
        tmp_path,
        [{"id": " item-1 ", "text": "Text."}],
    )

    with pytest.raises(ConfigError, match="surrounding whitespace"):
        load_benchmark_items(items_path)


@pytest.mark.parametrize(
    ("contents", "error"),
    [
        (b"", "No benchmark items"),
        (b"\n", "blank JSONL"),
        (
            b'{"benchmark_item_id":"item-1","content":"Text",'
            b'"student_knowledge":{}}\n\n',
            "blank JSONL",
        ),
        (b"not-json\n", "invalid JSON"),
        (b"[]\n", "must be an object"),
        (
            b'{"benchmark_item_id":"item-1","content":" ","student_knowledge":{}}\n',
            "content",
        ),
        (
            b'{"benchmark_item_id":"item-1","content":"Text","student_knowledge":[]}\n',
            "student_knowledge",
        ),
        (b"\xff", "UTF-8"),
    ],
)
def test_load_benchmark_items_rejects_invalid_artifacts(
    tmp_path: Path,
    contents: bytes,
    error: str,
) -> None:
    items_path = tmp_path / "invalid.jsonl"
    items_path.write_bytes(contents)

    with pytest.raises(ConfigError, match=error):
        load_benchmark_items(items_path)


def test_load_benchmark_items_rejects_duplicate_ids(tmp_path: Path) -> None:
    items_path = _write_items_jsonl(
        tmp_path,
        [
            {"id": "item-1", "text": "First."},
            {"id": "item-1", "text": "Second."},
        ],
    )

    with pytest.raises(ConfigError, match="duplicate benchmark_item_id.*item-1"):
        load_benchmark_items(items_path)


def test_execute_run_owns_identity_and_default_model_setup(tmp_path: Path) -> None:
    """The orchestration boundary prepares FDLLM and run provenance."""

    config = load_run_config(_write_config(tmp_path))
    expected_output = {"run_id": config.run_id}
    model_configs = {"model-a": {"Api_Interface": "OpenAI"}}

    with (
        patch("phonics_benchmark.runner.prepare_fdllm_for_run") as prepare,
        patch(
            "phonics_benchmark.runner.snapshot_model_configs",
            return_value=model_configs,
        ) as snapshot,
        patch(
            "phonics_benchmark.runner.run_benchmark",
            return_value=expected_output,
        ) as run,
    ):
        output = execute_run(config)

    assert output == expected_output
    prepare.assert_called_once_with(config)
    snapshot.assert_called_once_with(config.models)
    run.assert_called_once_with(
        config,
        config_id=config_identity_sha256(config),
        evaluator=None,
        model_configs=model_configs,
    )


def test_execute_run_skips_fdllm_setup_for_supplied_evaluator(
    tmp_path: Path,
) -> None:
    """Injected evaluators do not require the FDLLM model registry."""

    config = load_run_config(_write_config(tmp_path))
    evaluator = lambda *_args: {"output": "first"}  # noqa: E731

    with (
        patch("phonics_benchmark.runner.prepare_fdllm_for_run") as prepare,
        patch("phonics_benchmark.runner.snapshot_model_configs") as snapshot,
        patch("phonics_benchmark.runner.run_benchmark", return_value={}) as run,
    ):
        execute_run(config, evaluator=evaluator)

    prepare.assert_not_called()
    snapshot.assert_not_called()
    assert run.call_args.kwargs["evaluator"] is evaluator
    assert run.call_args.kwargs["model_configs"] is None


def test_run_from_config_file_uses_completed_jsonl(tmp_path: Path) -> None:
    """The runner loads every completed JSONL item in artifact order."""

    config_path = _write_config(
        tmp_path,
        models=["model-a"],
        items=[
            {"id": "item-1", "text": "First text."},
            {"id": "item-2", "text": "Second text."},
        ],
    )
    seen_items = []

    def evaluator(
        _config: BenchmarkRunConfig,
        _model: str,
        item: dict,
        _prompt: str,
    ) -> dict:
        seen_items.append(item["benchmark_item_id"])
        return {"output": "[]"}

    output = run_from_config_file(config_path, evaluator=evaluator)

    assert output["total_expected_results"] == 2
    assert seen_items == ["item-1", "item-2"]


def test_run_from_config_file_writes_linked_results(tmp_path: Path) -> None:
    """The runner writes one linked result per model/item pair."""

    config_path = _write_config(tmp_path)
    calls: list[tuple[str, str, str, str]] = []

    def fake_evaluator(
        config: BenchmarkRunConfig,
        model_name: str,
        item: dict,
        prompt: str,
    ) -> dict:
        calls.append(
            (config.language, config.task, model_name, item["benchmark_item_id"])
        )
        return {
            "output": f"{model_name}:{item['benchmark_item_id']}",
            "latency": 0.1,
            "total_tokens": 12,
            "seen_prompt": prompt,
        }

    with patch("phonics_benchmark.runner.git_hash", return_value="abc123"):
        output = run_from_config_file(
            config_path,
            evaluator=fake_evaluator,
        )

    rows = _read_all_results(output["results_paths"])
    assert output["total_expected_results"] == 4
    assert output["executed_count"] == 4
    assert output["skipped_count"] == 0
    assert output["error_count"] == 0
    assert output["results_dir"] == output["run_dir"] / "responses"
    assert {path.name for path in output["results_paths"].values()} == {
        "responses_model-a.jsonl",
        "responses_model-b.jsonl",
    }
    assert len(rows) == 4
    assert calls == [
        ("sw", "decodability", "model-a", "item-1"),
        ("sw", "decodability", "model-a", "item-2"),
        ("sw", "decodability", "model-b", "item-1"),
        ("sw", "decodability", "model-b", "item-2"),
    ]

    result_ids = {row["result_id"] for row in rows}
    model_call_ids = {row["model_call_id"] for row in rows}
    assert len(result_ids) == 4
    assert len(model_call_ids) == 4
    assert {row["config_id"] for row in rows} == {output["config_id"]}
    assert {row["config_snapshot_path"] for row in rows} == {
        str(output["config_snapshot_path"])
    }
    assert rows[0]["run_id"] == "experiment-a"
    assert rows[0]["language"] == "sw"
    assert rows[0]["success"] is True
    assert rows[0]["call_status"] == "success"
    assert "responses_timestamp-utc" not in rows[0]
    assert "responses_git-hash" not in rows[0]
    assert "A cat sat." in rows[0]["prompt"]
    assert '- Known graphemes: "a", "t"' in rows[0]["prompt"]
    assert rows[0]["seen_prompt"] == rows[0]["prompt"]
    assert "item" not in rows[0]
    assert "item_id" not in rows[0]
    assert rows[0]["benchmark_item_id"] == "item-1"
    assert "started_at" not in rows[0]
    assert "completed_at" not in rows[0]
    assert "elapsed_seconds" not in rows[0]

    snapshot = json.loads(output["config_snapshot_path"].read_text(encoding="utf-8"))
    assert snapshot["config_id"] == output["config_id"]
    assert snapshot["code_git_hash"] == "abc123"
    assert snapshot["raw_config"]["run_id"] == "experiment-a"
    assert snapshot["resolved_config"]["items_path"] == str(
        (tmp_path / "items.jsonl").resolve()
    )
    assert (
        snapshot["resolved_config"]["items_sha256"]
        == sha256((tmp_path / "items.jsonl").read_bytes()).hexdigest()
    )
    assert isinstance(snapshot["model_configs"], dict)

    assert "manifest_path" not in output
    assert not (output["run_dir"] / "manifest.json").exists()

    log = json.loads((output["run_dir"] / "log.json").read_text(encoding="utf-8"))
    assert set(log) == {"model-a", "model-b"}
    assert "responses_timestamp-utc" in log["model-a"]
    assert log["model-a"]["responses_git-hash"] == "abc123"


def test_run_from_config_file_parallelizes_models_not_items(tmp_path: Path) -> None:
    """Two workers overlap models while preserving item order within each model."""

    config_path = _write_config(
        tmp_path,
        models=["model-a", "model-b"],
        items=[
            {"id": "item-1", "text": "A cat sat."},
            {"id": "item-2", "text": "The dog ran."},
        ],
        model_workers=2,
    )
    first_call_barrier = Barrier(2, timeout=2)
    calls_by_model = {"model-a": [], "model-b": []}

    def evaluator(
        _config: BenchmarkRunConfig,
        model_name: str,
        item: dict,
        _prompt: str,
    ) -> dict:
        calls_by_model[model_name].append(item["benchmark_item_id"])
        if item["benchmark_item_id"] == "item-1":
            first_call_barrier.wait()
        return {"output": f"{model_name}:{item['benchmark_item_id']}"}

    output = run_from_config_file(config_path, evaluator=evaluator)

    assert output["executed_count"] == 4
    assert output["error_count"] == 0
    assert calls_by_model == {
        "model-a": ["item-1", "item-2"],
        "model-b": ["item-1", "item-2"],
    }


def test_models_in_one_provider_group_do_not_overlap(tmp_path: Path) -> None:
    custom_models_path = tmp_path / "custom_models.yaml"
    custom_models_path.write_text(
        '{"shared-provider":{"models":{"model-a":{},"model-b":{}}}}',
        encoding="utf-8",
    )
    config_path = _write_config(
        tmp_path,
        models=["model-a", "model-b"],
        items=[{"id": "item-1", "text": "Mama."}],
        custom_models_path=custom_models_path,
        model_workers=2,
    )
    lock = Lock()
    active_calls = 0
    maximum_active_calls = 0

    def evaluator(*_args) -> dict:
        nonlocal active_calls, maximum_active_calls
        with lock:
            active_calls += 1
            maximum_active_calls = max(maximum_active_calls, active_calls)
        sleep(0.01)
        with lock:
            active_calls -= 1
        return {"output": "[]"}

    output = run_from_config_file(config_path, evaluator=evaluator)

    assert output["executed_count"] == 2
    assert maximum_active_calls == 1


def test_model_workers_caps_active_provider_groups(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        models=["model-a", "model-b", "model-c"],
        items=[{"id": "item-1", "text": "Mama."}],
        model_workers=2,
    )
    lock = Lock()
    active_calls = 0
    maximum_active_calls = 0

    def evaluator(*_args) -> dict:
        nonlocal active_calls, maximum_active_calls
        with lock:
            active_calls += 1
            maximum_active_calls = max(maximum_active_calls, active_calls)
        sleep(0.02)
        with lock:
            active_calls -= 1
        return {"output": "[]"}

    output = run_from_config_file(config_path, evaluator=evaluator)

    assert output["executed_count"] == 3
    assert maximum_active_calls == 2


@pytest.mark.parametrize(
    ("registry", "error"),
    [
        ({"provider-a": {"models": {"other-model": {}}}}, "missing"),
        (
            {
                "provider-a": {"models": {"model-a": {}}},
                "provider-b": {"models": {"model-a": {}}},
            },
            "multiple",
        ),
    ],
)
def test_selected_models_require_one_provider_group(
    tmp_path: Path,
    registry: dict,
    error: str,
) -> None:
    custom_models_path = tmp_path / "custom_models.yaml"
    custom_models_path.write_text(json.dumps(registry), encoding="utf-8")
    config_path = _write_config(
        tmp_path,
        models=["model-a"],
        items=[{"id": "item-1", "text": "Mama."}],
        custom_models_path=custom_models_path,
    )

    with pytest.raises(ConfigError, match=error):
        run_from_config_file(
            config_path,
            evaluator=lambda *_args: {"output": "unexpected"},
        )


def test_prompts_are_rendered_once_per_item_before_workers(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        models=["model-a", "model-b"],
        model_workers=2,
    )

    with patch(
        "phonics_benchmark.runner.build_decodability_prompt",
        wraps=build_decodability_prompt,
    ) as build_prompt:
        output = run_from_config_file(
            config_path,
            evaluator=lambda *_args: {"output": "[]"},
        )

    assert output["executed_count"] == 4
    assert build_prompt.call_count == 2


def test_unknown_language_fails_before_model_execution(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        models=["model-a"],
        items=[{"id": "item-1", "text": "Mama."}],
    )
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    raw_config["language"] = "xx"
    config_path.write_text(json.dumps(raw_config), encoding="utf-8")
    calls = []

    with pytest.raises(ConfigError, match="Unknown language registry key 'xx'"):
        run_from_config_file(
            config_path,
            evaluator=lambda *_args: calls.append("called") or {"output": "[]"},
        )

    assert calls == []


def test_run_from_config_file_skips_existing_successes(tmp_path: Path) -> None:
    """A repeated run does not call the evaluator again for successful rows."""

    config_path = _write_config(
        tmp_path,
        models=["model-a"],
        items=[{"id": "item-1", "text": "A cat sat."}],
    )

    output = run_from_config_file(
        config_path,
        evaluator=lambda _config, _model, _item, _prompt: {"output": "first"},
    )

    def fail_if_called(
        _config: BenchmarkRunConfig,
        _model_name: str,
        _item: dict,
        _prompt: str,
    ) -> dict:
        raise AssertionError("successful rows should be skipped")

    second_output = run_from_config_file(
        config_path,
        evaluator=fail_if_called,
    )

    rows = _read_jsonl(output["results_paths"]["model-a"])
    assert second_output["executed_count"] == 0
    assert second_output["skipped_count"] == 1
    assert rows[0]["output"] == "first"


def test_run_from_config_file_adding_model_keeps_existing_model_cache(
    tmp_path: Path,
) -> None:
    """Adding a model does not invalidate already-cached calls for other models."""

    config_path = _write_config(
        tmp_path,
        models=["model-a"],
        items=[{"id": "item-1", "text": "A cat sat."}],
    )

    first_output = run_from_config_file(
        config_path,
        evaluator=lambda _config, _model, _item, _prompt: {"output": "first"},
    )
    first_rows = _read_jsonl(first_output["results_paths"]["model-a"])

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["models"] = ["model-a", "model-b"]
    config_path.write_text(json.dumps(config), encoding="utf-8")
    called_models = []

    def evaluator(
        _config: BenchmarkRunConfig,
        model_name: str,
        _item: dict,
        _prompt: str,
    ) -> dict:
        called_models.append(model_name)
        return {"output": f"{model_name}:second"}

    second_output = run_from_config_file(
        config_path,
        evaluator=evaluator,
    )

    assert first_output["config_id"] == second_output["config_id"]
    assert second_output["executed_count"] == 1
    assert second_output["skipped_count"] == 1
    assert called_models == ["model-b"]
    assert _read_jsonl(second_output["results_paths"]["model-a"]) == first_rows
    assert _read_jsonl(second_output["results_paths"]["model-b"])[0]["output"] == (
        "model-b:second"
    )


def test_run_from_config_file_provenance_change_keeps_model_cache(
    tmp_path: Path,
) -> None:
    """Provenance and future config fields do not force model calls to rerun."""

    config_path = _write_config(
        tmp_path,
        models=["model-a"],
        items=[{"id": "item-1", "text": "A cat sat."}],
    )

    first_output = run_from_config_file(
        config_path,
        evaluator=lambda _config, _model, _item, _prompt: {"output": "first"},
    )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["metadata"] = {"owner": "benchmarks"}
    config["future_field"] = "not-a-model-call-input"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    def fail_if_called(
        _config: BenchmarkRunConfig,
        _model_name: str,
        _item: dict,
        _prompt: str,
    ) -> dict:
        raise AssertionError("provenance-only changes should use cached rows")

    second_output = run_from_config_file(
        config_path,
        evaluator=fail_if_called,
    )

    rows = _read_jsonl(second_output["results_paths"]["model-a"])
    assert first_output["config_id"] == second_output["config_id"]
    assert second_output["executed_count"] == 0
    assert second_output["skipped_count"] == 1
    assert rows[0]["output"] == "first"


def test_benchmark_item_id_is_call_identity_and_requires_recomputation(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path,
        models=["model-a"],
        items=[{"id": "old-item-id", "text": "Mama."}],
    )
    first_output = run_from_config_file(
        config_path,
        evaluator=lambda *_args: {"output": "first"},
    )
    first_row = _read_jsonl(first_output["results_paths"]["model-a"])[0]

    _write_items_jsonl(
        tmp_path,
        [{"id": "new-item-id", "text": "Mama."}],
    )
    second_output = run_from_config_file(
        config_path,
        evaluator=lambda *_args: {"output": "second"},
    )
    second_row = _read_jsonl(second_output["results_paths"]["model-a"])[0]

    assert second_output["config_id"] == first_output["config_id"]
    assert second_output["executed_count"] == 1
    assert second_output["skipped_count"] == 0
    assert second_row["model_call_id"] != first_row["model_call_id"]
    assert second_row["result_id"] != first_row["result_id"]
    assert second_row["benchmark_item_id"] == "new-item-id"
    assert second_row["output"] == "second"


@pytest.mark.parametrize(
    "provider_error",
    [
        OpenAIAPIConnectionError(request=httpx.Request("POST", "https://example.test")),
        AnthropicAPIConnectionError(
            request=httpx.Request("POST", "https://example.test")
        ),
        GoogleServerError(
            503,
            {"error": {"message": "unavailable", "status": "UNAVAILABLE"}},
        ),
        EndpointConnectionError(endpoint_url="https://example.test"),
        MistralConnectionException("connection failed"),
    ],
    ids=["openai", "anthropic", "google", "bedrock", "mistral"],
)
def test_run_from_config_file_saves_provider_errors(
    tmp_path: Path,
    provider_error: Exception,
) -> None:
    """Known FDLLM provider failures are saved as resumable error rows."""

    config_path = _write_config(
        tmp_path,
        models=["model-a"],
        items=[{"id": "item-1", "text": "A cat sat."}],
    )

    def evaluator(*_args) -> dict:
        raise provider_error

    output = run_from_config_file(config_path, evaluator=evaluator)

    rows = _read_jsonl(output["results_paths"]["model-a"])
    assert output["executed_count"] == 1
    assert output["error_count"] == 1
    assert rows[0]["call_status"] == "error"
    assert rows[0]["error_type"] == type(provider_error).__name__
    assert rows[0]["error_message"] == str(provider_error)


def test_run_from_config_file_saves_empty_responses_as_errors(tmp_path: Path) -> None:
    """An empty model response is saved without treating the call as successful."""

    config_path = _write_config(
        tmp_path,
        models=["model-a"],
        items=[{"id": "item-1", "text": "A cat sat."}],
    )

    output = run_from_config_file(
        config_path,
        evaluator=lambda *_args: {
            "output": "",
            "timestamp_utc": "20260716T120000_000000Z",
            "latency_seconds": 1.25,
        },
    )

    rows = _read_jsonl(output["results_paths"]["model-a"])
    assert output["error_count"] == 1
    assert rows[0]["call_status"] == "error"
    assert rows[0]["error_type"] == "EmptyLLMResponse"
    assert "timestamp_utc" not in rows[0]
    assert "latency_seconds" not in rows[0]


@pytest.mark.parametrize(
    "unexpected_error",
    [
        RuntimeError("programming error"),
        ValueError("programming error"),
        KeyError("programming error"),
    ],
    ids=["runtime", "value", "key"],
)
def test_run_from_config_file_propagates_unexpected_evaluator_errors(
    tmp_path: Path,
    unexpected_error: Exception,
) -> None:
    """Non-provider exceptions are not converted into model-call errors."""

    config_path = _write_config(
        tmp_path,
        models=["model-a"],
        items=[{"id": "item-1", "text": "A cat sat."}],
    )

    with pytest.raises(type(unexpected_error), match="programming error"):
        run_from_config_file(
            config_path,
            evaluator=lambda *_args: (_ for _ in ()).throw(unexpected_error),
        )


def test_run_from_config_file_retries_only_cached_errors(tmp_path: Path) -> None:
    """Changing retry_errors does not change result IDs for cached rows."""

    config_path = _write_config(
        tmp_path,
        models=["model-a"],
        items=[
            {"id": "item-1", "text": "A cat sat."},
            {"id": "item-2", "text": "The dog ran."},
        ],
    )

    def first_evaluator(
        _config: BenchmarkRunConfig,
        _model_name: str,
        item: dict,
        _prompt: str,
    ) -> dict:
        if item["benchmark_item_id"] == "item-1":
            raise MistralConnectionException("transient")
        return {
            "output": "success",
            "timestamp_utc": "20260716T120000_000000Z",
            "latency_seconds": 1.25,
        }

    first_output = run_from_config_file(
        config_path,
        evaluator=first_evaluator,
    )
    first_rows = _read_jsonl(first_output["results_paths"]["model-a"])
    error_row = next(row for row in first_rows if row["call_status"] == "error")
    assert "timestamp_utc" not in error_row
    assert "latency_seconds" not in error_row
    assert "started_at" not in error_row
    assert "completed_at" not in error_row
    assert "elapsed_seconds" not in error_row

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["retry_errors"] = True
    config_path.write_text(json.dumps(config), encoding="utf-8")
    retried_items = []

    def retry_evaluator(
        _config: BenchmarkRunConfig,
        _model_name: str,
        item: dict,
        _prompt: str,
    ) -> dict:
        retried_items.append(item["benchmark_item_id"])
        return {"output": "retried"}

    second_output = run_from_config_file(
        config_path,
        evaluator=retry_evaluator,
    )

    rows = _read_jsonl(second_output["results_paths"]["model-a"])
    assert first_output["config_id"] == second_output["config_id"]
    assert retried_items == ["item-1"]
    assert second_output["executed_count"] == 1
    assert second_output["skipped_count"] == 1
    assert [row["output"] for row in rows] == ["retried", "success"]
    log = json.loads(
        (second_output["run_dir"] / "log.json").read_text(encoding="utf-8")
    )
    assert "responses_retry_timestamp-utc" in log["model-a"]
    assert "responses_retry_git-hash" in log["model-a"]


def test_retry_errors_rejects_uncached_items_before_starting_workers(
    tmp_path: Path,
) -> None:
    """Retry-only mode does not execute newly added semantic calls."""

    config_path = _write_config(
        tmp_path,
        models=["model-a", "model-b"],
        items=[{"id": "item-1", "text": "A cat sat."}],
        model_workers=2,
    )
    run_from_config_file(
        config_path,
        evaluator=lambda _config, model, _item, _prompt: {"output": model},
    )

    config_path = _write_config(
        tmp_path,
        models=["model-a", "model-b"],
        items=[
            {"id": "item-1", "text": "A cat sat."},
            {"id": "item-2", "text": "The dog ran."},
        ],
        model_workers=2,
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["retry_errors"] = True
    config_path.write_text(json.dumps(config), encoding="utf-8")
    called_models = []

    with pytest.raises(
        ConfigError,
        match=r"missing calls by model: model-a \(1\), model-b \(1\)",
    ):
        run_from_config_file(
            config_path,
            evaluator=lambda _config, model, _item, _prompt: (
                called_models.append(model) or {"output": "unexpected"}
            ),
        )

    assert called_models == []


def test_retry_errors_rejects_changed_model_config_before_calling_model(
    tmp_path: Path,
) -> None:
    """Retry-only mode requires the current semantic model-call identities."""

    config_path = _write_config(
        tmp_path,
        models=["model-a"],
        items=[{"id": "item-1", "text": "A cat sat."}],
    )
    config = load_run_config(config_path)
    run_benchmark(
        config,
        config_id="config-1",
        evaluator=lambda *_args: {"output": "first"},
        model_configs={"model-a": {"Token_Window": 100}},
    )
    retry_config = replace(config, retry_errors=True)
    calls = []

    with pytest.raises(ConfigError, match="Run normally first"):
        run_benchmark(
            retry_config,
            config_id="config-2",
            evaluator=lambda *_args: calls.append("called") or {"output": "unexpected"},
            model_configs={"model-a": {"Token_Window": 200}},
        )

    assert calls == []


def test_retry_errors_rejects_invalid_cached_call_status(tmp_path: Path) -> None:
    """Retry-only mode fails on cache rows that cannot be classified."""

    config_path = _write_config(
        tmp_path,
        models=["model-a"],
        items=[{"id": "item-1", "text": "A cat sat."}],
    )
    output = run_from_config_file(
        config_path,
        evaluator=lambda *_args: {"output": "first"},
    )
    results_path = output["results_paths"]["model-a"]
    rows = _read_jsonl(results_path)
    rows[0]["call_status"] = "pending"
    results_path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["retry_errors"] = True
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ConfigError, match="invalid call_status"):
        run_from_config_file(
            config_path,
            evaluator=lambda *_args: {"output": "unexpected"},
        )


def test_retry_errors_preflights_every_model_before_starting_workers(
    tmp_path: Path,
) -> None:
    """A missing model cache prevents all retry-only workers from starting."""

    config_path = _write_config(
        tmp_path,
        models=["model-a"],
        items=[{"id": "item-1", "text": "A cat sat."}],
    )
    run_from_config_file(
        config_path,
        evaluator=lambda *_args: (_ for _ in ()).throw(
            MistralConnectionException("transient")
        ),
    )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["models"] = ["model-a", "model-b"]
    config["model_workers"] = 2
    config["retry_errors"] = True
    config_path.write_text(json.dumps(config), encoding="utf-8")
    called_models = []

    with pytest.raises(FileNotFoundError, match="model-b"):
        run_from_config_file(
            config_path,
            evaluator=lambda _config, model, _item, _prompt: (
                called_models.append(model) or {"output": "unexpected"}
            ),
        )

    assert called_models == []


def test_run_from_config_file_recomputes_when_model_call_inputs_change(
    tmp_path: Path,
) -> None:
    """Model-call input changes create new rows instead of reusing cached rows."""

    config_path = _write_config(
        tmp_path,
        models=["model-a"],
        items=[{"id": "item-1", "text": "A cat sat."}],
    )
    first_output = run_from_config_file(
        config_path,
        evaluator=lambda _config, _model, _item, _prompt: {"output": "first"},
    )
    first_rows = _read_jsonl(first_output["results_paths"]["model-a"])

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["max_tokens"] = 100
    config_path.write_text(json.dumps(config), encoding="utf-8")
    calls = []

    def evaluator(
        config: BenchmarkRunConfig,
        _model_name: str,
        item: dict,
        _prompt: str,
    ) -> dict:
        calls.append((config.max_tokens, item["benchmark_item_id"]))
        return {"output": "second"}

    second_output = run_from_config_file(
        config_path,
        evaluator=evaluator,
    )

    rows = _read_jsonl(second_output["results_paths"]["model-a"])
    assert first_output["config_id"] != second_output["config_id"]
    assert calls == [(100, "item-1")]
    assert second_output["executed_count"] == 1
    assert second_output["skipped_count"] == 0
    assert len(rows) == 1
    assert rows[0]["output"] == "second"
    assert rows[0]["result_id"] != first_rows[0]["result_id"]
    assert rows[0]["model_call_id"] != first_rows[0]["model_call_id"]


def test_explicit_default_max_tokens_keeps_model_cache(tmp_path: Path) -> None:
    """Omitted and explicit 10,000-token limits identify the same model call."""

    config_path = _write_config(
        tmp_path,
        models=["model-a"],
        items=[{"id": "item-1", "text": "A cat sat."}],
    )
    first_output = run_from_config_file(
        config_path,
        evaluator=lambda _config, _model, _item, _prompt: {"output": "first"},
    )
    first_rows = _read_jsonl(first_output["results_paths"]["model-a"])

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["max_tokens"] = DEFAULT_MAX_TOKENS
    config_path.write_text(json.dumps(config), encoding="utf-8")

    def fail_if_called(*_args) -> dict:
        raise AssertionError("equivalent effective max_tokens should use the cache")

    second_output = run_from_config_file(
        config_path,
        evaluator=fail_if_called,
    )
    second_rows = _read_jsonl(second_output["results_paths"]["model-a"])

    assert first_output["config_id"] == second_output["config_id"]
    assert second_output["executed_count"] == 0
    assert second_output["skipped_count"] == 1
    assert second_rows[0]["model_call_id"] == first_rows[0]["model_call_id"]


def test_run_from_config_file_uses_safe_model_result_filename(tmp_path: Path) -> None:
    """Model names with path separators stay inside the responses directory."""

    config_path = _write_config(
        tmp_path,
        models=["provider/model"],
        items=[{"id": "item-1", "text": "A cat sat."}],
    )

    output = run_from_config_file(
        config_path,
        evaluator=lambda _config, _model, _item, _prompt: {"output": "first"},
    )

    results_path = output["results_paths"]["provider/model"]
    assert results_path.parent == output["results_dir"]
    assert results_path.name.startswith("responses_provider_model-")
    assert results_path.name.endswith(".jsonl")
    assert results_path.exists()


def test_run_benchmark_writes_supplied_model_configs(tmp_path: Path) -> None:
    """Run snapshots include sanitized model configs supplied by the caller."""

    custom_models_path = tmp_path / "custom_models.yaml"
    custom_models_path.write_text(
        '{"provider-a":{"models":{"model-a":{}}}}', encoding="utf-8"
    )
    config_path = _write_config(
        tmp_path,
        models=["model-a"],
        items=[{"id": "item-1", "text": "A cat sat."}],
        custom_models_path=custom_models_path,
    )

    config = load_run_config(config_path)
    output = run_benchmark(
        config,
        config_id="config-123",
        evaluator=lambda _config, _model, _item, _prompt: {"output": "first"},
        model_configs={
            "model-a": {
                "Api_Interface": "OpenAI",
                "Api_Key_Env_Var": "OPENAI_API_KEY",
                "Token_Window": 128000,
            }
        },
    )

    snapshot = json.loads(output["config_snapshot_path"].read_text(encoding="utf-8"))
    assert snapshot["model_configs"] == {
        "model-a": {
            "Api_Interface": "OpenAI",
            "Api_Key_Env_Var": "OPENAI_API_KEY",
            "Token_Window": 128000,
        }
    }
    rows = _read_jsonl(output["results_paths"]["model-a"])
    assert rows[0]["model"] == "model-a"
    assert "model_configs" not in rows[0]


def test_run_benchmark_recomputes_when_model_config_changes(tmp_path: Path) -> None:
    """Model config changes are part of the per-call cache identity."""

    custom_models_path = tmp_path / "custom_models.yaml"
    custom_models_path.write_text(
        '{"provider-a":{"models":{"model-a":{}}}}', encoding="utf-8"
    )
    config_path = _write_config(
        tmp_path,
        models=["model-a"],
        items=[{"id": "item-1", "text": "A cat sat."}],
        custom_models_path=custom_models_path,
    )
    config = load_run_config(config_path)

    first_output = run_benchmark(
        config,
        config_id="config-123",
        evaluator=lambda _config, _model, _item, _prompt: {"output": "first"},
        model_configs={"model-a": {"Token_Window": 100}},
    )
    first_rows = _read_jsonl(first_output["results_paths"]["model-a"])

    second_output = run_benchmark(
        config,
        config_id="config-123",
        evaluator=lambda _config, _model, _item, _prompt: {"output": "second"},
        model_configs={"model-a": {"Token_Window": 200}},
    )

    rows = _read_jsonl(second_output["results_paths"]["model-a"])
    assert first_output["config_id"] == second_output["config_id"]
    assert first_output["config_snapshot_path"] == second_output["config_snapshot_path"]
    assert second_output["executed_count"] == 1
    assert second_output["skipped_count"] == 0
    assert len(rows) == 1
    assert rows[0]["output"] == "second"
    assert rows[0]["result_id"] != first_rows[0]["result_id"]
    assert rows[0]["model_call_id"] != first_rows[0]["model_call_id"]


def test_snapshot_model_configs_reads_registered_fdllm_models() -> None:
    """Model config snapshots come from FDLLM's registered model state."""

    with patch("fdllm.list_models") as list_models:
        list_models.return_value = {
            "model-a": {
                "Api_Interface": "OpenAI",
                "Api_Key_Env_Var": "OPENAI_API_KEY",
                "Client_Args": {
                    "base_url": "https://example.test/v1",
                    "api_key": "secret-value",
                    "Authorization": "Bearer secret-value",
                },
                "Api_Model_Name": "provider-model-a",
                "Token_Window": 128000,
            }
        }

        snapshot = snapshot_model_configs(["model-a"])

    list_models.assert_called_once_with(full_info=True)
    assert snapshot == {
        "model-a": {
            "Api_Interface": "OpenAI",
            "Api_Key_Env_Var": "OPENAI_API_KEY",
            "Client_Args": {"base_url": "https://example.test/v1"},
            "Api_Model_Name": "provider-model-a",
            "Token_Window": 128000,
        }
    }


def test_download_default_custom_models_anonymously_and_atomically(
    tmp_path: Path,
) -> None:
    """A validated anonymous download atomically replaces the model registry."""

    destination = tmp_path / "custom_models.yaml"
    destination.write_text("OldProvider: {}\n", encoding="utf-8")

    def write_downloaded_models(**kwargs) -> None:
        Path(kwargs["write_path"]).write_text(
            "NewProvider:\n  models: {}\n",
            encoding="utf-8",
        )

    with (
        patch("phonics_benchmark.runner.DEFAULT_CUSTOM_MODELS_PATH", destination),
        patch(
            "fab_benchmark_publisher.download_file",
            side_effect=write_downloaded_models,
        ) as download_file,
    ):
        result = download_default_custom_models()

    call_args = download_file.call_args.kwargs
    temporary_path = Path(call_args.pop("write_path"))
    assert call_args == {
        "bucket_name": "fab-open-configs",
        "file_name": "custom_models.yaml",
        "require_auth": False,
    }
    assert temporary_path.parent == destination.parent
    assert temporary_path != destination
    assert not temporary_path.exists()
    assert destination.read_text(encoding="utf-8") == ("NewProvider:\n  models: {}\n")
    assert result == destination


def test_download_default_custom_models_preserves_existing_file_on_failure(
    tmp_path: Path,
) -> None:
    """A failed transfer cannot leave a partial registry at the final path."""

    destination = tmp_path / "custom_models.yaml"
    destination.write_text("OldProvider: {}\n", encoding="utf-8")

    def fail_after_partial_download(**kwargs) -> None:
        Path(kwargs["write_path"]).write_text("partial", encoding="utf-8")
        raise ConnectionError("download interrupted")

    with (
        patch("phonics_benchmark.runner.DEFAULT_CUSTOM_MODELS_PATH", destination),
        patch(
            "fab_benchmark_publisher.download_file",
            side_effect=fail_after_partial_download,
        ),
        pytest.raises(ConnectionError, match="interrupted"),
    ):
        download_default_custom_models()

    assert destination.read_text(encoding="utf-8") == "OldProvider: {}\n"
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize(
    ("downloaded_contents", "error"),
    [
        ("[", "invalid"),
        ("[]", "non-empty YAML mapping"),
    ],
)
def test_download_default_custom_models_rejects_invalid_registry(
    tmp_path: Path,
    downloaded_contents: str,
    error: str,
) -> None:
    """Invalid downloaded YAML does not replace the existing registry."""

    destination = tmp_path / "custom_models.yaml"
    destination.write_text("OldProvider: {}\n", encoding="utf-8")

    def write_invalid_download(**kwargs) -> None:
        Path(kwargs["write_path"]).write_text(
            downloaded_contents,
            encoding="utf-8",
        )

    with (
        patch("phonics_benchmark.runner.DEFAULT_CUSTOM_MODELS_PATH", destination),
        patch(
            "fab_benchmark_publisher.download_file",
            side_effect=write_invalid_download,
        ),
        pytest.raises(ConfigError, match=error),
    ):
        download_default_custom_models()

    assert destination.read_text(encoding="utf-8") == "OldProvider: {}\n"
    assert list(tmp_path.iterdir()) == [destination]


def test_prepare_fdllm_uses_default_custom_models_path(tmp_path: Path) -> None:
    """Model preparation registers the existing shared model registry."""

    config = BenchmarkRunConfig(
        run_id="experiment-a",
        language="sw",
        task="decodability",
        output_dir=tmp_path / "outputs",
        models=["model-a"],
        items_path=tmp_path / "items.jsonl",
    )

    with (
        patch("phonics_benchmark.runner._load_environment") as load_environment,
        patch("phonics_benchmark.runner._register_custom_models") as register_models,
        patch("phonics_benchmark.runner.validate_registered_models") as validate_models,
    ):
        prepare_fdllm_for_run(config)

    load_environment.assert_called_once_with(None)
    register_models.assert_called_once_with(DEFAULT_CUSTOM_MODELS_PATH)
    validate_models.assert_called_once_with(["model-a"])


def test_prepare_fdllm_preserves_explicit_custom_models_path(tmp_path: Path) -> None:
    """An explicit model registry remains the pinned registry used by the run."""

    custom_models_path = tmp_path / "custom_models.yaml"
    custom_models_path.write_text(
        '{"provider-a":{"models":{"model-a":{}}}}', encoding="utf-8"
    )
    config = BenchmarkRunConfig(
        run_id="experiment-a",
        language="sw",
        task="decodability",
        output_dir=tmp_path / "outputs",
        models=["model-a"],
        items_path=tmp_path / "items.jsonl",
        custom_models_path=custom_models_path,
    )

    with (
        patch("phonics_benchmark.runner._load_environment"),
        patch("phonics_benchmark.runner._register_custom_models") as register_models,
        patch("phonics_benchmark.runner.validate_registered_models"),
    ):
        prepare_fdllm_for_run(config)

    register_models.assert_called_once_with(custom_models_path)


def test_default_evaluator_uses_qfb_call_record(tmp_path: Path) -> None:
    """The default evaluator maps QFB call metadata into the runner result."""

    config = BenchmarkRunConfig(
        run_id="experiment-a",
        language="sw",
        task="decodability",
        output_dir=tmp_path / "outputs",
        models=["model-a"],
        items_path=tmp_path / "items.jsonl",
    )

    call_record = {
        "response_text": '["cat"]',
        "max_tokens": DEFAULT_MAX_TOKENS,
        "tokens_used_completion": 7,
        "timestamp_utc": "20260716T120000_000000Z",
        "latency_seconds": 1.25,
    }
    with patch(
        "phonics_benchmark.runner.call_llm", return_value=call_record
    ) as call_llm:
        payload = evaluate_item_with_fdllm(
            config,
            "model-a",
            {"benchmark_item_id": "item-1", "content": "A cat sat."},
            "rendered prompt",
        )

    call_llm.assert_called_once_with(
        "rendered prompt",
        model="model-a",
        system_prompt=None,
        max_tokens=DEFAULT_MAX_TOKENS,
        should_log=False,
        tag="item-1",
    )
    assert payload == {
        "output": '["cat"]',
        "max_tokens": DEFAULT_MAX_TOKENS,
        "tokens_used_completion": 7,
        "timestamp_utc": "20260716T120000_000000Z",
        "latency_seconds": 1.25,
    }


def test_default_evaluator_uses_configured_max_tokens(tmp_path: Path) -> None:
    """An explicit token limit overrides the shared QFB default."""

    config = BenchmarkRunConfig(
        run_id="experiment-a",
        language="sw",
        task="decodability",
        output_dir=tmp_path / "outputs",
        models=["model-a"],
        items_path=tmp_path / "items.jsonl",
        max_tokens=321,
    )
    call_record = {
        "response_text": "[]",
        "max_tokens": 321,
        "tokens_used_completion": 1,
        "timestamp_utc": "20260716T120000_000000Z",
        "latency_seconds": 0.5,
    }

    with patch(
        "phonics_benchmark.runner.call_llm", return_value=call_record
    ) as call_llm:
        evaluate_item_with_fdllm(
            config,
            "model-a",
            {"benchmark_item_id": "item-1", "content": "A cat sat."},
            "rendered prompt",
        )

    assert call_llm.call_args.kwargs["max_tokens"] == 321


def test_run_from_config_file_saves_through_result_function(tmp_path: Path) -> None:
    """The runner persists each model's results through the result seam."""

    custom_models_path = tmp_path / "custom_models.yaml"
    custom_models_path.write_text(
        '{"provider-a":{"models":{"model-a":{}}}}', encoding="utf-8"
    )
    config_path = _write_config(
        tmp_path,
        models=["model-a"],
        items=[{"id": "item-1", "text": "A cat sat."}],
        custom_models_path=custom_models_path,
    )

    with patch("phonics_benchmark.runner.save_results", wraps=save_results) as save:
        run_from_config_file(
            config_path,
            evaluator=lambda _config, _model, _item, _prompt: {"output": "first"},
        )

    save.assert_called()
    assert len(save.call_args.args) == 2
    assert save.call_args.args[1].name == "responses_model-a.jsonl"


def test_run_from_config_file_checkpoints_after_each_chunk(tmp_path: Path) -> None:
    """Pending items are saved in configured chunks, not only at run end."""

    config_path = _write_config(
        tmp_path,
        models=["model-a"],
        items=[
            {"id": "item-1", "text": "A cat sat."},
            {"id": "item-2", "text": "The dog ran."},
            {"id": "item-3", "text": "A fish swam."},
        ],
        chunk_size=2,
    )

    with patch("phonics_benchmark.runner.save_results", wraps=save_results) as save:
        output = run_from_config_file(
            config_path,
            evaluator=lambda _config, _model, item, _prompt: {
                "output": item["benchmark_item_id"]
            },
        )

    assert save.call_count == 2
    saved_lengths = [len(call.args[0]) for call in save.call_args_list]
    assert saved_lengths == [2, 3]
    rows = _read_jsonl(output["results_paths"]["model-a"])
    assert [row["benchmark_item_id"] for row in rows] == ["item-1", "item-2", "item-3"]


def test_run_from_config_file_can_save_once_per_model(tmp_path: Path) -> None:
    """A null chunk_size waits until all pending items for the model finish."""

    config_path = _write_config(
        tmp_path,
        models=["model-a"],
        items=[
            {"id": "item-1", "text": "A cat sat."},
            {"id": "item-2", "text": "The dog ran."},
            {"id": "item-3", "text": "A fish swam."},
        ],
        chunk_size=None,
    )

    with patch("phonics_benchmark.runner.save_results", wraps=save_results) as save:
        output = run_from_config_file(
            config_path,
            evaluator=lambda _config, _model, item, _prompt: {
                "output": item["benchmark_item_id"]
            },
        )

    assert save.call_count == 1
    assert len(save.call_args.args[0]) == 3
    assert len(_read_jsonl(output["results_paths"]["model-a"])) == 3


def _write_config(
    tmp_path: Path,
    *,
    models: list | None = None,
    items: list[dict] | None = None,
    custom_models_path: Path | None = None,
    chunk_size: int | None | object = _CHUNK_SIZE_UNSET,
    model_workers: int = 1,
) -> Path:
    if models is None:
        models = ["model-a", "model-b"]
    if items is None:
        items = [
            {
                "id": "item-1",
                "text": "A cat sat.",
                "student_knowledge": {
                    "graphemes": ["a", "t"],
                    "whole_words": ["a"],
                },
            },
            {
                "id": "item-2",
                "text": "The dog ran.",
                "student_knowledge": {
                    "graphemes": ["d", "o", "g"],
                    "whole_words": [],
                },
            },
        ]

    _write_items_jsonl(tmp_path, items)

    if custom_models_path is None:
        custom_models_path = tmp_path / "custom_models.yaml"
        registry_models = list(dict.fromkeys([*models, "model-a", "model-b"]))
        provider_registry = {
            f"provider-{index}": {"models": {model_name: {}}}
            for index, model_name in enumerate(registry_models, start=1)
        }
        custom_models_path.write_text(
            json.dumps(provider_registry),
            encoding="utf-8",
        )

    config_path = tmp_path / "run.json"
    config = {
        "run_id": "experiment-a",
        "language": "sw",
        "items_path": "items.jsonl",
        "output_dir": "outputs",
        "models": models,
        "model_workers": model_workers,
        "custom_models_path": str(custom_models_path),
    }
    if chunk_size is not _CHUNK_SIZE_UNSET:
        config["chunk_size"] = chunk_size
    config_path.write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    return config_path


def _write_items_jsonl(tmp_path: Path, items: list[dict]) -> Path:
    items_path = tmp_path / "items.jsonl"
    records = [
        {
            "benchmark_item_id": item.get("benchmark_item_id", item.get("id")),
            "content": item.get("content", item.get("text")),
            "student_knowledge": item.get(
                "student_knowledge",
                {"graphemes": ["a"], "whole_words": []},
            ),
            "source_id": f"source-{index}",
            "student_knowledge_id": f"student-{index}",
        }
        for index, item in enumerate(items, start=1)
    ]
    items_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return items_path


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_all_results(results_paths: dict[str, Path]) -> list[dict]:
    rows = []
    for path in results_paths.values():
        rows.extend(_read_jsonl(path))
    return rows
