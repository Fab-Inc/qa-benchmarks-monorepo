"""Tests for the config-driven phonics benchmark runner."""

import csv
import json
from pathlib import Path
from threading import Barrier
from unittest.mock import patch

import httpx
import pytest
from anthropic import APIConnectionError as AnthropicAPIConnectionError
from botocore.exceptions import EndpointConnectionError
from google.genai.errors import ServerError as GoogleServerError
from mistralai.exceptions import MistralConnectionException
from openai import APIConnectionError as OpenAIAPIConnectionError
from phonics_benchmark.config import BenchmarkRunConfig, load_run_config
from phonics_benchmark.items import load_benchmark_items
from phonics_benchmark.llm import DEFAULT_MAX_TOKENS
from phonics_benchmark.results import save_results
from phonics_benchmark.runner import (
    DEFAULT_CUSTOM_MODELS_PATH,
    download_default_custom_models,
    evaluate_item_with_fdllm,
    prepare_fdllm_for_run,
    run_benchmark,
    run_from_config_file,
    snapshot_model_configs,
)

_CHUNK_SIZE_UNSET = object()


def test_load_benchmark_items_reads_composed_csv(tmp_path: Path) -> None:
    """QFB-style composed CSV rows are adapted to the production item shape."""

    items_path = tmp_path / "items.csv"
    items_path.write_text(
        "\n".join(
            [
                (
                    "benchmark_item_id,language_iso_code,curriculum_stage,content,"
                    "student_knowledge_id,graphemes,whole_words,non_decodable_words"
                ),
                (
                    'item-1,sw,2,Mama anaona kuku.,student-1,"[""a"", ""m""]",'
                    '"[""mama""]","[""kuku""]"'
                ),
                (
                    'item-2,sw,4,Tai ni ndege.,student-2,"[""a"", ""i""]",'
                    '"[]","[""ndege""]"'
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    items = load_benchmark_items(items_path)

    assert len(items) == 2
    assert items.loc[0, "item_id"] == "item-1"
    assert items.loc[0, "text"] == "Mama anaona kuku."
    assert items.loc[0, "student_knowledge"] == {
        "graphemes": ["a", "m"],
        "whole_words": ["mama"],
    }
    assert items.loc[0, "metadata"]["curriculum_stage"] == "2"
    assert items.loc[0, "metadata"]["non_decodable_words"] == ["kuku"]


def test_run_from_config_file_uses_composed_csv(tmp_path: Path) -> None:
    """The runner loads every row from the configured composed items CSV."""

    items_path = tmp_path / "items.csv"
    items_path.write_text(
        "\n".join(
            [
                (
                    "benchmark_item_id,content,student_knowledge_id,"
                    "graphemes,whole_words"
                ),
                'item-1,First text.,student-1,"[""a""]","[]"',
                'item-2,Second text.,student-2,"[""b""]","[]"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    custom_models_path = tmp_path / "custom_models.yaml"
    custom_models_path.write_text("models: []", encoding="utf-8")
    config_path = tmp_path / "run.json"
    config_path.write_text(
        json.dumps(
            {
                "language": "kiswahili",
                "models": ["model-a"],
                "items_path": "items.csv",
                "custom_models_path": "custom_models.yaml",
                "output_dir": "outputs",
            }
        ),
        encoding="utf-8",
    )
    seen_items = []

    def evaluator(
        _config: BenchmarkRunConfig,
        _model: str,
        item: dict,
        _prompt: str,
    ) -> dict:
        seen_items.append(item["item_id"])
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
        calls.append((config.language, config.task, model_name, item["item_id"]))
        return {
            "output": f"{model_name}:{item['item_id']}",
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
    assert len(rows) == 4
    assert calls == [
        ("english", "decodability", "model-a", "item-1"),
        ("english", "decodability", "model-a", "item-2"),
        ("english", "decodability", "model-b", "item-1"),
        ("english", "decodability", "model-b", "item-2"),
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
    assert rows[0]["language"] == "english"
    assert rows[0]["success"] is True
    assert rows[0]["call_status"] == "success"
    assert "answers_timestamp-utc" not in rows[0]
    assert "answers_git-hash" not in rows[0]
    assert "A cat sat." in rows[0]["prompt"]
    assert '["a", "t"]' in rows[0]["prompt"]
    assert rows[0]["seen_prompt"] == rows[0]["prompt"]
    assert "started_at" not in rows[0]
    assert "completed_at" not in rows[0]
    assert "elapsed_seconds" not in rows[0]

    snapshot = json.loads(output["config_snapshot_path"].read_text(encoding="utf-8"))
    assert snapshot["config_id"] == output["config_id"]
    assert snapshot["code_git_hash"] == "abc123"
    assert snapshot["raw_config"]["run_id"] == "experiment-a"
    assert snapshot["resolved_config"]["items_path"] == str(
        (tmp_path / "items.csv").resolve()
    )
    assert isinstance(snapshot["model_configs"], dict)

    assert "manifest_path" not in output
    assert not (output["run_dir"] / "manifest.json").exists()

    log = json.loads((output["run_dir"] / "log.json").read_text(encoding="utf-8"))
    assert set(log) == {"model-a", "model-b"}
    assert "answers_timestamp-utc" in log["model-a"]
    assert log["model-a"]["answers_git-hash"] == "abc123"


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
        calls_by_model[model_name].append(item["item_id"])
        if item["item_id"] == "item-1":
            first_call_barrier.wait()
        return {"output": f"{model_name}:{item['item_id']}"}

    output = run_from_config_file(config_path, evaluator=evaluator)

    assert output["executed_count"] == 4
    assert output["error_count"] == 0
    assert calls_by_model == {
        "model-a": ["item-1", "item-2"],
        "model-b": ["item-1", "item-2"],
    }


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

    assert first_output["config_id"] != second_output["config_id"]
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
    assert first_output["config_id"] != second_output["config_id"]
    assert second_output["executed_count"] == 0
    assert second_output["skipped_count"] == 1
    assert rows[0]["output"] == "first"


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
        if item["item_id"] == "item-1":
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
        retried_items.append(item["item_id"])
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
        calls.append((config.max_tokens, item["item_id"]))
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

    assert first_output["config_id"] != second_output["config_id"]
    assert second_output["executed_count"] == 0
    assert second_output["skipped_count"] == 1
    assert second_rows[0]["model_call_id"] == first_rows[0]["model_call_id"]


def test_run_from_config_file_uses_safe_model_result_filename(tmp_path: Path) -> None:
    """Model names with path separators stay inside the answers directory."""

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
    assert results_path.name.startswith("resps_provider_model-")
    assert results_path.name.endswith(".jsonl")
    assert results_path.exists()


def test_run_benchmark_writes_supplied_model_configs(tmp_path: Path) -> None:
    """Run snapshots include sanitized model configs supplied by the caller."""

    custom_models_path = tmp_path / "custom_models.yaml"
    custom_models_path.write_text("models: []", encoding="utf-8")
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
    custom_models_path.write_text("models: []", encoding="utf-8")
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


def test_download_default_custom_models_anonymously() -> None:
    """The shared model registry is refreshed without requiring credentials."""

    with patch("fab_benchmark_publisher.download_file") as download_file:
        result = download_default_custom_models()

    download_file.assert_called_once_with(
        bucket_name="fab-open-configs",
        file_name="custom_models.yaml",
        write_path=DEFAULT_CUSTOM_MODELS_PATH,
        require_auth=False,
    )
    assert result == DEFAULT_CUSTOM_MODELS_PATH


def test_prepare_fdllm_uses_default_custom_models_path(tmp_path: Path) -> None:
    """Model preparation registers the existing shared model registry."""

    config = BenchmarkRunConfig(
        run_id="experiment-a",
        language="english",
        task="decodability",
        output_dir=tmp_path / "outputs",
        models=["model-a"],
        items_path=tmp_path / "items.csv",
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
    custom_models_path.write_text("models: []", encoding="utf-8")
    config = BenchmarkRunConfig(
        run_id="experiment-a",
        language="english",
        task="decodability",
        output_dir=tmp_path / "outputs",
        models=["model-a"],
        items_path=tmp_path / "items.csv",
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
        language="english",
        task="decodability",
        output_dir=tmp_path / "outputs",
        models=["model-a"],
        items_path=tmp_path / "items.csv",
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
            {"item_id": "item-1", "text": "A cat sat."},
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
        language="english",
        task="decodability",
        output_dir=tmp_path / "outputs",
        models=["model-a"],
        items_path=tmp_path / "items.csv",
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
            {"item_id": "item-1", "text": "A cat sat."},
            "rendered prompt",
        )

    assert call_llm.call_args.kwargs["max_tokens"] == 321


def test_run_from_config_file_saves_through_result_function(tmp_path: Path) -> None:
    """The runner passes config metadata to the result persistence seam."""

    custom_models_path = tmp_path / "custom_models.yaml"
    custom_models_path.write_text("models: []", encoding="utf-8")
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
    _, _, config_representation, run_id, custom_models_arg = save.call_args.args
    assert config_representation["language"] == "english"
    assert run_id == "experiment-a"
    assert custom_models_arg == custom_models_path.resolve()
    assert save.call_args.args[1].name == "resps_model-a.jsonl"


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
                "output": item["item_id"]
            },
        )

    assert save.call_count == 2
    saved_lengths = [len(call.args[0]) for call in save.call_args_list]
    assert saved_lengths == [2, 3]
    rows = _read_jsonl(output["results_paths"]["model-a"])
    assert [row["item_id"] for row in rows] == ["item-1", "item-2", "item-3"]


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
                "output": item["item_id"]
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

    items_path = tmp_path / "items.csv"
    fieldnames = [
        "benchmark_item_id",
        "content",
        "student_knowledge_id",
        "graphemes",
        "whole_words",
    ]
    with items_path.open("w", encoding="utf-8", newline="") as items_file:
        writer = csv.DictWriter(items_file, fieldnames=fieldnames)
        writer.writeheader()
        for index, item in enumerate(items, start=1):
            student_knowledge = item.get("student_knowledge", {})
            writer.writerow(
                {
                    "benchmark_item_id": item["id"],
                    "content": item["text"],
                    "student_knowledge_id": f"student-{index}",
                    "graphemes": json.dumps(student_knowledge.get("graphemes", [])),
                    "whole_words": json.dumps(student_knowledge.get("whole_words", [])),
                }
            )

    config_path = tmp_path / "run.json"
    config = {
        "run_id": "experiment-a",
        "language": "english",
        "items_path": "items.csv",
        "output_dir": "outputs",
        "models": models,
        "model_workers": model_workers,
    }
    if custom_models_path:
        config["custom_models_path"] = str(custom_models_path)
    if chunk_size is not _CHUNK_SIZE_UNSET:
        config["chunk_size"] = chunk_size
    config_path.write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    return config_path


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
