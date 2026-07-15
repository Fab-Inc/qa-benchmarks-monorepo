"""Tests for the config-driven phonics benchmark runner."""

import json
from pathlib import Path
from unittest.mock import patch

from phonics_benchmark.config import BenchmarkRunConfig, load_run_config
from phonics_benchmark.items import load_benchmark_items
from phonics_benchmark.results import save_results
from phonics_benchmark.runner import (
    DEFAULT_CUSTOM_MODELS_PATH,
    PLACEHOLDER_LLM_TEXT,
    evaluate_item_with_fdllm,
    prepare_fdllm_for_run,
    run_benchmark,
    run_from_config_file,
    snapshot_model_configs,
)

_CHUNK_SIZE_UNSET = object()


def test_load_benchmark_items_joins_data_files(tmp_path: Path) -> None:
    """The item adapter joins text and student profile records."""

    data_dir = tmp_path / "data" / "english"
    _write_dataset(
        data_dir,
        [
            {
                "id": "item-1",
                "text": "A cat sat.",
                "student_knowledge": {"gpcs": ["a", "t"]},
            }
        ],
    )

    items = load_benchmark_items("english", tmp_path / "data")

    assert len(items) == 1
    assert items.loc[0, "item_id"] == "item-1"
    assert items.loc[0, "text"] == "A cat sat."
    assert items.loc[0, "student_knowledge"] == {"gpcs": ["a", "t"]}
    assert items.loc[0, "metadata"]["decodable_text"]["text_id"] == "text-1"


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
            default_data_root=tmp_path / "data",
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
    assert rows[0]["prompt"] == PLACEHOLDER_LLM_TEXT
    assert rows[0]["seen_prompt"] == PLACEHOLDER_LLM_TEXT

    snapshot = json.loads(output["config_snapshot_path"].read_text(encoding="utf-8"))
    assert snapshot["config_id"] == output["config_id"]
    assert snapshot["code_git_hash"] == "abc123"
    assert snapshot["raw_config"]["run_id"] == "experiment-a"
    assert "items_path" not in snapshot["resolved_config"]
    assert snapshot["resolved_config"]["data_root"] == str(
        (tmp_path / "data").resolve()
    )
    assert isinstance(snapshot["model_configs"], dict)

    assert "manifest_path" not in output
    assert not (output["run_dir"] / "manifest.json").exists()

    log = json.loads((output["run_dir"] / "log.json").read_text(encoding="utf-8"))
    assert set(log) == {"model-a", "model-b"}
    assert "answers_timestamp-utc" in log["model-a"]
    assert log["model-a"]["answers_git-hash"] == "abc123"


def test_run_from_config_file_skips_existing_successes(tmp_path: Path) -> None:
    """A repeated run does not call the evaluator again for successful rows."""

    config_path = _write_config(
        tmp_path,
        models=["model-a"],
        items=[{"id": "item-1", "text": "A cat sat."}],
    )

    output = run_from_config_file(
        config_path,
        default_data_root=tmp_path / "data",
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
        default_data_root=tmp_path / "data",
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
        default_data_root=tmp_path / "data",
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
        default_data_root=tmp_path / "data",
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
        default_data_root=tmp_path / "data",
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
        default_data_root=tmp_path / "data",
        evaluator=fail_if_called,
    )

    rows = _read_jsonl(second_output["results_paths"]["model-a"])
    assert first_output["config_id"] != second_output["config_id"]
    assert second_output["executed_count"] == 0
    assert second_output["skipped_count"] == 1
    assert rows[0]["output"] == "first"


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
            raise RuntimeError("transient")
        return {"output": "success"}

    first_output = run_from_config_file(
        config_path,
        default_data_root=tmp_path / "data",
        evaluator=first_evaluator,
    )

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
        default_data_root=tmp_path / "data",
        evaluator=retry_evaluator,
    )

    rows = _read_jsonl(second_output["results_paths"]["model-a"])
    assert first_output["config_id"] == second_output["config_id"]
    assert retried_items == ["item-1"]
    assert second_output["executed_count"] == 1
    assert second_output["skipped_count"] == 1
    assert [row["output"] for row in rows] == ["retried", "success"]


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
        default_data_root=tmp_path / "data",
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
        default_data_root=tmp_path / "data",
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


def test_run_from_config_file_uses_safe_model_result_filename(tmp_path: Path) -> None:
    """Model names with path separators stay inside the answers directory."""

    config_path = _write_config(
        tmp_path,
        models=["provider/model"],
        items=[{"id": "item-1", "text": "A cat sat."}],
    )

    output = run_from_config_file(
        config_path,
        default_data_root=tmp_path / "data",
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

    config = load_run_config(config_path, default_data_root=tmp_path / "data")
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
    config = load_run_config(config_path, default_data_root=tmp_path / "data")

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


def test_prepare_fdllm_downloads_default_custom_models_anonymously(
    tmp_path: Path,
) -> None:
    """Default model setup refreshes the public shared model registry."""

    config = BenchmarkRunConfig(
        run_id="experiment-a",
        language="english",
        task="decodability",
        data_root=tmp_path / "data",
        output_dir=tmp_path / "outputs",
        models=["model-a"],
    )

    with (
        patch("phonics_benchmark.runner._load_environment") as load_environment,
        patch("fab_benchmark_publisher.download_file") as download_file,
        patch("phonics_benchmark.runner._register_custom_models") as register_models,
        patch("phonics_benchmark.runner.validate_registered_models") as validate_models,
    ):
        prepare_fdllm_for_run(config)

    load_environment.assert_called_once_with(None)
    download_file.assert_called_once_with(
        bucket_name="fab-open-configs",
        file_name="custom_models.yaml",
        write_path=DEFAULT_CUSTOM_MODELS_PATH,
        require_auth=False,
    )
    register_models.assert_called_once_with(DEFAULT_CUSTOM_MODELS_PATH)
    validate_models.assert_called_once_with(["model-a"])


def test_prepare_fdllm_preserves_explicit_custom_models_path(tmp_path: Path) -> None:
    """An explicit model registry remains a pinned, non-downloaded override."""

    custom_models_path = tmp_path / "custom_models.yaml"
    custom_models_path.write_text("models: []", encoding="utf-8")
    config = BenchmarkRunConfig(
        run_id="experiment-a",
        language="english",
        task="decodability",
        data_root=tmp_path / "data",
        output_dir=tmp_path / "outputs",
        models=["model-a"],
        custom_models_path=custom_models_path,
    )

    with (
        patch("phonics_benchmark.runner._load_environment"),
        patch("fab_benchmark_publisher.download_file") as download_file,
        patch("phonics_benchmark.runner._register_custom_models") as register_models,
        patch("phonics_benchmark.runner.validate_registered_models"),
    ):
        prepare_fdllm_for_run(config)

    download_file.assert_not_called()
    register_models.assert_called_once_with(custom_models_path)


def test_default_evaluator_returns_placeholder(tmp_path: Path) -> None:
    """The temporary default evaluator does not build prompts or call FDLLM."""

    config = BenchmarkRunConfig(
        run_id="experiment-a",
        language="english",
        task="decodability",
        data_root=tmp_path / "data",
        output_dir=tmp_path / "outputs",
        models=["model-a"],
    )

    payload = evaluate_item_with_fdllm(
        config,
        "model-a",
        {"item_id": "item-1", "text": "A cat sat."},
        PLACEHOLDER_LLM_TEXT,
    )

    assert payload == {"output": PLACEHOLDER_LLM_TEXT}


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
            default_data_root=tmp_path / "data",
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
            default_data_root=tmp_path / "data",
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
            default_data_root=tmp_path / "data",
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
) -> Path:
    if models is None:
        models = ["model-a", "model-b"]
    if items is None:
        items = [
            {
                "id": "item-1",
                "text": "A cat sat.",
                "student_knowledge": {"gpcs": ["a", "t"]},
            },
            {
                "id": "item-2",
                "text": "The dog ran.",
                "student_knowledge": {"gpcs": ["d", "o", "g"]},
            },
        ]

    data_dir = tmp_path / "data" / "english"
    _write_dataset(data_dir, items)
    config_path = tmp_path / "run.json"
    config = {
        "run_id": "experiment-a",
        "language": "english",
        "output_dir": "outputs",
        "models": models,
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


def _write_dataset(data_dir: Path, items: list[dict]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    texts = []
    students = []
    benchmark_items = []
    for index, item in enumerate(items, start=1):
        text_id = f"text-{index}"
        student_id = f"student-{index}"
        texts.append(
            {
                "text_id": text_id,
                "language_iso_code": "en",
                "content": item["text"],
            }
        )
        students.append(
            {
                "student_knowledge_id": student_id,
                "language_iso_code": "en",
                "student_knowledge": item.get("student_knowledge", {}),
            }
        )
        benchmark_items.append(
            {
                "benchmark_item_id": item["id"],
                "text_id": text_id,
                "student_knowledge_id": student_id,
            }
        )

    _write_jsonl(data_dir / "decodable_texts.jsonl", texts)
    _write_jsonl(data_dir / "student_knowledge_profiles.jsonl", students)
    _write_jsonl(data_dir / "benchmark_items.jsonl", benchmark_items)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


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
