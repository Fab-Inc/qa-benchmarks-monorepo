"""Tests for the config-driven phonics benchmark runner."""

import json
from pathlib import Path
from unittest.mock import patch

from phonics_benchmark.config import BenchmarkRunConfig, load_run_config
from phonics_benchmark.items import load_benchmark_items
from phonics_benchmark.results import save_results
from phonics_benchmark.runner import (
    run_benchmark,
    run_from_config_file,
    snapshot_model_configs,
)

_CHUNK_SIZE_UNSET = object()


def test_load_benchmark_items_supports_wrapped_json(tmp_path: Path) -> None:
    """Item files can use an {'items': [...]} wrapper."""

    items_path = tmp_path / "items.json"
    items_path.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "id": "item-1",
                        "text": "A cat sat.",
                        "student_knowledge": {"gpcs": ["a", "t"]},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    items = load_benchmark_items("english", items_path)

    assert len(items) == 1
    assert items.loc[0, "item_id"] == "item-1"
    assert items.loc[0, "student_knowledge"] == {"gpcs": ["a", "t"]}


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
        output = run_from_config_file(config_path, evaluator=fake_evaluator)

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
    assert len(result_ids) == 4
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
    assert '"gpcs": ["a", "t"]' in rows[0]["prompt"]

    snapshot = json.loads(output["config_snapshot_path"].read_text(encoding="utf-8"))
    assert snapshot["config_id"] == output["config_id"]
    assert snapshot["code_git_hash"] == "abc123"
    assert snapshot["raw_config"]["run_id"] == "experiment-a"
    assert snapshot["resolved_config"]["items_path"] == str(
        (tmp_path / "items.json").resolve()
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
        evaluator=lambda _config, _model, _item, _prompt: {"output": "first"},
    )

    def fail_if_called(
        _config: BenchmarkRunConfig,
        _model_name: str,
        _item: dict,
        _prompt: str,
    ) -> dict:
        raise AssertionError("successful rows should be skipped")

    second_output = run_from_config_file(config_path, evaluator=fail_if_called)

    rows = _read_jsonl(output["results_paths"]["model-a"])
    assert second_output["executed_count"] == 0
    assert second_output["skipped_count"] == 1
    assert rows[0]["output"] == "first"


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

    items_path = tmp_path / "items.json"
    items_path.write_text(json.dumps({"items": items}), encoding="utf-8")
    config_path = tmp_path / "run.json"
    config = {
        "run_id": "experiment-a",
        "language": "english",
        "task": "decodability",
        "items_path": "items.json",
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
