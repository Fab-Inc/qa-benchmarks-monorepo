"""Tests for the decodable-text-generation CLI."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from decodable_text_generation.cmd import BENCHMARK, app
from fab_benchmark_runner import execute_run, load_run_config
from typer.testing import CliRunner

FIXTURE = Path(__file__).parent / "fixtures" / "generation_benchmark_items.jsonl"


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def _write_config(
    tmp_path: Path,
    *,
    items_path: Path,
    task: str,
    prompt_name: str | None = None,
) -> Path:
    custom_models_path = tmp_path / "custom_models.yaml"
    custom_models_path.write_text(
        json.dumps({"provider-a": {"models": {"model-a": {}}}}),
        encoding="utf-8",
    )
    config = {
        "run_id": "experiment-a",
        "language": "en",
        "task": task,
        "items_path": str(items_path),
        "output_dir": str(tmp_path / "outputs"),
        "models": ["model-a"],
        "model_workers": 1,
        "custom_models_path": str(custom_models_path),
    }
    if prompt_name is not None:
        config["prompt_name"] = prompt_name
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def _fake_output(run_dir: Path) -> dict:
    return {
        "run_id": run_dir.name,
        "config_id": "x",
        "run_dir": run_dir,
        "config_snapshot_path": run_dir / "config.json",
        "results_dir": run_dir / "responses",
        "results_paths": {},
        "total_expected_results": 0,
        "executed_count": 0,
        "skipped_count": 0,
        "error_count": 0,
    }


def _child_items(tmp_path: Path) -> Path:
    """The fixture's rows with target_knowledge stripped: a child file."""
    items = [
        {
            key: value
            for key, value in json.loads(line).items()
            if key != "target_knowledge"
        }
        for line in FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    items_path = tmp_path / "items.jsonl"
    items_path.write_text(
        "\n".join(json.dumps(item) for item in items), encoding="utf-8"
    )
    return items_path


def _curriculum_items(tmp_path: Path) -> Path:
    """The fixture's target-bearing rows, as a homogeneous curriculum file."""
    lines = [
        line
        for line in FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("target_knowledge")
    ]
    items_path = tmp_path / "items.jsonl"
    items_path.write_text("\n".join(lines), encoding="utf-8")
    return items_path


def test_main_shows_help_without_command(runner: CliRunner) -> None:
    result = runner.invoke(app)
    assert result.exit_code == 0
    assert "Usage:" in result.stdout


def test_run_binds_generation_params(runner: CliRunner, tmp_path: Path) -> None:
    """Config keys become validated params; prompt_name defaults per task."""
    items_path = _child_items(tmp_path)
    config_path = _write_config(
        tmp_path, items_path=items_path, task="generation-child"
    )

    with patch(
        "fab_benchmark_runner.cli.execute_with_optional_sync",
        return_value=_fake_output(tmp_path / "outputs" / "experiment-a"),
    ) as run:
        result = runner.invoke(app, ["run", "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    run_config = run.call_args.args[0]
    assert run_config.params.language == "en"
    assert run_config.params.task == "generation-child"
    assert run_config.params.prompt_name == "decodable_text_generation"


def test_run_curriculum_defaults(runner: CliRunner, tmp_path: Path) -> None:
    """The curriculum task defaults to the {target}-bearing template name."""
    items_path = _curriculum_items(tmp_path)
    config_path = _write_config(
        tmp_path, items_path=items_path, task="generation-curriculum"
    )

    with patch(
        "fab_benchmark_runner.cli.execute_with_optional_sync",
        return_value=_fake_output(tmp_path / "outputs" / "experiment-a"),
    ) as run:
        result = runner.invoke(app, ["run", "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    run_config = run.call_args.args[0]
    assert run_config.params.prompt_name == "decodable_text_generation_curriculum"


def test_run_requires_task(runner: CliRunner, tmp_path: Path) -> None:
    """A config without ``task`` is rejected."""
    items_path = _child_items(tmp_path)
    config_path = _write_config(
        tmp_path, items_path=items_path, task="generation-child"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    del config["task"]
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = runner.invoke(app, ["run", "--config", str(config_path)])

    assert result.exit_code == 1
    assert "task" in result.output


def test_run_rejects_unknown_prompt_name(runner: CliRunner, tmp_path: Path) -> None:
    items_path = _child_items(tmp_path)
    config_path = _write_config(
        tmp_path,
        items_path=items_path,
        task="generation-child",
        prompt_name="bogus-prompt",
    )

    result = runner.invoke(app, ["run", "--config", str(config_path)])

    assert result.exit_code == 1
    assert "prompt_name" in result.output


def test_run_rejects_task_item_mismatch(runner: CliRunner, tmp_path: Path) -> None:
    """Curriculum items under a child task fail before any model call."""
    items_path = _curriculum_items(tmp_path)
    config_path = _write_config(
        tmp_path, items_path=items_path, task="generation-child"
    )

    with (
        patch("fab_benchmark_runner.runner.prepare_fdllm_for_run"),
        patch("fab_benchmark_runner.runner.snapshot_model_configs", return_value={}),
    ):
        result = runner.invoke(app, ["run", "--config", str(config_path)])

    assert result.exit_code == 1
    assert "target_knowledge" in result.output


def test_run_rejects_mixed_items(runner: CliRunner, tmp_path: Path) -> None:
    """A file mixing target-bearing and target-free items fails loudly."""
    items_path = tmp_path / "items.jsonl"
    items_path.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    config_path = _write_config(
        tmp_path, items_path=items_path, task="generation-child"
    )

    with (
        patch("fab_benchmark_runner.runner.prepare_fdllm_for_run"),
        patch("fab_benchmark_runner.runner.snapshot_model_configs", return_value={}),
    ):
        result = runner.invoke(app, ["run", "--config", str(config_path)])

    assert result.exit_code == 1
    assert "target_knowledge" in result.output


def test_dry_run_reports_plan(runner: CliRunner, tmp_path: Path) -> None:
    items_path = _child_items(tmp_path)
    config_path = _write_config(
        tmp_path, items_path=items_path, task="generation-child"
    )

    with (
        patch("fab_benchmark_runner.runner.prepare_fdllm_for_run"),
        patch("fab_benchmark_runner.runner.snapshot_model_configs", return_value={}),
    ):
        result = runner.invoke(app, ["run", "--config", str(config_path), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "model-a: 2 planned, 0 reused, 2 to run" in result.output


def test_run_end_to_end_with_fake_evaluator(tmp_path: Path) -> None:
    """A full run: items load, the baked prompt is used, rows record params."""
    items_path = _child_items(tmp_path)
    config_path = _write_config(
        tmp_path, items_path=items_path, task="generation-child"
    )

    prompts_seen: list[str] = []

    def evaluator(_config, _model, _item, prompt: str) -> dict:
        prompts_seen.append(prompt)
        return {"output": "<text>the cat sat</text>"}

    output = execute_run(
        load_run_config(config_path, BENCHMARK),
        evaluator=evaluator,
    )

    assert output["executed_count"] == 2
    assert output["error_count"] == 0
    assert len(prompts_seen) == 2
    assert "Generate a list of 20 words" in prompts_seen[0]
    assert "<text>" in prompts_seen[0]

    rows = [
        json.loads(line)
        for path in output["results_paths"].values()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert {row["task"] for row in rows} == {"generation-child"}
    assert {row["language"] for row in rows} == {"en"}
    assert {row["prompt_name"] for row in rows} == {"decodable_text_generation"}
    assert {row["benchmark_item_id"] for row in rows} == {
        "ufli-en-l008_words",
        "child-jagged-01_story",
    }
