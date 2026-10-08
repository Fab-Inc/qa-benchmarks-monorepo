"""Tests for the decodable-text-generation CLI."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from decodable_text_generation.cmd import app
from decodable_text_generation.items import load_generation_items
from decodable_text_generation.prompts import prompt_from_item
from phonics_benchmark.config import ConfigError
from phonics_benchmark.runner import run_from_config_file
from typer.testing import CliRunner

FIXTURE = Path(__file__).parent / "fixtures" / "generation_benchmark_items.jsonl"


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def _write_config(
    tmp_path: Path,
    *,
    items_path: Path,
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


def test_main_shows_help_without_command(runner: CliRunner) -> None:
    result = runner.invoke(app)
    assert result.exit_code == 0
    assert "Usage:" in result.stdout


@pytest.mark.parametrize(
    ("command", "task", "default_prompt"),
    [
        ("run-generation-child", "generation-child", "decodable_text_generation"),
        (
            "run-generation-curriculum",
            "generation-curriculum",
            "decodable_text_generation_curriculum",
        ),
    ],
)
def test_run_commands_set_task_and_generation_defaults(
    runner: CliRunner,
    tmp_path: Path,
    command: str,
    task: str,
    default_prompt: str,
) -> None:
    items_path = tmp_path / "items.jsonl"
    items_path.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    config_path = _write_config(tmp_path, items_path=items_path)

    with (
        patch(
            "decodable_text_generation.cmd.download_default_custom_models"
        ) as download_models,
        patch(
            "decodable_text_generation.cmd._execute_with_optional_sync",
            return_value=_fake_output(tmp_path / "outputs" / "experiment-a"),
        ) as run,
    ):
        result = runner.invoke(app, [command, "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    download_models.assert_called_once_with()
    run.assert_called_once()
    run_config = run.call_args.args[0]
    assert run_config.task == task
    assert run_config.prompt_name == default_prompt
    assert run.call_args.kwargs["prompt_builder"] is prompt_from_item
    # The item loader is the generation loader bound to the right variant.
    item_loader = run.call_args.kwargs["item_loader"]
    assert item_loader.func is load_generation_items
    assert item_loader.keywords["curriculum"] == (task == "generation-curriculum")


def test_run_command_curriculum_loader_requires_target(
    runner: CliRunner, tmp_path: Path
) -> None:
    """The curriculum command's item loader rejects items without a target."""
    items_path = tmp_path / "items.jsonl"
    items_path.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    config_path = _write_config(tmp_path, items_path=items_path)

    with (
        patch("decodable_text_generation.cmd.download_default_custom_models"),
        patch(
            "decodable_text_generation.cmd._execute_with_optional_sync",
            return_value=_fake_output(tmp_path / "outputs" / "experiment-a"),
        ) as run,
    ):
        result = runner.invoke(
            app, ["run-generation-curriculum", "--config", str(config_path)]
        )

    assert result.exit_code == 0, result.output
    item_loader = run.call_args.kwargs["item_loader"]
    with pytest.raises(ConfigError, match="target"):
        item_loader(items_path)


def test_run_end_to_end_with_fake_evaluator(tmp_path: Path) -> None:
    """A full run: items load, the baked prompt is used, rows record the task."""
    items_path = tmp_path / "items.jsonl"
    items_path.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    config_path = _write_config(tmp_path, items_path=items_path)

    prompts_seen: list[str] = []

    def evaluator(_config, _model, _item, prompt: str) -> dict:
        prompts_seen.append(prompt)
        return {"output": "<text>the cat sat</text>"}

    output = run_from_config_file(
        config_path,
        task="generation-child",
        default_prompt_name="decodable_text_generation",
        evaluator=evaluator,
        item_loader=load_generation_items,
        prompt_builder=prompt_from_item,
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
    assert {row["benchmark_item_id"] for row in rows} == {
        "ufli-en-l008_words",
        "child-jagged-01_story",
    }
