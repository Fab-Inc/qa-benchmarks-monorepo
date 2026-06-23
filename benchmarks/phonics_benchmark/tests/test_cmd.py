"""Test cases for the command-line interface."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from phonics_benchmark.cmd import DEFAULT_DATA_ROOT, app
from phonics_benchmark.config import ConfigError
from typer.testing import CliRunner


@pytest.fixture
def runner() -> CliRunner:
    """Fixture for invoking command-line interfaces."""
    return CliRunner()


def test_main_succeeds(runner: CliRunner) -> None:
    """It shows help when invoked without a command."""

    result = runner.invoke(app)
    assert result.exit_code == 0
    assert "Usage:" in result.stdout


def test_run_command_uses_config(runner: CliRunner, tmp_path: Path) -> None:
    """The run command delegates to the config-driven runner."""

    config_path = tmp_path / "run.json"
    config_path.write_text(
        json.dumps(
            {
                "run_id": "experiment-a",
                "language": "english",
                "models": ["model-a"],
            }
        ),
        encoding="utf-8",
    )
    output = _run_output(tmp_path / "outputs", "experiment-a", total=2)

    with (
        patch("phonics_benchmark.cmd.prepare_fdllm_for_run") as prepare,
        patch(
            "phonics_benchmark.cmd.snapshot_model_configs",
            return_value={"model-a": {"Api_Interface": "OpenAI"}},
        ) as snapshot,
        patch("phonics_benchmark.cmd.run_benchmark", return_value=output) as run,
    ):
        result = runner.invoke(app, ["run", "--config", str(config_path)])

    assert result.exit_code == 0
    prepare.assert_called_once()
    snapshot.assert_called_once_with(["model-a"])
    run.assert_called_once()
    run_config = run.call_args.args[0]
    assert run_config.models == ["model-a"]
    assert run_config.data_root == DEFAULT_DATA_ROOT.resolve()
    assert "fdllm_context" not in run.call_args.kwargs
    assert run.call_args.kwargs["model_configs"] == {
        "model-a": {"Api_Interface": "OpenAI"}
    }
    assert "Run ID: experiment-a" in result.stdout
    assert "Results:" in result.stdout


def test_run_command_uses_cli_args(runner: CliRunner, tmp_path: Path) -> None:
    """The run command can be driven directly by command-line options."""

    output_dir = tmp_path / "outputs"
    data_root = tmp_path / "data"
    data_root.mkdir()
    custom_models_path = tmp_path / "custom_models.yaml"
    custom_models_path.write_text("models: []", encoding="utf-8")
    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_API_KEY=test", encoding="utf-8")
    output = _run_output(output_dir, "experiment-a", total=2)

    with (
        patch("phonics_benchmark.cmd.prepare_fdllm_for_run") as prepare,
        patch(
            "phonics_benchmark.cmd.snapshot_model_configs",
            return_value={"model-a": {}, "model-b": {}},
        ) as snapshot,
        patch("phonics_benchmark.cmd.run_benchmark", return_value=output) as run,
    ):
        result = runner.invoke(
            app,
            [
                "run",
                "--run-id",
                "experiment-a",
                "--language",
                "english",
                "--models",
                "model-a,model-b",
                "--data-root",
                str(data_root),
                "--output-dir",
                str(output_dir),
                "--task",
                "decodability",
                "--env-path",
                str(env_path),
                "--custom-models-path",
                str(custom_models_path),
                "--max-tokens",
                "100",
                "--chunk-size",
                "25",
                "--force",
            ],
        )

    assert result.exit_code == 0
    prepare.assert_called_once()
    snapshot.assert_called_once_with(["model-a", "model-b"])
    run.assert_called_once()
    run_config = run.call_args.args[0]
    assert run_config.run_id == "experiment-a"
    assert run_config.language == "english"
    assert run_config.data_root == data_root.resolve()
    assert run_config.output_dir == output_dir.resolve()
    assert run_config.models == ["model-a", "model-b"]
    assert run_config.task == "decodability"
    assert run_config.env_path == env_path.resolve()
    assert run_config.custom_models_path == custom_models_path.resolve()
    assert run_config.max_tokens == 100
    assert run_config.chunk_size == 25
    assert run_config.force is True
    assert "config_id" in run.call_args.kwargs
    assert "Run ID: experiment-a" in result.stdout


def test_run_command_requires_cli_args_without_config(runner: CliRunner) -> None:
    """The CLI reports missing direct-run options when no config is supplied."""

    result = runner.invoke(app, ["run"])

    assert result.exit_code == 1
    assert "Missing required options" in result.stderr


def test_run_command_defaults_run_id(runner: CliRunner, tmp_path: Path) -> None:
    """Direct CLI runs can omit --run-id."""

    output = _run_output(tmp_path / "results", "default", total=1)

    with (
        patch("phonics_benchmark.cmd.prepare_fdllm_for_run"),
        patch("phonics_benchmark.cmd.snapshot_model_configs", return_value={}),
        patch("phonics_benchmark.cmd.run_benchmark", return_value=output) as run,
    ):
        result = runner.invoke(
            app,
            [
                "run",
                "--language",
                "english",
                "--models",
                "model-a",
            ],
        )

    assert result.exit_code == 0
    run_config = run.call_args.args[0]
    assert run_config.run_id == "default"
    assert run_config.data_root == DEFAULT_DATA_ROOT.resolve()


def test_run_command_timestamped_output_generates_run_id(
    runner: CliRunner, tmp_path: Path
) -> None:
    """--timestamped-output fills run_id when one is not supplied."""

    output = _run_output(tmp_path / "results", "20260622T120000Z", total=1)

    with (
        patch("phonics_benchmark.cmd._timestamp_run_id", return_value=output["run_id"]),
        patch("phonics_benchmark.cmd.prepare_fdllm_for_run"),
        patch("phonics_benchmark.cmd.snapshot_model_configs", return_value={}),
        patch("phonics_benchmark.cmd.run_benchmark", return_value=output) as run,
    ):
        result = runner.invoke(
            app,
            [
                "run",
                "--language",
                "english",
                "--models",
                "model-a",
                "--timestamped-output",
            ],
        )

    assert result.exit_code == 0
    run_config = run.call_args.args[0]
    assert run_config.run_id == "20260622T120000Z"


def test_run_command_accepts_models_json_list(
    runner: CliRunner, tmp_path: Path
) -> None:
    """The --models option also accepts a JSON list string."""

    output = _run_output(tmp_path / "outputs", "experiment-a", total=2)

    with (
        patch("phonics_benchmark.cmd.prepare_fdllm_for_run"),
        patch("phonics_benchmark.cmd.snapshot_model_configs", return_value={}),
        patch("phonics_benchmark.cmd.run_benchmark", return_value=output) as run,
    ):
        result = runner.invoke(
            app,
            [
                "run",
                "--run-id",
                "experiment-a",
                "--language",
                "english",
                "--models",
                '["model-a", "model-b"]',
            ],
        )

    assert result.exit_code == 0
    run_config = run.call_args.args[0]
    assert run_config.models == ["model-a", "model-b"]


def test_run_command_preflights_registered_models(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Model registration errors are reported before the runner starts."""

    with (
        patch(
            "phonics_benchmark.cmd.prepare_fdllm_for_run",
            side_effect=ConfigError("Configured model(s) are not registered: model-a"),
        ) as prepare,
        patch("phonics_benchmark.cmd.run_benchmark") as run,
    ):
        result = runner.invoke(
            app,
            [
                "run",
                "--language",
                "english",
                "--models",
                "model-a",
            ],
        )

    assert result.exit_code == 1
    prepare.assert_called_once()
    run.assert_not_called()
    assert "not registered" in result.stderr


def _run_output(output_dir: Path, run_id: str, *, total: int) -> dict:
    run_dir = output_dir / run_id
    return {
        "run_id": run_id,
        "config_id": "abc123",
        "run_dir": run_dir,
        "config_snapshot_path": run_dir / "config.json",
        "results_dir": run_dir / "answers",
        "results_paths": {
            "model-a": run_dir / "answers" / "resps_model-a.jsonl",
            "model-b": run_dir / "answers" / "resps_model-b.jsonl",
        },
        "total_expected_results": total,
        "executed_count": total,
        "skipped_count": 0,
        "error_count": 0,
    }
