"""Test cases for the command-line interface."""

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from phonics_benchmark.cmd import _timestamp_run_id, app
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


def test_run_decodability_command_uses_config(
    runner: CliRunner, tmp_path: Path
) -> None:
    """The decodability command delegates to the config-driven runner."""

    config_path = tmp_path / "run.json"
    config_path.write_text(
        json.dumps(
            {
                "run_id": "experiment-a",
                "language": "english",
                "items_path": "items.csv",
                "models": ["model-a"],
            }
        ),
        encoding="utf-8",
    )
    output = _run_output(tmp_path / "outputs", "experiment-a", total=2)

    with (
        patch(
            "phonics_benchmark.cmd.download_default_custom_models"
        ) as download_models,
        patch("phonics_benchmark.cmd.prepare_fdllm_for_run") as prepare,
        patch(
            "phonics_benchmark.cmd.snapshot_model_configs",
            return_value={"model-a": {"Api_Interface": "OpenAI"}},
        ) as snapshot,
        patch("phonics_benchmark.cmd.run_benchmark", return_value=output) as run,
    ):
        result = runner.invoke(app, ["run-decodability", "--config", str(config_path)])

    assert result.exit_code == 0
    download_models.assert_called_once_with()
    prepare.assert_called_once()
    snapshot.assert_called_once_with(["model-a"])
    run.assert_called_once()
    run_config = run.call_args.args[0]
    assert run_config.models == ["model-a"]
    assert run_config.items_path == (tmp_path / "items.csv").resolve()
    assert "fdllm_context" not in run.call_args.kwargs
    assert run.call_args.kwargs["model_configs"] == {
        "model-a": {"Api_Interface": "OpenAI"}
    }
    assert "Run ID: experiment-a" in result.stdout
    assert "Results:" in result.stdout


def test_run_decodability_command_fails_when_calls_have_errors(
    runner: CliRunner, tmp_path: Path
) -> None:
    """The command reports a failing status when any model call failed."""

    config_path = tmp_path / "run.json"
    config_path.write_text(
        json.dumps(
            {
                "run_id": "experiment-a",
                "language": "english",
                "items_path": "items.csv",
                "models": ["model-a"],
            }
        ),
        encoding="utf-8",
    )
    output = _run_output(tmp_path / "outputs", "experiment-a", total=2, errors=1)

    with (
        patch("phonics_benchmark.cmd.download_default_custom_models"),
        patch("phonics_benchmark.cmd.prepare_fdllm_for_run"),
        patch("phonics_benchmark.cmd.snapshot_model_configs", return_value={}),
        patch("phonics_benchmark.cmd.run_benchmark", return_value=output),
    ):
        result = runner.invoke(app, ["run-decodability", "--config", str(config_path)])

    assert result.exit_code == 1
    assert "Errors: 1" in result.stdout


@pytest.mark.parametrize(
    ("flag", "expected_force", "expected_retry_errors"),
    [
        ("--force", True, False),
        ("--retry-errors", False, True),
    ],
)
def test_run_decodability_command_execution_flags_override_config(
    runner: CliRunner,
    tmp_path: Path,
    flag: str,
    expected_force: bool,
    expected_retry_errors: bool,
) -> None:
    """CLI execution flags take precedence when a config file is supplied."""

    config_path = tmp_path / "run.json"
    config_path.write_text(
        json.dumps(
            {
                "run_id": "experiment-a",
                "language": "english",
                "items_path": "items.csv",
                "models": ["model-a"],
                "force": False,
                "retry_errors": False,
            }
        ),
        encoding="utf-8",
    )
    output = _run_output(tmp_path / "outputs", "experiment-a", total=1)

    with (
        patch("phonics_benchmark.cmd.download_default_custom_models"),
        patch("phonics_benchmark.cmd.prepare_fdllm_for_run"),
        patch("phonics_benchmark.cmd.snapshot_model_configs", return_value={}),
        patch("phonics_benchmark.cmd.run_benchmark", return_value=output) as run,
    ):
        result = runner.invoke(
            app,
            ["run-decodability", "--config", str(config_path), flag],
        )

    assert result.exit_code == 0
    run_config = run.call_args.args[0]
    assert run_config.force is expected_force
    assert run_config.retry_errors is expected_retry_errors


@pytest.mark.parametrize(
    ("cli_args", "expected_model_workers"),
    [
        ([], 3),
        (["--model-workers", "1"], 1),
        (["--model-workers", "2"], 2),
    ],
)
def test_run_decodability_command_model_workers_overrides_config_when_supplied(
    runner: CliRunner,
    tmp_path: Path,
    cli_args: list[str],
    expected_model_workers: int,
) -> None:
    """The config worker count is retained unless the CLI supplies a value."""

    config_path = tmp_path / "run.json"
    config_path.write_text(
        json.dumps(
            {
                "run_id": "experiment-a",
                "language": "english",
                "items_path": "items.csv",
                "models": ["model-a"],
                "model_workers": 3,
            }
        ),
        encoding="utf-8",
    )
    output = _run_output(tmp_path / "outputs", "experiment-a", total=1)

    with (
        patch("phonics_benchmark.cmd.download_default_custom_models"),
        patch("phonics_benchmark.cmd.prepare_fdllm_for_run"),
        patch("phonics_benchmark.cmd.snapshot_model_configs", return_value={}),
        patch("phonics_benchmark.cmd.run_benchmark", return_value=output) as run,
    ):
        result = runner.invoke(
            app,
            ["run-decodability", "--config", str(config_path), *cli_args],
        )

    assert result.exit_code == 0
    assert run.call_args.args[0].model_workers == expected_model_workers


def test_run_decodability_command_rejects_conflicting_config_and_cli_modes(
    runner: CliRunner, tmp_path: Path
) -> None:
    """An execution flag cannot combine with the opposite config mode."""

    config_path = tmp_path / "run.json"
    config_path.write_text(
        json.dumps(
            {
                "run_id": "experiment-a",
                "language": "english",
                "items_path": "items.csv",
                "models": ["model-a"],
                "force": True,
            }
        ),
        encoding="utf-8",
    )

    with patch(
        "phonics_benchmark.cmd.download_default_custom_models"
    ) as download_models:
        result = runner.invoke(
            app,
            [
                "run-decodability",
                "--config",
                str(config_path),
                "--retry-errors",
            ],
        )

    assert result.exit_code == 1
    assert "cannot be combined" in result.stderr
    download_models.assert_not_called()


def test_run_decodability_command_uses_cli_args(
    runner: CliRunner, tmp_path: Path
) -> None:
    """The decodability command accepts direct command-line options."""

    output_dir = tmp_path / "outputs"
    items_path = tmp_path / "items.csv"
    items_path.write_text("benchmark_item_id\nitem-1\n", encoding="utf-8")
    custom_models_path = tmp_path / "custom_models.yaml"
    custom_models_path.write_text("models: []", encoding="utf-8")
    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_API_KEY=test", encoding="utf-8")
    output = _run_output(output_dir, "experiment-a", total=2)

    with (
        patch(
            "phonics_benchmark.cmd.download_default_custom_models"
        ) as download_models,
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
                "run-decodability",
                "--run-id",
                "experiment-a",
                "--language",
                "english",
                "--models",
                "model-a,model-b",
                "--output-dir",
                str(output_dir),
                "--items-path",
                str(items_path),
                "--env-path",
                str(env_path),
                "--custom-models-path",
                str(custom_models_path),
                "--max-tokens",
                "100",
                "--chunk-size",
                "25",
                "--model-workers",
                "2",
                "--force",
            ],
        )

    assert result.exit_code == 0
    download_models.assert_called_once_with()
    prepare.assert_called_once()
    snapshot.assert_called_once_with(["model-a", "model-b"])
    run.assert_called_once()
    run_config = run.call_args.args[0]
    assert run_config.run_id == "experiment-a"
    assert run_config.language == "english"
    assert run_config.output_dir == output_dir.resolve()
    assert run_config.models == ["model-a", "model-b"]
    assert run_config.task == "decodability"
    assert run_config.items_path == items_path.resolve()
    assert run_config.env_path == env_path.resolve()
    assert run_config.custom_models_path == custom_models_path.resolve()
    assert run_config.max_tokens == 100
    assert run_config.chunk_size == 25
    assert run_config.model_workers == 2
    assert run_config.force is True
    assert "config_id" in run.call_args.kwargs
    assert "Run ID: experiment-a" in result.stdout


def test_run_decodability_command_requires_cli_args_without_config(
    runner: CliRunner,
) -> None:
    """The CLI reports missing direct-run options when no config is supplied."""

    with patch(
        "phonics_benchmark.cmd.download_default_custom_models"
    ) as download_models:
        result = runner.invoke(app, ["run-decodability"])

    assert result.exit_code == 1
    download_models.assert_not_called()
    assert "Missing required options" in result.stderr


def test_run_decodability_command_defaults_run_id(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Direct CLI runs can omit --run-id."""

    output = _run_output(tmp_path / "results", "default", total=1)
    items_path = _write_items_csv(tmp_path)

    with (
        patch("phonics_benchmark.cmd.download_default_custom_models"),
        patch("phonics_benchmark.cmd.prepare_fdllm_for_run"),
        patch("phonics_benchmark.cmd.snapshot_model_configs", return_value={}),
        patch("phonics_benchmark.cmd.run_benchmark", return_value=output) as run,
    ):
        result = runner.invoke(
            app,
            [
                "run-decodability",
                "--language",
                "english",
                "--models",
                "model-a",
                "--items-path",
                str(items_path),
            ],
        )

    assert result.exit_code == 0
    run_config = run.call_args.args[0]
    assert run_config.run_id == "default"
    assert run_config.items_path == items_path.resolve()
    assert run_config.model_workers == 1


def test_run_decodability_command_timestamped_output_generates_run_id(
    runner: CliRunner, tmp_path: Path
) -> None:
    """--timestamped-output fills run_id when one is not supplied."""

    output = _run_output(tmp_path / "results", "20260622T120000Z", total=1)
    items_path = _write_items_csv(tmp_path)

    with (
        patch("phonics_benchmark.cmd._timestamp_run_id", return_value=output["run_id"]),
        patch("phonics_benchmark.cmd.download_default_custom_models"),
        patch("phonics_benchmark.cmd.prepare_fdllm_for_run"),
        patch("phonics_benchmark.cmd.snapshot_model_configs", return_value={}),
        patch("phonics_benchmark.cmd.run_benchmark", return_value=output) as run,
    ):
        result = runner.invoke(
            app,
            [
                "run-decodability",
                "--language",
                "english",
                "--models",
                "model-a",
                "--items-path",
                str(items_path),
                "--timestamped-output",
            ],
        )

    assert result.exit_code == 0
    run_config = run.call_args.args[0]
    assert run_config.run_id == "20260622T120000Z"


def test_timestamp_run_id_is_precise_and_randomized() -> None:
    """Timestamped runs include enough entropy to avoid output collisions."""

    with (
        patch("phonics_benchmark.cmd.datetime") as datetime_mock,
        patch(
            "phonics_benchmark.cmd.uuid4",
            return_value=SimpleNamespace(hex="deadbeef" * 4),
        ),
    ):
        datetime_mock.now.return_value = datetime(
            2026,
            7,
            27,
            12,
            34,
            56,
            123456,
            tzinfo=UTC,
        )
        run_id = _timestamp_run_id()

    assert run_id == "20260727T123456123456Z-deadbeef"


@pytest.mark.parametrize("max_tokens", ["0", "-1"])
def test_run_decodability_command_rejects_non_positive_max_tokens(
    runner: CliRunner,
    tmp_path: Path,
    max_tokens: str,
) -> None:
    """Direct CLI runs require a positive response-token limit."""

    items_path = _write_items_csv(tmp_path)
    with patch(
        "phonics_benchmark.cmd.download_default_custom_models"
    ) as download_models:
        result = runner.invoke(
            app,
            [
                "run-decodability",
                "--language",
                "english",
                "--models",
                "model-a",
                "--items-path",
                str(items_path),
                "--max-tokens",
                max_tokens,
            ],
        )

    assert result.exit_code == 2
    assert "--max-tokens" in result.output
    download_models.assert_not_called()


def test_run_decodability_command_accepts_models_json_list(
    runner: CliRunner, tmp_path: Path
) -> None:
    """The --models option also accepts a JSON list string."""

    output = _run_output(tmp_path / "outputs", "experiment-a", total=2)
    items_path = _write_items_csv(tmp_path)

    with (
        patch("phonics_benchmark.cmd.download_default_custom_models"),
        patch("phonics_benchmark.cmd.prepare_fdllm_for_run"),
        patch("phonics_benchmark.cmd.snapshot_model_configs", return_value={}),
        patch("phonics_benchmark.cmd.run_benchmark", return_value=output) as run,
    ):
        result = runner.invoke(
            app,
            [
                "run-decodability",
                "--run-id",
                "experiment-a",
                "--language",
                "english",
                "--models",
                '["model-a", "model-b"]',
                "--items-path",
                str(items_path),
            ],
        )

    assert result.exit_code == 0
    run_config = run.call_args.args[0]
    assert run_config.models == ["model-a", "model-b"]


def test_run_decodability_command_preflights_registered_models(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Model registration errors are reported before the runner starts."""

    items_path = _write_items_csv(tmp_path)
    with (
        patch(
            "phonics_benchmark.cmd.download_default_custom_models"
        ) as download_models,
        patch(
            "phonics_benchmark.cmd.prepare_fdllm_for_run",
            side_effect=ConfigError("Configured model(s) are not registered: model-a"),
        ) as prepare,
        patch("phonics_benchmark.cmd.run_benchmark") as run,
    ):
        result = runner.invoke(
            app,
            [
                "run-decodability",
                "--language",
                "english",
                "--models",
                "model-a",
                "--items-path",
                str(items_path),
            ],
        )

    assert result.exit_code == 1
    download_models.assert_called_once_with()
    prepare.assert_called_once()
    run.assert_not_called()
    assert "not registered" in result.stderr


def test_run_decodability_command_can_skip_models_download(
    runner: CliRunner, tmp_path: Path
) -> None:
    """The operational CLI flag suppresses only the shared model-file refresh."""

    output = _run_output(tmp_path / "results", "default", total=1)
    items_path = _write_items_csv(tmp_path)

    with (
        patch(
            "phonics_benchmark.cmd.download_default_custom_models"
        ) as download_models,
        patch("phonics_benchmark.cmd.prepare_fdllm_for_run") as prepare,
        patch("phonics_benchmark.cmd.snapshot_model_configs", return_value={}),
        patch("phonics_benchmark.cmd.run_benchmark", return_value=output),
    ):
        result = runner.invoke(
            app,
            [
                "run-decodability",
                "--language",
                "english",
                "--models",
                "model-a",
                "--items-path",
                str(items_path),
                "--skip-models-download",
            ],
        )

    assert result.exit_code == 0
    download_models.assert_not_called()
    prepare.assert_called_once()


def test_run_decodability_help_does_not_download_models(runner: CliRunner) -> None:
    """Rendering command help does not refresh the shared model registry."""

    with patch(
        "phonics_benchmark.cmd.download_default_custom_models"
    ) as download_models:
        result = runner.invoke(app, ["run-decodability", "--help"])

    assert result.exit_code == 0
    download_models.assert_not_called()


def _write_items_csv(tmp_path: Path) -> Path:
    items_path = tmp_path / "items.csv"
    items_path.write_text(
        (
            "benchmark_item_id,content,student_knowledge_id,graphemes,whole_words\n"
            'item-1,A cat sat.,student-1,"[""a""]","[]"\n'
        ),
        encoding="utf-8",
    )
    return items_path


def _run_output(
    output_dir: Path,
    run_id: str,
    *,
    total: int,
    errors: int = 0,
) -> dict:
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
        "error_count": errors,
    }
