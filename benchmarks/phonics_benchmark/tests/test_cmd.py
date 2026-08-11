"""Test cases for the command-line interface."""

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from phonics_benchmark.cmd import _load_local_run_config, _timestamp_run_id, app
from phonics_benchmark.config import ConfigError
from typer.testing import CliRunner

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _unstyle(value: str) -> str:
    """Remove terminal control sequences without depending on external Click."""

    return ANSI_ESCAPE_RE.sub("", value)


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
                "language": "sw",
                "items_path": "items.jsonl",
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
        patch("phonics_benchmark.cmd.execute_run", return_value=output) as run,
    ):
        result = runner.invoke(app, ["run-decodability", "--config", str(config_path)])

    assert result.exit_code == 0
    download_models.assert_called_once_with()
    run.assert_called_once()
    run_config = run.call_args.args[0]
    assert run_config.models == ["model-a"]
    assert run_config.items_path == (tmp_path / "items.jsonl").resolve()
    assert run_config.inputs_dir is None
    assert not (tmp_path / "inputs").exists()
    assert "Run ID: experiment-a" in result.stdout
    assert "Results:" in result.stdout


def test_run_decodability_materializes_gcs_items_with_local_results(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """A local config can read GCS inputs without enabling result uploads."""

    items_uri = "gs://phonics-benchmark/phonics/datasets/items-v1.jsonl"
    config_path = tmp_path / "run.json"
    config_path.write_text(
        json.dumps(
            {
                "run_id": "experiment-a",
                "language": "sw",
                "items_path": items_uri,
                "output_dir": "outputs",
                "models": ["model-a"],
            }
        ),
        encoding="utf-8",
    )
    output = _run_output(tmp_path / "outputs", "experiment-a", total=1)

    def fake_download(**kwargs) -> int:
        Path(kwargs["write_path"]).write_text(
            json.dumps(
                {
                    "benchmark_item_id": "item-1",
                    "content": "Mama.",
                    "student_knowledge": {"graphemes": ["m", "a"]},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return 42

    with (
        patch(
            "phonics_benchmark.cmd.download_file_with_generation",
            side_effect=fake_download,
        ) as download_items,
        patch("phonics_benchmark.cmd.download_default_custom_models"),
        patch("phonics_benchmark.cmd.execute_run", return_value=output) as run,
    ):
        result = runner.invoke(
            app,
            ["run-decodability", "--config", str(config_path)],
        )

    assert result.exit_code == 0
    call = download_items.call_args.kwargs
    assert (call["bucket_name"], call["file_name"]) == (
        "phonics-benchmark",
        "phonics/datasets/items-v1.jsonl",
    )
    run_config = run.call_args.args[0]
    assert run_config.items_path.read_text(encoding="utf-8").endswith("\n")
    assert run_config.output_dir == (tmp_path / "outputs").resolve()
    assert run_config.items_path == (
        tmp_path / "inputs" / "experiment-a" / "benchmark-items.jsonl"
    )
    assert not run_config.items_path.is_relative_to(
        tmp_path / "outputs" / "experiment-a"
    )
    assert run_config.inputs_dir == tmp_path / "inputs"
    assert run_config.results_dest is None
    assert run_config.metadata == {
        "items_source_uri": items_uri,
        "items_source_generation": 42,
    }
    assert run_config.raw_config["items_path"] == items_uri
    assert run_config.config_path == config_path


def test_gcs_items_honor_configured_inputs_directory(tmp_path: Path) -> None:
    config_path = tmp_path / "run.json"
    config_path.write_text(
        json.dumps(
            {
                "run_id": "experiment-a",
                "language": "sw",
                "items_path": "gs://datasets/items.jsonl",
                "inputs_dir": "downloaded-inputs",
                "env_path": "secrets/.env",
                "custom_models_path": "models/custom_models.yaml",
                "models": ["model-a"],
            }
        ),
        encoding="utf-8",
    )

    def fake_download(**kwargs) -> int:
        Path(kwargs["write_path"]).write_text("items\n", encoding="utf-8")
        return 7

    with patch(
        "phonics_benchmark.cmd.download_file_with_generation",
        side_effect=fake_download,
    ):
        config = _load_local_run_config(config_path)

    assert config.inputs_dir == (tmp_path / "downloaded-inputs").resolve()
    assert config.output_dir == (tmp_path / "results").resolve()
    assert config.env_path == (tmp_path / "secrets/.env").resolve()
    assert (
        config.custom_models_path == (tmp_path / "models/custom_models.yaml").resolve()
    )
    assert (
        config.items_path
        == (
            tmp_path / "downloaded-inputs" / "experiment-a" / "benchmark-items.jsonl"
        ).resolve()
    )


def test_run_decodability_command_fails_when_calls_have_errors(
    runner: CliRunner, tmp_path: Path
) -> None:
    """The command reports a failing status when any model call failed."""

    config_path = tmp_path / "run.json"
    config_path.write_text(
        json.dumps(
            {
                "run_id": "experiment-a",
                "language": "sw",
                "items_path": "items.jsonl",
                "models": ["model-a"],
            }
        ),
        encoding="utf-8",
    )
    output = _run_output(tmp_path / "outputs", "experiment-a", total=2, errors=1)

    with (
        patch("phonics_benchmark.cmd.download_default_custom_models"),
        patch("phonics_benchmark.cmd.execute_run", return_value=output),
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
                "language": "sw",
                "items_path": "items.jsonl",
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
        patch("phonics_benchmark.cmd.execute_run", return_value=output) as run,
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
                "language": "sw",
                "items_path": "items.jsonl",
                "models": ["model-a"],
                "model_workers": 3,
            }
        ),
        encoding="utf-8",
    )
    output = _run_output(tmp_path / "outputs", "experiment-a", total=1)

    with (
        patch("phonics_benchmark.cmd.download_default_custom_models"),
        patch("phonics_benchmark.cmd.execute_run", return_value=output) as run,
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
                "language": "sw",
                "items_path": "items.jsonl",
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
    items_path = _write_items_jsonl(tmp_path)
    custom_models_path = tmp_path / "custom_models.yaml"
    custom_models_path.write_text(
        '{"provider-a":{"models":{"model-a":{},"model-b":{}}}}',
        encoding="utf-8",
    )
    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_API_KEY=test", encoding="utf-8")
    output = _run_output(output_dir, "experiment-a", total=2)

    with (
        patch(
            "phonics_benchmark.cmd.download_default_custom_models"
        ) as download_models,
        patch("phonics_benchmark.cmd.execute_run", return_value=output) as run,
    ):
        result = runner.invoke(
            app,
            [
                "run-decodability",
                "--run-id",
                "experiment-a",
                "--language",
                "sw",
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
    run.assert_called_once()
    run_config = run.call_args.args[0]
    assert run_config.run_id == "experiment-a"
    assert run_config.language == "sw"
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
    assert "Run ID: experiment-a" in result.stdout


def test_run_decodability_command_requires_cli_args_without_config(
    runner: CliRunner,
) -> None:
    """Direct runs use the shared config validation for required fields."""

    with patch(
        "phonics_benchmark.cmd.download_default_custom_models"
    ) as download_models:
        result = runner.invoke(app, ["run-decodability"])

    assert result.exit_code == 1
    download_models.assert_not_called()
    assert "Missing required run fields: language, models, items_path" in result.stderr


def test_run_decodability_command_preserves_missing_items_path_for_validation(
    runner: CliRunner,
) -> None:
    """A missing CLI path remains None instead of becoming a path named 'None'."""

    with patch(
        "phonics_benchmark.cmd.download_default_custom_models"
    ) as download_models:
        result = runner.invoke(
            app,
            [
                "run-decodability",
                "--language",
                "sw",
                "--models",
                "model-a",
            ],
        )

    assert result.exit_code == 1
    download_models.assert_not_called()
    assert "Missing required run fields: items_path" in result.stderr


@pytest.mark.parametrize(
    ("models", "error"),
    [
        ("model-a,model-a", "duplicate: model-a"),
        ("model-a,,model-b", "non-empty strings"),
        ('["model-a", 1]', "string name"),
    ],
)
def test_run_decodability_command_uses_config_model_validation(
    runner: CliRunner,
    tmp_path: Path,
    models: str,
    error: str,
) -> None:
    """CLI model syntax is translated without bypassing config semantics."""

    items_path = _write_items_jsonl(tmp_path)
    with patch(
        "phonics_benchmark.cmd.download_default_custom_models"
    ) as download_models:
        result = runner.invoke(
            app,
            [
                "run-decodability",
                "--language",
                "sw",
                "--models",
                models,
                "--items-path",
                str(items_path),
            ],
        )

    assert result.exit_code == 1
    download_models.assert_not_called()
    assert error in result.stderr


def test_run_decodability_command_defaults_run_id(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Direct CLI runs can omit --run-id."""

    output = _run_output(tmp_path / "results", "default", total=1)
    items_path = _write_items_jsonl(tmp_path)

    with (
        patch("phonics_benchmark.cmd.download_default_custom_models"),
        patch("phonics_benchmark.cmd.execute_run", return_value=output) as run,
    ):
        result = runner.invoke(
            app,
            [
                "run-decodability",
                "--language",
                "sw",
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
    assert run_config.chunk_size == 20
    assert run_config.model_workers == 4


def test_run_decodability_command_timestamped_output_generates_run_id(
    runner: CliRunner, tmp_path: Path
) -> None:
    """--timestamped-output fills run_id when one is not supplied."""

    output = _run_output(tmp_path / "results", "20260622T120000Z", total=1)
    items_path = _write_items_jsonl(tmp_path)

    with (
        patch("phonics_benchmark.cmd._timestamp_run_id", return_value=output["run_id"]),
        patch("phonics_benchmark.cmd.download_default_custom_models"),
        patch("phonics_benchmark.cmd.execute_run", return_value=output) as run,
    ):
        result = runner.invoke(
            app,
            [
                "run-decodability",
                "--language",
                "sw",
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

    items_path = _write_items_jsonl(tmp_path)
    with patch(
        "phonics_benchmark.cmd.download_default_custom_models"
    ) as download_models:
        result = runner.invoke(
            app,
            [
                "run-decodability",
                "--language",
                "sw",
                "--models",
                "model-a",
                "--items-path",
                str(items_path),
                "--max-tokens",
                max_tokens,
            ],
        )

    assert result.exit_code == 2
    assert "--max-tokens" in _unstyle(result.output)
    download_models.assert_not_called()


def test_run_decodability_command_accepts_models_json_list(
    runner: CliRunner, tmp_path: Path
) -> None:
    """The --models option also accepts a JSON list string."""

    output = _run_output(tmp_path / "outputs", "experiment-a", total=2)
    items_path = _write_items_jsonl(tmp_path)

    with (
        patch("phonics_benchmark.cmd.download_default_custom_models"),
        patch("phonics_benchmark.cmd.execute_run", return_value=output) as run,
    ):
        result = runner.invoke(
            app,
            [
                "run-decodability",
                "--run-id",
                "experiment-a",
                "--language",
                "sw",
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
    """Model registration errors are reported before benchmark execution."""

    items_path = _write_items_jsonl(tmp_path)
    with (
        patch(
            "phonics_benchmark.cmd.download_default_custom_models"
        ) as download_models,
        patch(
            "phonics_benchmark.cmd.execute_run",
            side_effect=ConfigError("Configured model(s) are not registered: model-a"),
        ) as run,
    ):
        result = runner.invoke(
            app,
            [
                "run-decodability",
                "--language",
                "sw",
                "--models",
                "model-a",
                "--items-path",
                str(items_path),
            ],
        )

    assert result.exit_code == 1
    download_models.assert_called_once_with()
    run.assert_called_once()
    assert "not registered" in result.stderr


def test_run_decodability_command_can_skip_models_download(
    runner: CliRunner, tmp_path: Path
) -> None:
    """The operational CLI flag suppresses only the shared model-file refresh."""

    output = _run_output(tmp_path / "results", "default", total=1)
    items_path = _write_items_jsonl(tmp_path)

    with (
        patch(
            "phonics_benchmark.cmd.download_default_custom_models"
        ) as download_models,
        patch("phonics_benchmark.cmd.execute_run", return_value=output) as run,
    ):
        result = runner.invoke(
            app,
            [
                "run-decodability",
                "--language",
                "sw",
                "--models",
                "model-a",
                "--items-path",
                str(items_path),
                "--skip-models-download",
            ],
        )

    assert result.exit_code == 0
    download_models.assert_not_called()
    run.assert_called_once()


def test_run_decodability_help_does_not_download_models(runner: CliRunner) -> None:
    """Rendering command help does not refresh the shared model registry."""

    with patch(
        "phonics_benchmark.cmd.download_default_custom_models"
    ) as download_models:
        result = runner.invoke(app, ["run-decodability", "--help"])

    assert result.exit_code == 0
    download_models.assert_not_called()
    help_text = _unstyle(result.output)
    assert "JSONL containing" in help_text
    assert "completed benchmark" in help_text
    assert "provider groups to run" in help_text


def _write_items_jsonl(tmp_path: Path) -> Path:
    items_path = tmp_path / "items.jsonl"
    items_path.write_text(
        json.dumps(
            {
                "benchmark_item_id": "item-1",
                "content": "Mama.",
                "student_knowledge": {
                    "graphemes": ["a", "m"],
                    "whole_words": ["mama"],
                },
            }
        )
        + "\n",
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
        "results_dir": run_dir / "responses",
        "results_paths": {
            "model-a": run_dir / "responses" / "responses_model-a.jsonl",
            "model-b": run_dir / "responses" / "responses_model-b.jsonl",
        },
        "total_expected_results": total,
        "executed_count": total,
        "skipped_count": 0,
        "error_count": errors,
    }
