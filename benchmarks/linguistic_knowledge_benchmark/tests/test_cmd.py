"""Test cases for the command-line interface."""

from unittest.mock import MagicMock

import pandas as pd
import pytest
from conftest import VALID_MODEL
from linguistic_knowledge_benchmark import cmd
from typer.testing import CliRunner


@pytest.fixture
def runner() -> CliRunner:
    """Fixture for invoking command-line interfaces."""
    return CliRunner()


# --------------------------------------------------------------------------- #
# help / structure
# --------------------------------------------------------------------------- #
def test_help_lists_commands(runner: CliRunner) -> None:
    result = runner.invoke(cmd.app, ["--help"])
    assert result.exit_code == 0
    assert "run-bench" in result.output
    assert "judge-bench" in result.output
    assert "run-llpk-eval" not in result.output


def test_datasets_map_is_parameterized() -> None:
    for entry in cmd.DATASETS.values():
        assert {"path", "language_name"} <= entry.keys()


# --------------------------------------------------------------------------- #
# run-bench
# --------------------------------------------------------------------------- #
def test_run_bench_rejects_unknown_language(runner: CliRunner) -> None:
    result = runner.invoke(cmd.app, ["run-bench", "--language", "klingon"])
    assert result.exit_code == 1
    assert "Invalid language" in result.output


def test_run_bench_rejects_unknown_model(runner: CliRunner) -> None:
    result = runner.invoke(
        cmd.app, ["run-bench", "--model", "bogus", "--language", "luganda"]
    )
    assert result.exit_code == 1
    assert "not found" in result.output


def test_run_bench_rejects_invalid_run_id(runner: CliRunner) -> None:
    result = runner.invoke(
        cmd.app,
        [
            "run-bench",
            "--model",
            VALID_MODEL,
            "--language",
            "luganda",
            "--run-id",
            "my_run",
        ],
    )
    assert result.exit_code == 1
    assert "invalid" in result.output.lower()


def test_run_bench_single_model_forwards_params(runner, monkeypatch) -> None:
    mock = MagicMock()
    monkeypatch.setattr(cmd, "run_benchmark", mock)
    result = runner.invoke(
        cmd.app, ["run-bench", "--model", VALID_MODEL, "--language", "luganda"]
    )
    assert result.exit_code == 0
    mock.assert_called_once()
    kwargs = mock.call_args.kwargs
    assert kwargs["model_name"] == VALID_MODEL
    assert kwargs["language_name"] == "Luganda"
    assert kwargs["retry_errors"] is False


def test_run_bench_all_models(runner, monkeypatch) -> None:
    mock = MagicMock()
    monkeypatch.setattr(cmd, "run_benchmark", mock)
    result = runner.invoke(cmd.app, ["run-bench", "--language", "luganda"])
    assert result.exit_code == 0
    assert mock.call_count == len(cmd.EVAL_MODELS)


def test_run_bench_forwards_retry_and_run_options(runner, monkeypatch) -> None:
    mock = MagicMock()
    monkeypatch.setattr(cmd, "run_benchmark", mock)
    result = runner.invoke(
        cmd.app,
        [
            "run-bench",
            "--model",
            VALID_MODEL,
            "--language",
            "luganda",
            "--retry-errors",
            "--run-id",
            "run-123",
        ],
    )
    assert result.exit_code == 0
    kwargs = mock.call_args.kwargs
    assert kwargs["retry_errors"] is True
    assert kwargs["run_id"] == "run-123"


def test_run_bench_timestamped_output_generates_run_id(runner, monkeypatch) -> None:
    mock = MagicMock()
    monkeypatch.setattr(cmd, "run_benchmark", mock)
    monkeypatch.setattr(cmd, "_timestamp_run_id", lambda: "20260608T120000Z")
    result = runner.invoke(
        cmd.app,
        [
            "run-bench",
            "--model",
            VALID_MODEL,
            "--language",
            "luganda",
            "--timestamped-output",
        ],
    )
    assert result.exit_code == 0
    assert "Run ID: 20260608T120000Z" in result.output
    assert mock.call_args.kwargs["run_id"] == "20260608T120000Z"


def test_run_bench_rejects_force_with_retry_errors(runner) -> None:
    result = runner.invoke(
        cmd.app,
        [
            "run-bench",
            "--model",
            VALID_MODEL,
            "--language",
            "luganda",
            "--force",
            "--retry-errors",
        ],
    )
    assert result.exit_code == 1
    assert "cannot be combined" in result.output


def test_run_bench_propagates_error(runner, monkeypatch) -> None:
    monkeypatch.setattr(
        cmd, "run_benchmark", MagicMock(side_effect=RuntimeError("kaboom"))
    )
    result = runner.invoke(
        cmd.app, ["run-bench", "--model", VALID_MODEL, "--language", "luganda"]
    )
    assert result.exit_code == 1
    assert "ERROR" in result.output


# --------------------------------------------------------------------------- #
# judge-bench
# --------------------------------------------------------------------------- #
def _patch_summary(monkeypatch):
    monkeypatch.setattr(
        cmd,
        "compute_summary",
        lambda df, model: {
            "model": model,
            "accuracy": 80.0,
            "accuracy_mc": 90.0,
            "accuracy_short_form": 70.0,
            "accuracy_true_false": 100.0,
        },
    )


def test_judge_bench_rejects_unknown_language(runner: CliRunner) -> None:
    result = runner.invoke(cmd.app, ["judge-bench", "--language", "klingon"])
    assert result.exit_code == 1
    assert "Invalid language" in result.output


def test_judge_bench_rejects_unknown_model(runner: CliRunner) -> None:
    result = runner.invoke(
        cmd.app, ["judge-bench", "--model", "bogus", "--language", "luganda"]
    )
    assert result.exit_code == 1
    assert "not found" in result.output


def test_judge_bench_rejects_invalid_run_id(runner: CliRunner) -> None:
    result = runner.invoke(
        cmd.app,
        [
            "judge-bench",
            "--model",
            VALID_MODEL,
            "--language",
            "luganda",
            "--run-id",
            "my_run",
        ],
    )
    assert result.exit_code == 1
    assert "invalid" in result.output.lower()


def test_judge_bench_standard_writes_summary(runner, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cmd, "BENCH_DIR", tmp_path)
    monkeypatch.setattr(cmd, "judge_results", MagicMock(return_value=pd.DataFrame()))
    _patch_summary(monkeypatch)
    result = runner.invoke(
        cmd.app, ["judge-bench", "--model", VALID_MODEL, "--language", "luganda"]
    )
    assert result.exit_code == 0
    assert (
        tmp_path / "results" / "summary_linguistic-knowledge_luganda_default.csv"
    ).exists()


def test_judge_bench_run_id_writes_run_specific_summary(
    runner, monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(cmd, "BENCH_DIR", tmp_path)
    judge = MagicMock(return_value=pd.DataFrame())
    monkeypatch.setattr(cmd, "judge_results", judge)
    _patch_summary(monkeypatch)
    result = runner.invoke(
        cmd.app,
        [
            "judge-bench",
            "--model",
            VALID_MODEL,
            "--language",
            "luganda",
            "--run-id",
            "run-123",
        ],
    )
    assert result.exit_code == 0
    assert "Run ID: run-123" in result.output
    assert judge.call_args.kwargs["run_id"] == "run-123"
    assert (
        tmp_path / "results" / "summary_linguistic-knowledge_luganda_run-123.csv"
    ).exists()


def test_judge_bench_all_models(runner, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cmd, "BENCH_DIR", tmp_path)
    judge = MagicMock(return_value=pd.DataFrame())
    monkeypatch.setattr(cmd, "judge_results", judge)
    _patch_summary(monkeypatch)
    result = runner.invoke(cmd.app, ["judge-bench", "--language", "luganda"])
    assert result.exit_code == 0
    assert judge.call_count == len(cmd.EVAL_MODELS)


def test_judge_bench_fallback_path(runner, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cmd, "BENCH_DIR", tmp_path)
    fallback = MagicMock(return_value=pd.DataFrame())
    monkeypatch.setattr(cmd, "judge_results_fallback", fallback)
    _patch_summary(monkeypatch)
    result = runner.invoke(
        cmd.app,
        [
            "judge-bench",
            "--model",
            VALID_MODEL,
            "--language",
            "luganda",
            "--judge-fallback",
        ],
    )
    assert result.exit_code == 0
    fallback.assert_called_once()
    name = "summary_linguistic-knowledge_luganda_default_fallback.csv"
    assert (tmp_path / "results" / name).exists()


def test_judge_bench_skips_on_missing_results(runner, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cmd, "BENCH_DIR", tmp_path)
    monkeypatch.setattr(
        cmd, "judge_results", MagicMock(side_effect=FileNotFoundError("no file"))
    )
    result = runner.invoke(
        cmd.app, ["judge-bench", "--model", VALID_MODEL, "--language", "luganda"]
    )
    assert result.exit_code == 0
    assert not (tmp_path / "results").exists()


def test_judge_bench_propagates_error(runner, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cmd, "BENCH_DIR", tmp_path)
    monkeypatch.setattr(
        cmd, "judge_results", MagicMock(side_effect=RuntimeError("kaboom"))
    )
    result = runner.invoke(
        cmd.app, ["judge-bench", "--model", VALID_MODEL, "--language", "luganda"]
    )
    assert result.exit_code == 1
    assert "ERROR" in result.output


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #
def test_main_invokes_app(monkeypatch) -> None:
    mock_app = MagicMock()
    monkeypatch.setattr(cmd, "app", mock_app)
    cmd.main()
    mock_app.assert_called_once()
