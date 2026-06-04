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


def test_run_bench_all_models(runner, monkeypatch) -> None:
    mock = MagicMock()
    monkeypatch.setattr(cmd, "run_benchmark", mock)
    result = runner.invoke(cmd.app, ["run-bench", "--language", "luganda"])
    assert result.exit_code == 0
    assert mock.call_count == len(cmd.EVAL_MODELS)


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


def test_judge_bench_standard_writes_summary(runner, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cmd, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cmd, "judge_results", MagicMock(return_value=pd.DataFrame()))
    _patch_summary(monkeypatch)
    result = runner.invoke(
        cmd.app, ["judge-bench", "--model", VALID_MODEL, "--language", "luganda"]
    )
    assert result.exit_code == 0
    assert (tmp_path / "results" / "summary_linguistic_knowledge_luganda.csv").exists()


def test_judge_bench_all_models(runner, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cmd, "PROJECT_ROOT", tmp_path)
    judge = MagicMock(return_value=pd.DataFrame())
    monkeypatch.setattr(cmd, "judge_results", judge)
    _patch_summary(monkeypatch)
    result = runner.invoke(cmd.app, ["judge-bench", "--language", "luganda"])
    assert result.exit_code == 0
    assert judge.call_count == len(cmd.EVAL_MODELS)


def test_judge_bench_fallback_path(runner, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cmd, "PROJECT_ROOT", tmp_path)
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
    name = "summary_linguistic_knowledge_luganda_fallback.csv"
    assert (tmp_path / "results" / name).exists()


def test_judge_bench_skips_on_missing_results(runner, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cmd, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        cmd, "judge_results", MagicMock(side_effect=FileNotFoundError("no file"))
    )
    result = runner.invoke(
        cmd.app, ["judge-bench", "--model", VALID_MODEL, "--language", "luganda"]
    )
    assert result.exit_code == 0
    assert "SKIP" in result.output
    assert not (tmp_path / "results").exists()


def test_judge_bench_propagates_error(runner, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cmd, "PROJECT_ROOT", tmp_path)
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
