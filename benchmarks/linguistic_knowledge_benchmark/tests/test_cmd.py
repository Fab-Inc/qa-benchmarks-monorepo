"""Test cases for the command-line interface."""

import json
import re
from unittest.mock import MagicMock

import pandas as pd
import pytest
from conftest import VALID_MODEL
from linguistic_knowledge_benchmark import cmd, shared_utils
from linguistic_knowledge_benchmark.judge_bench import DEFAULT_JUDGE_MODELS
from typer.testing import CliRunner

# judge-bench defaults to a single judge, DEFAULT_JUDGE_MODELS[0], when no
# --judge-model/--num-judges is given; mocked judge_results()/judge_results_fallback()
# return values need this column so the stale-cache judge-column check doesn't
# flag them as missing the (default) requested judge.
DEFAULT_JUDGE_COL = f"is_correct_{DEFAULT_JUDGE_MODELS[0]}"
DEFAULT_JUDGE_FALLBACK_COL = f"is_correct_fallback_{DEFAULT_JUDGE_MODELS[0]}"


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


def test_timestamp_experiment_id_format() -> None:
    assert re.match(r"^\d{8}T\d{6}Z$", cmd._timestamp_experiment_id())


# --------------------------------------------------------------------------- #
# run-bench
# --------------------------------------------------------------------------- #
def test_run_bench_rejects_unknown_language(runner: CliRunner) -> None:
    result = runner.invoke(cmd.app, ["run-bench", "--language", "klingon"])
    assert result.exit_code == 1
    assert "Invalid language" in result.output


def test_run_bench_rejects_unknown_model(runner: CliRunner) -> None:
    result = runner.invoke(
        cmd.app, ["run-bench", "--model", "bogus", "--language", "kiswahili"]
    )
    assert result.exit_code == 1
    assert "not found" in result.output


def test_run_bench_rejects_invalid_experiment_id(runner: CliRunner) -> None:
    result = runner.invoke(
        cmd.app,
        [
            "run-bench",
            "--model",
            VALID_MODEL,
            "--language",
            "kiswahili",
            "--experiment-id",
            "my_run",
        ],
    )
    assert result.exit_code == 1
    assert "invalid" in result.output.lower()


def test_run_bench_single_model_forwards_params(runner, monkeypatch) -> None:
    mock = MagicMock()
    monkeypatch.setattr(cmd, "run_benchmark", mock)
    result = runner.invoke(
        cmd.app, ["run-bench", "--model", VALID_MODEL, "--language", "kiswahili"]
    )
    assert result.exit_code == 0
    mock.assert_called_once()
    assert "Experiment ID: default (no --experiment-id supplied)" in result.output
    kwargs = mock.call_args.kwargs
    assert kwargs["model_name"] == VALID_MODEL
    assert kwargs["language_name"] == "Kiswahili"
    assert kwargs["retry_errors"] is False


def test_run_bench_all_models(runner, monkeypatch) -> None:
    mock = MagicMock()
    monkeypatch.setattr(cmd, "run_benchmark", mock)
    result = runner.invoke(cmd.app, ["run-bench", "--language", "kiswahili"])
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
            "kiswahili",
            "--retry-errors",
            "--experiment-id",
            "experiment-123",
        ],
    )
    assert result.exit_code == 0
    kwargs = mock.call_args.kwargs
    assert kwargs["retry_errors"] is True
    assert kwargs["experiment_id"] == "experiment-123"


def test_run_bench_timestamped_output_generates_experiment_id(
    runner, monkeypatch
) -> None:
    mock = MagicMock()
    monkeypatch.setattr(cmd, "run_benchmark", mock)
    monkeypatch.setattr(cmd, "_timestamp_experiment_id", lambda: "20260608T120000Z")
    result = runner.invoke(
        cmd.app,
        [
            "run-bench",
            "--model",
            VALID_MODEL,
            "--language",
            "kiswahili",
            "--timestamped-output",
        ],
    )
    assert result.exit_code == 0
    assert "Experiment ID: 20260608T120000Z" in result.output
    assert mock.call_args.kwargs["experiment_id"] == "20260608T120000Z"


def test_run_bench_rejects_force_with_retry_errors(runner) -> None:
    result = runner.invoke(
        cmd.app,
        [
            "run-bench",
            "--model",
            VALID_MODEL,
            "--language",
            "kiswahili",
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
        cmd.app, ["run-bench", "--model", VALID_MODEL, "--language", "kiswahili"]
    )
    assert result.exit_code == 1
    assert "ERROR" in result.output


def test_run_bench_dataset_override_prints_note_and_forwards_path(
    runner, monkeypatch
) -> None:
    mock = MagicMock()
    monkeypatch.setattr(cmd, "run_benchmark", mock)
    result = runner.invoke(
        cmd.app,
        [
            "run-bench",
            "--model",
            VALID_MODEL,
            "--language",
            "kiswahili",
            "--dataset",
            "linguistic_knowledge_eval_kiswahili_20x.json",
        ],
    )
    assert result.exit_code == 0
    assert "NOTE: --dataset override in effect" in result.output
    assert (
        mock.call_args.kwargs["dataset_path"]
        == cmd.DATA_DIR / "linguistic_knowledge_eval_kiswahili_20x.json"
    )


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
            "accuracy_mcq": 90.0,
            "accuracy_short_form": 70.0,
        },
    )


def test_judge_bench_rejects_unknown_language(runner: CliRunner) -> None:
    result = runner.invoke(cmd.app, ["judge-bench", "--language", "klingon"])
    assert result.exit_code == 1
    assert "Invalid language" in result.output


def test_judge_bench_rejects_unknown_model(runner: CliRunner) -> None:
    result = runner.invoke(
        cmd.app, ["judge-bench", "--model", "bogus", "--language", "kiswahili"]
    )
    assert result.exit_code == 1
    assert "not found" in result.output


def test_judge_bench_rejects_invalid_experiment_id(runner: CliRunner) -> None:
    result = runner.invoke(
        cmd.app,
        [
            "judge-bench",
            "--model",
            VALID_MODEL,
            "--language",
            "kiswahili",
            "--experiment-id",
            "my_run",
        ],
    )
    assert result.exit_code == 1
    assert "invalid" in result.output.lower()


def test_judge_bench_standard_writes_summary(runner, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cmd, "BENCH_DIR", tmp_path)
    monkeypatch.setattr(shared_utils, "CACHE_BASE_DIR", tmp_path)
    monkeypatch.setattr(
        cmd,
        "judge_results",
        MagicMock(return_value=pd.DataFrame({DEFAULT_JUDGE_COL: []})),
    )
    _patch_summary(monkeypatch)
    result = runner.invoke(
        cmd.app, ["judge-bench", "--model", VALID_MODEL, "--language", "kiswahili"]
    )
    assert result.exit_code == 0
    assert "Experiment ID: default (no --experiment-id supplied)" in result.output
    summary_path = (
        tmp_path / "results" / "summary_linguistic-knowledge_kiswahili_default.csv"
    )
    assert summary_path.exists()
    df = pd.read_csv(summary_path)
    assert "responses_timestamp-utc" in df.columns
    assert "responses_judged_git-hash" in df.columns


def test_judge_bench_skips_model_over_call_error_rate(
    runner, monkeypatch, tmp_path
) -> None:
    """A model refused by the error-rate gate is announced twice (inline and in the
    end-of-run block) and, with no other model to judge, produces no summary at all."""
    monkeypatch.setattr(cmd, "BENCH_DIR", tmp_path)
    monkeypatch.setattr(shared_utils, "CACHE_BASE_DIR", tmp_path)
    responses = tmp_path / "kiswahili" / "experiments" / "default" / "responses"
    responses.mkdir(parents=True)
    monkeypatch.setattr(
        cmd,
        "judge_results",
        MagicMock(
            side_effect=cmd.TooManyCallErrorsError("20/100 rows failed to call (20.0%)")
        ),
    )
    _patch_summary(monkeypatch)

    result = runner.invoke(
        cmd.app, ["judge-bench", "--model", VALID_MODEL, "--language", "kiswahili"]
    )
    assert result.exit_code == 0
    assert f"SKIPPING {VALID_MODEL}" in result.output
    assert "NOT JUDGED" in result.output
    assert "every model with results exceeded" in result.output
    assert not (
        tmp_path / "results" / "summary_linguistic-knowledge_kiswahili_default.csv"
    ).exists()


def test_judge_bench_summary_excludes_only_the_over_limit_model(
    runner, monkeypatch, tmp_path
) -> None:
    """The gate skips per model: a healthy model still gets judged and summarized."""
    monkeypatch.setattr(cmd, "BENCH_DIR", tmp_path)
    monkeypatch.setattr(shared_utils, "CACHE_BASE_DIR", tmp_path)
    monkeypatch.setattr(cmd, "EVAL_MODELS", {VALID_MODEL: "Valid", "bad-model": "Bad"})

    def _judge(model_name, **kwargs):
        if model_name == "bad-model":
            raise cmd.TooManyCallErrorsError("30/100 rows failed to call (30.0%)")
        return pd.DataFrame({DEFAULT_JUDGE_COL: []})

    monkeypatch.setattr(cmd, "judge_results", _judge)
    _patch_summary(monkeypatch)

    result = runner.invoke(cmd.app, ["judge-bench", "--language", "kiswahili"])
    assert result.exit_code == 0
    assert "SKIPPING bad-model" in result.output
    df = pd.read_csv(
        tmp_path / "results" / "summary_linguistic-knowledge_kiswahili_default.csv"
    )
    assert df["model"].tolist() == [VALID_MODEL]


def test_judge_bench_standard_reads_existing_log_for_provenance(
    runner, monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(cmd, "BENCH_DIR", tmp_path)
    monkeypatch.setattr(shared_utils, "CACHE_BASE_DIR", tmp_path)
    monkeypatch.setattr(
        cmd,
        "judge_results",
        MagicMock(return_value=pd.DataFrame({DEFAULT_JUDGE_COL: []})),
    )
    _patch_summary(monkeypatch)

    experiment_dir = tmp_path / "kiswahili" / "experiments" / "default"
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "log.json").write_text(
        json.dumps(
            {VALID_MODEL: {"responses_timestamp-utc": "2020-01-01T00:00:00+00:00"}}
        )
    )

    result = runner.invoke(
        cmd.app, ["judge-bench", "--model", VALID_MODEL, "--language", "kiswahili"]
    )
    assert result.exit_code == 0
    summary_path = (
        tmp_path / "results" / "summary_linguistic-knowledge_kiswahili_default.csv"
    )
    df = pd.read_csv(summary_path)
    assert df.loc[0, "responses_timestamp-utc"] == "2020-01-01T00:00:00+00:00"


def test_judge_bench_experiment_id_writes_experiment_specific_summary(
    runner, monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(cmd, "BENCH_DIR", tmp_path)
    judge = MagicMock(return_value=pd.DataFrame({DEFAULT_JUDGE_COL: []}))
    monkeypatch.setattr(cmd, "judge_results", judge)
    _patch_summary(monkeypatch)
    result = runner.invoke(
        cmd.app,
        [
            "judge-bench",
            "--model",
            VALID_MODEL,
            "--language",
            "kiswahili",
            "--experiment-id",
            "experiment-123",
        ],
    )
    assert result.exit_code == 0
    assert "Experiment ID: experiment-123" in result.output
    assert judge.call_args.kwargs["experiment_id"] == "experiment-123"
    assert (
        tmp_path
        / "results"
        / "summary_linguistic-knowledge_kiswahili_experiment-123.csv"
    ).exists()


def test_judge_bench_all_models(runner, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cmd, "BENCH_DIR", tmp_path)
    judge = MagicMock(return_value=pd.DataFrame({DEFAULT_JUDGE_COL: []}))
    monkeypatch.setattr(cmd, "judge_results", judge)
    _patch_summary(monkeypatch)
    result = runner.invoke(cmd.app, ["judge-bench", "--language", "kiswahili"])
    assert result.exit_code == 0
    assert judge.call_count == len(cmd.EVAL_MODELS)


def test_judge_bench_fallback_path(runner, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cmd, "BENCH_DIR", tmp_path)
    fallback = MagicMock(return_value=pd.DataFrame({DEFAULT_JUDGE_FALLBACK_COL: []}))
    monkeypatch.setattr(cmd, "judge_results_fallback", fallback)
    _patch_summary(monkeypatch)
    result = runner.invoke(
        cmd.app,
        [
            "judge-bench",
            "--model",
            VALID_MODEL,
            "--language",
            "kiswahili",
            "--judge-fallback",
        ],
    )
    assert result.exit_code == 0
    fallback.assert_called_once()
    name = "summary_linguistic-knowledge_kiswahili_default_fallback.csv"
    assert (tmp_path / "results" / name).exists()


def test_timestamp_experiment_id_is_filesystem_safe() -> None:
    assert re.fullmatch(r"\d{8}T\d{6}Z", cmd._timestamp_experiment_id())


def test_judge_bench_merges_existing_log_fields(runner, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cmd, "BENCH_DIR", tmp_path)
    monkeypatch.setattr(shared_utils, "CACHE_BASE_DIR", tmp_path)
    monkeypatch.setattr(
        cmd,
        "judge_results",
        MagicMock(return_value=pd.DataFrame({DEFAULT_JUDGE_COL: []})),
    )
    _patch_summary(monkeypatch)
    log_path = tmp_path / "kiswahili" / "experiments" / "default" / "log.json"
    log_path.parent.mkdir(parents=True)
    log_path.write_text(json.dumps({VALID_MODEL: {"responses_git-hash": "deadbee"}}))

    result = runner.invoke(
        cmd.app, ["judge-bench", "--model", VALID_MODEL, "--language", "kiswahili"]
    )
    assert result.exit_code == 0
    summary = pd.read_csv(
        tmp_path / "results" / "summary_linguistic-knowledge_kiswahili_default.csv"
    )
    assert summary.loc[0, "responses_git-hash"] == "deadbee"


def test_judge_bench_skips_on_missing_results(runner, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cmd, "BENCH_DIR", tmp_path)
    monkeypatch.setattr(shared_utils, "CACHE_BASE_DIR", tmp_path)
    monkeypatch.setattr(
        cmd, "judge_results", MagicMock(side_effect=FileNotFoundError("no file"))
    )
    result = runner.invoke(
        cmd.app, ["judge-bench", "--model", VALID_MODEL, "--language", "kiswahili"]
    )
    assert result.exit_code == 0
    assert not (tmp_path / "results").exists()
    # When the directory is missing, show only the not-found diagnostic — the
    # per-model "Skipped" line would be redundant noise.
    assert "Nothing judged: expected directory not found" in result.output
    assert "Skipped" not in result.output


def test_judge_bench_reports_empty_directory(runner, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cmd, "BENCH_DIR", tmp_path)
    monkeypatch.setattr(shared_utils, "CACHE_BASE_DIR", tmp_path)
    # The responses directory exists but holds no result files for the model.
    (tmp_path / "kiswahili" / "experiments" / "default" / "responses").mkdir(
        parents=True
    )
    monkeypatch.setattr(
        cmd, "judge_results", MagicMock(side_effect=FileNotFoundError("no file"))
    )
    result = runner.invoke(
        cmd.app, ["judge-bench", "--model", VALID_MODEL, "--language", "kiswahili"]
    )
    assert result.exit_code == 0
    assert "Nothing judged: no result files" in result.output


def test_judge_bench_propagates_error(runner, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cmd, "BENCH_DIR", tmp_path)
    monkeypatch.setattr(
        cmd, "judge_results", MagicMock(side_effect=RuntimeError("kaboom"))
    )
    result = runner.invoke(
        cmd.app, ["judge-bench", "--model", VALID_MODEL, "--language", "kiswahili"]
    )
    assert result.exit_code == 1
    assert "ERROR" in result.output


def test_judge_bench_rejects_num_judges_out_of_range(runner: CliRunner) -> None:
    result = runner.invoke(
        cmd.app,
        [
            "judge-bench",
            "--model",
            VALID_MODEL,
            "--language",
            "kiswahili",
            "--num-judges",
            "0",
        ],
    )
    assert result.exit_code == 1
    assert "--num-judges must be between" in result.output


def test_judge_bench_explicit_judge_models_override_num_judges(
    runner, monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(cmd, "BENCH_DIR", tmp_path)
    judges = list(DEFAULT_JUDGE_MODELS[:2])
    cols = {f"is_correct_{jm}": [] for jm in judges}
    judge = MagicMock(return_value=pd.DataFrame(cols))
    monkeypatch.setattr(cmd, "judge_results", judge)
    # >1 judge means compute_summary()'s judge_agreement_mean/judge_disagreement_count
    # keys are read unconditionally (see cmd.py's "if len(judge_models) > 1" branch) --
    # unlike _patch_summary's single-judge-shaped fake, this needs both present.
    monkeypatch.setattr(
        cmd,
        "compute_summary",
        lambda df, model: {
            "model": model,
            "accuracy": 80.0,
            "accuracy_mcq": 90.0,
            "accuracy_short_form": 70.0,
            "judge_agreement_mean": None,
            "judge_disagreement_count": 0,
        },
    )
    args = ["judge-bench", "--model", VALID_MODEL, "--language", "kiswahili"]
    for jm in judges:
        args += ["--judge-model", jm]
    result = runner.invoke(cmd.app, args)
    assert result.exit_code == 0
    assert judge.call_args.kwargs["judge_models"] == judges


def test_judge_bench_dataset_override_prints_note_and_forwards_path(
    runner, monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(cmd, "BENCH_DIR", tmp_path)
    # --dataset is filename-only, resolved as DATA_DIR/<dataset> (see cmd.py's help
    # text) -- point DATA_DIR at tmp_path and write a minimal real dataset file there,
    # rather than relying on a real (large, gitignored, not-committed) data/*_20x.json
    # existing on disk. build_detailed() below actually reads this file for real
    # (unlike judge_results, which is mocked), so it must be valid, parseable JSON
    # with a question_id column -- a nonexistent path would raise FileNotFoundError.
    monkeypatch.setattr(cmd, "DATA_DIR", tmp_path)
    dataset_path = tmp_path / "fake_dataset.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "question_id": 1,
                    "domain": "D1",
                    "answer_type": "multiple_choice",
                    "question": "Example question?",
                    "expected_answers": ["A"],
                }
            ]
        ),
        encoding="utf-8",
    )
    judge = MagicMock(return_value=pd.DataFrame({DEFAULT_JUDGE_COL: []}))
    monkeypatch.setattr(cmd, "judge_results", judge)
    _patch_summary(monkeypatch)
    result = runner.invoke(
        cmd.app,
        [
            "judge-bench",
            "--model",
            VALID_MODEL,
            "--language",
            "kiswahili",
            "--dataset",
            "fake_dataset.json",
        ],
    )
    assert result.exit_code == 0
    assert "NOTE: --dataset override in effect" in result.output
    assert judge.call_args.kwargs["dataset_path"] == dataset_path


def test_judge_bench_reports_missing_judge_columns(
    runner, monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(cmd, "BENCH_DIR", tmp_path)
    judges = list(DEFAULT_JUDGE_MODELS[:2])
    # Only the first requested judge's column is present -- simulates data already
    # judged with a narrower judge set, re-run with a wider one but no --force.
    judge = MagicMock(return_value=pd.DataFrame({f"is_correct_{judges[0]}": []}))
    monkeypatch.setattr(cmd, "judge_results", judge)
    args = ["judge-bench", "--model", VALID_MODEL, "--language", "kiswahili"]
    for jm in judges:
        args += ["--judge-model", jm]
    result = runner.invoke(cmd.app, args)
    assert result.exit_code == 1
    assert "was already judged but only has results for" in result.output
    assert "Re-run with --force to rejudge" in result.output


def test_judge_bench_prints_and_saves_pairwise_agreement(
    runner, monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(cmd, "BENCH_DIR", tmp_path)
    judges = list(DEFAULT_JUDGE_MODELS[:2])
    cols = {f"is_correct_{jm}": [] for jm in judges}
    judge = MagicMock(return_value=pd.DataFrame(cols))
    monkeypatch.setattr(cmd, "judge_results", judge)
    monkeypatch.setattr(
        cmd,
        "compute_summary",
        lambda df, model: {
            "model": model,
            "accuracy": 80.0,
            "accuracy_mcq": 90.0,
            "accuracy_short_form": 70.0,
            "judge_agreement_mean": 0.75,
            "judge_disagreement_count": 3,
        },
    )
    agreement_df = pd.DataFrame([[1.0, 0.8], [0.8, 1.0]], index=judges, columns=judges)
    monkeypatch.setattr(cmd, "compute_judge_agreement", lambda *a, **k: agreement_df)
    args = ["judge-bench", "--model", VALID_MODEL, "--language", "kiswahili"]
    for jm in judges:
        args += ["--judge-model", jm]
    result = runner.invoke(cmd.app, args)
    assert result.exit_code == 0
    assert "Judge agreement: 75.00%" in result.output
    assert "(3 disagreement(s))" in result.output
    assert "Pairwise judge agreement" in result.output
    name = "summary_linguistic-knowledge_kiswahili_default_judge_agreement.csv"
    assert (tmp_path / "results" / name).exists()


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #
def test_main_invokes_app(monkeypatch) -> None:
    mock_app = MagicMock()
    monkeypatch.setattr(cmd, "app", mock_app)
    cmd.main()
    mock_app.assert_called_once()
