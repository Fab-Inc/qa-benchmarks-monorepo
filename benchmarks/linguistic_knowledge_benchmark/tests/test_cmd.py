"""Test cases for the command-line interface."""

import json
import os
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from conftest import VALID_MODEL, sample_questions, write_dataset
from linguistic_knowledge_benchmark import cmd, shared_utils
from linguistic_knowledge_benchmark.cmd import DATASETS as REGISTERED_DATASETS
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


@pytest.fixture(autouse=True)
def stub_registered_datasets(monkeypatch, tmp_path_factory):
    """Point every registered language at a tiny on-disk dataset.

    The real data/linguistic_knowledge_eval_<language>.json item sets are gitignored
    (kept private ahead of open-sourcing the repo), so they are absent on every fresh
    clone and in CI. judge-bench reads the registered path for real -- build_detailed()
    is not mocked in these tests -- so without this every judge-bench test would pass
    locally and fail in CI with FileNotFoundError. Autouse so new tests inherit it.
    """
    dataset_path = write_dataset(
        tmp_path_factory.mktemp("registered_datasets") / "dataset.json",
        sample_questions(),
    )
    monkeypatch.setattr(
        cmd,
        "DATASETS",
        {key: {**entry, "path": dataset_path} for key, entry in cmd.DATASETS.items()},
    )


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
    # REGISTERED_DATASETS, not cmd.DATASETS: this checks the real registry, which
    # stub_registered_datasets replaces for every test in this module.
    for entry in REGISTERED_DATASETS.values():
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
    assert "responses_timestamp_utc" in df.columns
    assert "responses_judged_git_hash" in df.columns


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
            {VALID_MODEL: {"responses_timestamp_utc": "2020-01-01T00:00:00+00:00"}}
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
    assert df.loc[0, "responses_timestamp_utc"] == "2020-01-01T00:00:00+00:00"


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
    log_path.write_text(json.dumps({VALID_MODEL: {"responses_git_hash": "deadbee"}}))

    result = runner.invoke(
        cmd.app, ["judge-bench", "--model", VALID_MODEL, "--language", "kiswahili"]
    )
    assert result.exit_code == 0
    summary = pd.read_csv(
        tmp_path / "results" / "summary_linguistic-knowledge_kiswahili_default.csv"
    )
    assert summary.loc[0, "responses_git_hash"] == "deadbee"


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
    # cmd.py reads both passes' agreement keys unconditionally, gating each printed
    # line on its own mean being non-None, so every key has to be present here.
    monkeypatch.setattr(
        cmd,
        "compute_summary",
        lambda df, model: {
            "model": model,
            "accuracy": 80.0,
            "accuracy_mcq": 90.0,
            "accuracy_short_form": 70.0,
            "short_form_judge_agreement_mean": None,
            "short_form_judge_disagreement_count": 0,
            "mcq_judge_fallback_agreement_mean": None,
            "mcq_judge_fallback_disagreement_count": 0,
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
            "short_form_judge_agreement_mean": 0.75,
            "short_form_judge_disagreement_count": 3,
            "mcq_judge_fallback_agreement_mean": 1.0,
            "mcq_judge_fallback_disagreement_count": 0,
        },
    )
    agreement_df = pd.DataFrame([[1.0, 0.8], [0.8, 1.0]], index=judges, columns=judges)
    monkeypatch.setattr(cmd, "compute_judge_agreement", lambda *a, **k: agreement_df)
    args = ["judge-bench", "--model", VALID_MODEL, "--language", "kiswahili"]
    for jm in judges:
        args += ["--judge-model", jm]
    result = runner.invoke(cmd.app, args)
    assert result.exit_code == 0
    # Each pass reports on its own line, so neither can be mistaken for the other.
    assert "Short-form judge agreement: 75.00% (3 disagreement(s))" in result.output
    assert "MCQ fallback judge agreement: 100.00% (0 disagreement(s))" in result.output
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


# --------------------------------------------------------------------------- #
# shared config sync (custom_models.yaml + models.csv + providers.csv)
# --------------------------------------------------------------------------- #
# conftest's autouse fixture replaces cmd._download_shared_configs with a no-op so the
# rest of the suite stays offline. Bind the real function at import time -- before any
# fixture runs -- so the tests below still exercise it.
_download_shared_configs = cmd._download_shared_configs

VALID_CONTENT = {
    "custom_models.yaml": "SomeProvider:\n  models: {}\n",
    "models.csv": "model_id,provider\nm1,Anthropic\n",
    "providers.csv": "provider,color\nAnthropic,#e9967a\n",
}


def _configs_in(directory: Path) -> tuple[Path, ...]:
    """cmd.SHARED_CONFIGS relocated under `directory`, so a test never writes the real
    repo-root files."""
    return tuple(directory / path.name for path in cmd.SHARED_CONFIGS)


def _writes_valid_content(**kwargs) -> None:
    """download_file stand-in: write the good copy of whichever file was asked for."""
    Path(kwargs["write_path"]).write_text(
        VALID_CONTENT[kwargs["file_name"]], encoding="utf-8"
    )


def test_download_shared_configs_fetches_all_three(monkeypatch, tmp_path) -> None:
    """All three synced files are pulled anonymously from the same bucket."""
    monkeypatch.setattr(cmd, "SHARED_CONFIGS", _configs_in(tmp_path))
    with patch(
        "linguistic_knowledge_benchmark.cmd.download_file",
        side_effect=_writes_valid_content,
    ) as download_file:
        _download_shared_configs()

    requested = []
    for call in download_file.call_args_list:
        kwargs = dict(call.kwargs)
        file_name = kwargs.pop("file_name")
        requested.append(file_name)
        assert kwargs == {
            "bucket_name": "fab-open-configs",
            "write_path": tmp_path / file_name,
            "require_auth": False,
        }
    assert requested == ["custom_models.yaml", "models.csv", "providers.csv"]

    for name, content in VALID_CONTENT.items():
        assert (tmp_path / name).read_text(encoding="utf-8") == content


def test_download_shared_configs_replaces_previous_copy(monkeypatch, tmp_path) -> None:
    """A stale file on disk is replaced by the freshly downloaded one."""
    monkeypatch.setattr(cmd, "SHARED_CONFIGS", _configs_in(tmp_path))
    (tmp_path / "models.csv").write_text("model_id,provider\nstale,Old\n", "utf-8")
    with patch(
        "linguistic_knowledge_benchmark.cmd.download_file",
        side_effect=_writes_valid_content,
    ):
        _download_shared_configs()
    assert (tmp_path / "models.csv").read_text(encoding="utf-8") == (
        VALID_CONTENT["models.csv"]
    )


def test_download_shared_configs_propagates_failure(monkeypatch, tmp_path) -> None:
    """A failed download aborts the CLI rather than continuing with missing config.

    The sync runs in the app callback, so raising here is what stops a command from
    proceeding against a stale or absent models.csv.
    """
    monkeypatch.setattr(cmd, "SHARED_CONFIGS", _configs_in(tmp_path))

    def fail_on_models_csv(**kwargs):
        if kwargs["file_name"] == "models.csv":
            raise RuntimeError("network down")
        _writes_valid_content(**kwargs)

    with pytest.raises(RuntimeError, match="network down"):
        with patch(
            "linguistic_knowledge_benchmark.cmd.download_file",
            side_effect=fail_on_models_csv,
        ):
            _download_shared_configs()


def test_setup_syncs_shared_configs(monkeypatch, runner: CliRunner) -> None:
    """The app callback is what keeps the three files fresh on every invocation."""
    sync = MagicMock()
    monkeypatch.setattr(cmd, "_download_shared_configs", sync)
    monkeypatch.setattr(cmd, "register_models", MagicMock())
    monkeypatch.setattr(cmd, "run_benchmark", MagicMock())
    result = runner.invoke(cmd.app, ["run-bench", "--language", "kiswahili"])
    assert result.exit_code == 0
    sync.assert_called_once()


@pytest.mark.integration
def test_shared_configs_are_readable_from_the_bucket(tmp_path, monkeypatch) -> None:
    """Opt-in: the real objects exist in the bucket and are anonymously readable.

    Everything above mocks the network, so this is the only check that fab-open-configs
    still serves all three files with the columns the analysis code expects. Skipped
    unless LKB_TEST_GCS is set, following the same opt-in convention as the phonics
    integration tests -- it needs no credentials, but CI should not depend on the
    bucket being reachable.
    """
    if not os.environ.get("LKB_TEST_GCS"):
        pytest.skip("LKB_TEST_GCS not set -- set it to hit the real bucket")
    monkeypatch.setattr(cmd, "SHARED_CONFIGS", _configs_in(tmp_path))
    _download_shared_configs()
    assert (tmp_path / "custom_models.yaml").stat().st_size > 0
    assert "model_id" in pd.read_csv(tmp_path / "models.csv").columns
    assert "provider" in pd.read_csv(tmp_path / "providers.csv").columns
