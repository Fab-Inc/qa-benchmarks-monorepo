"""The fab-bench command line: options, output and exit codes."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from fab_benchmark_runner import parse_run_config, read_experiment
from fab_benchmark_runner.cli import app as fab_bench

REGISTRY = b"provider:\n  models:\n    model-a: {}\n"
CONFIG = parse_run_config(
    {"experiment": "gs://bucket/exp", "models": ["model-a"]}, base_dir=Path.cwd()
)


@pytest.fixture
def fdllm_free() -> Iterator[list[tuple[str, str]]]:
    """Run the real command path with FDLLM replaced by an echoing model."""

    calls: list[tuple[str, str]] = []

    def call(model, model_input, _max_tokens) -> dict[str, Any]:
        calls.append((model, model_input.benchmark_item_id))
        if model_input.prompt == "Fail?":
            return {"output": ""}
        return {"output": f"{model}:{model_input.benchmark_item_id}"}

    with (
        patch("fab_benchmark_runner.jobs.prepare_fdllm", return_value={}),
        patch("fab_benchmark_runner.jobs.call_model_with_fdllm", side_effect=call),
    ):
        yield calls


def _experiment(tmp_path: Path, *, fail: bool = False) -> Path:
    inputs = _write_experiment_inputs(tmp_path / "inputs.jsonl")
    if fail:
        inputs.write_text(
            inputs.read_text()
            + json.dumps({"benchmark_item_id": "c", "prompt": "Fail?"})
            + "\n"
        )
    result = CliRunner().invoke(
        fab_bench,
        [
            "create-experiment",
            "--inputs",
            str(inputs),
            "--dest",
            str(tmp_path / "exp"),
            "--benchmark",
            "lk",
            "--max-tokens",
            "100",
        ],
    )
    assert result.exit_code == 0, result.output
    (tmp_path / "custom_models.yaml").write_bytes(REGISTRY)
    config = tmp_path / "run.json"
    config.write_text(
        json.dumps(
            {
                "experiment": "exp",
                "models": ["model-a"],
                "custom_models_path": "custom_models.yaml",
            }
        )
    )
    return config


def test_run_then_dry_run(tmp_path: Path, fdllm_free: list) -> None:
    config = _experiment(tmp_path)

    ran = CliRunner().invoke(fab_bench, ["run", "--config", str(config)])
    planned = CliRunner().invoke(
        fab_bench, ["run", "--config", str(config), "--dry-run"]
    )

    assert ran.exit_code == 0, ran.output
    assert "Executed: 2" in ran.output
    assert planned.exit_code == 0, planned.output
    assert "model-a: 2 planned, 2 reused, 0 to run" in planned.output
    assert fdllm_free == [("model-a", "a"), ("model-a", "b")]


def test_run_exits_nonzero_when_calls_fail(tmp_path: Path, fdllm_free: list) -> None:
    config = _experiment(tmp_path, fail=True)

    result = CliRunner().invoke(fab_bench, ["run", "--config", str(config)])

    assert result.exit_code == 1
    assert "Errors: 1" in result.output


def test_run_applies_command_line_options(tmp_path: Path, fdllm_free: list) -> None:
    config = _experiment(tmp_path)
    CliRunner().invoke(fab_bench, ["run", "--config", str(config)])
    fdllm_free.clear()

    result = CliRunner().invoke(
        fab_bench, ["run", "--config", str(config), "--force", "--model-workers", "2"]
    )

    assert result.exit_code == 0, result.output
    assert len(fdllm_free) == 2
    both = CliRunner().invoke(
        fab_bench, ["run", "--config", str(config), "--force", "--retry-errors"]
    )
    assert both.exit_code == 1
    assert "cannot be combined" in both.output


def test_run_reports_config_errors(tmp_path: Path) -> None:
    config = tmp_path / "run.json"
    config.write_text(json.dumps({"experiment": "exp", "models": [], "max_tokens": 5}))

    result = CliRunner().invoke(fab_bench, ["run", "--config", str(config)])

    assert result.exit_code == 1
    assert "ERROR: Run configs no longer take 'max_tokens'" in result.output


def test_run_cloud_reads_its_config_from_gcs_into_its_workdir(tmp_path: Path) -> None:
    with (
        patch("fab_benchmark_runner.cli.load_run_config") as load,
        patch("fab_benchmark_runner.cli.run_job") as run,
    ):
        run.return_value.errors = 0
        run.return_value.version_warnings = ()
        result = CliRunner().invoke(
            fab_bench,
            [
                "run-cloud",
                "--config",
                "gs://bucket/run-configs/x.json",
                "--workdir",
                str(tmp_path),
                "--skip-models-download",
            ],
        )

    assert result.exit_code == 0, result.output
    load.assert_called_once_with("gs://bucket/run-configs/x.json")
    kwargs = run.call_args.kwargs
    assert kwargs["workdir"] == tmp_path
    assert kwargs["download_public_registry"] is False
    assert kwargs["public_registry_path"] == tmp_path.resolve() / "custom_models.yaml"
    local = CliRunner().invoke(fab_bench, ["run-cloud", "--config", "run.json"])
    assert local.exit_code == 1
    assert "gs:// URI" in local.output


def test_the_dry_run_prints_version_warnings(tmp_path: Path) -> None:
    from fab_benchmark_runner.runner import RunPlan

    plan = RunPlan("e", tmp_path, 100, (), ("fdllm: 0.29.0 in earlier results",))
    with (
        patch("fab_benchmark_runner.cli.load_run_config", return_value=CONFIG),
        patch("fab_benchmark_runner.cli.plan_job", return_value=plan),
    ):
        result = CliRunner().invoke(
            fab_bench, ["run", "--config", "run.json", "--dry-run"]
        )

    assert result.exit_code == 0
    assert "results from other versions" in result.output
    assert "fdllm: 0.29.0 in earlier results" in result.output


def test_the_dry_run_says_how_many_images_it_checked(tmp_path: Path) -> None:
    from fab_benchmark_runner.runner import RunPlan

    plan = RunPlan("e", tmp_path, None, (), images_to_send=3)
    with (
        patch("fab_benchmark_runner.cli.load_run_config", return_value=CONFIG),
        patch("fab_benchmark_runner.cli.plan_job", return_value=plan),
    ):
        result = CliRunner().invoke(
            fab_bench, ["run", "--config", "run.json", "--dry-run"]
        )

    assert result.exit_code == 0
    assert "Max tokens: none" in result.output
    assert "Images: 3 to send, each fetched and checked" in result.output


def test_raise_max_tokens(tmp_path: Path) -> None:
    _experiment(tmp_path)
    command = ["raise-max-tokens", "--experiment", str(tmp_path / "exp")]

    raised = CliRunner().invoke(fab_bench, [*command, "--max-tokens", "400"])
    same = CliRunner().invoke(fab_bench, [*command, "--max-tokens", "400"])
    lower = CliRunner().invoke(fab_bench, [*command, "--max-tokens", "200"])

    assert raised.exit_code == 0, raised.output
    assert "Raised max_tokens from 100 to 400" in raised.output
    assert "--retry-errors" in raised.output
    assert "already 400" in same.output
    assert lower.exit_code == 1
    assert "can only be raised" in lower.output
    assert read_experiment(tmp_path / "exp").max_tokens == 400


def test_fetch_results_and_upload_experiment(tmp_path: Path) -> None:
    with (
        patch("fab_benchmark_runner.cli.fetch_results", return_value=4) as fetch,
        patch("fab_benchmark_runner.cli.upload_experiment", return_value=3) as upload,
    ):
        fetched = CliRunner().invoke(
            fab_bench,
            [
                "fetch-results",
                "--experiment",
                "gs://bucket/exp",
                "--dest",
                str(tmp_path / "out"),
                "--model",
                "model-a",
            ],
        )
        uploaded = CliRunner().invoke(
            fab_bench, ["upload-experiment", str(tmp_path), "gs://bucket/exp"]
        )

    assert "Downloaded 4 files" in fetched.output
    fetch.assert_called_once_with("gs://bucket/exp", tmp_path / "out", model="model-a")
    assert "Uploaded 3 files to gs://bucket/exp" in uploaded.output
    upload.assert_called_once_with(tmp_path, "gs://bucket/exp")


def test_unexpected_exceptions_keep_their_traceback(tmp_path: Path) -> None:
    with (
        patch("fab_benchmark_runner.cli.load_run_config", return_value=CONFIG),
        patch("fab_benchmark_runner.cli.run_job", side_effect=KeyError("bug")),
    ):
        result = CliRunner().invoke(fab_bench, ["run", "--config", "run.json"])

    assert isinstance(result.exception, KeyError)


def test_fab_bench_without_a_command_shows_help() -> None:
    result = CliRunner().invoke(fab_bench, [])

    assert result.exit_code == 0
    assert "create-experiment" in result.output
    assert "raise-max-tokens" in result.output


def _write_experiment_inputs(path: Path) -> Path:
    rows = [
        {"benchmark_item_id": "a", "prompt": "First?", "system_prompt": "Be brief."},
        {"benchmark_item_id": "b", "prompt": "Second?"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_create_experiment_dry_run_then_create(tmp_path: Path) -> None:
    inputs = _write_experiment_inputs(tmp_path / "inputs.jsonl")
    dest = tmp_path / "exp"
    command = [
        "create-experiment",
        "--inputs",
        str(inputs),
        "--dest",
        str(dest),
        "--benchmark",
        "lk",
        "--max-tokens",
        "none",
        "--experiment-id",
        "sw-v1",
        "--label",
        "language=sw",
    ]

    planned = CliRunner().invoke(fab_bench, [*command, "--dry-run"])
    assert planned.exit_code == 0, planned.output
    assert "Experiment: sw-v1" in planned.output
    assert "Items: 2" in planned.output
    assert "Max tokens: none" in planned.output
    assert "Dry run: nothing was written. First inputs:" in planned.output
    assert '"prompt": "First?"' in planned.output
    assert not dest.exists()

    created = CliRunner().invoke(fab_bench, command)
    assert created.exit_code == 0, created.output
    assert "Created." in created.output
    record = json.loads((dest / "experiment.json").read_text(encoding="utf-8"))
    assert (record["experiment_id"], record["labels"]) == ("sw-v1", {"language": "sw"})
    assert record["max_tokens"] is None

    again = CliRunner().invoke(fab_bench, command)
    assert again.exit_code == 0, again.output
    assert "already exists; nothing was written" in again.output


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--max-tokens", "0"], "positive whole number or 'none'"),
        (["--max-tokens", "many"], "positive whole number or 'none'"),
        (["--max-tokens", "10", "--label", "subset"], "is not key=value"),
        (
            ["--max-tokens", "10", "--label", "a=1", "--label", "a=2"],
            "'a' is given twice",
        ),
    ],
)
def test_create_experiment_refuses_bad_options(
    tmp_path: Path, extra: list[str], message: str
) -> None:
    inputs = _write_experiment_inputs(tmp_path / "inputs.jsonl")

    result = CliRunner().invoke(
        fab_bench,
        [
            "create-experiment",
            "--inputs",
            str(inputs),
            "--dest",
            str(tmp_path / "exp"),
            "--benchmark",
            "lk",
            *extra,
        ],
    )

    assert result.exit_code == 2
    assert message in result.output
    assert not (tmp_path / "exp").exists()


def test_create_experiment_reports_a_folder_it_cannot_use(tmp_path: Path) -> None:
    inputs = _write_experiment_inputs(tmp_path / "inputs.jsonl")
    (tmp_path / "exp").mkdir()
    (tmp_path / "exp" / "log.json").write_text("{}", encoding="utf-8")

    result = CliRunner().invoke(
        fab_bench,
        [
            "create-experiment",
            "--inputs",
            str(inputs),
            "--dest",
            str(tmp_path / "exp"),
            "--benchmark",
            "lk",
            "--max-tokens",
            "10",
        ],
    )

    assert result.exit_code == 1
    assert "ERROR:" in result.output
    assert "holds other files (log.json)" in result.output


def test_create_experiment_reports_a_missing_inputs_file(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        fab_bench,
        [
            "create-experiment",
            "--inputs",
            str(tmp_path / "missing.jsonl"),
            "--dest",
            str(tmp_path / "exp"),
            "--benchmark",
            "lk",
            "--max-tokens",
            "100",
        ],
    )

    assert result.exit_code == 1
    assert "Could not read inputs file" in result.output
    assert not (tmp_path / "exp").exists()
