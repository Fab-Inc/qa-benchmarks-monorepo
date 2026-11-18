"""The generated command-line interface."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from hashlib import sha256
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from fab_benchmark_runner import Benchmark, build_app, load_run_config

REGISTRY = b"provider:\n  models:\n    model-a: {}\n"


@pytest.fixture
def fdllm_free() -> Iterator[list[tuple[str, str]]]:
    """Run the real CLI path with FDLLM replaced by an echoing evaluator."""

    calls: list[tuple[str, str]] = []

    def evaluate(_config, model, item, _prompt) -> dict[str, Any]:
        calls.append((model, item["benchmark_item_id"]))
        if item.get("fail"):
            return {"output": ""}
        return {"output": f"{model}:{item['benchmark_item_id']}"}

    with (
        patch("fab_benchmark_runner.runner.prepare_fdllm_for_run"),
        patch("fab_benchmark_runner.runner.snapshot_model_configs", return_value={}),
        patch(
            "fab_benchmark_runner.runner.evaluate_item_with_fdllm", side_effect=evaluate
        ),
    ):
        yield calls


def test_run_then_dry_run(
    benchmark: Benchmark,
    write_run: Callable[..., Path],
    fdllm_free: list[tuple[str, str]],
) -> None:
    app = build_app(benchmark)
    config_path = write_run()

    planned = CliRunner().invoke(
        app, ["run", "--config", str(config_path), "--dry-run"]
    )
    assert planned.exit_code == 0, planned.output
    assert "model-a: 2 planned, 0 reused, 2 to run" in planned.output
    assert "Total: 4 planned, 0 reused, 4 to run" in planned.output
    assert fdllm_free == []
    assert not load_run_config(config_path, benchmark).run_dir.exists()

    ran = CliRunner().invoke(app, ["run", "--config", str(config_path)])
    assert ran.exit_code == 0, ran.output
    assert "Executed: 4" in ran.output
    assert len(fdllm_free) == 4

    replanned = CliRunner().invoke(
        app, ["run", "--config", str(config_path), "--dry-run"]
    )
    assert "Total: 4 planned, 4 reused, 0 to run" in replanned.output


def test_run_exits_nonzero_when_calls_fail(
    benchmark: Benchmark,
    write_run: Callable[..., Path],
    fdllm_free: list[tuple[str, str]],
) -> None:
    config_path = write_run(
        items=[{"benchmark_item_id": "item-1", "content": "x", "fail": True}]
    )

    result = CliRunner().invoke(build_app(benchmark), ["run", "-c", str(config_path)])

    assert result.exit_code == 1
    assert "Errors: 2" in result.output


def test_run_reports_config_errors(
    benchmark: Benchmark, write_run: Callable[..., Path]
) -> None:
    config_path = write_run(language="")

    result = CliRunner().invoke(build_app(benchmark), ["run", "-c", str(config_path)])

    assert result.exit_code == 1
    assert "ERROR: Invalid test-bench run params" in result.output


def test_run_applies_command_line_overrides(
    benchmark: Benchmark, write_run: Callable[..., Path]
) -> None:
    seen = []
    with patch(
        "fab_benchmark_runner.cli.execute_with_optional_sync",
        side_effect=lambda config: seen.append(config) or _output(),
    ):
        result = CliRunner().invoke(
            build_app(benchmark),
            [
                "run",
                "-c",
                str(write_run()),
                "--force",
                "--model-workers",
                "3",
                "--results-dest",
                "gs://bucket/runs/a/",
                "--skip-models-download",
            ],
        )

    assert result.exit_code == 0, result.output
    [config] = seen
    assert (config.force, config.model_workers) == (True, 3)
    assert config.results_dest == "gs://bucket/runs/a/"


@pytest.mark.parametrize("skip_download", [False, True])
@pytest.mark.parametrize("valid_checksum", [False, True])
def test_run_cloud_uses_the_pinned_registry(
    tmp_path: Path,
    benchmark: Benchmark,
    write_run: Callable[..., Path],
    skip_download: bool,
    valid_checksum: bool,
) -> None:
    config = load_run_config(
        write_run(
            custom_models_path=None,
            custom_models_uri="gs://results/model-configs/experiment-a.yaml",
            custom_models_sha256=sha256(REGISTRY).hexdigest(),
            inputs_dir="work/inputs",
        ),
        benchmark,
    )
    pinned_path = tmp_path / "work" / "inputs" / "experiment-a" / "custom_models.yaml"
    pinned_path.parent.mkdir(parents=True)
    pinned_path.write_bytes(b"previous registry")

    def fake_download(**kwargs: Any) -> None:
        Path(kwargs["write_path"]).write_bytes(
            REGISTRY if valid_checksum else b"changed"
        )

    args = ["run-cloud", "--config", "gs://results/run-configs/experiment-a.json"]
    if skip_download:
        args.append("--skip-models-download")
    with (
        patch("fab_benchmark_runner.cli.load_cloud_run_config", return_value=config),
        patch("fab_benchmark_runner.gcs.download_file", side_effect=fake_download),
        patch("fab_benchmark_runner.registry.download_public_registry") as public,
        patch(
            "fab_benchmark_runner.cli.execute_with_optional_sync",
            return_value=_output(),
        ) as execute,
    ):
        result = CliRunner().invoke(build_app(benchmark), args)

    public.assert_not_called()
    assert not list(pinned_path.parent.glob(".custom_models.yaml.*"))
    if valid_checksum:
        assert result.exit_code == 0, result.output
        [call] = execute.call_args_list
        assert call.args[0].custom_models_path == pinned_path.resolve()
        assert pinned_path.read_bytes() == REGISTRY
    else:
        assert result.exit_code == 1
        assert "SHA-256 does not match" in result.output
        execute.assert_not_called()
        assert pinned_path.read_bytes() == b"previous registry"


def test_run_cloud_refreshes_the_public_registry_into_the_workdir(
    tmp_path: Path, benchmark: Benchmark, write_run: Callable[..., Path]
) -> None:
    config = load_run_config(write_run(custom_models_path=None), benchmark)
    workdir = tmp_path / "work"

    with (
        patch("fab_benchmark_runner.cli.load_cloud_run_config", return_value=config),
        patch("fab_benchmark_runner.registry.download_public_registry") as public,
        patch(
            "fab_benchmark_runner.cli.execute_with_optional_sync",
            return_value=_output(),
        ) as execute,
    ):
        result = CliRunner().invoke(
            build_app(benchmark),
            ["run-cloud", "-c", "gs://r/runs/a/config.json", "--workdir", str(workdir)],
        )

    assert result.exit_code == 0, result.output
    expected = workdir.resolve() / "inputs" / "custom_models.yaml"
    public.assert_called_once_with(expected)
    assert execute.call_args.args[0].custom_models_path == expected


def test_run_cloud_dry_run_plans_instead_of_running(
    benchmark: Benchmark, write_run: Callable[..., Path]
) -> None:
    config = load_run_config(write_run(), benchmark)

    with (
        patch("fab_benchmark_runner.cli.load_cloud_run_config", return_value=config),
        patch("fab_benchmark_runner.cli.execute_with_optional_sync") as execute,
        patch("fab_benchmark_runner.cli.plan_with_optional_sync") as plan,
    ):
        plan.return_value.models = ()
        plan.return_value.planned = plan.return_value.skipped = 0
        plan.return_value.to_run = 0
        result = CliRunner().invoke(
            build_app(benchmark), ["run-cloud", "-c", "gs://r/a/c.json", "--dry-run"]
        )

    assert result.exit_code == 0, result.output
    execute.assert_not_called()
    plan.assert_called_once()
    assert "Dry run" in result.output


def test_unexpected_exceptions_keep_their_traceback(
    benchmark: Benchmark, write_run: Callable[..., Path]
) -> None:
    with patch(
        "fab_benchmark_runner.cli.execute_with_optional_sync",
        side_effect=RuntimeError("bug"),
    ):
        result = CliRunner().invoke(
            build_app(benchmark),
            ["run", "-c", str(write_run()), "--skip-models-download"],
        )

    assert isinstance(result.exception, RuntimeError)


def test_fetch_results_can_select_one_model(
    tmp_path: Path, benchmark: Benchmark
) -> None:
    with patch("fab_benchmark_runner.cli.RunSync") as run_sync:
        run_sync.return_value.pull.return_value = 1
        result = CliRunner().invoke(
            build_app(benchmark),
            [
                "fetch-results",
                "--src",
                "gs://r/runs/a/",
                "--run-dir",
                str(tmp_path / "a"),
                "--model",
                "openrouter/qwen",
            ],
        )

    assert result.exit_code == 0, result.output
    run_sync.assert_called_once_with(tmp_path / "a", "gs://r/runs/a/")
    run_sync.return_value.pull.assert_called_once_with(
        relative_paths={
            "responses/responses_openrouter_qwen-fff03286.jsonl",
            "models/openrouter_qwen-fff03286.json",
        }
    )
    assert "Downloaded: 1 from gs://r/runs/a/" in result.output


def test_upload_results_reconciles_the_run_directory(
    tmp_path: Path, benchmark: Benchmark
) -> None:
    with patch("fab_benchmark_runner.cli.RunSync") as run_sync:
        run_sync.return_value.push_missing_or_stale.return_value = 3
        result = CliRunner().invoke(
            build_app(benchmark),
            ["upload-results", "--run-dir", str(tmp_path), "--dest", "gs://r/runs/a/"],
        )

    assert result.exit_code == 0, result.output
    assert "Uploaded: 3 to gs://r/runs/a/" in result.output


def test_app_without_a_command_shows_help(benchmark: Benchmark) -> None:
    result = CliRunner().invoke(build_app(benchmark), [])

    assert result.exit_code == 0
    assert "run-cloud" in result.output


def _output() -> dict[str, Any]:
    return {
        "run_id": "experiment-a",
        "config_id": "c",
        "results_dir": "r",
        "executed_count": 0,
        "skipped_count": 0,
        "error_count": 0,
    }
