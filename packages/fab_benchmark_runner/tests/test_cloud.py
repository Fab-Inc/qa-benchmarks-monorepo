"""Cloud configs, GCS items, resume checks and the sync lifecycle."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from fab_benchmark_runner import (
    Benchmark,
    ConfigError,
    RunConfig,
    config_identity_sha256,
    execute_run,
    load_cloud_run_config,
    load_local_run_config,
    load_run_config,
)
from fab_benchmark_runner.cloud import (
    download_items_file,
    execute_with_optional_sync,
    plan_with_optional_sync,
    validate_resume_snapshot,
)
from fab_benchmark_runner.run_sync import RunSyncConflictError

ITEMS_JSONL = '{"benchmark_item_id": "item-1", "content": "Mama."}\n'
CONFIG_URI = "gs://results/test-bench/runs/experiment-a/config.json"


@pytest.fixture
def synced_config(benchmark: Benchmark, write_run: Callable[..., Path]) -> RunConfig:
    return load_run_config(
        write_run(results_dest="gs://results/test-bench/runs/experiment-a/"),
        benchmark,
    )


class FakeSync:
    """Records the sync lifecycle; optionally fails pushes or reconciliation."""

    events: list[str]
    fail_push: dict[str, Exception]
    fail_reconcile: Exception | None

    def __init__(self, run_dir: Path, dest: str) -> None:
        self.run_dir = run_dir
        self.dest = dest

    def pull(self, **_kwargs: Any) -> int:
        self.events.append("pull")
        return 0

    def push_file(self, local_path: Path) -> None:
        self.events.append(f"push:{local_path.name}")
        if local_path.name in self.fail_push:
            raise self.fail_push[local_path.name]

    def push_missing_or_stale(self) -> int:
        self.events.append("reconcile")
        if self.fail_reconcile is not None:
            raise self.fail_reconcile
        return 0


@pytest.fixture
def fake_sync() -> type[FakeSync]:
    FakeSync.events = []
    FakeSync.fail_push = {}
    FakeSync.fail_reconcile = None
    return FakeSync


def _execute_writing_artifacts(events: list[str]):
    def execute(config: RunConfig, *, on_artifact_write) -> dict:
        events.append("execute")
        run_dir = config.run_dir
        for relative in (
            "config.json",
            "responses/responses_model-a.jsonl",
            "log.json",
        ):
            path = run_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")
            on_artifact_write(path)
        return {"run_id": config.run_id}

    return execute


def test_synced_run_pulls_then_pushes_checkpoints_and_reconciles(
    synced_config: RunConfig,
    fake_sync: type[FakeSync],
    capsys: pytest.CaptureFixture[str],
) -> None:
    with patch("fab_benchmark_runner.cloud.RunSync", fake_sync):
        output = execute_with_optional_sync(
            synced_config, execute=_execute_writing_artifacts(fake_sync.events)
        )

    assert output == {"run_id": "experiment-a"}
    assert fake_sync.events == [
        "pull",
        "execute",
        "push:config.json",
        "push:responses_model-a.jsonl",
        "push:log.json",
        "push:log.json",
        "reconcile",
    ]
    stdout = capsys.readouterr().out
    assert "Reading existing results from gs://results/test-bench/runs/" in stdout
    assert "Wrote results to gs://results/test-bench/runs/experiment-a/" in stdout


def test_transient_checkpoint_failures_warn_and_reconcile_later(
    synced_config: RunConfig,
    fake_sync: type[FakeSync],
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_sync.fail_push = {"responses_model-a.jsonl": RuntimeError("network")}

    with patch("fab_benchmark_runner.cloud.RunSync", fake_sync):
        execute_with_optional_sync(
            synced_config, execute=_execute_writing_artifacts(fake_sync.events)
        )

    assert fake_sync.events[-1] == "reconcile"
    assert "WARNING: could not upload" in capsys.readouterr().err


def test_generation_conflicts_abort_the_run(
    synced_config: RunConfig, fake_sync: type[FakeSync]
) -> None:
    fake_sync.fail_push = {"config.json": RunSyncConflictError("changed")}

    with (
        patch("fab_benchmark_runner.cloud.RunSync", fake_sync),
        pytest.raises(RunSyncConflictError),
    ):
        execute_with_optional_sync(
            synced_config, execute=_execute_writing_artifacts(fake_sync.events)
        )

    assert "push:responses_model-a.jsonl" not in fake_sync.events
    assert fake_sync.events[-1] == "reconcile"


def test_reconcile_failure_does_not_mask_the_run_failure(
    synced_config: RunConfig,
    fake_sync: type[FakeSync],
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_sync.fail_reconcile = RuntimeError("reconcile failed")

    def fail(_config: RunConfig, **_kwargs: Any) -> dict:
        raise KeyError("run failed")

    with (
        patch("fab_benchmark_runner.cloud.RunSync", fake_sync),
        pytest.raises(KeyError, match="run failed"),
    ):
        execute_with_optional_sync(synced_config, execute=fail)

    assert "sync cleanup failed" in capsys.readouterr().err


def test_reconcile_failure_after_a_successful_run_propagates(
    synced_config: RunConfig, fake_sync: type[FakeSync]
) -> None:
    fake_sync.fail_reconcile = RuntimeError("reconcile failed")

    with (
        patch("fab_benchmark_runner.cloud.RunSync", fake_sync),
        pytest.raises(RuntimeError, match="reconcile failed"),
    ):
        execute_with_optional_sync(
            synced_config, execute=_execute_writing_artifacts(fake_sync.events)
        )


def test_unsynced_runs_execute_locally(
    benchmark: Benchmark, write_run: Callable[..., Path]
) -> None:
    config = load_run_config(write_run(), benchmark)
    calls: list[RunConfig] = []

    execute_with_optional_sync(config, execute=lambda c: calls.append(c) or {})

    assert calls == [config]


def test_dry_run_pulls_existing_results_but_never_pushes(
    synced_config: RunConfig, fake_sync: type[FakeSync]
) -> None:
    with patch("fab_benchmark_runner.cloud.RunSync", fake_sync):
        plan = plan_with_optional_sync(synced_config, plan=lambda _c: "the plan")

    assert plan == "the plan"
    assert fake_sync.events == ["pull"]


def test_resume_guard_rejects_changed_config_or_items(
    tmp_path: Path, benchmark: Benchmark, write_run: Callable[..., Path]
) -> None:
    config = load_run_config(write_run(), benchmark)
    execute_run(config, evaluator=lambda *_a: {"output": "x"})
    snapshot_path = config.run_dir / "config.json"
    validate_resume_snapshot(config, snapshot_path)

    changed = load_run_config(write_run(prompt_name="clusters"), benchmark)
    with pytest.raises(ConfigError, match="config_id does not match"):
        validate_resume_snapshot(changed, snapshot_path)

    (tmp_path / "items.jsonl").write_text(ITEMS_JSONL, encoding="utf-8")
    with pytest.raises(ConfigError, match="items_sha256 does not match"):
        validate_resume_snapshot(config, snapshot_path)


def test_resume_guard_accepts_the_uploaded_source_config(
    tmp_path: Path, benchmark: Benchmark, write_run: Callable[..., Path]
) -> None:
    config = load_run_config(write_run(), benchmark)
    snapshot_path = tmp_path / "config.json"
    snapshot_path.write_text(json.dumps({"run_id": "experiment-a"}), encoding="utf-8")

    validate_resume_snapshot(config, snapshot_path)


@pytest.mark.parametrize("as_snapshot", [False, True])
def test_cloud_config_downloads_items_and_resumes_from_snapshots(
    tmp_path: Path,
    benchmark: Benchmark,
    as_snapshot: bool,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw_config = {
        "run_id": "experiment-a",
        "language": "sw",
        "items_path": "gs://datasets/test-bench/items.jsonl",
        "models": ["model-a"],
    }
    payload = (
        {
            "config_id": "prior",
            "resolved_config": {},
            "raw_config": {
                **raw_config,
                "items_path": "/workspace/inputs/items.jsonl",
                "output_dir": "/workspace/outputs",
                "metadata": {"items_source_uri": raw_config["items_path"]},
            },
        }
        if as_snapshot
        else raw_config
    )

    config = _load_cloud(tmp_path, benchmark, payload)

    workdir = tmp_path / "work"
    assert (
        config.items_path
        == (workdir / "inputs" / "experiment-a" / "benchmark-items.jsonl").resolve()
    )
    assert config.items_path.read_text(encoding="utf-8") == ITEMS_JSONL
    assert config.output_dir == (workdir / "outputs").resolve()
    assert config.inputs_dir == (workdir / "inputs").resolve()
    assert config.custom_models_path is None
    assert config.results_dest == "gs://results/test-bench/runs/experiment-a/"
    assert config.metadata["items_source_uri"] == raw_config["items_path"]
    assert config.metadata["items_source_generation"] == 17
    assert config.raw_config["items_path"] == raw_config["items_path"]
    assert config.config_path == workdir / "inputs" / "run-config-source.json"
    stdout = capsys.readouterr().out
    assert f"Reading config from {CONFIG_URI}" in stdout
    assert "Reading items from gs://datasets/test-bench/items.jsonl" in stdout


def test_cloud_config_keeps_the_items_file_extension(
    tmp_path: Path, benchmark: Benchmark
) -> None:
    config = _load_cloud(
        tmp_path,
        benchmark,
        {
            "language": "sw",
            "items_path": "gs://datasets/level1.parquet",
            "models": ["model-a"],
        },
    )

    assert config.items_path.name == "benchmark-items.parquet"


def test_cloud_config_ignores_a_local_registry_path(
    tmp_path: Path, benchmark: Benchmark, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _load_cloud(
        tmp_path,
        benchmark,
        {
            "language": "sw",
            "items_path": "gs://datasets/items.jsonl",
            "models": ["model-a"],
            "custom_models_path": "/home/me/custom_models.yaml",
            "results_dest": "gs://elsewhere/runs/a/",
        },
    )

    assert config.custom_models_path is None
    assert config.results_dest == "gs://elsewhere/runs/a/"
    assert "Ignoring custom_models_path" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("config_uri", "message"),
    [
        ("not-a-gcs-uri", "Invalid cloud config URI"),
        ("gs://results/config.json", "must be inside a run prefix"),
    ],
)
def test_cloud_config_rejects_invalid_uris(
    tmp_path: Path, benchmark: Benchmark, config_uri: str, message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        load_cloud_run_config(config_uri, benchmark, workdir=tmp_path)


def test_cloud_snapshot_needs_an_object_raw_config(
    tmp_path: Path, benchmark: Benchmark
) -> None:
    with pytest.raises(ConfigError, match="raw_config must be an object"):
        _load_cloud(
            tmp_path, benchmark, {"resolved_config": {}, "raw_config": ["not", "dict"]}
        )


def test_local_config_downloads_gcs_items_into_its_inputs_folder(
    tmp_path: Path, benchmark: Benchmark, write_registry: Callable[..., Path]
) -> None:
    write_registry()
    config_path = tmp_path / "run.json"
    config_path.write_text(
        json.dumps(
            {
                "run_id": "experiment-a",
                "language": "sw",
                "items_path": "gs://datasets/items.jsonl",
                "models": ["model-a"],
            }
        ),
        encoding="utf-8",
    )

    with patch(
        "fab_benchmark_runner.gcs.download_file_with_generation",
        side_effect=_fake_items_download,
    ):
        config = load_local_run_config(config_path, benchmark)

    assert config.items_path == (
        tmp_path / "inputs" / "experiment-a" / "benchmark-items.jsonl"
    )
    assert config.raw_config["items_path"] == "gs://datasets/items.jsonl"
    assert config.config_path == config_path


def test_local_config_with_local_items_downloads_nothing(
    benchmark: Benchmark, write_run: Callable[..., Path]
) -> None:
    with patch("fab_benchmark_runner.gcs.download_file_with_generation") as download:
        config = load_local_run_config(write_run(), benchmark)

    download.assert_not_called()
    assert config == load_run_config(write_run(), benchmark)


def test_items_download_is_atomic(tmp_path: Path) -> None:
    destination = tmp_path / "items.jsonl"
    destination.write_text("old\n", encoding="utf-8")

    def fail(**kwargs: Any) -> int:
        Path(kwargs["write_path"]).write_text("partial", encoding="utf-8")
        raise ConnectionError("interrupted")

    with (
        patch(
            "fab_benchmark_runner.gcs.download_file_with_generation", side_effect=fail
        ),
        pytest.raises(ConnectionError),
    ):
        download_items_file("gs://datasets/items.jsonl", destination)

    assert destination.read_text(encoding="utf-8") == "old\n"
    assert list(tmp_path.iterdir()) == [destination]
    with pytest.raises(ConfigError, match="Invalid cloud items_path"):
        download_items_file("https://datasets/items.jsonl", destination)


def test_snapshot_resume_keeps_the_config_id(
    tmp_path: Path, benchmark: Benchmark
) -> None:
    fresh = _load_cloud(
        tmp_path,
        benchmark,
        {"language": "sw", "items_path": "gs://d/items.jsonl", "models": ["m"]},
    )
    resumed = _load_cloud(
        tmp_path,
        benchmark,
        {"config_id": "x", "resolved_config": {}, "raw_config": fresh.raw_config},
    )

    assert config_identity_sha256(resumed) == config_identity_sha256(fresh)
    assert replace(resumed, metadata={}) == replace(fresh, metadata={})


def _fake_items_download(**kwargs: Any) -> int:
    assert (kwargs["bucket_name"], kwargs["file_name"]) in {
        ("datasets", "test-bench/items.jsonl"),
        ("datasets", "items.jsonl"),
        ("datasets", "level1.parquet"),
        ("d", "items.jsonl"),
    }
    Path(kwargs["write_path"]).write_text(ITEMS_JSONL, encoding="utf-8")
    return 17


def _load_cloud(tmp_path: Path, benchmark: Benchmark, payload: Any) -> RunConfig:
    def fake_config_download(**kwargs: Any) -> None:
        assert (kwargs["bucket_name"], kwargs["file_name"]) == (
            "results",
            "test-bench/runs/experiment-a/config.json",
        )
        assert kwargs.get("require_auth", True) is True
        Path(kwargs["write_path"]).write_text(json.dumps(payload), encoding="utf-8")

    with (
        patch(
            "fab_benchmark_runner.gcs.download_file", side_effect=fake_config_download
        ),
        patch(
            "fab_benchmark_runner.gcs.download_file_with_generation",
            side_effect=_fake_items_download,
        ),
    ):
        return load_cloud_run_config(CONFIG_URI, benchmark, workdir=tmp_path / "work")
