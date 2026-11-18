"""Cloud configs, GCS items, the sync lifecycle, and jobs sharing a run folder."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from fab_benchmark_runner import (
    Benchmark,
    ConfigError,
    RunConfig,
    execute_run,
    load_cloud_run_config,
    load_local_run_config,
    load_run_config,
)
from fab_benchmark_runner.cloud import (
    download_items_file,
    execute_with_optional_sync,
    plan_with_optional_sync,
)
from fab_benchmark_runner.run_sync import RunSync, RunSyncConflictError

ITEMS_JSONL = '{"benchmark_item_id": "item-1", "content": "Mama."}\n'
CONFIG_URI = "gs://results/test-bench/runs/experiment-a/run-config.json"
RUN_PREFIX = "test-bench/runs/experiment-a/"
RESULTS_DEST = f"gs://results/{RUN_PREFIX}"


def _echo(_config: RunConfig, model: str, item: dict, _prompt: str) -> dict:
    return {"output": f"{model}:{item['benchmark_item_id']}"}


class FakeSync:
    """Records the sync lifecycle; optionally fails pushes or the final retry."""

    events: list[str]
    fail_push: dict[str, Exception]
    fail_pending: Exception | None

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

    def push_pending(self) -> int:
        self.events.append("push_pending")
        if self.fail_pending is not None:
            raise self.fail_pending
        return 0


@pytest.fixture
def fake_sync() -> type[FakeSync]:
    FakeSync.events = []
    FakeSync.fail_push = {}
    FakeSync.fail_pending = None
    return FakeSync


@pytest.fixture
def synced_config(benchmark: Benchmark, write_run: Callable[..., Path]) -> RunConfig:
    return load_run_config(write_run(results_dest=RESULTS_DEST), benchmark)


def _execute_writing_artifacts(events: list[str]):
    def execute(config: RunConfig, *, on_artifact_write) -> dict:
        events.append("execute")
        for relative in (
            "config.json",
            "responses/responses_model-a.jsonl",
            "models/model-a.json",
        ):
            path = config.run_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")
            on_artifact_write(path)
        return {"run_id": config.run_id}

    return execute


def test_synced_run_pulls_then_pushes_its_own_files(
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
        "push:model-a.json",
        "push_pending",
    ]
    stdout = capsys.readouterr().out
    assert f"Reading existing results from {RESULTS_DEST}" in stdout
    assert f"Wrote results to {RESULTS_DEST}" in stdout


def test_transient_upload_failures_warn_and_are_retried_at_the_end(
    synced_config: RunConfig,
    fake_sync: type[FakeSync],
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_sync.fail_push = {"responses_model-a.jsonl": RuntimeError("network")}

    with patch("fab_benchmark_runner.cloud.RunSync", fake_sync):
        execute_with_optional_sync(
            synced_config, execute=_execute_writing_artifacts(fake_sync.events)
        )

    assert fake_sync.events[-1] == "push_pending"
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
    assert fake_sync.events[-1] == "push_pending"


def test_a_failed_final_retry_does_not_mask_the_run_failure(
    synced_config: RunConfig,
    fake_sync: type[FakeSync],
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_sync.fail_pending = RuntimeError("retry failed")

    def fail(_config: RunConfig, **_kwargs: Any) -> dict:
        raise KeyError("run failed")

    with (
        patch("fab_benchmark_runner.cloud.RunSync", fake_sync),
        pytest.raises(KeyError, match="run failed"),
    ):
        execute_with_optional_sync(synced_config, execute=fail)

    assert "sync cleanup failed" in capsys.readouterr().err


def test_a_failed_final_retry_after_a_successful_run_propagates(
    synced_config: RunConfig, fake_sync: type[FakeSync]
) -> None:
    fake_sync.fail_pending = RuntimeError("retry failed")

    with (
        patch("fab_benchmark_runner.cloud.RunSync", fake_sync),
        pytest.raises(RuntimeError, match="retry failed"),
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


# Jobs sharing one run folder, against an in-memory GCS


@pytest.fixture
def shared_folder(
    benchmark: Benchmark,
    write_run: Callable[..., Path],
    storage,
):
    """Load configs for jobs that each have a local folder but share one GCS run."""

    groups = {f"provider-{m}": [m] for m in ("model-a", "model-b", "model-c")}

    def job_config(name: str, models: list[str], **overrides: Any) -> RunConfig:
        config_path = write_run(
            models=models,
            groups=groups,
            output_dir=f"job-{name}",
            results_dest=RESULTS_DEST,
            **overrides,
        )
        return load_run_config(config_path, benchmark)

    def make_sync(run_dir: Path, dest: str) -> RunSync:
        return RunSync(run_dir, dest, storage_client=storage)

    with patch("fab_benchmark_runner.cloud.RunSync", make_sync):
        yield job_config


def _run_job(config: RunConfig, evaluator=_echo) -> dict:
    return execute_with_optional_sync(
        config,
        execute=lambda c, **kwargs: execute_run(c, evaluator=evaluator, **kwargs),
    )


def _remote(storage) -> dict[str, bytes]:
    return {
        name.removeprefix(RUN_PREFIX): content
        for name, (_generation, content) in storage.objects.get("results", {}).items()
    }


def test_jobs_running_other_models_share_a_run_folder(shared_folder, storage) -> None:
    """Job B runs model-b from start to finish while job A is running model-a."""

    job_a = shared_folder("a", ["model-a"])
    job_b = shared_folder("b", ["model-b"])
    b_outputs: list[dict] = []

    def evaluator_a(config: RunConfig, model: str, item: dict, prompt: str) -> dict:
        if not b_outputs:
            b_outputs.append(_run_job(job_b))
        return _echo(config, model, item, prompt)

    output_a = _run_job(job_a, evaluator=evaluator_a)

    assert b_outputs[0]["executed_count"] == 2
    assert output_a["executed_count"] == 2
    assert b_outputs[0]["config_id"] == output_a["config_id"]
    remote = _remote(storage)
    assert sorted(remote) == [
        "config.json",
        "models/model-a.json",
        "models/model-b.json",
        "responses/responses_model-a.jsonl",
        "responses/responses_model-b.jsonl",
    ]
    assert json.loads(remote["config.json"])["config_id"] == output_a["config_id"]
    assert len(remote["responses/responses_model-b.jsonl"].splitlines()) == 2


def test_adding_a_model_later_reuses_the_folder(shared_folder, storage) -> None:
    _run_job(shared_folder("first", ["model-a", "model-b"]))
    record_before = _remote(storage)["config.json"]
    calls: list[str] = []

    def evaluator(config: RunConfig, model: str, item: dict, prompt: str) -> dict:
        calls.append(model)
        return _echo(config, model, item, prompt)

    output = _run_job(
        shared_folder("later", ["model-a", "model-b", "model-c"]), evaluator=evaluator
    )

    assert calls == ["model-c", "model-c"]
    assert output["skipped_count"] == 4
    remote = _remote(storage)
    assert remote["config.json"] == record_before
    assert "models/model-c.json" in remote


def test_jobs_running_the_same_model_conflict(shared_folder, storage) -> None:
    """Two jobs paying for the same calls are stopped, not merged."""

    job_a = shared_folder("a", ["model-a"])
    job_b = shared_folder("b", ["model-a"])
    b_outputs: list[dict] = []

    def evaluator_a(config: RunConfig, model: str, item: dict, prompt: str) -> dict:
        if not b_outputs:
            b_outputs.append(_run_job(job_b))
        return _echo(config, model, item, prompt)

    with pytest.raises(RunSyncConflictError, match="responses_model-a.jsonl"):
        _run_job(job_a, evaluator=evaluator_a)

    assert len(_remote(storage)["responses/responses_model-a.jsonl"].splitlines()) == 2


def test_a_job_with_another_config_id_is_refused(shared_folder, storage) -> None:
    _run_job(shared_folder("a", ["model-a"]))
    changed = shared_folder("changed", ["model-b"], prompt_name="clusters")

    with pytest.raises(ConfigError, match="config_id does not match"):
        _run_job(changed)

    assert "models/model-b.json" not in _remote(storage)


# Loading cloud and local configs


def test_cloud_config_downloads_its_items(
    tmp_path: Path, benchmark: Benchmark, capsys: pytest.CaptureFixture[str]
) -> None:
    raw_config = {
        "run_id": "experiment-a",
        "language": "sw",
        "items_path": "gs://datasets/test-bench/items.jsonl",
        "models": ["model-a"],
    }

    config = _load_cloud(tmp_path, benchmark, raw_config)

    workdir = tmp_path / "work"
    assert (
        config.items_path
        == (workdir / "inputs" / "experiment-a" / "benchmark-items.jsonl").resolve()
    )
    assert config.items_path.read_text(encoding="utf-8") == ITEMS_JSONL
    assert config.output_dir == (workdir / "outputs").resolve()
    assert config.inputs_dir == (workdir / "inputs").resolve()
    assert config.custom_models_path is None
    assert config.results_dest == RESULTS_DEST
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


def test_a_run_record_is_not_a_config(tmp_path: Path, benchmark: Benchmark) -> None:
    record = {"benchmark": "test-bench", "config_id": "abc", "items_sha256": "def"}

    with pytest.raises(ConfigError, match="is a run folder's record, not a run config"):
        _load_cloud(tmp_path, benchmark, record)


def test_a_config_cannot_live_where_its_run_record_goes(
    tmp_path: Path, benchmark: Benchmark
) -> None:
    raw_config = {
        "language": "sw",
        "items_path": "gs://datasets/items.jsonl",
        "models": ["model-a"],
    }

    with pytest.raises(ConfigError, match="keeps its run record"):
        _load_cloud(
            tmp_path, benchmark, raw_config, config_uri=f"{RESULTS_DEST}config.json"
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


def _fake_items_download(**kwargs: Any) -> int:
    assert (kwargs["bucket_name"], kwargs["file_name"]) in {
        ("datasets", "test-bench/items.jsonl"),
        ("datasets", "items.jsonl"),
        ("datasets", "level1.parquet"),
    }
    Path(kwargs["write_path"]).write_text(ITEMS_JSONL, encoding="utf-8")
    return 17


def _load_cloud(
    tmp_path: Path,
    benchmark: Benchmark,
    payload: Any,
    *,
    config_uri: str = CONFIG_URI,
) -> RunConfig:
    def fake_config_download(**kwargs: Any) -> None:
        assert f"gs://{kwargs['bucket_name']}/{kwargs['file_name']}" == config_uri
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
        return load_cloud_run_config(config_uri, benchmark, workdir=tmp_path / "work")
