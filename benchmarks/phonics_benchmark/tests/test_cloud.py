"""Tests for GCS synchronization and the container-facing command flow."""

from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import pytest
from fab_benchmark_publisher import RunSyncConflictError
from phonics_benchmark.cmd import (
    _download_items_file,
    _execute_with_optional_sync,
    _items_source_uri,
    _load_cloud_run_config,
    _materialize_run_config,
    _read_config_object,
    _validate_resume_snapshot,
    app,
)
from phonics_benchmark.config import (
    BenchmarkRunConfig,
    ConfigError,
    build_run_config,
    config_identity_sha256,
)
from phonics_benchmark.items import load_benchmark_items
from typer.testing import CliRunner


def test_synced_lifecycle_pulls_then_pushes_checkpoints_and_reconciles(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _build_config(tmp_path)
    events: list[str] = []

    class FakeSync:
        def __init__(self, run_dir: Path, dest: str) -> None:
            assert run_dir == config.output_dir / config.run_id
            assert dest == config.results_dest
            self.run_dir = run_dir

        def pull(self) -> int:
            events.append("pull")
            return 0

        def push_file(self, local_path: Path) -> None:
            events.append(f"push:{local_path.name}")

        def push_missing_or_stale(self) -> int:
            events.append("reconcile")
            return 0

    def fake_execute(
        run_config: BenchmarkRunConfig,
        *,
        on_artifact_write,
    ) -> dict:
        events.append("execute")
        run_dir = run_config.output_dir / run_config.run_id
        response_path = run_dir / "responses" / "responses_model-a.jsonl"
        response_path.parent.mkdir(parents=True)
        config_path = run_dir / "config.json"
        config_path.write_text("{}\n", encoding="utf-8")
        on_artifact_write(config_path)
        response_path.write_text("{}\n", encoding="utf-8")
        on_artifact_write(response_path)
        log_path = run_dir / "log.json"
        log_path.write_text("{}\n", encoding="utf-8")
        on_artifact_write(log_path)
        return _output(run_config)

    with (
        patch("phonics_benchmark.cmd.RunSync", FakeSync),
        patch("phonics_benchmark.cmd.execute_run", side_effect=fake_execute),
    ):
        output = _execute_with_optional_sync(config)

    assert output["run_id"] == "experiment-a"
    assert events == [
        "pull",
        "execute",
        "push:config.json",
        "push:responses_model-a.jsonl",
        "push:log.json",
        "push:log.json",
        "reconcile",
    ]
    stdout = capsys.readouterr().out
    assert (
        "Reading existing results from "
        "gs://results/phonics/runs/experiment-a/" in stdout
    )
    assert "Writing results to gs://results/phonics/runs/experiment-a/" in stdout
    assert "Wrote results to gs://results/phonics/runs/experiment-a/" in stdout


def test_transient_checkpoint_upload_warns_and_finally_reconciles(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _build_config(tmp_path)
    events: list[str] = []

    class FakeSync:
        def __init__(self, run_dir: Path, _dest: str) -> None:
            self.run_dir = run_dir

        def pull(self) -> int:
            return 0

        def push_file(self, local_path: Path) -> None:
            if local_path.name == "config.json":
                raise RuntimeError("temporary outage")

        def push_missing_or_stale(self) -> int:
            events.append("reconcile")
            return 1

    def fake_execute(
        run_config: BenchmarkRunConfig,
        *,
        on_artifact_write,
    ) -> dict:
        config_path = run_config.output_dir / run_config.run_id / "config.json"
        config_path.parent.mkdir(parents=True)
        config_path.write_text("{}\n", encoding="utf-8")
        on_artifact_write(config_path)
        events.append("continued")
        return _output(run_config)

    with (
        patch("phonics_benchmark.cmd.RunSync", FakeSync),
        patch("phonics_benchmark.cmd.execute_run", side_effect=fake_execute),
    ):
        _execute_with_optional_sync(config)

    assert events == ["continued", "reconcile"]
    stderr = capsys.readouterr().err
    assert "WARNING" in stderr
    assert "temporary outage" in stderr


def test_generation_conflict_from_checkpoint_aborts_run(tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    reconciled = False

    class FakeSync:
        def __init__(self, run_dir: Path, _dest: str) -> None:
            self.run_dir = run_dir

        def pull(self) -> int:
            return 0

        def push_file(self, _local_path: Path) -> None:
            raise RunSyncConflictError("stale process")

        def push_missing_or_stale(self) -> int:
            nonlocal reconciled
            reconciled = True
            return 0

    def fake_execute(
        run_config: BenchmarkRunConfig,
        *,
        on_artifact_write,
    ) -> dict:
        config_path = run_config.output_dir / run_config.run_id / "config.json"
        config_path.parent.mkdir(parents=True)
        config_path.write_text("{}\n", encoding="utf-8")
        on_artifact_write(config_path)
        raise AssertionError("the conflict should have propagated")

    with (
        patch("phonics_benchmark.cmd.RunSync", FakeSync),
        patch("phonics_benchmark.cmd.execute_run", side_effect=fake_execute),
        pytest.raises(RunSyncConflictError, match="stale process"),
    ):
        _execute_with_optional_sync(config)

    assert reconciled is True


def test_reconcile_failure_does_not_mask_execution_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _build_config(tmp_path)

    class FakeSync:
        def __init__(self, _run_dir: Path, _dest: str) -> None:
            pass

        def pull(self) -> int:
            return 0

        def push_missing_or_stale(self) -> int:
            raise RuntimeError("GCS unavailable during cleanup")

    with (
        patch("phonics_benchmark.cmd.RunSync", FakeSync),
        patch(
            "phonics_benchmark.cmd.execute_run",
            side_effect=KeyError("evaluator bug"),
        ),
        pytest.raises(KeyError, match="evaluator bug"),
    ):
        _execute_with_optional_sync(config)

    assert "GCS unavailable during cleanup" in capsys.readouterr().err


def test_reconcile_failure_propagates_after_successful_execution(
    tmp_path: Path,
) -> None:
    config = _build_config(tmp_path)

    class FakeSync:
        def __init__(self, _run_dir: Path, _dest: str) -> None:
            pass

        def pull(self) -> int:
            return 0

        def push_missing_or_stale(self) -> int:
            raise RuntimeError("GCS unavailable during cleanup")

    with (
        patch("phonics_benchmark.cmd.RunSync", FakeSync),
        patch("phonics_benchmark.cmd.execute_run", return_value=_output(config)),
        pytest.raises(RuntimeError, match="GCS unavailable during cleanup"),
    ):
        _execute_with_optional_sync(config)


def test_resume_guards_reject_config_and_items_mismatches(tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    snapshot_path = tmp_path / "snapshot.json"
    _items, items_sha256 = load_benchmark_items(config.items_path)

    snapshot_path.write_text(
        json.dumps(
            {
                "config_id": "different",
                "resolved_config": {"items_sha256": items_sha256},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="config_id"):
        _validate_resume_snapshot(config, snapshot_path)

    snapshot_path.write_text(
        json.dumps(
            {
                "config_id": config_identity_sha256(config),
                "resolved_config": {"items_sha256": "different"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="items_sha256"):
        _validate_resume_snapshot(config, snapshot_path)


def test_resume_guard_accepts_initial_raw_config(tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    raw_config_path = tmp_path / "config.json"
    raw_config_path.write_text(
        json.dumps({"language": "sw", "models": ["model-a"]}),
        encoding="utf-8",
    )

    _validate_resume_snapshot(config, raw_config_path)


def test_local_execution_rejects_changed_items_from_existing_snapshot(
    tmp_path: Path,
) -> None:
    config = replace(_build_config(tmp_path), results_dest=None)
    snapshot_path = config.output_dir / config.run_id / "config.json"
    snapshot_path.parent.mkdir(parents=True)
    snapshot_path.write_text(
        json.dumps(
            {
                "config_id": config_identity_sha256(config),
                "resolved_config": {"items_sha256": "previous-items"},
            }
        ),
        encoding="utf-8",
    )

    with (
        patch("phonics_benchmark.cmd.execute_run") as execute,
        pytest.raises(ConfigError, match="items_sha256"),
    ):
        _execute_with_optional_sync(config)

    execute.assert_not_called()


@pytest.mark.parametrize("as_snapshot", [False, True])
def test_cloud_config_downloads_gcs_items_and_supports_snapshot_resume(
    tmp_path: Path,
    as_snapshot: bool,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw_config = {
        "run_id": "experiment-a",
        "language": "sw",
        "items_path": "gs://datasets/phonics/items.jsonl",
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
                "metadata": {"items_source_uri": "gs://datasets/phonics/items.jsonl"},
            },
        }
        if as_snapshot
        else raw_config
    )

    def fake_config_download(
        *,
        bucket_name: str,
        file_name: str,
        write_path: Path,
        require_auth: bool = True,
    ) -> None:
        assert require_auth is True
        destination = Path(write_path)
        assert bucket_name == "results"
        assert file_name == "phonics/runs/experiment-a/config.json"
        destination.write_text(json.dumps(payload), encoding="utf-8")

    def fake_items_download(
        *,
        bucket_name: str,
        file_name: str,
        write_path: Path,
    ) -> int:
        assert (bucket_name, file_name) == (
            "datasets",
            "phonics/items.jsonl",
        )
        Path(write_path).write_text(_item_jsonl(), encoding="utf-8")
        return 17

    with (
        patch(
            "phonics_benchmark.cmd.download_file",
            side_effect=fake_config_download,
        ),
        patch(
            "phonics_benchmark.cmd.download_file_with_generation",
            side_effect=fake_items_download,
        ),
    ):
        config = _load_cloud_run_config(
            "gs://results/phonics/runs/experiment-a/config.json",
            workdir=tmp_path,
        )

    assert config.items_path.read_text(encoding="utf-8") == _item_jsonl()
    assert config.output_dir == (tmp_path / "outputs").resolve()
    assert config.inputs_dir == (tmp_path / "inputs").resolve()
    assert (
        config.items_path
        == (tmp_path / "inputs" / "experiment-a" / "benchmark-items.jsonl").resolve()
    )
    assert (
        config.custom_models_path
        == (tmp_path / "inputs" / "custom_models.yaml").resolve()
    )
    assert config.results_dest == "gs://results/phonics/runs/experiment-a/"
    assert config.metadata["items_source_uri"] == ("gs://datasets/phonics/items.jsonl")
    assert config.metadata["items_source_generation"] == 17
    assert config.raw_config["items_path"] == "gs://datasets/phonics/items.jsonl"
    stdout = capsys.readouterr().out
    assert (
        "Reading config from "
        "gs://results/phonics/runs/experiment-a/config.json" in stdout
    )
    assert "Reading items from gs://datasets/phonics/items.jsonl" in stdout


@pytest.mark.parametrize(
    ("config_uri", "message"),
    [
        ("not-a-gcs-uri", "Invalid cloud config URI"),
        ("gs://results/config.json", "must be inside a run prefix"),
    ],
)
def test_cloud_config_rejects_invalid_uri_or_root_object(
    tmp_path: Path,
    config_uri: str,
    message: str,
) -> None:
    with pytest.raises(ConfigError, match=message):
        _load_cloud_run_config(config_uri, workdir=tmp_path)


def test_cloud_config_rejects_snapshot_without_object_raw_config(
    tmp_path: Path,
) -> None:
    def fake_config_download(**kwargs) -> None:
        Path(kwargs["write_path"]).write_text(
            json.dumps({"resolved_config": {}, "raw_config": []}),
            encoding="utf-8",
        )

    with (
        patch(
            "phonics_benchmark.cmd.download_file",
            side_effect=fake_config_download,
        ),
        pytest.raises(ConfigError, match="raw_config must be an object"),
    ):
        _load_cloud_run_config(
            "gs://results/phonics/runs/experiment-a/config.json",
            workdir=tmp_path,
        )


def test_materialization_rejects_non_object_metadata(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="metadata.*object"):
        _materialize_run_config(
            {
                "run_id": "experiment-a",
                "language": "sw",
                "items_path": "gs://datasets/items.jsonl",
                "models": ["model-a"],
                "metadata": "invalid",
            },
            source_config_path=tmp_path / "config.json",
            inputs_dir=tmp_path / "inputs" / "experiment-a",
            use_metadata_fallback=False,
        )


def test_items_source_fallback_rejects_non_object_metadata() -> None:
    with pytest.raises(ConfigError, match="metadata.*object"):
        _items_source_uri(
            {"items_path": "items.jsonl", "metadata": "invalid"},
            use_metadata_fallback=True,
        )


def test_items_download_rejects_invalid_gcs_uri(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Invalid cloud items_path"):
        _download_items_file("not-a-gcs-uri", tmp_path / "items.jsonl")


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("{", "Invalid JSON in config file"),
        ("[]", "Config file must be a JSON object"),
    ],
)
def test_read_config_object_rejects_invalid_documents(
    tmp_path: Path,
    contents: str,
    message: str,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        _read_config_object(config_path, description="config file")


def test_items_download_is_atomic_on_failure(tmp_path: Path) -> None:
    destination = tmp_path / "inputs" / "benchmark-items.jsonl"
    destination.parent.mkdir()
    destination.write_text("existing\n", encoding="utf-8")

    def fail_after_partial_download(**kwargs) -> int:
        Path(kwargs["write_path"]).write_text("partial", encoding="utf-8")
        raise RuntimeError("download interrupted")

    with (
        patch(
            "phonics_benchmark.cmd.download_file_with_generation",
            side_effect=fail_after_partial_download,
        ),
        pytest.raises(RuntimeError, match="interrupted"),
    ):
        _download_items_file(
            "gs://datasets/phonics/items.jsonl",
            destination,
        )

    assert destination.read_text(encoding="utf-8") == "existing\n"
    assert list(destination.parent.iterdir()) == [destination]


def test_fetch_results_command_filters_to_one_model_file(tmp_path: Path) -> None:
    calls: list[set[str] | None] = []

    class FakeSync:
        def __init__(self, run_dir: Path, src: str) -> None:
            assert run_dir == tmp_path / "downloaded"
            assert src == "gs://results/phonics/runs/experiment-a/"

        def pull(self, *, relative_paths=None) -> int:
            calls.append(relative_paths)
            return 1

    with patch("phonics_benchmark.cmd.RunSync", FakeSync):
        result = CliRunner().invoke(
            app,
            [
                "fetch-results",
                "--src",
                "gs://results/phonics/runs/experiment-a/",
                "--run-dir",
                str(tmp_path / "downloaded"),
                "--model",
                "provider/model-a",
            ],
        )

    assert result.exit_code == 0
    assert calls == [{"responses/responses_provider_model-a-6f6ae619.jsonl"}]
    assert "Downloaded: 1" in result.stdout
    assert "from gs://results/phonics/runs/experiment-a/" in result.stdout


def test_upload_results_command_reconciles_local_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    reconciled = False

    class FakeSync:
        def __init__(self, selected_run_dir: Path, dest: str) -> None:
            assert selected_run_dir == run_dir
            assert dest == "gs://results/phonics/runs/run-a/"

        def push_missing_or_stale(self) -> int:
            nonlocal reconciled
            reconciled = True
            return 3

    with patch("phonics_benchmark.cmd.RunSync", FakeSync):
        result = CliRunner().invoke(
            app,
            [
                "upload-results",
                "--run-dir",
                str(run_dir),
                "--dest",
                "gs://results/phonics/runs/run-a/",
            ],
        )

    assert result.exit_code == 0
    assert reconciled is True
    assert "Uploaded: 3" in result.stdout
    assert "to gs://results/phonics/runs/run-a/" in result.stdout


@pytest.mark.parametrize(
    ("command", "patch_target"),
    [
        (
            [
                "upload-results",
                "--run-dir",
                "{run_dir}",
                "--dest",
                "gs://results/phonics/runs/run-a/",
            ],
            "phonics_benchmark.cmd.RunSync",
        ),
        (
            [
                "fetch-results",
                "--src",
                "gs://results/phonics/runs/run-a/",
                "--run-dir",
                "{run_dir}",
            ],
            "phonics_benchmark.cmd.RunSync",
        ),
        (
            [
                "run-cloud",
                "--config",
                "gs://results/phonics/runs/run-a/config.json",
                "--workdir",
                "{run_dir}",
            ],
            "phonics_benchmark.cmd._load_cloud_run_config",
        ),
    ],
)
def test_cloud_commands_preserve_unexpected_exception_tracebacks(
    tmp_path: Path,
    command: list[str],
    patch_target: str,
) -> None:
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    args = [argument.format(run_dir=run_dir) for argument in command]

    with patch(patch_target, side_effect=KeyError("items_sha256")):
        result = CliRunner().invoke(app, args)

    assert result.exit_code == 1
    assert isinstance(result.exception, KeyError)
    assert "ERROR:" not in result.output


def test_run_cloud_downloads_models_to_writable_workdir(tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    models_path = tmp_path / "inputs" / "custom_models.yaml"
    config = replace(config, custom_models_path=models_path)

    with (
        patch("phonics_benchmark.cmd._load_cloud_run_config", return_value=config),
        patch("phonics_benchmark.cmd.download_default_custom_models") as download,
        patch(
            "phonics_benchmark.cmd._execute_with_optional_sync",
            return_value=_output(config),
        ),
    ):
        result = CliRunner().invoke(
            app,
            [
                "run-cloud",
                "--config",
                "gs://results/phonics/runs/experiment-a/config.json",
                "--workdir",
                str(tmp_path),
            ],
        )

    assert result.exit_code == 0
    download.assert_called_once_with(models_path)


@pytest.mark.parametrize("skip_download", [False, True])
@pytest.mark.parametrize("valid_checksum", [False, True])
def test_run_cloud_uses_pinned_registry_and_stops_on_changed_content(
    tmp_path: Path, skip_download: bool, valid_checksum: bool
) -> None:
    config = _build_config(tmp_path)
    registry = b"provider:\n  models:\n    model-a: {}\n"
    models_path = tmp_path / "inputs" / "custom_models.yaml"
    models_path.parent.mkdir()
    models_path.write_bytes(b"previous registry")
    config = replace(
        config,
        custom_models_path=models_path,
        raw_config={
            **config.raw_config,
            "custom_models_uri": "gs://results/model-configs/experiment-a.yaml",
            "custom_models_sha256": sha256(registry).hexdigest(),
        },
    )

    def fake_download(*, bucket_name, file_name, write_path):
        assert (bucket_name, file_name) == (
            "results",
            "model-configs/experiment-a.yaml",
        )
        Path(write_path).write_bytes(registry if valid_checksum else b"changed")

    with (
        patch("phonics_benchmark.cmd._load_cloud_run_config", return_value=config),
        patch("phonics_benchmark.cmd.download_file", side_effect=fake_download),
        patch("phonics_benchmark.cmd.download_default_custom_models") as refresh,
        patch(
            "phonics_benchmark.cmd._execute_with_optional_sync",
            return_value=_output(config),
        ) as execute,
    ):
        args = ["run-cloud", "--config", "gs://results/configs/experiment-a.json"]
        if skip_download:
            args.append("--skip-models-download")
        result = CliRunner().invoke(app, args)

    refresh.assert_not_called()
    assert not list(models_path.parent.glob(".custom-models.*"))
    if valid_checksum:
        assert result.exit_code == 0, result.output
        execute.assert_called_once_with(config)
        assert models_path.read_bytes() == registry
    else:
        assert result.exit_code == 1
        assert "SHA-256 does not match" in result.output
        execute.assert_not_called()
        assert models_path.read_bytes() == b"previous registry"


def _build_config(tmp_path: Path) -> BenchmarkRunConfig:
    items_path = tmp_path / "items.jsonl"
    items_path.write_text(_item_jsonl(), encoding="utf-8")
    return build_run_config(
        run_id="experiment-a",
        language="sw",
        output_dir=tmp_path / "outputs",
        models=["model-a"],
        base_dir=tmp_path,
        task="decodability",
        items_path=items_path,
        results_dest="gs://results/phonics/runs/experiment-a/",
    )


def _item_jsonl() -> str:
    return (
        json.dumps(
            {
                "benchmark_item_id": "item-1",
                "content": "Mama.",
                "student_knowledge": {"graphemes": ["m", "a"]},
            }
        )
        + "\n"
    )


def _output(config: BenchmarkRunConfig) -> dict:
    run_dir = config.output_dir / config.run_id
    return {
        "run_id": config.run_id,
        "config_id": config_identity_sha256(config),
        "run_dir": run_dir,
        "config_snapshot_path": run_dir / "config.json",
        "results_dir": run_dir / "responses",
        "results_paths": {},
        "total_expected_results": 1,
        "executed_count": 1,
        "skipped_count": 0,
        "error_count": 0,
    }
