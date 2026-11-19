"""Validating run configs: what each key accepts and how paths resolve."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fab_benchmark_runner import (
    ConfigError,
    apply_execution_overrides,
    load_run_config,
    parse_run_config,
)
from fab_benchmark_runner.config import DEFAULT_CHUNK_SIZE, DEFAULT_MODEL_WORKERS

SHA = "a" * 64


def _parse(tmp_path: Path, **raw: Any):
    return parse_run_config(
        {"experiment": "exp", "models": ["model-a"], **raw}, base_dir=tmp_path
    )


def test_defaults_and_relative_paths(tmp_path: Path) -> None:
    config = _parse(tmp_path, custom_models_path="models.yaml", env_path="../.env")

    assert config.experiment == str((tmp_path / "exp").resolve())
    assert config.custom_models_path == (tmp_path / "models.yaml").resolve()
    assert config.env_path == (tmp_path.parent / ".env").resolve()
    assert (config.model_workers, config.chunk_size) == (
        DEFAULT_MODEL_WORKERS,
        DEFAULT_CHUNK_SIZE,
    )
    assert (config.force, config.retry_errors) == (False, False)
    assert config.models == ("model-a",)


def test_an_experiment_on_gcs_keeps_its_uri(tmp_path: Path) -> None:
    config = _parse(tmp_path, experiment="gs://bucket/lk/sw-v1/")

    assert config.experiment == "gs://bucket/lk/sw-v1"
    assert config.experiment_is_gcs


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"experiment": ""}, "'experiment' must name"),
        ({"experiment": "gs://bucket"}, "Invalid 'experiment'"),
        ({"models": []}, "'models' must be a non-empty list"),
        ({"models": [{"name": "a"}]}, "custom_models.yaml"),
        ({"models": ["a", "a"]}, "duplicate: a"),
        ({"models": [" "]}, "non-empty strings"),
        ({"chunk_size": 0}, "'chunk_size'"),
        ({"chunk_size": True}, "'chunk_size'"),
        ({"model_workers": 0}, "'model_workers'"),
        ({"force": "yes"}, "booleans"),
        ({"force": True, "retry_errors": True}, "cannot be combined"),
        ({"custom_models_sha256": SHA}, "requires custom_models_uri"),
        ({"custom_models_uri": "gs://b/m.yaml"}, "lowercase SHA-256"),
        ({"custom_models_uri": "s3://b/m.yaml", "custom_models_sha256": SHA}, "gs://"),
        (
            {
                "custom_models_uri": "gs://b/m.yaml",
                "custom_models_sha256": SHA,
                "custom_models_path": "m.yaml",
            },
            "not both",
        ),
        ({"env_path": ""}, "'env_path'"),
        ({"results_dest": "gs://b/r"}, "experiment's folder"),
        ({"output_dir": "out"}, "experiment's folder"),
        ({"model_config": {}}, "Unknown run config keys: model_config"),
    ],
)
def test_invalid_configs_are_refused(
    tmp_path: Path, raw: dict[str, Any], message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        _parse(tmp_path, **raw)


def test_a_config_from_gcs_can_only_use_gcs(tmp_path: Path) -> None:
    def parse(**raw: Any):
        return parse_run_config(
            {"experiment": "gs://bucket/exp", "models": ["model-a"], **raw},
            base_dir=None,
            source="gs://bucket/run-configs/x.json",
        )

    assert parse(env_path=".env").env_path is None
    with pytest.raises(ConfigError, match="custom_models_uri"):
        parse(custom_models_path="models.yaml")
    with pytest.raises(ConfigError, match="upload-experiment"):
        parse(experiment="exp")


def test_load_reads_local_files_and_gcs_uris(tmp_path: Path, storage: Any) -> None:
    raw = {"experiment": "gs://bucket/exp", "models": ["model-a"]}
    local = tmp_path / "run.json"
    local.write_text(json.dumps(raw), encoding="utf-8")
    storage.objects["bucket"] = {"run-configs/x.json": (3, json.dumps(raw).encode())}

    from_file = load_run_config(local)
    from_gcs = load_run_config("gs://bucket/run-configs/x.json", storage_client=storage)

    assert from_file.source == str(local.resolve())
    assert from_gcs.source == "gs://bucket/run-configs/x.json"
    assert from_file.experiment == from_gcs.experiment == "gs://bucket/exp"


def test_load_reports_missing_and_malformed_configs(
    tmp_path: Path, storage: Any
) -> None:
    (tmp_path / "list.json").write_text("[]", encoding="utf-8")
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    storage.objects["bucket"] = {}

    with pytest.raises(ConfigError, match="does not exist"):
        load_run_config(tmp_path / "missing.json")
    with pytest.raises(ConfigError, match="must be a JSON object"):
        load_run_config(tmp_path / "list.json")
    with pytest.raises(ConfigError, match="Could not read"):
        load_run_config(tmp_path / "bad.json")
    with pytest.raises(ConfigError, match="does not exist"):
        load_run_config("gs://bucket/missing.json", storage_client=storage)


def test_command_line_overrides(tmp_path: Path) -> None:
    config = _parse(tmp_path)

    overridden = apply_execution_overrides(config, retry_errors=True, model_workers=2)

    assert (overridden.retry_errors, overridden.model_workers) == (True, 2)
    with pytest.raises(ConfigError, match="cannot be combined"):
        apply_execution_overrides(overridden, force=True)
