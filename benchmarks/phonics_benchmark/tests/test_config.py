"""Tests for phonics benchmark run configuration."""

import json
from pathlib import Path

import pytest
from phonics_benchmark.config import (
    ConfigError,
    config_identity_sha256,
    load_run_config,
)


def test_load_run_config_resolves_paths_and_models(tmp_path: Path) -> None:
    """Config paths are resolved relative to the config file."""

    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_API_KEY=test", encoding="utf-8")
    data_root = tmp_path / "data"
    config_path = tmp_path / "configs" / "run.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "run_id": "experiment-a",
                "language": "english",
                "task": "decodability",
                "data_root": "../data",
                "output_dir": "../outputs",
                "env_path": "../.env",
                "chunk_size": 50,
                "models": [
                    "model-a",
                    "model-b",
                ],
                "metadata": {"owner": "benchmarks"},
                "future_field": "preserved",
            }
        ),
        encoding="utf-8",
    )

    config = load_run_config(config_path)

    assert config.run_id == "experiment-a"
    assert config.language == "english"
    assert config.data_root == data_root.resolve()
    assert config.output_dir == (tmp_path / "outputs").resolve()
    assert config.env_path == env_path.resolve()
    assert config.chunk_size == 50
    assert config.models == ["model-a", "model-b"]
    assert config.extra == {"future_field": "preserved"}


def test_load_run_config_requires_non_empty_models(tmp_path: Path) -> None:
    """A run without models is rejected early."""

    config_path = tmp_path / "run.json"
    config_path.write_text(
        json.dumps(
            {
                "run_id": "experiment-a",
                "language": "english",
                "models": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="models"):
        load_run_config(config_path, default_data_root=tmp_path / "data")


def test_load_run_config_defaults_run_id(tmp_path: Path) -> None:
    """A missing run_id defaults to the reusable default run."""

    config_path = tmp_path / "run.json"
    config_path.write_text(
        json.dumps(
            {
                "language": "english",
                "models": ["model-a"],
            }
        ),
        encoding="utf-8",
    )

    config = load_run_config(config_path, default_data_root=tmp_path / "data")

    assert config.run_id == "default"
    assert config.data_root == (tmp_path / "data").resolve()
    assert config.chunk_size == 100


def test_config_identity_includes_default_data_root(tmp_path: Path) -> None:
    """A caller-provided default data root is part of result identity."""

    config_path = tmp_path / "run.json"
    config_path.write_text(
        json.dumps(
            {
                "language": "english",
                "models": ["model-a"],
            }
        ),
        encoding="utf-8",
    )

    first = load_run_config(config_path, default_data_root=tmp_path / "data-a")
    second = load_run_config(config_path, default_data_root=tmp_path / "data-b")

    assert config_identity_sha256(first) != config_identity_sha256(second)


def test_config_identity_excludes_retry_controls(tmp_path: Path) -> None:
    """Retry and checkpoint controls do not change cached row identity."""

    config_path = tmp_path / "run.json"
    config_path.write_text(
        json.dumps(
            {
                "language": "english",
                "models": ["model-a"],
                "chunk_size": 100,
                "retry_errors": False,
            }
        ),
        encoding="utf-8",
    )
    original = load_run_config(config_path, default_data_root=tmp_path / "data")

    config_path.write_text(
        json.dumps(
            {
                "language": "english",
                "models": ["model-a"],
                "chunk_size": 1,
                "retry_errors": True,
            }
        ),
        encoding="utf-8",
    )
    retry = load_run_config(config_path, default_data_root=tmp_path / "data")

    assert config_identity_sha256(original) == config_identity_sha256(retry)


def test_load_run_config_requires_data_root_without_default(tmp_path: Path) -> None:
    """A config without data_root needs a caller-provided default."""

    config_path = tmp_path / "run.json"
    config_path.write_text(
        json.dumps(
            {
                "language": "english",
                "models": ["model-a"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="data_root"):
        load_run_config(config_path)


def test_load_run_config_rejects_items_path(tmp_path: Path) -> None:
    """Benchmark data is selected by language, not by config path."""

    config_path = tmp_path / "run.json"
    config_path.write_text(
        json.dumps(
            {
                "language": "kiswahili",
                "items_path": "items.json",
                "models": ["model-a"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="items_path"):
        load_run_config(config_path, default_data_root=tmp_path / "data")


def test_load_run_config_allows_save_at_end_chunk_size(tmp_path: Path) -> None:
    """A null chunk_size means each model saves once after all pending items."""

    config_path = tmp_path / "run.json"
    config_path.write_text(
        json.dumps(
            {
                "language": "english",
                "models": ["model-a"],
                "chunk_size": None,
            }
        ),
        encoding="utf-8",
    )

    config = load_run_config(config_path, default_data_root=tmp_path / "data")

    assert config.chunk_size is None


def test_load_run_config_rejects_invalid_chunk_size(tmp_path: Path) -> None:
    """Chunk size must be positive when provided."""

    config_path = tmp_path / "run.json"
    config_path.write_text(
        json.dumps(
            {
                "language": "english",
                "models": ["model-a"],
                "chunk_size": 0,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="chunk_size"):
        load_run_config(config_path, default_data_root=tmp_path / "data")


def test_load_run_config_rejects_invalid_run_id(tmp_path: Path) -> None:
    """Run IDs are restricted to alphanumeric characters and hyphens."""

    config_path = tmp_path / "run.json"
    config_path.write_text(
        json.dumps(
            {
                "run_id": "bad_run",
                "language": "english",
                "models": ["model-a"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="run_id"):
        load_run_config(config_path, default_data_root=tmp_path / "data")


def test_load_run_config_rejects_model_objects(tmp_path: Path) -> None:
    """Model provider parameters must live in custom_models.yaml."""

    config_path = tmp_path / "run.json"
    config_path.write_text(
        json.dumps(
            {
                "run_id": "experiment-a",
                "language": "english",
                "models": [{"name": "model-a", "parameters": {"temperature": 0}}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="custom_models.yaml"):
        load_run_config(config_path, default_data_root=tmp_path / "data")
