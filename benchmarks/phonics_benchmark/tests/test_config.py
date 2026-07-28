"""Tests for phonics benchmark run configuration."""

import json
from hashlib import sha256
from pathlib import Path

import pytest
from phonics_benchmark.config import (
    ConfigError,
    apply_execution_overrides,
    config_identity_sha256,
    config_object_sha256,
    config_sha256,
    load_run_config,
)


def _write_config(tmp_path: Path, **overrides: object) -> Path:
    config_path = tmp_path / "run.json"
    config = {
        "language": "english",
        "items_path": "items.csv",
        "models": ["model-a"],
        **overrides,
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def test_load_run_config_resolves_paths_and_models(tmp_path: Path) -> None:
    """Config paths are resolved relative to the config file."""

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_API_KEY=test", encoding="utf-8")
    config_path = config_dir / "run.json"
    config_path.write_text(
        json.dumps(
            {
                "run_id": "experiment-a",
                "language": "english",
                "items_path": "../items.csv",
                "output_dir": "../outputs",
                "env_path": "../.env",
                "chunk_size": 50,
                "model_workers": 2,
                "models": ["model-a", "model-b"],
                "metadata": {"owner": "benchmarks"},
                "future_field": "preserved",
            }
        ),
        encoding="utf-8",
    )

    config = load_run_config(config_path)

    assert config.run_id == "experiment-a"
    assert config.language == "english"
    assert config.task == "decodability"
    assert config.items_path == (tmp_path / "items.csv").resolve()
    assert config.output_dir == (tmp_path / "outputs").resolve()
    assert config.env_path == env_path.resolve()
    assert config.chunk_size == 50
    assert config.model_workers == 2
    assert config.models == ["model-a", "model-b"]
    assert config.extra == {"future_field": "preserved"}


def test_load_run_config_requires_non_empty_models(tmp_path: Path) -> None:
    """A run without models is rejected early."""

    config_path = _write_config(tmp_path, models=[])

    with pytest.raises(ConfigError, match="models"):
        load_run_config(config_path)


def test_load_run_config_reports_all_missing_required_fields(tmp_path: Path) -> None:
    """File configs use the same centralized required-field validation."""

    config_path = tmp_path / "run.json"
    config_path.write_text("{}", encoding="utf-8")

    with pytest.raises(
        ConfigError,
        match="Missing required run fields: language, models, items_path",
    ):
        load_run_config(config_path)


def test_load_run_config_defaults_run_id(tmp_path: Path) -> None:
    """A missing run_id defaults to the reusable default run."""

    config = load_run_config(_write_config(tmp_path))

    assert config.run_id == "default"
    assert config.chunk_size == 100
    assert config.model_workers == 1


def test_config_identity_includes_items_path(tmp_path: Path) -> None:
    """Selecting another composed-item file changes run identity."""

    first = load_run_config(_write_config(tmp_path, items_path="items-a.csv"))
    second = load_run_config(_write_config(tmp_path, items_path="items-b.csv"))

    assert config_identity_sha256(first) != config_identity_sha256(second)


def test_config_identity_excludes_operational_controls(tmp_path: Path) -> None:
    """Concurrency, retry, and checkpoint controls do not change run identity."""

    original = load_run_config(
        _write_config(
            tmp_path,
            chunk_size=100,
            model_workers=1,
            retry_errors=False,
        )
    )
    retry = load_run_config(
        _write_config(
            tmp_path,
            chunk_size=1,
            model_workers=4,
            retry_errors=True,
        )
    )

    assert config_identity_sha256(original) == config_identity_sha256(retry)
    assert config_object_sha256(original) != config_object_sha256(retry)


def test_apply_execution_overrides_preserves_unspecified_values(
    tmp_path: Path,
) -> None:
    """Only explicitly supplied execution options replace config values."""

    config = load_run_config(
        _write_config(tmp_path, model_workers=3, force=False, retry_errors=False)
    )

    overridden = apply_execution_overrides(
        config,
        force=True,
        retry_errors=False,
        model_workers=None,
    )

    assert overridden.force is True
    assert overridden.retry_errors is False
    assert overridden.model_workers == 3


def test_apply_execution_overrides_rejects_conflicting_modes(tmp_path: Path) -> None:
    """Config and supplied execution modes must remain mutually exclusive."""

    config = load_run_config(_write_config(tmp_path, force=True))

    with pytest.raises(ConfigError, match="cannot be combined"):
        apply_execution_overrides(
            config,
            force=False,
            retry_errors=True,
            model_workers=None,
        )


def test_apply_execution_overrides_validates_model_workers(tmp_path: Path) -> None:
    """Overrides cannot bypass the config's worker-count invariant."""

    config = load_run_config(_write_config(tmp_path))

    with pytest.raises(ConfigError, match="model_workers"):
        apply_execution_overrides(
            config,
            force=False,
            retry_errors=False,
            model_workers=0,
        )


def test_config_sha256_hashes_exact_file_contents(tmp_path: Path) -> None:
    """The source-config hash represents the exact bytes on disk."""

    config_path = _write_config(tmp_path)

    assert config_sha256(config_path) == sha256(config_path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("contents", "error"),
    [
        ("{", "Invalid JSON"),
        ("[]", "JSON object"),
    ],
)
def test_load_run_config_rejects_invalid_document_shape(
    tmp_path: Path,
    contents: str,
    error: str,
) -> None:
    """Config documents must be valid JSON objects."""

    config_path = tmp_path / "run.json"
    config_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ConfigError, match=error):
        load_run_config(config_path)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"language": ""}, "language"),
        ({"output_dir": ""}, "output_dir"),
        ({"env_path": 123}, "env_path"),
        ({"custom_models_path": 123}, "custom_models_path"),
        ({"max_tokens": "100"}, "max_tokens"),
        ({"max_tokens": True}, "max_tokens"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"max_tokens": -1}, "max_tokens"),
        ({"force": "yes"}, "force"),
        ({"retry_errors": "yes"}, "retry_errors"),
        ({"force": True, "retry_errors": True}, "cannot be combined"),
        ({"metadata": []}, "metadata"),
        ({"run_id": 123}, "run_id"),
    ],
)
def test_load_run_config_rejects_invalid_field_values(
    tmp_path: Path,
    overrides: dict[str, object],
    error: str,
) -> None:
    """Each typed config field rejects values outside its schema."""

    with pytest.raises(ConfigError, match=error):
        load_run_config(_write_config(tmp_path, **overrides))


def test_load_run_config_rejects_empty_command_task(tmp_path: Path) -> None:
    """The command must provide the task associated with the run."""

    with pytest.raises(ConfigError, match="command task"):
        load_run_config(_write_config(tmp_path), task="")


def test_load_run_config_requires_items_path(tmp_path: Path) -> None:
    """Every run must point to already-composed benchmark items."""

    config_path = tmp_path / "run.json"
    config_path.write_text(
        json.dumps({"language": "english", "models": ["model-a"]}),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="items_path"):
        load_run_config(config_path)


def test_load_run_config_rejects_data_root(tmp_path: Path) -> None:
    """The runner no longer composes items from language-level data files."""

    config_path = _write_config(tmp_path, data_root="data")

    with pytest.raises(ConfigError, match="no longer supported"):
        load_run_config(config_path)


def test_load_run_config_rejects_task_selected_in_config(tmp_path: Path) -> None:
    """The CLI command, rather than the config file, selects the benchmark task."""

    config_path = _write_config(tmp_path, task="phonics_annotation")

    with pytest.raises(ConfigError, match="selected by the CLI command"):
        load_run_config(config_path)


def test_load_run_config_allows_save_at_end_chunk_size(tmp_path: Path) -> None:
    """A null chunk_size means each model saves once after all pending items."""

    config = load_run_config(_write_config(tmp_path, chunk_size=None))

    assert config.chunk_size is None


def test_load_run_config_rejects_invalid_chunk_size(tmp_path: Path) -> None:
    """Chunk size must be positive when provided."""

    with pytest.raises(ConfigError, match="chunk_size"):
        load_run_config(_write_config(tmp_path, chunk_size=0))


@pytest.mark.parametrize("model_workers", [0, -1, True, None, "2"])
def test_load_run_config_rejects_invalid_model_workers(
    tmp_path: Path,
    model_workers: object,
) -> None:
    """Model concurrency must be represented by a positive integer."""

    with pytest.raises(ConfigError, match="model_workers"):
        load_run_config(_write_config(tmp_path, model_workers=model_workers))


def test_load_run_config_rejects_invalid_run_id(tmp_path: Path) -> None:
    """Run IDs are restricted to alphanumeric characters and hyphens."""

    with pytest.raises(ConfigError, match="run_id"):
        load_run_config(_write_config(tmp_path, run_id="bad_run"))


def test_load_run_config_rejects_model_objects(tmp_path: Path) -> None:
    """Model provider parameters must live in custom_models.yaml."""

    models = [{"name": "model-a", "parameters": {"temperature": 0}}]

    with pytest.raises(ConfigError, match="custom_models.yaml"):
        load_run_config(_write_config(tmp_path, models=models))


def test_load_run_config_rejects_duplicate_models(tmp_path: Path) -> None:
    """Each model owns one result file and therefore may appear only once."""

    with pytest.raises(ConfigError, match="duplicate"):
        load_run_config(_write_config(tmp_path, models=["model-a", "model-a"]))
