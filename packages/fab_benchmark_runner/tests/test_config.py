"""Run config parsing: shared keys, benchmark params and pinned registries."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from fab_benchmark_runner import (
    Benchmark,
    ConfigError,
    apply_execution_overrides,
    config_identity,
    config_identity_sha256,
    load_run_config,
    parse_run_config,
)

SHA = "a" * 64


def _parse(raw: dict[str, Any], benchmark: Benchmark, tmp_path: Path):
    return parse_run_config(raw, benchmark, base_dir=tmp_path)


def _minimal(**overrides: Any) -> dict[str, Any]:
    return {
        "language": "sw",
        "models": ["model-a"],
        "items_path": "items.jsonl",
        **overrides,
    }


def test_load_run_config_resolves_paths_and_parses_params(
    tmp_path: Path, benchmark: Benchmark, write_run: Callable[..., Path]
) -> None:
    config = load_run_config(write_run(prompt_name="clusters"), benchmark)

    assert config.benchmark is benchmark
    assert config.run_id == "experiment-a"
    assert config.items_path == (tmp_path / "items.jsonl").resolve()
    assert config.output_dir == (tmp_path / "outputs").resolve()
    assert config.custom_models_path == (tmp_path / "custom_models.yaml").resolve()
    assert config.run_dir == config.output_dir / "experiment-a"
    assert config.params.language == "sw"
    assert config.params.prompt_name == "clusters"
    assert config.config_path == (tmp_path / "run.json").resolve()
    assert config.raw_config["prompt_name"] == "clusters"


def test_defaults(tmp_path: Path, benchmark: Benchmark) -> None:
    config = _parse(_minimal(), benchmark, tmp_path)

    assert config.run_id == "default"
    assert config.output_dir == (tmp_path / "results").resolve()
    assert config.params.prompt_name == "default"
    assert config.max_tokens is None
    assert config.effective_max_tokens == 10_000
    assert config.chunk_size == 20
    assert config.model_workers == 4
    assert config.extra == {}


def test_missing_core_and_param_fields_are_reported_together(
    tmp_path: Path, benchmark: Benchmark
) -> None:
    with pytest.raises(
        ConfigError, match="Missing required run fields: models, items_path, language"
    ):
        _parse({"models": []}, benchmark, tmp_path)


def test_param_validation_errors_become_config_errors(
    tmp_path: Path, benchmark: Benchmark
) -> None:
    with pytest.raises(ConfigError, match="Invalid test-bench run params.*language"):
        _parse(_minimal(language="  "), benchmark, tmp_path)


def test_unknown_keys_are_kept_as_extra(tmp_path: Path, benchmark: Benchmark) -> None:
    config = _parse(_minimal(data_root="data", notes="x"), benchmark, tmp_path)

    assert config.extra == {"data_root": "data", "notes": "x"}
    assert "prompt_name" not in config.extra


def test_benchmark_without_params(tmp_path: Path) -> None:
    plain = Benchmark(name="plain", build_prompt=lambda item, _params: "p")

    config = _parse(_minimal(), plain, tmp_path)

    assert config.params is None
    assert config.extra == {"language": "sw"}
    assert config_identity(config) == {"max_tokens": 10_000}


def test_benchmark_params_may_not_reuse_core_keys() -> None:
    @dataclass(frozen=True)
    class Clashing:
        models: str = "x"

    with pytest.raises(ValueError, match="reuse core run config keys: models"):
        Benchmark(name="bad", params_type=Clashing, build_prompt=lambda *_: "")


def test_benchmark_params_type_must_be_a_dataclass() -> None:
    with pytest.raises(TypeError, match="dataclass"):
        Benchmark(name="bad", params_type=dict, build_prompt=lambda *_: "")


def test_identity_defaults_to_all_params(tmp_path: Path, benchmark: Benchmark) -> None:
    config = _parse(_minimal(max_tokens=50), benchmark, tmp_path)

    assert config_identity(config) == {
        "language": "sw",
        "prompt_name": "default",
        "max_tokens": 50,
    }


def test_identity_ignores_models_items_and_operational_controls(
    tmp_path: Path, benchmark: Benchmark
) -> None:
    first = _parse(_minimal(), benchmark, tmp_path)
    second = _parse(
        _minimal(
            models=["model-b", "model-c"],
            items_path="other.jsonl",
            run_id="other",
            chunk_size=5,
            model_workers=1,
            force=True,
            metadata={"note": "x"},
        ),
        benchmark,
        tmp_path,
    )

    assert config_identity_sha256(first) == config_identity_sha256(second)


def test_explicit_default_max_tokens_keeps_the_config_id(
    tmp_path: Path, benchmark: Benchmark
) -> None:
    implicit = _parse(_minimal(), benchmark, tmp_path)
    explicit = _parse(_minimal(max_tokens=10_000), benchmark, tmp_path)

    assert config_identity_sha256(implicit) == config_identity_sha256(explicit)


def test_identity_may_not_override_max_tokens(tmp_path: Path) -> None:
    clashing = Benchmark(
        name="clash",
        build_prompt=lambda *_: "",
        identity=lambda _params: {"max_tokens": 1},
    )
    config = _parse(_minimal(), clashing, tmp_path)

    with pytest.raises(ConfigError, match="must not include 'max_tokens'"):
        config_identity(config)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"run_id": "bad id"}, "run_id"),
        ({"run_id": ""}, "run_id"),
        ({"models": "model-a"}, "'models' must be a non-empty list"),
        ({"models": [{"name": "m"}]}, "Each model must be a string"),
        ({"models": ["m", "m"]}, "duplicate: m"),
        ({"models": [" "]}, "non-empty"),
        ({"items_path": 3}, "items_path"),
        ({"items_path": "gs://bucket/items.jsonl"}, "must be downloaded first"),
        ({"output_dir": ""}, "output_dir"),
        ({"env_path": 3}, "env_path"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"max_tokens": True}, "max_tokens"),
        ({"chunk_size": 0}, "chunk_size"),
        ({"model_workers": 0}, "model_workers"),
        ({"force": "yes"}, "force"),
        ({"retry_errors": 1}, "retry_errors"),
        ({"force": True, "retry_errors": True}, "cannot be combined"),
        ({"metadata": []}, "metadata"),
        ({"results_dest": "s3://bucket/run/"}, "results_dest"),
    ],
)
def test_invalid_core_values_are_rejected(
    tmp_path: Path, benchmark: Benchmark, overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        _parse(_minimal(**overrides), benchmark, tmp_path)


def test_chunk_size_null_saves_once_per_model(
    tmp_path: Path, benchmark: Benchmark
) -> None:
    assert _parse(_minimal(chunk_size=None), benchmark, tmp_path).chunk_size is None


def test_pinned_registry_is_parsed(tmp_path: Path, benchmark: Benchmark) -> None:
    config = _parse(
        _minimal(
            custom_models_uri="gs://bucket/model-configs/run.yaml",
            custom_models_sha256=SHA,
        ),
        benchmark,
        tmp_path,
    )

    assert config.custom_models_uri == "gs://bucket/model-configs/run.yaml"
    assert config.custom_models_sha256 == SHA
    assert config.extra == {}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"custom_models_sha256": SHA}, "requires custom_models_uri"),
        ({"custom_models_uri": 3, "custom_models_sha256": SHA}, "gs:// URI"),
        ({"custom_models_uri": "https://x/y", "custom_models_sha256": SHA}, "Invalid"),
        ({"custom_models_uri": "gs://b/r.yaml"}, "lowercase SHA-256"),
        (
            {"custom_models_uri": "gs://b/r.yaml", "custom_models_sha256": SHA.upper()},
            "lowercase SHA-256",
        ),
        (
            {"custom_models_uri": "gs://b/r.yaml", "custom_models_sha256": SHA[:-1]},
            "lowercase SHA-256",
        ),
        (
            {
                "custom_models_uri": "gs://b/r.yaml",
                "custom_models_sha256": SHA,
                "custom_models_path": "custom_models.yaml",
            },
            "not both",
        ),
    ],
)
def test_invalid_pinned_registries_are_rejected(
    tmp_path: Path, benchmark: Benchmark, overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        _parse(_minimal(**overrides), benchmark, tmp_path)


def test_execution_overrides(tmp_path: Path, benchmark: Benchmark) -> None:
    config = _parse(_minimal(model_workers=2), benchmark, tmp_path)

    unchanged = apply_execution_overrides(config)
    overridden = apply_execution_overrides(
        config,
        force=True,
        model_workers=3,
        results_dest="gs://bucket/runs/a/",
    )

    assert unchanged == config
    assert overridden.force is True
    assert overridden.model_workers == 3
    assert overridden.results_dest == "gs://bucket/runs/a/"
    with pytest.raises(ConfigError, match="cannot be combined"):
        apply_execution_overrides(overridden, retry_errors=True)
    with pytest.raises(ConfigError, match="model_workers"):
        apply_execution_overrides(config, model_workers=0)


@pytest.mark.parametrize("payload", ["[]", "not json"])
def test_load_run_config_rejects_invalid_documents(
    tmp_path: Path, benchmark: Benchmark, payload: str
) -> None:
    config_path = tmp_path / "run.json"
    config_path.write_text(payload, encoding="utf-8")

    with pytest.raises(ConfigError):
        load_run_config(config_path, benchmark)
