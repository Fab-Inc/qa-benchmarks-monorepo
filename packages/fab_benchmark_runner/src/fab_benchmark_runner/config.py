"""Run configs: shared core keys plus the benchmark's own params."""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from fab_benchmark_runner.benchmark import CORE_CONFIG_KEYS, Benchmark
from fab_benchmark_runner.errors import ConfigError
from fab_benchmark_runner.gcs import parse_gs_uri
from fab_benchmark_runner.ids import sha256_json

DEFAULT_RUN_ID = "default"
DEFAULT_OUTPUT_DIR = "results"
DEFAULT_MAX_TOKENS = 10_000
DEFAULT_CHUNK_SIZE = 20
DEFAULT_MODEL_WORKERS = 4
RUN_ID_RE = re.compile(r"^[a-zA-Z0-9-]+$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class RunConfig:
    """A validated run config, bound to the benchmark that parsed it."""

    benchmark: Benchmark
    run_id: str
    output_dir: Path
    models: list[str]
    items_path: Path
    params: Any = None
    inputs_dir: Path | None = None
    env_path: Path | None = None
    custom_models_path: Path | None = None
    custom_models_uri: str | None = None
    custom_models_sha256: str | None = None
    max_tokens: int | None = None
    chunk_size: int | None = DEFAULT_CHUNK_SIZE
    model_workers: int = DEFAULT_MODEL_WORKERS
    force: bool = False
    retry_errors: bool = False
    results_dest: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    raw_config: dict[str, Any] = field(default_factory=dict)
    config_path: Path | None = None

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.run_id

    @property
    def effective_max_tokens(self) -> int:
        """Return the token limit actually sent to providers."""

        return self.max_tokens if self.max_tokens is not None else DEFAULT_MAX_TOKENS

    def identity_fields(self) -> dict[str, Any]:
        """Return the benchmark params that define this run's semantics."""

        return self.benchmark.identity_fields(self.params)


def config_identity(config: RunConfig) -> dict[str, Any]:
    """Return the payload behind the config ID: what every call shares."""

    fields = config.identity_fields()
    if "max_tokens" in fields:
        raise ConfigError("Benchmark identity fields must not include 'max_tokens'")
    return {**fields, "max_tokens": config.effective_max_tokens}


def config_identity_sha256(config: RunConfig) -> str:
    """Return a stable hash of the semantic inputs of a run."""

    return sha256_json(config_identity(config))


def load_run_config(config_path: str | Path, benchmark: Benchmark) -> RunConfig:
    """Load and validate a local JSON run config."""

    path = Path(config_path).expanduser().resolve()
    raw_config = read_config_object(path, description="config file")
    return parse_run_config(
        raw_config, benchmark, base_dir=path.parent, config_path=path
    )


def read_config_object(path: Path, *, description: str) -> dict[str, Any]:
    """Read a JSON document that must be an object."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON in {description}: {e}") from e
    if not isinstance(payload, dict):
        raise ConfigError(f"{description.capitalize()} must be a JSON object")
    return payload


def parse_run_config(
    raw_config: dict[str, Any],
    benchmark: Benchmark,
    *,
    base_dir: Path,
    config_path: Path | None = None,
) -> RunConfig:
    """Validate core keys, then hand the benchmark's keys to its params type."""

    missing_fields = [
        name
        for name in ("models", "items_path")
        if raw_config.get(name) is None or raw_config.get(name) == []
    ]
    missing_fields += _missing_param_fields(benchmark, raw_config)
    if missing_fields:
        raise ConfigError("Missing required run fields: " + ", ".join(missing_fields))

    run_id = parse_run_id(raw_config.get("run_id"))
    models = _validate_models(raw_config.get("models"))

    items_path = raw_config.get("items_path")
    if not isinstance(items_path, str) or not items_path.strip():
        raise ConfigError("'items_path' must be a non-empty string")
    if items_path.startswith("gs://"):
        raise ConfigError(
            "A gs:// items_path must be downloaded first; load the config with "
            "load_local_run_config or load_cloud_run_config"
        )

    output_dir = raw_config.get("output_dir", DEFAULT_OUTPUT_DIR)
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ConfigError("'output_dir' must be a non-empty string")

    inputs_dir = _optional_path_string(raw_config, "inputs_dir")
    env_path = _optional_path_string(raw_config, "env_path")
    custom_models_path = _optional_path_string(raw_config, "custom_models_path")
    custom_models_uri, custom_models_sha256 = _validate_pinned_registry(
        raw_config, custom_models_path=custom_models_path
    )

    max_tokens = raw_config.get("max_tokens")
    if isinstance(max_tokens, bool) or (
        max_tokens is not None and (not isinstance(max_tokens, int) or max_tokens < 1)
    ):
        raise ConfigError("'max_tokens' must be a positive integer when provided")

    chunk_size = raw_config.get("chunk_size", DEFAULT_CHUNK_SIZE)
    if isinstance(chunk_size, bool) or (
        chunk_size is not None and (not isinstance(chunk_size, int) or chunk_size < 1)
    ):
        raise ConfigError("'chunk_size' must be a positive integer or null")

    model_workers = validate_model_workers(
        raw_config.get("model_workers", DEFAULT_MODEL_WORKERS)
    )

    force = raw_config.get("force", False)
    if not isinstance(force, bool):
        raise ConfigError("'force' must be a boolean when provided")
    retry_errors = raw_config.get("retry_errors", False)
    if not isinstance(retry_errors, bool):
        raise ConfigError("'retry_errors' must be a boolean when provided")
    if force and retry_errors:
        raise ConfigError("'force' and 'retry_errors' cannot be combined")

    metadata = raw_config.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ConfigError("'metadata' must be an object when provided")

    results_dest = raw_config.get("results_dest")
    if results_dest is not None:
        results_dest = validate_results_dest(results_dest)

    params = _parse_params(benchmark, raw_config)
    known_keys = set(benchmark.param_keys) | CORE_CONFIG_KEYS
    extra = {key: value for key, value in raw_config.items() if key not in known_keys}

    return RunConfig(
        benchmark=benchmark,
        run_id=run_id,
        output_dir=_resolve_path(output_dir, base_dir),
        models=models,
        items_path=_resolve_path(items_path, base_dir),
        params=params,
        inputs_dir=_resolve_path(inputs_dir, base_dir) if inputs_dir else None,
        env_path=_resolve_path(env_path, base_dir) if env_path else None,
        custom_models_path=(
            _resolve_path(custom_models_path, base_dir) if custom_models_path else None
        ),
        custom_models_uri=custom_models_uri,
        custom_models_sha256=custom_models_sha256,
        max_tokens=max_tokens,
        chunk_size=chunk_size,
        model_workers=model_workers,
        force=force,
        retry_errors=retry_errors,
        results_dest=results_dest,
        metadata=metadata,
        extra=extra,
        raw_config=raw_config,
        config_path=config_path,
    )


def apply_execution_overrides(
    config: RunConfig,
    *,
    force: bool = False,
    retry_errors: bool = False,
    model_workers: int | None = None,
    results_dest: str | None = None,
) -> RunConfig:
    """Apply explicitly supplied command-line options to a validated config."""

    effective_force = force or config.force
    effective_retry_errors = retry_errors or config.retry_errors
    if effective_force and effective_retry_errors:
        raise ConfigError("'force' and 'retry_errors' cannot be combined")
    return replace(
        config,
        force=effective_force,
        retry_errors=effective_retry_errors,
        model_workers=(
            validate_model_workers(model_workers)
            if model_workers is not None
            else config.model_workers
        ),
        results_dest=(
            validate_results_dest(results_dest)
            if results_dest is not None
            else config.results_dest
        ),
    )


def parse_run_id(value: Any) -> str:
    """Validate a run ID, which names the run's output folder."""

    if value is None:
        return DEFAULT_RUN_ID
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("'run_id' must be a non-empty string when provided")
    if not RUN_ID_RE.match(value):
        raise ConfigError("'run_id' must use only alphanumeric characters and hyphens")
    return value


def validate_model_workers(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError("'model_workers' must be a positive integer")
    return value


def validate_results_dest(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("'results_dest' must be a non-empty gs:// URI")
    try:
        parse_gs_uri(value)
    except ValueError as e:
        raise ConfigError(f"Invalid 'results_dest': {e}") from e
    return value


def _missing_param_fields(
    benchmark: Benchmark, raw_config: dict[str, Any]
) -> list[str]:
    if benchmark.params_type is None:
        return []
    return [
        param.name
        for param in dataclasses.fields(benchmark.params_type)
        if param.init
        and param.default is dataclasses.MISSING
        and param.default_factory is dataclasses.MISSING
        and raw_config.get(param.name) is None
    ]


def _parse_params(benchmark: Benchmark, raw_config: dict[str, Any]) -> Any:
    if benchmark.params_type is None:
        return None
    values = {
        param.name: raw_config[param.name]
        for param in dataclasses.fields(benchmark.params_type)
        if param.init and param.name in raw_config
    }
    try:
        return benchmark.params_type(**values)
    except ConfigError:
        raise
    except (TypeError, ValueError) as e:
        raise ConfigError(f"Invalid {benchmark.name} run params: {e}") from e


def _validate_models(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ConfigError("'models' must be a non-empty list")
    models: list[str] = []
    for model in value:
        if not isinstance(model, str):
            raise ConfigError(
                "Each model must be a string name. Provider parameters belong in "
                "custom_models.yaml."
            )
        model_name = model.strip()
        if not model_name:
            raise ConfigError("Model names must be non-empty strings")
        if model_name in models:
            raise ConfigError(f"Model names must be unique; duplicate: {model_name}")
        models.append(model_name)
    return models


def _validate_pinned_registry(
    raw_config: dict[str, Any],
    *,
    custom_models_path: str | None,
) -> tuple[str | None, str | None]:
    source_uri = raw_config.get("custom_models_uri")
    expected_hash = raw_config.get("custom_models_sha256")
    if source_uri is None:
        if expected_hash is not None:
            raise ConfigError("custom_models_sha256 requires custom_models_uri")
        return None, None
    if not isinstance(source_uri, str):
        raise ConfigError("custom_models_uri must be a gs:// URI")
    try:
        parse_gs_uri(source_uri)
    except ValueError as e:
        raise ConfigError(f"Invalid custom_models_uri: {e}") from e
    if not isinstance(expected_hash, str) or not SHA256_RE.match(expected_hash):
        raise ConfigError(
            "custom_models_uri requires a lowercase SHA-256 custom_models_sha256"
        )
    if custom_models_path is not None:
        raise ConfigError(
            "Set either custom_models_path or custom_models_uri, not both"
        )
    return source_uri, expected_hash


def _optional_path_string(raw_config: dict[str, Any], key: str) -> str | None:
    value = raw_config.get(key)
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ConfigError(f"'{key}' must be a non-empty string when provided")
    return value


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()
