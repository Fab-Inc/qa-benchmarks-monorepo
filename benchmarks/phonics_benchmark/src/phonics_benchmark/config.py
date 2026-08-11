"""Configuration loading for phonics benchmark runs."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

DEFAULT_RUN_ID = "default"
DEFAULT_MAX_TOKENS = 10_000
DEFAULT_CHUNK_SIZE = 20
DEFAULT_MODEL_WORKERS = 4
DECODABILITY_TASK = "decodability"
DEFAULT_PROMPT_NAME = "decodability_evaluation"
RUN_ID_RE = re.compile(r"^[a-zA-Z0-9-]+$")


class ConfigError(ValueError):
    """Raised when a benchmark run config is invalid."""


@dataclass(frozen=True)
class BenchmarkRunConfig:
    """Validated benchmark run config."""

    run_id: str
    language: str
    task: str
    output_dir: Path
    models: list[str]
    items_path: Path
    prompt_name: str = DEFAULT_PROMPT_NAME
    env_path: Path | None = None
    custom_models_path: Path | None = None
    max_tokens: int | None = None
    chunk_size: int | None = DEFAULT_CHUNK_SIZE
    model_workers: int = DEFAULT_MODEL_WORKERS
    force: bool = False
    retry_errors: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    raw_config: dict[str, Any] = field(default_factory=dict)
    config_path: Path | None = None

    def to_resolved_dict(self) -> dict[str, Any]:
        """Return the config with resolved paths for snapshotting."""

        return {
            "run_id": self.run_id,
            "language": self.language,
            "task": self.task,
            "output_dir": str(self.output_dir),
            "models": self.models,
            "items_path": str(self.items_path),
            "prompt_name": self.prompt_name,
            "env_path": str(self.env_path) if self.env_path else None,
            "custom_models_path": (
                str(self.custom_models_path) if self.custom_models_path else None
            ),
            "max_tokens": self.max_tokens,
            "chunk_size": self.chunk_size,
            "model_workers": self.model_workers,
            "force": self.force,
            "retry_errors": self.retry_errors,
            "metadata": self.metadata,
            "extra": self.extra,
        }

    def to_identity_dict(self) -> dict[str, Any]:
        """Return the exact shared call-shaping configuration payload."""

        return {
            "language": self.language,
            "task": self.task,
            "prompt_name": self.prompt_name,
            "max_tokens": effective_max_tokens(self.max_tokens),
        }


def config_sha256(config_path: Path) -> str:
    """Return a sha256 hash of the exact config file bytes."""

    return hashlib.sha256(config_path.read_bytes()).hexdigest()


def config_object_sha256(config: BenchmarkRunConfig) -> str:
    """Return a stable hash for a constructed run config."""

    payload = json.dumps(
        config.to_resolved_dict(),
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def config_identity_sha256(config: BenchmarkRunConfig) -> str:
    """Return a stable hash for the semantic inputs of a benchmark run."""

    payload = json.dumps(
        config.to_identity_dict(),
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def effective_max_tokens(configured_max_tokens: int | None) -> int:
    """Resolve the maximum-token value actually passed to providers."""

    return (
        configured_max_tokens
        if configured_max_tokens is not None
        else DEFAULT_MAX_TOKENS
    )


def apply_execution_overrides(
    config: BenchmarkRunConfig,
    *,
    force: bool,
    retry_errors: bool,
    model_workers: int | None,
) -> BenchmarkRunConfig:
    """Apply explicitly supplied execution options to a validated config."""

    effective_force = force or config.force
    effective_retry_errors = retry_errors or config.retry_errors
    if effective_force and effective_retry_errors:
        raise ConfigError("'force' and 'retry_errors' cannot be combined")
    return replace(
        config,
        force=effective_force,
        retry_errors=effective_retry_errors,
        model_workers=(
            _validate_model_workers(model_workers)
            if model_workers is not None
            else config.model_workers
        ),
    )


def load_run_config(
    config_path: str | Path,
    *,
    task: str = DECODABILITY_TASK,
) -> BenchmarkRunConfig:
    """Load and validate a phonics benchmark run config from JSON."""

    path = Path(config_path).expanduser().resolve()
    try:
        raw_config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON in config file: {e}") from e

    if not isinstance(raw_config, dict):
        raise ConfigError("Run config must be a JSON object")

    base_dir = path.parent
    return _parse_run_config(
        raw_config,
        task=task,
        base_dir=base_dir,
        config_path=path,
    )


def build_run_config(
    *,
    run_id: str | None,
    language: str | None,
    output_dir: str | Path,
    models: list[Any],
    base_dir: Path,
    task: str,
    items_path: str | Path | None,
    prompt_name: str = DEFAULT_PROMPT_NAME,
    env_path: str | Path | None = None,
    custom_models_path: str | Path | None = None,
    max_tokens: int | None = None,
    chunk_size: int | None = DEFAULT_CHUNK_SIZE,
    model_workers: int = DEFAULT_MODEL_WORKERS,
    force: bool = False,
    retry_errors: bool = False,
    metadata: dict[str, Any] | None = None,
) -> BenchmarkRunConfig:
    """Build a validated run config from command-line values."""

    raw_config: dict[str, Any] = {
        "run_id": run_id,
        "language": language,
        "output_dir": str(output_dir),
        "models": models,
        "items_path": str(items_path) if items_path is not None else None,
        "prompt_name": prompt_name,
        "env_path": str(env_path) if env_path else None,
        "custom_models_path": str(custom_models_path) if custom_models_path else None,
        "max_tokens": max_tokens,
        "chunk_size": chunk_size,
        "model_workers": model_workers,
        "force": force,
        "retry_errors": retry_errors,
        "metadata": metadata or {},
    }
    return _parse_run_config(
        raw_config,
        task=task,
        base_dir=base_dir.resolve(),
        config_path=None,
    )


def _parse_run_config(
    raw_config: dict[str, Any],
    *,
    task: str,
    base_dir: Path,
    config_path: Path | None,
) -> BenchmarkRunConfig:
    run_id = _parse_run_id(raw_config.get("run_id"))

    required_fields = ("language", "models", "items_path")
    missing_fields = [
        field_name
        for field_name in required_fields
        if raw_config.get(field_name) is None
        or (field_name == "models" and raw_config.get(field_name) == [])
    ]
    if missing_fields:
        raise ConfigError("Missing required run fields: " + ", ".join(missing_fields))

    language = raw_config.get("language")
    if not isinstance(language, str) or not language.strip():
        raise ConfigError("'language' must be a non-empty string")

    if not isinstance(task, str) or not task.strip():
        raise ConfigError("The command task must be a non-empty string")

    if "task" in raw_config:
        raise ConfigError(
            "'task' is selected by the CLI command and must not be set in config"
        )

    output_dir = raw_config.get("output_dir", "results")
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ConfigError("'output_dir' must be a non-empty string")

    models_raw = raw_config.get("models")
    models = _validate_config_models(models_raw)

    items_path = raw_config.get("items_path")
    if not isinstance(items_path, str) or not items_path.strip():
        raise ConfigError("'items_path' must be a non-empty string")

    env_path = raw_config.get("env_path")
    if env_path is not None and not isinstance(env_path, str):
        raise ConfigError("'env_path' must be a string when provided")

    custom_models_path = raw_config.get("custom_models_path")
    if custom_models_path is not None and not isinstance(custom_models_path, str):
        raise ConfigError("'custom_models_path' must be a string when provided")

    prompt_name = raw_config.get("prompt_name", DEFAULT_PROMPT_NAME)
    if not isinstance(prompt_name, str) or not prompt_name.strip():
        raise ConfigError("'prompt_name' must be a non-empty string when provided")

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

    model_workers = _validate_model_workers(
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

    known_keys = {
        "run_id",
        "language",
        "output_dir",
        "models",
        "items_path",
        "prompt_name",
        "env_path",
        "custom_models_path",
        "max_tokens",
        "chunk_size",
        "model_workers",
        "force",
        "retry_errors",
        "metadata",
    }
    extra = {key: value for key, value in raw_config.items() if key not in known_keys}

    return BenchmarkRunConfig(
        run_id=run_id,
        language=language,
        task=task,
        output_dir=_resolve_path(output_dir, base_dir),
        models=models,
        items_path=_resolve_path(items_path, base_dir),
        prompt_name=prompt_name,
        env_path=_resolve_path(env_path, base_dir) if env_path else None,
        custom_models_path=(
            _resolve_path(custom_models_path, base_dir) if custom_models_path else None
        ),
        max_tokens=max_tokens,
        chunk_size=chunk_size,
        model_workers=model_workers,
        force=force,
        retry_errors=retry_errors,
        metadata=metadata,
        extra=extra,
        raw_config=raw_config,
        config_path=config_path,
    )


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _validate_config_models(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ConfigError("'models' must be a non-empty list")

    models = []
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


def _validate_model_workers(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError("'model_workers' must be a positive integer")
    return value


def _parse_run_id(value: Any) -> str:
    if value is None:
        return DEFAULT_RUN_ID
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("'run_id' must be a non-empty string when provided")
    if not RUN_ID_RE.match(value):
        raise ConfigError("'run_id' must use only alphanumeric characters and hyphens")
    return value
