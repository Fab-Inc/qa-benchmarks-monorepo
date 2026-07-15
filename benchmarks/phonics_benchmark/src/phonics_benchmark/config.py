"""Configuration loading for phonics benchmark runs."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_RUN_ID = "default"
DECODABILITY_TASK = "decodability"
RUN_ID_RE = re.compile(r"^[a-zA-Z0-9-]+$")


class ConfigError(ValueError):
    """Raised when a benchmark run config is invalid."""


@dataclass(frozen=True)
class BenchmarkRunConfig:
    """Validated benchmark run config."""

    run_id: str
    language: str
    task: str
    data_root: Path
    output_dir: Path
    models: list[str]
    env_path: Path | None = None
    custom_models_path: Path | None = None
    max_tokens: int | None = None
    chunk_size: int | None = 100
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
            "data_root": str(self.data_root),
            "output_dir": str(self.output_dir),
            "models": self.models,
            "env_path": str(self.env_path) if self.env_path else None,
            "custom_models_path": (
                str(self.custom_models_path) if self.custom_models_path else None
            ),
            "max_tokens": self.max_tokens,
            "chunk_size": self.chunk_size,
            "force": self.force,
            "retry_errors": self.retry_errors,
            "metadata": self.metadata,
            "extra": self.extra,
        }

    def to_identity_dict(self) -> dict[str, Any]:
        """Return the semantic run inputs used to identify result rows."""

        resolved = self.to_resolved_dict()
        for operational_key in ("chunk_size", "force", "retry_errors"):
            resolved.pop(operational_key, None)
        return resolved


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


def load_run_config(
    config_path: str | Path,
    *,
    task: str = DECODABILITY_TASK,
    default_data_root: str | Path | None = None,
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
    resolved_default_data_root = (
        Path(default_data_root).expanduser().resolve()
        if default_data_root is not None
        else None
    )
    return _parse_run_config(
        raw_config,
        task=task,
        base_dir=base_dir,
        config_path=path,
        default_data_root=resolved_default_data_root,
    )


def build_run_config(
    *,
    run_id: str | None,
    language: str,
    data_root: str | Path,
    output_dir: str | Path,
    models: list[str],
    base_dir: Path,
    task: str,
    env_path: str | Path | None = None,
    custom_models_path: str | Path | None = None,
    max_tokens: int | None = None,
    chunk_size: int | None = 100,
    force: bool = False,
    retry_errors: bool = False,
    metadata: dict[str, Any] | None = None,
) -> BenchmarkRunConfig:
    """Build a validated run config from command-line values."""

    raw_config: dict[str, Any] = {
        "run_id": run_id,
        "language": language,
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "models": models,
        "env_path": str(env_path) if env_path else None,
        "custom_models_path": str(custom_models_path) if custom_models_path else None,
        "max_tokens": max_tokens,
        "chunk_size": chunk_size,
        "force": force,
        "retry_errors": retry_errors,
        "metadata": metadata or {},
    }
    return _parse_run_config(
        raw_config,
        task=task,
        base_dir=base_dir.resolve(),
        config_path=None,
        default_data_root=None,
    )


def _parse_run_config(
    raw_config: dict[str, Any],
    *,
    task: str,
    base_dir: Path,
    config_path: Path | None,
    default_data_root: str | Path | None,
) -> BenchmarkRunConfig:
    run_id = _parse_run_id(raw_config.get("run_id"))

    language = raw_config.get("language")
    if not isinstance(language, str) or not language.strip():
        raise ConfigError("Config must include a non-empty 'language'")

    if not isinstance(task, str) or not task.strip():
        raise ConfigError("The command task must be a non-empty string")

    if "task" in raw_config:
        raise ConfigError(
            "'task' is selected by the CLI command and must not be set in config"
        )

    if "items_path" in raw_config:
        raise ConfigError(
            "'items_path' is no longer supported; benchmark data is selected by "
            "'language'"
        )

    data_root = raw_config.get("data_root")
    if data_root is None:
        data_root = str(default_data_root) if default_data_root is not None else None
    if not isinstance(data_root, str) or not data_root.strip():
        raise ConfigError("Config must include a non-empty 'data_root'")

    output_dir = raw_config.get("output_dir", "results")
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ConfigError("'output_dir' must be a non-empty string")

    models_raw = raw_config.get("models")
    models = _parse_models(models_raw)

    env_path = raw_config.get("env_path")
    if env_path is not None and not isinstance(env_path, str):
        raise ConfigError("'env_path' must be a string when provided")

    custom_models_path = raw_config.get("custom_models_path")
    if custom_models_path is not None and not isinstance(custom_models_path, str):
        raise ConfigError("'custom_models_path' must be a string when provided")

    max_tokens = raw_config.get("max_tokens")
    if max_tokens is not None and not isinstance(max_tokens, int):
        raise ConfigError("'max_tokens' must be an integer when provided")

    chunk_size = raw_config.get("chunk_size", 100)
    if isinstance(chunk_size, bool) or (
        chunk_size is not None and (not isinstance(chunk_size, int) or chunk_size < 1)
    ):
        raise ConfigError("'chunk_size' must be a positive integer or null")

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
        "data_root",
        "output_dir",
        "models",
        "env_path",
        "custom_models_path",
        "max_tokens",
        "chunk_size",
        "force",
        "retry_errors",
        "metadata",
    }
    extra = {key: value for key, value in raw_config.items() if key not in known_keys}

    return BenchmarkRunConfig(
        run_id=run_id,
        language=language,
        task=task,
        data_root=_resolve_path(data_root, base_dir),
        output_dir=_resolve_path(output_dir, base_dir),
        models=models,
        env_path=_resolve_path(env_path, base_dir) if env_path else None,
        custom_models_path=(
            _resolve_path(custom_models_path, base_dir) if custom_models_path else None
        ),
        max_tokens=max_tokens,
        chunk_size=chunk_size,
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


def _parse_models(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ConfigError("Config must include a non-empty 'models' list")

    models = []
    for model in value:
        if isinstance(model, str) and model.strip():
            models.append(model.strip())
            continue
        raise ConfigError(
            "Each model must be a string name. Provider parameters belong in "
            "custom_models.yaml."
        )
    return models


def _parse_run_id(value: Any) -> str:
    if value is None:
        return DEFAULT_RUN_ID
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("'run_id' must be a non-empty string when provided")
    if not RUN_ID_RE.match(value):
        raise ConfigError("'run_id' must use only alphanumeric characters and hyphens")
    return value
