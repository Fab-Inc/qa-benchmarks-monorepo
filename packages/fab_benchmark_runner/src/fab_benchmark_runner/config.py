"""Run configs: which experiment, which models, and how to run them.

A run config never describes inputs: those belong to the experiment, which every
run reads as stored. Adding a model means writing another run config; nothing
shared is rewritten. ``max_tokens`` belongs to the experiment too, because it
shapes every response.
"""

from __future__ import annotations

import json
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from google.api_core.exceptions import NotFound

from fab_benchmark_runner.errors import ConfigError
from fab_benchmark_runner.gcs import (
    create_storage_client,
    download_to_path,
    parse_gs_uri,
)

DEFAULT_CHUNK_SIZE = 20
DEFAULT_MODEL_WORKERS = 4
RUN_CONFIG_KEYS = frozenset(
    {
        "chunk_size",
        "custom_models_path",
        "custom_models_sha256",
        "custom_models_uri",
        "env_path",
        "experiment",
        "force",
        "model_workers",
        "models",
        "retry_errors",
    }
)
# Keys of the run configs used before experiments, with what replaced them.
_REPLACED_KEYS = {
    "items_path": "inputs now belong to an experiment; create one with "
    "fab-bench create-experiment and name it in 'experiment'",
    "max_tokens": "max_tokens belongs to the experiment; set it when creating the "
    "experiment",
    "output_dir": "results are written into the experiment's folder",
    "results_dest": "results are written into the experiment's folder",
    "run_id": "a run is named by its experiment and models",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class RunConfig:
    """A validated run config.

    ``experiment`` is a ``gs://`` URI or an absolute local path. ``source`` is
    where the config was read from, and ``raw`` its content, which each model's
    record keeps.
    """

    experiment: str
    models: tuple[str, ...]
    custom_models_path: Path | None = None
    custom_models_uri: str | None = None
    custom_models_sha256: str | None = None
    env_path: Path | None = None
    model_workers: int = DEFAULT_MODEL_WORKERS
    chunk_size: int | None = DEFAULT_CHUNK_SIZE
    force: bool = False
    retry_errors: bool = False
    source: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def experiment_is_gcs(self) -> bool:
        return self.experiment.startswith("gs://")


def load_run_config(
    source: str | Path, *, storage_client: Any | None = None
) -> RunConfig:
    """Read and validate a run config from a local file or a ``gs://`` URI.

    Relative paths in a local config resolve against its folder. A config on GCS
    is meant for Cloud Run: it must name an experiment on GCS, and its registry can
    only be pinned with ``custom_models_uri`` or left to the public one. Its
    ``env_path`` is ignored, because jobs get provider keys from their environment.
    """

    text = str(source)
    if text.startswith("gs://"):
        try:
            parse_gs_uri(text)
        except ValueError as e:
            raise ConfigError(f"Invalid run config URI: {e}") from e
        with tempfile.TemporaryDirectory(prefix="fab-bench-config-") as temporary:
            local_copy = Path(temporary) / "run.json"
            try:
                download_to_path(
                    text,
                    local_copy,
                    storage_client=storage_client or create_storage_client(),
                )
            except NotFound as e:
                raise ConfigError(f"Run config {text} does not exist") from e
            raw = read_config_object(local_copy, description=f"run config {text}")
        return parse_run_config(raw, base_dir=None, source=text)

    path = Path(source).expanduser().resolve()
    raw = read_config_object(path, description=f"run config {path}")
    return parse_run_config(raw, base_dir=path.parent, source=str(path))


def read_config_object(path: Path, *, description: str) -> dict[str, Any]:
    """Read a JSON document that must be an object."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise ConfigError(f"{description.capitalize()} does not exist") from e
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ConfigError(f"Could not read {description}: {e}") from e
    if not isinstance(payload, dict):
        raise ConfigError(f"{description.capitalize()} must be a JSON object")
    return payload


def parse_run_config(
    raw: Mapping[str, Any], *, base_dir: Path | None, source: str | None = None
) -> RunConfig:
    """Validate a run config. ``base_dir`` is None for a config read from GCS.

    Unknown keys are refused, so a misspelled setting can't be silently ignored.
    """

    raw = dict(raw)
    replaced = sorted(set(raw) & set(_REPLACED_KEYS))
    if replaced:
        key = replaced[0]
        raise ConfigError(f"Run configs no longer take '{key}': {_REPLACED_KEYS[key]}")
    unknown = sorted(set(raw) - RUN_CONFIG_KEYS)
    if unknown:
        raise ConfigError(f"Unknown run config keys: {', '.join(unknown)}")

    on_gcs = base_dir is None
    experiment = raw.get("experiment")
    if not isinstance(experiment, str) or not experiment.strip():
        raise ConfigError("'experiment' must name an experiment folder or gs:// URI")
    if experiment.startswith("gs://"):
        try:
            parse_gs_uri(experiment.rstrip("/") + "/")
        except ValueError as e:
            raise ConfigError(f"Invalid 'experiment': {e}") from e
        experiment = experiment.rstrip("/")
    elif on_gcs:
        raise ConfigError(
            "A run config on GCS runs on Cloud Run, which can't read the local "
            f"experiment {experiment!r}; upload it with fab-bench upload-experiment"
        )
    else:
        experiment = str(_resolve_path(experiment, base_dir))

    custom_models_path = _optional_path(raw, "custom_models_path")
    custom_models_uri, custom_models_sha256 = _validate_pinned_registry(raw)
    if custom_models_path is not None and custom_models_uri is not None:
        raise ConfigError(
            "Set either custom_models_path or custom_models_uri, not both"
        )
    if on_gcs and custom_models_path is not None:
        raise ConfigError(
            "A run config on GCS can't use a local custom_models_path; pin the "
            "registry with custom_models_uri and custom_models_sha256"
        )
    env_path = _optional_path(raw, "env_path")

    chunk_size = raw.get("chunk_size", DEFAULT_CHUNK_SIZE)
    if chunk_size is not None and not _positive_int(chunk_size):
        raise ConfigError("'chunk_size' must be a positive integer or null")
    force = raw.get("force", False)
    retry_errors = raw.get("retry_errors", False)
    if not isinstance(force, bool) or not isinstance(retry_errors, bool):
        raise ConfigError("'force' and 'retry_errors' must be booleans")
    if force and retry_errors:
        raise ConfigError("'force' and 'retry_errors' cannot be combined")

    return RunConfig(
        experiment=experiment,
        models=_validate_models(raw.get("models")),
        custom_models_path=(
            None
            if custom_models_path is None or base_dir is None
            else _resolve_path(custom_models_path, base_dir)
        ),
        custom_models_uri=custom_models_uri,
        custom_models_sha256=custom_models_sha256,
        env_path=(
            None
            if env_path is None or base_dir is None
            else _resolve_path(env_path, base_dir)
        ),
        model_workers=validate_model_workers(
            raw.get("model_workers", DEFAULT_MODEL_WORKERS)
        ),
        chunk_size=chunk_size,
        force=force,
        retry_errors=retry_errors,
        source=source,
        raw=raw,
    )


def apply_execution_overrides(
    config: RunConfig,
    *,
    force: bool = False,
    retry_errors: bool = False,
    model_workers: int | None = None,
) -> RunConfig:
    """Apply command-line options on top of a validated config."""

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
    )


def validate_model_workers(value: Any) -> int:
    if not _positive_int(value):
        raise ConfigError("'model_workers' must be a positive integer")
    return value


def _validate_models(value: Any) -> tuple[str, ...]:
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
    return tuple(models)


def _validate_pinned_registry(
    raw: Mapping[str, Any],
) -> tuple[str | None, str | None]:
    source_uri = raw.get("custom_models_uri")
    expected_hash = raw.get("custom_models_sha256")
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
    return source_uri, expected_hash


def _optional_path(raw: Mapping[str, Any], key: str) -> str | None:
    value = raw.get(key)
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ConfigError(f"'{key}' must be a non-empty string when provided")
    return value


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()
