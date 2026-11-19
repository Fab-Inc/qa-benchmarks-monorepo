"""The FDLLM model registry (custom_models.yaml): download, pin and register."""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import yaml

from fab_benchmark_runner import gcs
from fab_benchmark_runner.config import RunConfig
from fab_benchmark_runner.errors import ConfigError
from fab_benchmark_runner.results import file_sha256

CUSTOM_MODELS_BUCKET = "fab-open-configs"
CUSTOM_MODELS_FILE_NAME = "custom_models.yaml"
ENV_FILE_NAME = ".env"
PINNED_REGISTRY_DIR = "registry"


@dataclass(frozen=True)
class ModelRegistry:
    """The registry file a run uses, its SHA-256, and the pinned URI it came from."""

    path: Path
    sha256: str
    uri: str | None = None

    def record(self) -> dict[str, Any]:
        """Return what a model's record keeps about the registry."""

        return {"sha256": self.sha256, "uri": self.uri}


def prepare_model_registry(
    config: RunConfig,
    *,
    workdir: Path,
    download_public: bool,
    public_path: Path | None = None,
) -> ModelRegistry:
    """Make the run's model registry available locally.

    - ``custom_models_uri``: download the pinned file into ``workdir`` and verify
      its SHA-256. This always happens, because a fresh container has no copy.
    - ``custom_models_path``: use that file as it is.
    - Neither: use the shared public registry at ``public_path`` (by default
      ``custom_models.yaml`` in the working directory), refreshing it first when
      ``download_public`` is set.
    """

    if config.custom_models_uri is not None:
        if config.custom_models_sha256 is None:  # pragma: no cover - parse-time check
            raise ConfigError("custom_models_uri requires custom_models_sha256")
        destination = (
            workdir.expanduser().resolve()
            / PINNED_REGISTRY_DIR
            / f"custom_models-{config.custom_models_sha256}.yaml"
        )
        download_pinned_registry(
            config.custom_models_uri,
            expected_sha256=config.custom_models_sha256,
            destination=destination,
        )
        return ModelRegistry(
            destination, config.custom_models_sha256, config.custom_models_uri
        )
    if config.custom_models_path is not None:
        path = config.custom_models_path
    else:
        path = public_path or Path.cwd() / CUSTOM_MODELS_FILE_NAME
        if download_public:
            download_public_registry(path)
    if not path.exists():
        raise FileNotFoundError(f"Custom models file not found: {path}")
    return ModelRegistry(path, file_sha256(path))


def prepare_fdllm(
    registry: ModelRegistry, model_names: Iterable[str], *, env_path: Path | None
) -> dict[str, Any]:
    """Load provider keys, register the registry, and return the models' configs."""

    model_names = list(model_names)
    load_environment(env_path)
    register_custom_models(registry.path)
    validate_registered_models(model_names)
    return snapshot_model_configs(model_names)


def download_public_registry(destination: Path) -> Path:
    """Anonymously download the shared custom_models.yaml and replace it atomically."""

    destination = destination.expanduser().resolve()
    with _temporary_sibling(destination) as temporary_path:
        gcs.download_file(
            bucket_name=CUSTOM_MODELS_BUCKET,
            file_name=CUSTOM_MODELS_FILE_NAME,
            write_path=temporary_path,
            require_auth=False,
        )
        try:
            downloaded_models = yaml.safe_load(
                temporary_path.read_text(encoding="utf-8")
            )
        except (UnicodeDecodeError, yaml.YAMLError) as e:
            raise ConfigError(f"Downloaded custom_models.yaml is invalid: {e}") from e
        if not isinstance(downloaded_models, dict) or not downloaded_models:
            raise ConfigError(
                "Downloaded custom_models.yaml must contain a non-empty YAML mapping"
            )
        os.replace(temporary_path, destination)
    return destination


def download_pinned_registry(
    source_uri: str,
    *,
    expected_sha256: str,
    destination: Path,
) -> Path:
    """Download a pinned registry and fail before replacing anything on a mismatch."""

    try:
        bucket_name, object_name = gcs.parse_gs_uri(source_uri)
    except ValueError as e:
        raise ConfigError(f"Invalid custom_models_uri: {e}") from e
    print(f"Reading pinned model registry from {source_uri}")
    destination = destination.expanduser().resolve()
    with _temporary_sibling(destination) as temporary_path:
        gcs.download_file(
            bucket_name=bucket_name,
            file_name=object_name,
            write_path=temporary_path,
        )
        if sha256(temporary_path.read_bytes()).hexdigest() != expected_sha256:
            raise ConfigError("Pinned model registry SHA-256 does not match config")
        os.replace(temporary_path, destination)
    return destination


def load_environment(env_path: Path | None) -> Path | None:
    """Load provider API keys from a .env file without overriding the environment.

    Without ``env_path``, a ``.env`` in the working directory is used if present.
    """

    path = env_path or Path.cwd() / ENV_FILE_NAME
    if env_path is not None and not path.exists():
        raise FileNotFoundError(f"Environment file not found: {path}")
    if not path.exists():
        return None

    from dotenv import load_dotenv

    load_dotenv(path, override=False)
    return path


def register_custom_models(custom_models_path: Path) -> None:
    from fdllm import register_models

    register_models(str(custom_models_path))


def validate_registered_models(model_names: list[str]) -> None:
    """Raise ConfigError if any configured model is not registered with FDLLM."""

    from fdllm import list_models

    registered_models = set(list_models())
    missing_models = [name for name in model_names if name not in registered_models]
    if missing_models:
        raise ConfigError(
            "Configured model(s) are not registered with FDLLM: "
            + ", ".join(missing_models)
        )


def snapshot_model_configs(model_names: list[str]) -> dict[str, Any]:
    """Return sanitized model configs from FDLLM's registered model state.

    These are FDLLM's expanded configs: the registry entry merged with provider
    defaults and FDLLM's built-in model list. They feed every model call ID.
    """

    from fdllm import list_models

    model_configs = list_models(full_info=True)
    return {
        model_name: sanitize_model_config(model_configs[model_name])
        for model_name in model_names
        if model_name in model_configs
    }


def sanitize_model_config(value: Any) -> Any:
    """Drop secret-looking keys, recursively, keeping the API key's variable name."""

    if isinstance(value, Mapping):
        return {
            key: sanitize_model_config(nested)
            for key, nested in value.items()
            if not _is_secret_model_config_key(str(key))
        }
    if isinstance(value, list):
        return [sanitize_model_config(item) for item in value]
    return value


def check_provider_groups(custom_models_path: Path, model_names: list[str]) -> None:
    """Require each selected model to appear in exactly one top-level provider group.

    FDLLM merges the groups into one model list, so a model listed under two groups
    would silently take one group's settings.
    """

    try:
        raw_registry = yaml.safe_load(custom_models_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as e:
        raise ConfigError(
            f"Could not read custom models YAML {custom_models_path}: {e}"
        ) from e
    if not isinstance(raw_registry, Mapping) or not raw_registry:
        raise ConfigError(
            f"Custom models YAML {custom_models_path} must contain a non-empty mapping"
        )

    memberships: dict[str, list[str]] = defaultdict(list)
    for provider_group, provider_config in raw_registry.items():
        if not isinstance(provider_config, Mapping):
            raise ConfigError(
                f"Provider group '{provider_group}' in {custom_models_path} "
                "must be an object"
            )
        models = provider_config.get("models")
        if models is None:
            continue
        if not isinstance(models, Mapping):
            raise ConfigError(
                f"Provider group '{provider_group}' in {custom_models_path} "
                "must define 'models' as an object"
            )
        for model_name in models:
            if isinstance(model_name, str):
                memberships[model_name].append(str(provider_group))

    missing = [name for name in model_names if not memberships[name]]
    if missing:
        raise ConfigError(
            "Selected model(s) are missing from custom_models.yaml provider groups: "
            + ", ".join(missing)
        )
    ambiguous = {
        name: memberships[name] for name in model_names if len(memberships[name]) > 1
    }
    if ambiguous:
        details = ", ".join(
            f"{name} ({', '.join(groups)})" for name, groups in ambiguous.items()
        )
        raise ConfigError(
            "Selected model(s) belong to multiple custom_models.yaml provider "
            f"groups: {details}"
        )


def _is_secret_model_config_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    if normalized == "api_key_env_var":
        return False
    secret_markers = (
        "api_key",
        "apikey",
        "secret",
        "password",
        "authorization",
        "access_token",
        "refresh_token",
        "id_token",
    )
    return any(marker in normalized for marker in secret_markers)


@contextmanager
def _temporary_sibling(destination: Path) -> Iterator[Path]:
    """Yield a temporary file next to ``destination``; always remove it afterwards."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        yield temporary_path
    finally:
        temporary_path.unlink(missing_ok=True)
