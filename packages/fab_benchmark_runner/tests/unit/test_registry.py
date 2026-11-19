"""The model registry: public and pinned downloads, FDLLM registration, groups."""

from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from fab_benchmark_runner import ConfigError, parse_run_config
from fab_benchmark_runner.registry import (
    ModelRegistry,
    check_provider_groups,
    download_pinned_registry,
    download_public_registry,
    load_environment,
    prepare_fdllm,
    prepare_model_registry,
    snapshot_model_configs,
)

REGISTRY = b"provider:\n  models:\n    model-a: {}\n"
PINNED_URI = "gs://results/model-configs/experiment-a.yaml"


def _fake_download(content: bytes, calls: list[dict] | None = None):
    def download(**kwargs) -> None:
        if calls is not None:
            calls.append(dict(kwargs))
        Path(kwargs["write_path"]).write_bytes(content)

    return download


def _config(tmp_path: Path, **raw: Any):
    return parse_run_config(
        {"experiment": "exp", "models": ["model-a", "model-b"], **raw},
        base_dir=tmp_path,
    )


def test_public_download_is_anonymous_and_atomic(tmp_path: Path) -> None:
    destination = tmp_path / "custom_models.yaml"
    destination.write_text("OldProvider: {}\n", encoding="utf-8")
    calls: list[dict] = []

    with patch(
        "fab_benchmark_runner.gcs.download_file",
        side_effect=_fake_download(b"NewProvider:\n  models: {}\n", calls),
    ):
        result = download_public_registry(destination)

    [call] = calls
    temporary_path = Path(call.pop("write_path"))
    assert call == {
        "bucket_name": "fab-open-configs",
        "file_name": "custom_models.yaml",
        "require_auth": False,
    }
    assert temporary_path.parent == destination.parent
    assert not temporary_path.exists()
    assert result == destination.resolve()
    assert destination.read_text(encoding="utf-8") == "NewProvider:\n  models: {}\n"


@pytest.mark.parametrize(
    ("downloaded", "message"),
    [(b"[", "invalid"), (b"[]", "non-empty YAML mapping")],
)
def test_public_download_rejects_invalid_registries(
    tmp_path: Path, downloaded: bytes, message: str
) -> None:
    destination = tmp_path / "custom_models.yaml"
    destination.write_text("OldProvider: {}\n", encoding="utf-8")

    with (
        patch(
            "fab_benchmark_runner.gcs.download_file",
            side_effect=_fake_download(downloaded),
        ),
        pytest.raises(ConfigError, match=message),
    ):
        download_public_registry(destination)

    assert destination.read_text(encoding="utf-8") == "OldProvider: {}\n"
    assert list(tmp_path.iterdir()) == [destination]


def test_public_download_failure_keeps_the_existing_file(tmp_path: Path) -> None:
    destination = tmp_path / "custom_models.yaml"
    destination.write_text("OldProvider: {}\n", encoding="utf-8")

    def fail(**kwargs) -> None:
        Path(kwargs["write_path"]).write_text("partial", encoding="utf-8")
        raise ConnectionError("download interrupted")

    with (
        patch("fab_benchmark_runner.gcs.download_file", side_effect=fail),
        pytest.raises(ConnectionError),
    ):
        download_public_registry(destination)

    assert destination.read_text(encoding="utf-8") == "OldProvider: {}\n"
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("valid_checksum", [True, False])
def test_pinned_download_verifies_the_checksum(
    tmp_path: Path, valid_checksum: bool
) -> None:
    destination = tmp_path / "inputs" / "custom_models.yaml"
    destination.parent.mkdir()
    destination.write_bytes(b"previous registry")
    calls: list[dict] = []
    downloaded = REGISTRY if valid_checksum else b"changed"

    with patch(
        "fab_benchmark_runner.gcs.download_file",
        side_effect=_fake_download(downloaded, calls),
    ):
        if valid_checksum:
            download_pinned_registry(
                PINNED_URI,
                expected_sha256=sha256(REGISTRY).hexdigest(),
                destination=destination,
            )
        else:
            with pytest.raises(ConfigError, match="SHA-256 does not match"):
                download_pinned_registry(
                    PINNED_URI,
                    expected_sha256=sha256(REGISTRY).hexdigest(),
                    destination=destination,
                )

    assert (calls[0]["bucket_name"], calls[0]["file_name"]) == (
        "results",
        "model-configs/experiment-a.yaml",
    )
    assert "require_auth" not in calls[0]
    assert destination.read_bytes() == (
        REGISTRY if valid_checksum else b"previous registry"
    )
    assert not list(destination.parent.glob(".custom_models.yaml.*"))


@pytest.mark.parametrize("download_public", [True, False])
def test_prepare_uses_pinned_registry_even_when_skipping_downloads(
    tmp_path: Path, download_public: bool
) -> None:
    config = _config(
        tmp_path,
        custom_models_uri=PINNED_URI,
        custom_models_sha256=sha256(REGISTRY).hexdigest(),
    )

    with (
        patch(
            "fab_benchmark_runner.gcs.download_file",
            side_effect=_fake_download(REGISTRY),
        ),
        patch("fab_benchmark_runner.registry.download_public_registry") as public,
    ):
        registry = prepare_model_registry(
            config, workdir=tmp_path / "work", download_public=download_public
        )

    public.assert_not_called()
    assert registry.path.parent == (tmp_path / "work" / "registry").resolve()
    assert registry.path.read_bytes() == REGISTRY
    assert registry.record() == {
        "sha256": sha256(REGISTRY).hexdigest(),
        "uri": PINNED_URI,
    }


def test_prepare_keeps_an_explicit_registry_untouched(tmp_path: Path) -> None:
    (tmp_path / "models.yaml").write_bytes(REGISTRY)
    config = _config(tmp_path, custom_models_path="models.yaml")

    with patch("fab_benchmark_runner.registry.download_public_registry") as public:
        registry = prepare_model_registry(
            config, workdir=tmp_path, download_public=True
        )

    public.assert_not_called()
    assert registry == ModelRegistry(
        (tmp_path / "models.yaml").resolve(), sha256(REGISTRY).hexdigest()
    )


@pytest.mark.parametrize("download_public", [True, False])
def test_prepare_refreshes_the_public_registry_unless_skipped(
    tmp_path: Path, download_public: bool
) -> None:
    public_path = tmp_path / "workspace" / "custom_models.yaml"
    public_path.parent.mkdir()
    public_path.write_bytes(REGISTRY)

    with patch("fab_benchmark_runner.registry.download_public_registry") as public:
        registry = prepare_model_registry(
            _config(tmp_path),
            workdir=tmp_path,
            download_public=download_public,
            public_path=public_path,
        )

    assert public.called is download_public
    assert registry.path == public_path


def test_the_public_registry_defaults_to_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "custom_models.yaml").write_bytes(REGISTRY)

    registry = prepare_model_registry(
        _config(tmp_path), workdir=tmp_path, download_public=False
    )

    assert registry.path == tmp_path / "custom_models.yaml"


def test_a_missing_registry_file_is_reported(tmp_path: Path) -> None:
    config = _config(tmp_path, custom_models_path="missing.yaml")

    with pytest.raises(FileNotFoundError, match="Custom models file not found"):
        prepare_model_registry(config, workdir=tmp_path, download_public=False)


def test_prepare_fdllm_registers_the_registry_and_returns_model_configs(
    tmp_path: Path,
) -> None:
    registry = ModelRegistry(tmp_path / "models.yaml", "a" * 64)

    with (
        patch("fab_benchmark_runner.registry.load_environment") as load_env,
        patch("fab_benchmark_runner.registry.register_custom_models") as register,
        patch("fab_benchmark_runner.registry.validate_registered_models") as validate,
        patch(
            "fab_benchmark_runner.registry.snapshot_model_configs",
            return_value={"model-a": {}},
        ),
    ):
        configs = prepare_fdllm(registry, ("model-a",), env_path=tmp_path / ".env")

    load_env.assert_called_once_with(tmp_path / ".env")
    register.assert_called_once_with(registry.path)
    validate.assert_called_once_with(["model-a"])
    assert configs == {"model-a": {}}


def test_an_explicit_env_file_must_exist(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Environment file not found"):
        load_environment(tmp_path / "missing.env")


def test_env_file_does_not_override_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "keys.env").write_text("A_KEY=from-file\nB_KEY=from-file\n")
    monkeypatch.setenv("A_KEY", "from-env")
    monkeypatch.delenv("B_KEY", raising=False)

    assert load_environment(tmp_path / "keys.env") == tmp_path / "keys.env"

    assert os.environ["A_KEY"] == "from-env"
    assert os.environ["B_KEY"] == "from-file"
    monkeypatch.delenv("B_KEY")


def test_snapshot_sanitizes_fdllm_model_configs() -> None:
    with patch("fdllm.list_models") as list_models:
        list_models.return_value = {
            "model-a": {
                "Api_Interface": "OpenAI",
                "Api_Key_Env_Var": "OPENAI_API_KEY",
                "Client_Args": {
                    "base_url": "https://example.test/v1",
                    "api_key": "secret-value",
                    "Authorization": "Bearer secret-value",
                },
                "Headers": [{"X-Access-Token": "secret"}, {"Accept": "json"}],
            },
            "model-b": {},
        }

        snapshot = snapshot_model_configs(["model-a"])

    list_models.assert_called_once_with(full_info=True)
    assert snapshot == {
        "model-a": {
            "Api_Interface": "OpenAI",
            "Api_Key_Env_Var": "OPENAI_API_KEY",
            "Client_Args": {"base_url": "https://example.test/v1"},
            "Headers": [{}, {"Accept": "json"}],
        }
    }


def test_models_each_in_one_provider_group_pass(tmp_path: Path) -> None:
    registry_path = tmp_path / "custom_models.yaml"
    registry_path.write_text(
        json.dumps(
            {
                "shared": {"models": {"model-a": {}, "model-c": {}}},
                "other": {"models": {"model-b": {}}},
                "no-models": {"Api_Interface": "OpenAI"},
            }
        ),
        encoding="utf-8",
    )

    check_provider_groups(registry_path, ["model-a", "model-b", "model-c"])


@pytest.mark.parametrize(
    ("registry", "message"),
    [
        ({"p": {"models": {"model-b": {}}}}, "missing from custom_models.yaml"),
        (
            {"p": {"models": {"model-a": {}}}, "q": {"models": {"model-a": {}}}},
            "multiple custom_models.yaml provider groups: model-a \\(p, q\\)",
        ),
        ({"p": []}, "must be an object"),
        ({"p": {"models": []}}, "'models' as an object"),
        ({}, "non-empty mapping"),
    ],
)
def test_provider_group_check_rejects_invalid_registries(
    tmp_path: Path, registry: dict, message: str
) -> None:
    registry_path = tmp_path / "custom_models.yaml"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        check_provider_groups(registry_path, ["model-a"])
