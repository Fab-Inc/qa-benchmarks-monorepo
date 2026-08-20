"""Outer benchmark runner for config-driven phonics experiments."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, TypedDict

import pandas as pd
import yaml
from anthropic import APIError as AnthropicAPIError
from botocore.exceptions import (
    ClientError as BedrockAPIError,
)
from botocore.exceptions import (
    ConnectionError as BedrockConnectionError,
)
from botocore.exceptions import (
    HTTPClientError as BedrockHTTPClientError,
)
from google.genai.errors import APIError as GoogleAPIError
from mistralai.exceptions import MistralAPIException, MistralConnectionException
from openai import APIError as OpenAIAPIError

from phonics_benchmark.config import (
    BenchmarkRunConfig,
    ConfigError,
    config_identity_sha256,
    effective_max_tokens,
    load_run_config,
)
from phonics_benchmark.items import load_benchmark_items
from phonics_benchmark.llm import call_llm
from phonics_benchmark.prompts import build_decodability_prompt
from phonics_benchmark.results import load_results, save_results

CALL_STATUS_ERROR = "error"
CALL_STATUS_SUCCESS = "success"
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ENV_PATH = REPO_ROOT / ".env"
DEFAULT_CUSTOM_MODELS_PATH = REPO_ROOT / "custom_models.yaml"
CUSTOM_MODELS_BUCKET = "fab-open-configs"
CUSTOM_MODELS_FILE_NAME = "custom_models.yaml"
SYSTEM_PROMPT: str | None = None


RunOutput = dict[str, Any]
ArtifactWriteHook = Callable[[Path], None]


class _ModelRunOutput(TypedDict):
    model_name: str
    executed_count: int
    skipped_count: int
    error_count: int


class _PreparedItem(TypedDict):
    item: dict[str, Any]
    prompt: str


def run_from_config_file(
    config_path: str | Path,
    *,
    evaluator: Callable[
        [BenchmarkRunConfig, str, Mapping[str, Any], str], dict[str, Any]
    ]
    | None = None,
    on_artifact_write: ArtifactWriteHook | None = None,
) -> RunOutput:
    """Load a run config file and execute it without downloading model config."""

    config = load_run_config(config_path)
    return execute_run(
        config,
        evaluator=evaluator,
        on_artifact_write=on_artifact_write,
    )


def execute_run(
    config: BenchmarkRunConfig,
    *,
    evaluator: Callable[
        [BenchmarkRunConfig, str, Mapping[str, Any], str], dict[str, Any]
    ]
    | None = None,
    on_artifact_write: ArtifactWriteHook | None = None,
) -> RunOutput:
    """Prepare provenance and model state, then execute a validated run config."""

    config_id = config_identity_sha256(config)
    model_configs = None
    if evaluator is None:
        prepare_fdllm_for_run(config)
        model_configs = snapshot_model_configs(config.models)
    run_kwargs: dict[str, Any] = {
        "config_id": config_id,
        "evaluator": evaluator,
        "model_configs": model_configs,
    }
    if on_artifact_write is not None:
        run_kwargs["on_artifact_write"] = on_artifact_write
    return run_benchmark(config, **run_kwargs)


def run_benchmark(
    config: BenchmarkRunConfig,
    *,
    config_id: str,
    evaluator: Callable[
        [BenchmarkRunConfig, str, Mapping[str, Any], str], dict[str, Any]
    ]
    | None = None,
    model_configs: Mapping[str, Any] | None = None,
    on_artifact_write: ArtifactWriteHook | None = None,
) -> RunOutput:
    """Execute a phonics benchmark run from a validated config.

    Normal callers should use execute_run. Direct callers using the default
    FDLLM evaluator must call
    prepare_fdllm_for_run(config), then pass a model_configs snapshot from
    snapshot_model_configs(config.models), before entering this function.
    """

    if evaluator is None:
        evaluator = evaluate_item_with_fdllm

    custom_models_path = _resolve_custom_models_path(config.custom_models_path)
    code_git_hash = git_hash()
    code_git_dirty = git_dirty()
    items, items_sha256 = load_benchmark_items(config.items_path)
    prepared_items = _prepare_items(items, language=config.language)
    provider_groups = _provider_groups_for_models(custom_models_path, config.models)
    resolved_config = config.to_resolved_dict()
    resolved_config["custom_models_path"] = str(custom_models_path)
    resolved_config["items_sha256"] = items_sha256
    run_dir = config.output_dir / config.run_id
    results_dir = run_dir / "responses"
    config_snapshot_path = run_dir / "config.json"
    results_paths = {
        model_name: _model_results_path(results_dir, model_name)
        for model_name in config.models
    }
    sanitized_model_configs = _sanitize_model_config(dict(model_configs or {}))
    _preflight_retry_results(
        config,
        results_paths,
        model_configs=sanitized_model_configs,
        items=prepared_items,
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = {
        "config_id": config_id,
        "source_config_path": str(config.config_path) if config.config_path else None,
        "raw_config": config.raw_config,
        "resolved_config": resolved_config,
        "code_git_hash": code_git_hash,
        "code_git_dirty": code_git_dirty,
        "model_configs": sanitized_model_configs,
    }
    _write_json(config_snapshot_path, config_snapshot)
    if on_artifact_write is not None:
        on_artifact_write(config_snapshot_path)

    model_outputs = _run_models(
        config=config,
        config_id=config_id,
        config_snapshot_path=config_snapshot_path,
        model_configs=sanitized_model_configs,
        items=prepared_items,
        results_paths=results_paths,
        evaluator=evaluator,
        provider_groups=provider_groups,
        on_artifact_write=on_artifact_write,
    )
    executed_count = sum(result["executed_count"] for result in model_outputs)
    skipped_count = sum(result["skipped_count"] for result in model_outputs)
    error_count = sum(result["error_count"] for result in model_outputs)
    executed_models = {
        result["model_name"] for result in model_outputs if result["executed_count"] > 0
    }

    output: RunOutput = {
        "run_id": config.run_id,
        "config_id": config_id,
        "run_dir": run_dir,
        "config_snapshot_path": config_snapshot_path,
        "results_dir": results_dir,
        "results_paths": results_paths,
        "total_expected_results": len(config.models) * len(prepared_items),
        "executed_count": executed_count,
        "skipped_count": skipped_count,
        "error_count": error_count,
    }
    _update_run_log(
        run_dir,
        executed_models=executed_models,
        retry_errors=config.retry_errors,
        code_git_hash=code_git_hash,
        code_git_dirty=code_git_dirty,
        on_artifact_write=on_artifact_write,
    )
    return output


def _prepare_items(
    items: list[dict[str, Any]],
    *,
    language: str,
) -> list[_PreparedItem]:
    """Validate and render every item before any model execution starts."""

    return [
        {
            "item": item,
            "prompt": build_decodability_prompt(item, language),
        }
        for item in items
    ]


def _run_models(
    *,
    config: BenchmarkRunConfig,
    config_id: str,
    config_snapshot_path: Path,
    model_configs: Mapping[str, Any],
    items: list[_PreparedItem],
    results_paths: Mapping[str, Path],
    evaluator: Callable[
        [BenchmarkRunConfig, str, Mapping[str, Any], str], dict[str, Any]
    ],
    provider_groups: list[list[str]],
    on_artifact_write: ArtifactWriteHook | None,
) -> list[_ModelRunOutput]:
    """Run provider groups concurrently and their models sequentially."""

    worker_kwargs = {
        model_name: {
            "config": config,
            "config_id": config_id,
            "config_snapshot_path": config_snapshot_path,
            "model_name": model_name,
            "model_config": model_configs.get(model_name, {}),
            "items": deepcopy(items),
            "results_path": results_paths[model_name],
            "evaluator": evaluator,
            "on_artifact_write": on_artifact_write,
        }
        for model_name in config.models
    }
    group_worker_kwargs = [
        [worker_kwargs[model_name] for model_name in model_names]
        for model_names in provider_groups
    ]
    if config.model_workers == 1 or len(group_worker_kwargs) == 1:
        group_outputs = [
            _run_provider_group(model_workers) for model_workers in group_worker_kwargs
        ]
    else:
        max_workers = min(config.model_workers, len(group_worker_kwargs))
        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="phonics-provider",
        ) as executor:
            futures = [
                executor.submit(_run_provider_group, model_workers)
                for model_workers in group_worker_kwargs
            ]
            group_outputs = [future.result() for future in futures]

    outputs_by_model = {
        output["model_name"]: output
        for group_output in group_outputs
        for output in group_output
    }
    return [outputs_by_model[model_name] for model_name in config.models]


def _run_provider_group(
    model_workers: list[dict[str, Any]],
) -> list[_ModelRunOutput]:
    """Run all selected models for one provider group without overlap."""

    return [_run_model(**worker) for worker in model_workers]


def _run_model(
    *,
    config: BenchmarkRunConfig,
    config_id: str,
    config_snapshot_path: Path,
    model_name: str,
    model_config: Mapping[str, Any],
    items: list[_PreparedItem],
    results_path: Path,
    evaluator: Callable[
        [BenchmarkRunConfig, str, Mapping[str, Any], str], dict[str, Any]
    ],
    on_artifact_write: ArtifactWriteHook | None,
) -> _ModelRunOutput:
    """Run every pending item for one model, sequentially and with checkpoints."""

    existing_results = load_results(results_path, force=config.force)
    existing_rows_by_id = _results_by_id(existing_results)
    planned_items = _planned_model_items(
        config=config,
        model_name=model_name,
        model_config=model_config,
        items=items,
    )
    expected_result_ids = [
        str(planned_item["result_id"]) for planned_item in planned_items
    ]
    expected_result_id_set = set(expected_result_ids)
    stale_result_count = len(
        set(existing_rows_by_id).difference(expected_result_id_set)
    )
    rows_by_id = {
        result_id: existing_rows_by_id[result_id]
        for result_id in expected_result_ids
        if result_id in existing_rows_by_id
    }
    pending_items: list[dict[str, Any]] = []
    skipped_count = 0

    for planned_item in planned_items:
        result_id = str(planned_item["result_id"])
        existing = rows_by_id.get(result_id)
        if existing is None and config.retry_errors:
            raise ConfigError(
                f"--retry-errors found no cached result for model {model_name} "
                "item "
                f"{planned_item['item']['benchmark_item_id']}; run normally first."
            )
        if existing and _should_skip_existing(
            existing, retry_errors=config.retry_errors
        ):
            skipped_count += 1
            continue

        pending_items.append(
            {
                "item": planned_item["item"],
                "model_call_id": planned_item["model_call_id"],
                "prompt": planned_item["prompt"],
                "result_id": result_id,
            }
        )

    executed_count = 0
    for chunk in _chunked(pending_items, config.chunk_size):
        for pending in chunk:
            item = pending["item"]
            result_id = pending["result_id"]
            row = _evaluate_item(
                config=config,
                config_id=config_id,
                config_snapshot_path=config_snapshot_path,
                model_name=model_name,
                item=item,
                model_call_id=pending["model_call_id"],
                prompt=pending["prompt"],
                result_id=result_id,
                evaluator=evaluator,
            )
            rows_by_id[result_id] = row
            executed_count += 1

        results = _ordered_results(rows_by_id, expected_result_ids)
        save_results(results, results_path)
        if on_artifact_write is not None:
            on_artifact_write(results_path)
    if stale_result_count and not pending_items:
        results = _ordered_results(rows_by_id, expected_result_ids)
        save_results(results, results_path)
        if on_artifact_write is not None:
            on_artifact_write(results_path)

    results = _ordered_results(rows_by_id, expected_result_ids)
    return {
        "model_name": model_name,
        "executed_count": executed_count,
        "skipped_count": skipped_count,
        "error_count": _error_count(results),
    }


def _preflight_retry_results(
    config: BenchmarkRunConfig,
    results_paths: Mapping[str, Path],
    *,
    model_configs: Mapping[str, Any],
    items: list[_PreparedItem],
) -> None:
    """Validate every retry cache before starting any model worker."""

    if not config.retry_errors:
        return
    missing_models = [
        model_name
        for model_name in config.models
        if not results_paths[model_name].exists()
    ]
    if missing_models:
        raise FileNotFoundError(
            "No cached results found for "
            + ", ".join(missing_models)
            + "; cannot use --retry-errors."
        )

    missing_call_counts: dict[str, int] = {}
    invalid_status_counts: dict[str, int] = {}
    for model_name in config.models:
        existing_rows_by_id = _results_by_id(
            load_results(results_paths[model_name], force=False)
        )
        planned_items = _planned_model_items(
            config=config,
            model_name=model_name,
            model_config=model_configs.get(model_name, {}),
            items=items,
        )
        expected_result_ids = [
            str(planned_item["result_id"]) for planned_item in planned_items
        ]
        missing_count = sum(
            result_id not in existing_rows_by_id for result_id in expected_result_ids
        )
        if missing_count:
            missing_call_counts[model_name] = missing_count

        invalid_status_count = sum(
            existing_rows_by_id[result_id].get("call_status")
            not in {CALL_STATUS_ERROR, CALL_STATUS_SUCCESS}
            for result_id in expected_result_ids
            if result_id in existing_rows_by_id
        )
        if invalid_status_count:
            invalid_status_counts[model_name] = invalid_status_count

    if missing_call_counts:
        details = ", ".join(
            f"{model_name} ({count})"
            for model_name, count in missing_call_counts.items()
        )
        raise ConfigError(
            "--retry-errors requires cached results for every planned call; "
            f"missing calls by model: {details}. Run normally first to execute "
            "new or changed calls."
        )
    if invalid_status_counts:
        details = ", ".join(
            f"{model_name} ({count})"
            for model_name, count in invalid_status_counts.items()
        )
        raise ConfigError(
            "--retry-errors found cached rows with invalid call_status values "
            f"by model: {details}."
        )


def prepare_fdllm_for_run(config: BenchmarkRunConfig) -> None:
    """Load model environment, register the selected file, and validate models."""

    _load_environment(config.env_path)
    custom_models_path = _resolve_custom_models_path(config.custom_models_path)
    _register_custom_models(custom_models_path)
    validate_registered_models(config.models)


def validate_registered_models(model_names: list[str]) -> None:
    """Raise ConfigError if any configured model is not registered with FDLLM."""

    from fdllm import list_models

    registered_models = set(list_models())
    missing_models = [
        model_name for model_name in model_names if model_name not in registered_models
    ]
    if missing_models:
        raise ConfigError(
            "Configured model(s) are not registered with FDLLM: "
            + ", ".join(missing_models)
        )


def snapshot_model_configs(model_names: list[str]) -> dict[str, Any]:
    """Return sanitized model configs from FDLLM's registered model state."""

    from fdllm import list_models

    model_configs = list_models(full_info=True)
    return {
        model_name: _sanitize_model_config(model_configs[model_name])
        for model_name in model_names
        if model_name in model_configs
    }


def evaluate_item_with_fdllm(
    config: BenchmarkRunConfig,
    model_name: str,
    item: Mapping[str, Any],
    prompt: str,
) -> dict[str, Any]:
    """Call the configured model and return the fields stored in its result row."""

    call_record = call_llm(
        prompt,
        model=model_name,
        system_prompt=SYSTEM_PROMPT,
        max_tokens=_effective_max_tokens(config.max_tokens),
        should_log=False,
        tag=str(item["benchmark_item_id"]),
    )
    return {
        "output": call_record["response_text"],
        "max_tokens": call_record["max_tokens"],
        "tokens_used_completion": call_record["tokens_used_completion"],
        "timestamp_utc": call_record["timestamp_utc"],
        "latency_seconds": call_record["latency_seconds"],
    }


def _planned_model_items(
    *,
    config: BenchmarkRunConfig,
    model_name: str,
    model_config: Mapping[str, Any],
    items: list[_PreparedItem],
) -> list[dict[str, Any]]:
    planned_items = []
    for prepared_item in items:
        item = prepared_item["item"]
        prompt = prepared_item["prompt"]
        model_call_id = _model_call_id(
            benchmark_item_id=item["benchmark_item_id"],
            config=config,
            model_name=model_name,
            model_config=model_config,
            prompt=prompt,
        )
        planned_items.append(
            {
                "item": item,
                "model_call_id": model_call_id,
                "prompt": prompt,
                "result_id": _result_id(model_call_id=model_call_id),
            }
        )
    return planned_items


def _evaluate_item(
    *,
    config: BenchmarkRunConfig,
    config_id: str,
    config_snapshot_path: Path,
    model_name: str,
    item: Mapping[str, Any],
    model_call_id: str,
    prompt: str,
    result_id: str,
    evaluator: Callable[
        [BenchmarkRunConfig, str, Mapping[str, Any], str], dict[str, Any]
    ],
) -> dict[str, Any]:
    base_row: dict[str, Any] = {
        "result_id": result_id,
        "run_id": config.run_id,
        "config_id": config_id,
        "config_path": str(config.config_path) if config.config_path else None,
        "config_snapshot_path": str(config_snapshot_path),
        "language": config.language,
        "task": config.task,
        "model": model_name,
        "model_call_id": model_call_id,
        "benchmark_item_id": item["benchmark_item_id"],
        "prompt": prompt,
    }

    # FDLLM propagates provider SDK failures unchanged. Save only those known
    # call failures here so setup and result-processing bugs fail fast.
    try:
        payload = evaluator(config, model_name, item, prompt)
    except (
        AnthropicAPIError,
        BedrockAPIError,
        BedrockConnectionError,
        BedrockHTTPClientError,
        GoogleAPIError,
        MistralAPIException,
        MistralConnectionException,
        OpenAIAPIError,
    ) as e:
        return {
            **base_row,
            "output": "",
            "success": False,
            "call_status": CALL_STATUS_ERROR,
            "error_type": type(e).__name__,
            "error_message": str(e),
        }

    if not payload["output"]:
        return {
            **base_row,
            "output": "",
            "success": False,
            "call_status": CALL_STATUS_ERROR,
            "error_type": "EmptyLLMResponse",
            "error_message": (
                f"LLM response empty for item {item['benchmark_item_id']}"
            ),
        }

    return {
        **base_row,
        **payload,
        "success": True,
        "call_status": CALL_STATUS_SUCCESS,
        "error_type": "",
        "error_message": "",
    }


def _load_environment(env_path: Path | None) -> Path | None:
    resolved_env_path = env_path or DEFAULT_ENV_PATH
    if env_path is not None and not resolved_env_path.exists():
        raise FileNotFoundError(f"Environment file not found: {resolved_env_path}")
    if not resolved_env_path.exists():
        return None

    from dotenv import load_dotenv

    load_dotenv(resolved_env_path, override=False)
    return resolved_env_path


def _register_custom_models(custom_models_path: Path | None) -> Path:
    resolved_custom_models_path = _resolve_custom_models_path(custom_models_path)

    from fdllm import register_models

    register_models(str(resolved_custom_models_path))
    return resolved_custom_models_path


def download_default_custom_models(destination: Path | None = None) -> Path:
    """Anonymously refresh custom_models.yaml at the selected destination."""

    from fab_benchmark_publisher import download_file

    destination = (destination or DEFAULT_CUSTOM_MODELS_PATH).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{CUSTOM_MODELS_FILE_NAME}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)

    try:
        download_file(
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
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)

    return destination


def _resolve_custom_models_path(custom_models_path: Path | None) -> Path:
    resolved_custom_models_path = custom_models_path or DEFAULT_CUSTOM_MODELS_PATH
    if not resolved_custom_models_path.exists():
        raise FileNotFoundError(
            f"Custom models file not found: {resolved_custom_models_path}"
        )
    return resolved_custom_models_path


def _provider_groups_for_models(
    custom_models_path: Path,
    model_names: list[str],
) -> list[list[str]]:
    """Map selected models to their unique top-level provider groups."""

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

    missing = [model_name for model_name in model_names if not memberships[model_name]]
    ambiguous = {
        model_name: memberships[model_name]
        for model_name in model_names
        if len(memberships[model_name]) > 1
    }
    if missing:
        raise ConfigError(
            "Selected model(s) are missing from custom_models.yaml provider groups: "
            + ", ".join(missing)
        )
    if ambiguous:
        details = ", ".join(
            f"{model_name} ({', '.join(groups)})"
            for model_name, groups in ambiguous.items()
        )
        raise ConfigError(
            "Selected model(s) belong to multiple custom_models.yaml provider "
            f"groups: {details}"
        )

    groups_by_name: dict[str, list[str]] = {}
    for model_name in model_names:
        provider_group = memberships[model_name][0]
        groups_by_name.setdefault(provider_group, []).append(model_name)
    return list(groups_by_name.values())


def _should_skip_existing(row: dict[str, Any], *, retry_errors: bool) -> bool:
    if retry_errors:
        return row.get("call_status") != CALL_STATUS_ERROR
    return True


def _results_by_id(results: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if results.empty:
        return {}
    return {str(row["result_id"]): row for row in results.to_dict(orient="records")}


def _ordered_results(
    rows_by_id: dict[str, dict[str, Any]],
    ordered_ids: list[str],
) -> pd.DataFrame:
    return pd.DataFrame(
        [rows_by_id[row_id] for row_id in ordered_ids if row_id in rows_by_id]
    )


def _error_count(results: pd.DataFrame) -> int:
    if results.empty or "call_status" not in results.columns:
        return 0
    return int(results["call_status"].eq(CALL_STATUS_ERROR).sum())


def _chunked(
    items: list[dict[str, Any]],
    chunk_size: int | None,
) -> list[list[dict[str, Any]]]:
    if not items:
        return []
    if chunk_size is None:
        return [items]
    return [
        items[start : start + chunk_size] for start in range(0, len(items), chunk_size)
    ]


def _model_results_path(results_dir: Path, model_name: str) -> Path:
    return results_dir / model_results_filename(model_name)


def model_results_filename(model_name: str) -> str:
    """Return the stable per-model response artifact filename."""

    return f"responses_{_model_filename_key(model_name)}.jsonl"


def _model_filename_key(model_name: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("._-")
    if not safe_name:
        safe_name = "model"
    if safe_name == model_name:
        return safe_name
    suffix = sha256(model_name.encode("utf-8")).hexdigest()[:8]
    return f"{safe_name}-{suffix}"


def _sanitize_model_config(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _sanitize_model_config(nested)
            for key, nested in value.items()
            if not _is_secret_model_config_key(str(key))
        }
    if isinstance(value, list):
        return [_sanitize_model_config(item) for item in value]
    return value


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


def _update_run_log(
    run_dir: Path,
    *,
    executed_models: set[str],
    retry_errors: bool,
    code_git_hash: str | None,
    code_git_dirty: bool | None,
    on_artifact_write: ArtifactWriteHook | None,
) -> None:
    if not executed_models:
        return
    prefix = "responses_retry" if retry_errors else "responses"
    fields = {
        f"{prefix}_timestamp_utc": _utc_now(),
        f"{prefix}_git_hash": code_git_hash,
        f"{prefix}_git_dirty": code_git_dirty,
    }
    for model_name in sorted(executed_models):
        update_log(
            run_dir,
            model_name,
            fields,
            on_artifact_write=on_artifact_write,
        )


def update_log(
    run_dir: Path,
    model_name: str,
    fields: dict[str, str | bool | None],
    *,
    on_artifact_write: ArtifactWriteHook | None = None,
) -> None:
    """Merge-write model provenance fields into log.json; skip None values."""

    log_path = run_dir / "log.json"
    log_data: dict[str, dict[str, str | bool | None]] = {}
    if log_path.exists():
        log_data = json.loads(log_path.read_text(encoding="utf-8"))
    entry = log_data.setdefault(model_name, {})
    entry.update({key: value for key, value in fields.items() if value is not None})
    _write_json(log_path, log_data)
    if on_artifact_write is not None:
        on_artifact_write(log_path)


def git_hash() -> str | None:
    """Return the image revision or local short git hash, if available."""

    image_revision = os.environ.get("CODE_GIT_HASH", "").strip()
    if image_revision and image_revision != "unknown":
        return image_revision

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[4],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception as e:
        print(f"WARNING: could not read git hash: {e}", file=sys.stderr)
        return None


def git_dirty() -> bool | None:
    """Return whether the image source or local worktree is dirty, if known."""

    image_dirty = os.environ.get("CODE_GIT_DIRTY", "").strip().lower()
    if image_dirty == "true":
        return True
    if image_dirty == "false":
        return False

    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=Path(__file__).resolve().parents[4],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return bool(status.strip())
    except Exception as e:
        print(f"WARNING: could not read git dirty state: {e}", file=sys.stderr)
        return None


def _model_call_id(
    *,
    benchmark_item_id: str,
    config: BenchmarkRunConfig,
    model_name: str,
    model_config: Mapping[str, Any],
    prompt: str,
) -> str:
    return _sha256_json(
        {
            "benchmark_item_id": benchmark_item_id,
            "generation_parameters": {
                "max_tokens": _effective_max_tokens(config.max_tokens),
            },
            "model": model_name,
            "model_config": model_config,
            "prompt": prompt,
            "system_prompt": SYSTEM_PROMPT,
        }
    )


def _effective_max_tokens(configured_max_tokens: int | None) -> int:
    return effective_max_tokens(configured_max_tokens)


def _result_id(*, model_call_id: str) -> str:
    return _sha256_json(
        {
            "model_call_id": model_call_id,
        }
    )[:24]


def _sha256_json(value: Any) -> str:
    return sha256(_json_dumps(value).encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_dumps(payload, indent=2) + "\n", encoding="utf-8")


def _json_dumps(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=indent)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
