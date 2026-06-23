"""Outer benchmark runner for config-driven phonics experiments."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from phonics_benchmark.config import (
    BenchmarkRunConfig,
    ConfigError,
    config_sha256,
    load_run_config,
)
from phonics_benchmark.items import item_to_dict, load_benchmark_items
from phonics_benchmark.results import load_results, save_results

CALL_STATUS_ERROR = "error"
CALL_STATUS_SUCCESS = "success"
PLACEHOLDER_LLM_TEXT = "placeholder"
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ENV_PATH = REPO_ROOT / ".env"
DEFAULT_CUSTOM_MODELS_PATH = REPO_ROOT / "custom_models.yaml"


RunOutput = dict[str, Any]


def run_from_config_file(
    config_path: str | Path,
    *,
    default_data_root: str | Path | None = None,
    evaluator: Callable[
        [BenchmarkRunConfig, str, Mapping[str, Any], str], dict[str, Any]
    ]
    | None = None,
) -> RunOutput:
    """Load a run config file and execute the configured benchmark."""

    config = load_run_config(config_path, default_data_root=default_data_root)
    config_id = config_sha256(Path(config_path).expanduser().resolve())
    model_configs = None
    if evaluator is None:
        prepare_fdllm_for_run(config)
        model_configs = snapshot_model_configs(config.models)
    return run_benchmark(
        config,
        config_id=config_id,
        evaluator=evaluator,
        model_configs=model_configs,
    )


def run_benchmark(
    config: BenchmarkRunConfig,
    *,
    config_id: str,
    evaluator: Callable[
        [BenchmarkRunConfig, str, Mapping[str, Any], str], dict[str, Any]
    ]
    | None = None,
    model_configs: Mapping[str, Any] | None = None,
) -> RunOutput:
    """Execute a phonics benchmark run from a validated config.

    Callers using the default FDLLM evaluator should call
    prepare_fdllm_for_run(config), then pass a model_configs snapshot from
    snapshot_model_configs(config.models), before entering this function.
    """

    if evaluator is None:
        evaluator = evaluate_item_with_fdllm

    custom_models_path = _resolve_custom_models_path(config.custom_models_path)
    resolved_config = config.to_resolved_dict()
    resolved_config["custom_models_path"] = str(custom_models_path)
    code_git_hash = git_hash()
    items = load_benchmark_items(config.language, config.data_root)
    run_dir = config.output_dir / config.run_id
    results_dir = run_dir / "answers"
    config_snapshot_path = run_dir / "config.json"
    results_paths = {
        model_name: _model_results_path(results_dir, model_name)
        for model_name in config.models
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        config_snapshot_path,
        {
            "config_id": config_id,
            "source_config_path": (
                str(config.config_path) if config.config_path else None
            ),
            "raw_config": config.raw_config,
            "resolved_config": resolved_config,
            "code_git_hash": code_git_hash,
            "model_configs": _sanitize_model_config(dict(model_configs or {})),
        },
    )

    executed_count = 0
    skipped_count = 0
    executed_models: set[str] = set()
    error_count = 0

    for model_name in config.models:
        results_path = results_paths[model_name]
        if config.retry_errors and not results_path.exists():
            raise FileNotFoundError(
                f"No cached results found for {model_name}; cannot use --retry-errors."
            )
        existing_results = load_results(results_path, force=config.force)
        rows_by_id = _results_by_id(existing_results)
        ordered_ids = _ordered_result_ids(existing_results)
        pending_items: list[dict[str, Any]] = []

        for item in items.to_dict(orient="records"):
            item_id = str(item["item_id"])
            result_id = _result_id(
                config_id=config_id,
                run_id=config.run_id,
                model_name=model_name,
                item_id=item_id,
            )
            existing = rows_by_id.get(result_id)
            if existing and _should_skip_existing(
                existing, retry_errors=config.retry_errors
            ):
                skipped_count += 1
                continue

            pending_items.append(
                {
                    "item": item,
                    "result_id": result_id,
                }
            )

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
                    prompt=PLACEHOLDER_LLM_TEXT,
                    result_id=result_id,
                    evaluator=evaluator,
                )
                if result_id not in rows_by_id:
                    ordered_ids.append(result_id)
                rows_by_id[result_id] = row
                executed_count += 1
                executed_models.add(model_name)

            results = _ordered_results(rows_by_id, ordered_ids)
            _save_results(
                results_path,
                results,
                config=config,
                resolved_config=resolved_config,
                custom_models_path=custom_models_path,
            )
        error_count += _error_count(_ordered_results(rows_by_id, ordered_ids))

    output: RunOutput = {
        "run_id": config.run_id,
        "config_id": config_id,
        "run_dir": run_dir,
        "config_snapshot_path": config_snapshot_path,
        "results_dir": results_dir,
        "results_paths": results_paths,
        "total_expected_results": len(config.models) * len(items),
        "executed_count": executed_count,
        "skipped_count": skipped_count,
        "error_count": error_count,
    }
    _update_run_log(
        run_dir,
        executed_models=executed_models,
        retry_errors=config.retry_errors,
        code_git_hash=code_git_hash,
    )
    return output


def prepare_fdllm_for_run(config: BenchmarkRunConfig) -> None:
    """Load model environment, register model config, and validate model names."""

    _load_environment(config.env_path)
    _register_custom_models(config.custom_models_path)
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
    """Return the temporary model response payload for one model/item pair."""

    return {
        "output": call_fdllm(config, model_name, item, prompt),
    }


def call_fdllm(
    config: BenchmarkRunConfig,
    model_name: str,
    item: Mapping[str, Any],
    prompt: str,
) -> str:
    """Temporary placeholder for prompt loading and the eventual FDLLM call."""

    return PLACEHOLDER_LLM_TEXT


def _evaluate_item(
    *,
    config: BenchmarkRunConfig,
    config_id: str,
    config_snapshot_path: Path,
    model_name: str,
    item: Mapping[str, Any],
    prompt: str,
    result_id: str,
    evaluator: Callable[
        [BenchmarkRunConfig, str, Mapping[str, Any], str], dict[str, Any]
    ],
) -> dict[str, Any]:
    started_at = _utc_now()
    started_perf = time.perf_counter()
    base_row: dict[str, Any] = {
        "result_id": result_id,
        "run_id": config.run_id,
        "config_id": config_id,
        "config_path": str(config.config_path) if config.config_path else None,
        "config_snapshot_path": str(config_snapshot_path),
        "language": config.language,
        "task": config.task,
        "model": model_name,
        "item_id": item["item_id"],
        "item": item_to_dict(item),
        "prompt": prompt,
        "started_at": started_at,
    }

    try:
        payload = evaluator(config, model_name, item, prompt)
        elapsed_seconds = time.perf_counter() - started_perf
        return {
            **base_row,
            **payload,
            "success": True,
            "call_status": CALL_STATUS_SUCCESS,
            "error_type": "",
            "error_message": "",
            "elapsed_seconds": elapsed_seconds,
            "completed_at": _utc_now(),
        }
    except Exception as e:
        elapsed_seconds = time.perf_counter() - started_perf
        return {
            **base_row,
            "output": "",
            "success": False,
            "call_status": CALL_STATUS_ERROR,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "elapsed_seconds": elapsed_seconds,
            "completed_at": _utc_now(),
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


def _resolve_custom_models_path(custom_models_path: Path | None) -> Path:
    resolved_custom_models_path = custom_models_path or DEFAULT_CUSTOM_MODELS_PATH
    if not resolved_custom_models_path.exists():
        raise FileNotFoundError(
            f"Custom models file not found: {resolved_custom_models_path}"
        )
    return resolved_custom_models_path


def _should_skip_existing(row: dict[str, Any], *, retry_errors: bool) -> bool:
    if row.get("call_status") == CALL_STATUS_SUCCESS:
        return True
    return not retry_errors


def _save_results(
    results_path: Path,
    results: pd.DataFrame,
    *,
    config: BenchmarkRunConfig,
    resolved_config: dict[str, Any],
    custom_models_path: Path | None,
) -> None:
    save_results(
        results,
        results_path,
        resolved_config,
        config.run_id,
        custom_models_path,
    )


def _results_by_id(results: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if results.empty:
        return {}
    return {str(row["result_id"]): row for row in results.to_dict(orient="records")}


def _ordered_result_ids(results: pd.DataFrame) -> list[str]:
    if results.empty:
        return []
    return [str(result_id) for result_id in results["result_id"].tolist()]


def _ordered_results(
    rows_by_id: dict[str, dict[str, Any]],
    ordered_ids: list[str],
) -> pd.DataFrame:
    return pd.DataFrame([rows_by_id[row_id] for row_id in ordered_ids])


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
    return results_dir / f"resps_{model_name}.jsonl"


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
) -> None:
    if not executed_models:
        return
    prefix = "answers_retry" if retry_errors else "answers"
    fields = {
        f"{prefix}_timestamp-utc": _utc_now(),
        f"{prefix}_git-hash": code_git_hash,
    }
    for model_name in sorted(executed_models):
        update_log(run_dir, model_name, fields)


def update_log(run_dir: Path, model_name: str, fields: dict[str, str | None]) -> None:
    """Merge-write model provenance fields into log.json; skip None values."""

    log_path = run_dir / "log.json"
    log_data: dict[str, dict[str, str | None]] = {}
    if log_path.exists():
        log_data = json.loads(log_path.read_text(encoding="utf-8"))
    entry = log_data.setdefault(model_name, {})
    entry.update({key: value for key, value in fields.items() if value is not None})
    _write_json(log_path, log_data)


def git_hash() -> str | None:
    """Return the short git hash of HEAD, or None if unavailable."""

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


def _result_id(*, config_id: str, run_id: str, model_name: str, item_id: str) -> str:
    payload = _json_dumps(
        {
            "config_id": config_id,
            "run_id": run_id,
            "model": model_name,
            "item_id": item_id,
        }
    )
    return sha256(payload.encode("utf-8")).hexdigest()[:24]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_dumps(payload, indent=2) + "\n", encoding="utf-8")


def _json_dumps(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=indent)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
