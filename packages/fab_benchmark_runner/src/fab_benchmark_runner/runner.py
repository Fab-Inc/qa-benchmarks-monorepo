"""Plan and execute benchmark runs with resumable, per-model result files."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

from anthropic import APIError as AnthropicAPIError
from botocore.exceptions import ClientError as BedrockAPIError
from botocore.exceptions import ConnectionError as BedrockConnectionError
from botocore.exceptions import HTTPClientError as BedrockHTTPClientError
from google.genai.errors import APIError as GoogleAPIError
from mistralai.exceptions import MistralAPIException, MistralConnectionException
from openai import APIError as OpenAIAPIError

from fab_benchmark_runner import ids
from fab_benchmark_runner.config import RunConfig, config_identity_sha256
from fab_benchmark_runner.errors import ConfigError
from fab_benchmark_runner.items import ITEM_ID_KEY, validate_items
from fab_benchmark_runner.llm import evaluate_item_with_fdllm
from fab_benchmark_runner.provenance import git_dirty, git_hash
from fab_benchmark_runner.registry import (
    prepare_fdllm_for_run,
    provider_groups_for_models,
    resolve_custom_models_path,
    sanitize_model_config,
    snapshot_model_configs,
)
from fab_benchmark_runner.results import (
    CALL_STATUS_ERROR,
    CALL_STATUS_SUCCESS,
    error_count,
    load_results,
    model_results_filename,
    save_results,
)

RunOutput = dict[str, Any]
ArtifactWriteHook = Callable[[Path], None]
Evaluator = Callable[[RunConfig, str, Mapping[str, Any], str], dict[str, Any]]

# FDLLM propagates provider SDK failures unchanged. Only these known call failures
# become error rows, so setup and result-processing bugs still fail fast.
PROVIDER_ERRORS: tuple[type[BaseException], ...] = (
    AnthropicAPIError,
    BedrockAPIError,
    BedrockConnectionError,
    BedrockHTTPClientError,
    GoogleAPIError,
    MistralAPIException,
    MistralConnectionException,
    OpenAIAPIError,
)

# Columns every result row starts with; benchmark identity fields may not reuse them.
_BASE_ROW_KEYS = frozenset(
    {
        "result_id",
        "run_id",
        "config_id",
        "config_path",
        "config_snapshot_path",
        "model",
        "model_call_id",
        ITEM_ID_KEY,
        "prompt",
        "output",
        "success",
        "call_status",
        "error_type",
        "error_message",
    }
)


@dataclass(frozen=True)
class ModelPlan:
    """What a run would do for one model."""

    model: str
    results_path: Path
    planned: int
    skipped: int
    to_run: int
    retries: int
    stale: int


@dataclass(frozen=True)
class RunPlan:
    """What a run would do, without calling any model or writing results."""

    run_id: str
    config_id: str
    run_dir: Path
    models: tuple[ModelPlan, ...]

    @property
    def planned(self) -> int:
        return sum(model.planned for model in self.models)

    @property
    def skipped(self) -> int:
        return sum(model.skipped for model in self.models)

    @property
    def to_run(self) -> int:
        return sum(model.to_run for model in self.models)


class _ModelRunOutput(TypedDict):
    model_name: str
    executed_count: int
    skipped_count: int
    error_count: int


@dataclass(frozen=True)
class _PreparedItem:
    item: dict[str, Any]
    prompt: str


@dataclass(frozen=True)
class _PlannedCall:
    item: dict[str, Any]
    prompt: str
    model_call_id: str
    result_id: str


@dataclass
class _ModelState:
    model: str
    model_config: dict[str, Any]
    results_path: Path
    results_file_exists: bool
    calls: list[_PlannedCall]
    rows_by_id: dict[str, dict[str, Any]]
    pending: list[_PlannedCall] = field(default_factory=list)
    stale_count: int = 0
    retry_count: int = 0
    missing_cached_count: int = 0
    invalid_status_count: int = 0

    @property
    def expected_result_ids(self) -> list[str]:
        return [call.result_id for call in self.calls]

    def ordered_rows(self) -> list[dict[str, Any]]:
        return [
            self.rows_by_id[result_id]
            for result_id in self.expected_result_ids
            if result_id in self.rows_by_id
        ]

    def summary(self) -> ModelPlan:
        return ModelPlan(
            model=self.model,
            results_path=self.results_path,
            planned=len(self.calls),
            skipped=len(self.calls) - len(self.pending),
            to_run=len(self.pending),
            retries=self.retry_count,
            stale=self.stale_count,
        )


@dataclass(frozen=True)
class _RunContext:
    config_id: str
    custom_models_path: Path
    items_sha256: str
    model_configs: dict[str, Any]
    provider_groups: list[list[str]]
    states: dict[str, _ModelState]


def execute_run(
    config: RunConfig,
    *,
    evaluator: Evaluator | None = None,
    on_artifact_write: ArtifactWriteHook | None = None,
) -> RunOutput:
    """Run every planned call that has no reusable result, checkpointing as it goes.

    Without an ``evaluator`` the models are called through FDLLM, after registering
    the run's model registry. A supplied evaluator skips FDLLM entirely and the
    model configs in call IDs are empty, which tests rely on.
    """

    model_configs: dict[str, Any] | None = None
    if evaluator is None:
        prepare_fdllm_for_run(config)
        model_configs = snapshot_model_configs(config.models)
        evaluator = evaluate_item_with_fdllm
    context = _prepare_run(config, model_configs or {})

    code_git_hash = git_hash()
    code_git_dirty = git_dirty()
    run_dir = config.run_dir
    config_snapshot_path = run_dir / "config.json"
    resolved_config = config.to_resolved_dict()
    resolved_config["custom_models_path"] = str(context.custom_models_path)
    resolved_config["items_sha256"] = context.items_sha256
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        config_snapshot_path,
        {
            "benchmark": config.benchmark.name,
            "config_id": context.config_id,
            "source_config_path": (
                str(config.config_path) if config.config_path else None
            ),
            "raw_config": config.raw_config,
            "resolved_config": resolved_config,
            "code_git_hash": code_git_hash,
            "code_git_dirty": code_git_dirty,
            "model_configs": context.model_configs,
        },
    )
    if on_artifact_write is not None:
        on_artifact_write(config_snapshot_path)

    model_outputs = _run_provider_groups(
        config=config,
        context=context,
        config_snapshot_path=config_snapshot_path,
        evaluator=evaluator,
        on_artifact_write=on_artifact_write,
    )
    executed_models = {
        output["model_name"] for output in model_outputs if output["executed_count"]
    }
    _update_run_log(
        run_dir,
        executed_models=executed_models,
        retry_errors=config.retry_errors,
        code_git_hash=code_git_hash,
        code_git_dirty=code_git_dirty,
        on_artifact_write=on_artifact_write,
    )
    return {
        "run_id": config.run_id,
        "config_id": context.config_id,
        "run_dir": run_dir,
        "config_snapshot_path": config_snapshot_path,
        "results_dir": run_dir / "responses",
        "results_paths": {
            model_name: state.results_path
            for model_name, state in context.states.items()
        },
        "total_expected_results": sum(
            len(state.calls) for state in context.states.values()
        ),
        "executed_count": sum(output["executed_count"] for output in model_outputs),
        "skipped_count": sum(output["skipped_count"] for output in model_outputs),
        "error_count": sum(output["error_count"] for output in model_outputs),
    }


def plan_run(
    config: RunConfig,
    *,
    model_configs: Mapping[str, Any] | None = None,
) -> RunPlan:
    """Report which calls a run would make and which it would reuse.

    Nothing is called or written. Without ``model_configs`` the model registry is
    registered with FDLLM to read them, exactly as ``execute_run`` does; pass
    ``{}`` to plan a run that uses a supplied evaluator.
    """

    if model_configs is None:
        prepare_fdllm_for_run(config)
        model_configs = snapshot_model_configs(config.models)
    context = _prepare_run(config, model_configs)
    return RunPlan(
        run_id=config.run_id,
        config_id=context.config_id,
        run_dir=config.run_dir,
        models=tuple(state.summary() for state in context.states.values()),
    )


def _prepare_run(config: RunConfig, model_configs: Mapping[str, Any]) -> _RunContext:
    """Validate inputs and plan every model before any call or write."""

    config_id = config_identity_sha256(config)
    identity_fields = config.identity_fields()
    clashes = sorted(set(identity_fields) & _BASE_ROW_KEYS)
    if clashes:
        raise ConfigError(
            "Benchmark identity fields clash with result columns: " + ", ".join(clashes)
        )
    custom_models_path = resolve_custom_models_path(config)
    items, items_sha256 = config.benchmark.load_items(config.items_path)
    validate_items(items, source=config.items_path)
    prepared_items = [_prepare_item(config, item) for item in items]
    provider_groups = provider_groups_for_models(custom_models_path, config.models)
    sanitized_model_configs = sanitize_model_config(dict(model_configs))
    states = {
        model_name: _plan_model(
            config,
            model_name=model_name,
            model_config=sanitized_model_configs.get(model_name, {}),
            items=deepcopy(prepared_items),
        )
        for model_name in config.models
    }
    if config.retry_errors:
        _check_retry_caches(states)
    return _RunContext(
        config_id=config_id,
        custom_models_path=custom_models_path,
        items_sha256=items_sha256,
        model_configs=sanitized_model_configs,
        provider_groups=provider_groups,
        states=states,
    )


def _prepare_item(config: RunConfig, item: dict[str, Any]) -> _PreparedItem:
    prompt = config.benchmark.build_prompt(item, config.params)
    if not isinstance(prompt, str):
        raise ConfigError(
            f"Prompt builder returned {type(prompt).__name__} for item "
            f"{item[ITEM_ID_KEY]}; expected str"
        )
    return _PreparedItem(item=item, prompt=prompt)


def _plan_model(
    config: RunConfig,
    *,
    model_name: str,
    model_config: dict[str, Any],
    items: list[_PreparedItem],
) -> _ModelState:
    results_path = config.run_dir / "responses" / model_results_filename(model_name)
    existing_rows = {
        str(row["result_id"]): row
        for row in load_results(results_path, force=config.force)
    }
    calls = []
    for prepared in items:
        model_call_id = ids.model_call_id(
            benchmark_item_id=prepared.item[ITEM_ID_KEY],
            model=model_name,
            model_config=model_config,
            prompt=prepared.prompt,
            system_prompt=config.benchmark.system_prompt,
            max_tokens=config.effective_max_tokens,
        )
        calls.append(
            _PlannedCall(
                item=prepared.item,
                prompt=prepared.prompt,
                model_call_id=model_call_id,
                result_id=ids.result_id(model_call_id),
            )
        )

    expected_ids = {call.result_id for call in calls}
    state = _ModelState(
        model=model_name,
        model_config=model_config,
        results_path=results_path,
        results_file_exists=results_path.exists(),
        calls=calls,
        rows_by_id={
            call.result_id: existing_rows[call.result_id]
            for call in calls
            if call.result_id in existing_rows
        },
        stale_count=len(set(existing_rows) - expected_ids),
    )
    for call in calls:
        existing = state.rows_by_id.get(call.result_id)
        if existing is None:
            if config.retry_errors:
                state.missing_cached_count += 1
                continue
            state.pending.append(call)
            continue
        if config.retry_errors and existing.get("call_status") not in {
            CALL_STATUS_ERROR,
            CALL_STATUS_SUCCESS,
        }:
            state.invalid_status_count += 1
        if config.retry_errors and existing.get("call_status") == CALL_STATUS_ERROR:
            state.retry_count += 1
            state.pending.append(call)
    return state


def _check_retry_caches(states: Mapping[str, _ModelState]) -> None:
    """Validate every retry cache before any model worker starts."""

    missing_models = [
        name for name, state in states.items() if not state.results_file_exists
    ]
    if missing_models:
        raise FileNotFoundError(
            "No cached results found for "
            + ", ".join(missing_models)
            + "; cannot use --retry-errors."
        )
    missing_calls = {
        name: state.missing_cached_count
        for name, state in states.items()
        if state.missing_cached_count
    }
    if missing_calls:
        details = ", ".join(
            f"{name} ({count})" for name, count in missing_calls.items()
        )
        raise ConfigError(
            "--retry-errors requires cached results for every planned call; "
            f"missing calls by model: {details}. Run normally first to execute "
            "new or changed calls."
        )
    invalid_statuses = {
        name: state.invalid_status_count
        for name, state in states.items()
        if state.invalid_status_count
    }
    if invalid_statuses:
        details = ", ".join(
            f"{name} ({count})" for name, count in invalid_statuses.items()
        )
        raise ConfigError(
            "--retry-errors found cached rows with invalid call_status values "
            f"by model: {details}."
        )


def _run_provider_groups(
    *,
    config: RunConfig,
    context: _RunContext,
    config_snapshot_path: Path,
    evaluator: Evaluator,
    on_artifact_write: ArtifactWriteHook | None,
) -> list[_ModelRunOutput]:
    """Run provider groups concurrently and the models within a group in order."""

    def run_group(model_names: list[str]) -> list[_ModelRunOutput]:
        return [
            _run_model(
                config=config,
                config_id=context.config_id,
                config_snapshot_path=config_snapshot_path,
                state=context.states[model_name],
                evaluator=evaluator,
                on_artifact_write=on_artifact_write,
            )
            for model_name in model_names
        ]

    groups = context.provider_groups
    if config.model_workers == 1 or len(groups) == 1:
        group_outputs = [run_group(group) for group in groups]
    else:
        with ThreadPoolExecutor(
            max_workers=min(config.model_workers, len(groups)),
            thread_name_prefix="provider-group",
        ) as executor:
            futures = [executor.submit(run_group, group) for group in groups]
            group_outputs = [future.result() for future in futures]

    outputs_by_model = {
        output["model_name"]: output for outputs in group_outputs for output in outputs
    }
    return [outputs_by_model[model_name] for model_name in config.models]


def _run_model(
    *,
    config: RunConfig,
    config_id: str,
    config_snapshot_path: Path,
    state: _ModelState,
    evaluator: Evaluator,
    on_artifact_write: ArtifactWriteHook | None,
) -> _ModelRunOutput:
    """Run one model's pending calls in item order, checkpointing each chunk."""

    def checkpoint() -> None:
        save_results(state.ordered_rows(), state.results_path)
        if on_artifact_write is not None:
            on_artifact_write(state.results_path)

    executed_count = 0
    for chunk in _chunked(state.pending, config.chunk_size):
        for call in chunk:
            state.rows_by_id[call.result_id] = _evaluate_call(
                config=config,
                config_id=config_id,
                config_snapshot_path=config_snapshot_path,
                model_name=state.model,
                call=call,
                evaluator=evaluator,
            )
            executed_count += 1
        checkpoint()
    if state.stale_count and not state.pending:
        checkpoint()

    return {
        "model_name": state.model,
        "executed_count": executed_count,
        "skipped_count": len(state.calls) - len(state.pending),
        "error_count": error_count(state.ordered_rows()),
    }


def _evaluate_call(
    *,
    config: RunConfig,
    config_id: str,
    config_snapshot_path: Path,
    model_name: str,
    call: _PlannedCall,
    evaluator: Evaluator,
) -> dict[str, Any]:
    base_row: dict[str, Any] = {
        "result_id": call.result_id,
        "run_id": config.run_id,
        "config_id": config_id,
        "config_path": str(config.config_path) if config.config_path else None,
        "config_snapshot_path": str(config_snapshot_path),
        **config.identity_fields(),
        "model": model_name,
        "model_call_id": call.model_call_id,
        ITEM_ID_KEY: call.item[ITEM_ID_KEY],
        "prompt": call.prompt,
    }
    try:
        payload = evaluator(config, model_name, call.item, call.prompt)
    except PROVIDER_ERRORS as e:
        return _error_row(base_row, error_type=type(e).__name__, message=str(e))

    if not payload["output"]:
        return _error_row(
            base_row,
            error_type="EmptyLLMResponse",
            message=f"LLM response empty for item {call.item[ITEM_ID_KEY]}",
        )
    return {
        **base_row,
        **payload,
        "success": True,
        "call_status": CALL_STATUS_SUCCESS,
        "error_type": "",
        "error_message": "",
    }


def _error_row(
    base_row: dict[str, Any], *, error_type: str, message: str
) -> dict[str, Any]:
    return {
        **base_row,
        "output": "",
        "success": False,
        "call_status": CALL_STATUS_ERROR,
        "error_type": error_type,
        "error_message": message,
    }


def _chunked(calls: list[_PlannedCall], chunk_size: int | None) -> list[list[Any]]:
    if not calls:
        return []
    if chunk_size is None:
        return [calls]
    return [
        calls[start : start + chunk_size] for start in range(0, len(calls), chunk_size)
    ]


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
        update_log(run_dir, model_name, fields, on_artifact_write=on_artifact_write)


def update_log(
    run_dir: Path,
    model_name: str,
    fields: dict[str, str | bool | None],
    *,
    on_artifact_write: ArtifactWriteHook | None = None,
) -> None:
    """Merge-write model provenance fields into log.json, skipping None values."""

    log_path = run_dir / "log.json"
    log_data: dict[str, dict[str, Any]] = {}
    if log_path.exists():
        log_data = json.loads(log_path.read_text(encoding="utf-8"))
    entry = log_data.setdefault(model_name, {})
    entry.update({key: value for key, value in fields.items() if value is not None})
    _write_json(log_path, log_data)
    if on_artifact_write is not None:
        on_artifact_write(log_path)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ids.canonical_json(payload, indent=2) + "\n", encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
