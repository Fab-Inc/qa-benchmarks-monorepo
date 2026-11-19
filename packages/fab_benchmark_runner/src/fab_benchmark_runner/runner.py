"""Plan and execute benchmark runs with resumable, per-model result files."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

from anthropic import APIError as AnthropicAPIError
from botocore.exceptions import ClientError as BedrockAPIError
from botocore.exceptions import ConnectionError as BedrockConnectionError
from botocore.exceptions import HTTPClientError as BedrockHTTPClientError
from google.genai.errors import APIError as GoogleAPIError
from mistralai.exceptions import MistralAPIException, MistralConnectionException
from openai import APIError as OpenAIAPIError

try:
    from fdllm.errors import LLMResponseError
except ImportError:  # FDLLM's main branch has no typed response errors yet
    LLMResponseError = None

from fab_benchmark_runner import ids, records
from fab_benchmark_runner.config import RunConfig, config_identity_sha256
from fab_benchmark_runner.errors import ConfigError
from fab_benchmark_runner.items import ITEM_ID_KEY, file_sha256, validate_items
from fab_benchmark_runner.llm import evaluate_item_with_fdllm
from fab_benchmark_runner.provenance import git_dirty, git_hash
from fab_benchmark_runner.registry import (
    check_provider_groups,
    prepare_fdllm_for_run,
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

# FDLLM propagates provider SDK failures unchanged, and from its dev branch on it
# raises LLMResponseError for responses it can't use, such as an empty one. Only
# these known call failures become error rows, so setup and result-processing bugs
# still fail fast. The runner works with FDLLM versions with and without them.
PROVIDER_ERRORS: tuple[type[BaseException], ...] = (
    AnthropicAPIError,
    BedrockAPIError,
    BedrockConnectionError,
    BedrockHTTPClientError,
    GoogleAPIError,
    MistralAPIException,
    MistralConnectionException,
    OpenAIAPIError,
) + ((LLMResponseError,) if LLMResponseError is not None else ())

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
        "max_tokens",
        "output",
        "success",
        "call_status",
        "error_type",
        "error_message",
        "stop_reason",
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
    results_written: bool


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
    registry_sha256: str
    run_record: dict[str, Any]
    run_record_path: Path
    write_run_record: bool
    model_configs: dict[str, Any]
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

    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    if context.write_run_record:
        records.write_json(context.run_record_path, context.run_record)
        if on_artifact_write is not None:
            on_artifact_write(context.run_record_path)

    code_git_hash = git_hash()
    code_git_dirty = git_dirty()
    registry = {
        "path": str(context.custom_models_path),
        "uri": config.custom_models_uri,
        "sha256": context.registry_sha256,
    }

    def record_model(state: _ModelState, output: _ModelRunOutput) -> None:
        """Record a model's run, in its own worker, once its results are written."""

        if not output["results_written"]:
            return
        path = records.model_record_path(run_dir, state.model)
        records.update_model_record(
            path,
            model_name=state.model,
            config_id=context.config_id,
            model_config=state.model_config,
            registry=registry,
            rows=state.ordered_rows(),
            run_entry={
                "finished_utc": records.utc_now(),
                "run_id": config.run_id,
                "executed_calls": output["executed_count"],
                "dropped_stale_rows": state.stale_count,
                "force": config.force,
                "retry_errors": config.retry_errors,
                "code_git_hash": code_git_hash,
                "code_git_dirty": code_git_dirty,
                "run_config_path": (
                    str(config.config_path) if config.config_path else None
                ),
                "run_config": config.raw_config,
            },
        )
        if on_artifact_write is not None:
            on_artifact_write(path)

    model_outputs = _run_models_in_parallel(
        config=config,
        context=context,
        evaluator=evaluator,
        on_artifact_write=on_artifact_write,
        after_model=record_model,
    )
    return {
        "run_id": config.run_id,
        "config_id": context.config_id,
        "run_dir": run_dir,
        "config_snapshot_path": context.run_record_path,
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

    Nothing is called or written, but a run folder that belongs to a different run
    is still reported as an error. Without ``model_configs`` the model registry is
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
    """Validate inputs and the run folder, and plan every model, before any write."""

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
    check_provider_groups(custom_models_path, config.models)
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
    run_record = records.run_record(config, items_sha256=items_sha256)
    run_record_path = config.run_dir / records.RUN_RECORD_NAME
    write_run_record = records.check_run_record(run_record_path, run_record)
    return _RunContext(
        config_id=config_id,
        custom_models_path=custom_models_path,
        registry_sha256=file_sha256(custom_models_path),
        run_record=run_record,
        run_record_path=run_record_path,
        write_run_record=write_run_record,
        model_configs=sanitized_model_configs,
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


def _run_models_in_parallel(
    *,
    config: RunConfig,
    context: _RunContext,
    evaluator: Evaluator,
    on_artifact_write: ArtifactWriteHook | None,
    after_model: Callable[[_ModelState, _ModelRunOutput], None],
) -> list[_ModelRunOutput]:
    """Run up to ``model_workers`` models at once, each model's items in order.

    Models start in config order. Once a model fails with an exception, no further
    model starts; the models already running finish before the error propagates.
    """

    def run(model_name: str) -> _ModelRunOutput:
        state = context.states[model_name]
        output = _run_model(
            config=config,
            config_id=context.config_id,
            run_record_path=context.run_record_path,
            state=state,
            evaluator=evaluator,
            on_artifact_write=on_artifact_write,
        )
        after_model(state, output)
        return output

    waiting = list(config.models)
    workers = min(config.model_workers, len(waiting))
    outputs: dict[str, _ModelRunOutput] = {}
    if workers <= 1:
        for model_name in waiting:
            outputs[model_name] = run(model_name)
    else:
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="model"
        ) as pool:
            running: dict[Future[_ModelRunOutput], str] = {}

            def start_next() -> None:
                model_name = waiting.pop(0)
                running[pool.submit(run, model_name)] = model_name

            while waiting and len(running) < workers:
                start_next()
            while running:
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    outputs[running.pop(future)] = future.result()
                    if waiting:
                        start_next()
    return [outputs[model_name] for model_name in config.models]


def _run_model(
    *,
    config: RunConfig,
    config_id: str,
    run_record_path: Path,
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
    results_written = False
    for chunk in _chunked(state.pending, config.chunk_size):
        for call in chunk:
            state.rows_by_id[call.result_id] = _evaluate_call(
                config=config,
                config_id=config_id,
                run_record_path=run_record_path,
                model_name=state.model,
                call=call,
                evaluator=evaluator,
            )
            executed_count += 1
        checkpoint()
        results_written = True
    if state.stale_count and not state.pending:
        checkpoint()
        results_written = True

    return {
        "model_name": state.model,
        "executed_count": executed_count,
        "skipped_count": len(state.calls) - len(state.pending),
        "error_count": error_count(state.ordered_rows()),
        "results_written": results_written,
    }


def _evaluate_call(
    *,
    config: RunConfig,
    config_id: str,
    run_record_path: Path,
    model_name: str,
    call: _PlannedCall,
    evaluator: Evaluator,
) -> dict[str, Any]:
    base_row: dict[str, Any] = {
        "result_id": call.result_id,
        "run_id": config.run_id,
        "config_id": config_id,
        "config_path": str(config.config_path) if config.config_path else None,
        "config_snapshot_path": str(run_record_path),
        **config.identity_fields(),
        "model": model_name,
        "model_call_id": call.model_call_id,
        ITEM_ID_KEY: call.item[ITEM_ID_KEY],
        "prompt": call.prompt,
        "max_tokens": config.effective_max_tokens,
    }
    try:
        payload = evaluator(config, model_name, call.item, call.prompt)
    except PROVIDER_ERRORS as e:
        return _error_row(
            base_row,
            error_type=type(e).__name__,
            message=str(e),
            stop_reason=getattr(e, "stop_reason", None),
        )

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
    base_row: dict[str, Any],
    *,
    error_type: str,
    message: str,
    stop_reason: str | None = None,
) -> dict[str, Any]:
    """Build an error row; ``stop_reason`` is the provider's, when FDLLM reports it.

    A stop reason of ``length``, ``max_tokens`` or ``MAX_TOKENS`` means the response
    ran out of tokens.
    """

    return {
        **base_row,
        "output": "",
        "success": False,
        "call_status": CALL_STATUS_ERROR,
        "error_type": error_type,
        "error_message": message,
        "stop_reason": stop_reason,
    }


def _chunked(calls: list[_PlannedCall], chunk_size: int | None) -> list[list[Any]]:
    if not calls:
        return []
    if chunk_size is None:
        return [calls]
    return [
        calls[start : start + chunk_size] for start in range(0, len(calls), chunk_size)
    ]
