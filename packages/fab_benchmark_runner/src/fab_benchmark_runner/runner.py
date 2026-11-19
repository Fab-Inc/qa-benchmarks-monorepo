"""Plan and run models on an experiment, with resumable, per-model result files.

Everything here works on a local experiment folder. Jobs on an experiment on GCS
run on a local mirror of it; ``jobs`` handles the mirroring. Images are fetched and
checked by ``images.ImageFiles`` before any call.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
from fab_benchmark_runner.config import RunConfig
from fab_benchmark_runner.environment import run_environment, version_differences
from fab_benchmark_runner.errors import ConfigError
from fab_benchmark_runner.experiment import Experiment, read_experiment
from fab_benchmark_runner.images import ImageFiles
from fab_benchmark_runner.inputs import ITEM_ID_KEY, ImageRef, ModelInput
from fab_benchmark_runner.registry import sanitize_model_config
from fab_benchmark_runner.results import (
    CALL_STATUS_ERROR,
    CALL_STATUS_SUCCESS,
    error_count,
    load_results,
    save_results,
    utc_now,
)

# A call function gets the model name, the item's stored input with each image at a
# checked local copy, and the experiment's token limit (None for no limit). It
# returns the row fields of the response, such as "output".
CallModel = Callable[[str, ModelInput, "int | None"], Mapping[str, Any]]
ArtifactWriteHook = Callable[[Path], None]

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


@dataclass(frozen=True)
class ModelPlan:
    """What a run would do for one model."""

    model: str
    results_path: Path
    planned: int
    reused: int
    to_run: int
    retries: int
    stale: int


@dataclass(frozen=True)
class RunPlan:
    """What a run would do, without calling any model or writing anything.

    ``images_to_send`` counts the distinct images of the calls to make.
    ``version_warnings`` describe how the versions of earlier runs on the
    experiment differ from this one's.
    """

    experiment_id: str
    experiment_dir: Path
    max_tokens: int | None
    models: tuple[ModelPlan, ...]
    version_warnings: tuple[str, ...] = ()
    images_to_send: int = 0

    @property
    def planned(self) -> int:
        return sum(model.planned for model in self.models)

    @property
    def reused(self) -> int:
        return sum(model.reused for model in self.models)

    @property
    def to_run(self) -> int:
        return sum(model.to_run for model in self.models)


@dataclass(frozen=True)
class RunResult:
    """What a run did."""

    experiment_id: str
    experiment_dir: Path
    results_paths: dict[str, Path]
    planned: int
    executed: int
    reused: int
    errors: int
    version_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _PlannedCall:
    model_input: ModelInput
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

    def ordered_rows(self) -> list[dict[str, Any]]:
        return [
            self.rows_by_id[call.result_id]
            for call in self.calls
            if call.result_id in self.rows_by_id
        ]

    def summary(self) -> ModelPlan:
        return ModelPlan(
            model=self.model,
            results_path=self.results_path,
            planned=len(self.calls),
            reused=len(self.calls) - len(self.pending),
            to_run=len(self.pending),
            retries=self.retry_count,
            stale=self.stale_count,
        )


def plan_run(
    config: RunConfig,
    experiment_dir: Path,
    *,
    model_configs: Mapping[str, Any],
    image_files: ImageFiles | None = None,
) -> RunPlan:
    """Report which calls a run would make and which results it would reuse.

    ``model_configs`` are FDLLM's configs of the run's models, which enter every
    call ID; pass ``{}`` for models called some other way. With ``image_files``,
    the images of the calls to make are fetched and checked, as a run would.
    No model is called and the experiment is not written.
    """

    experiment = read_experiment(experiment_dir)
    states = _plan_models(config, experiment, experiment_dir, model_configs)
    images = _images_to_send(states)
    if image_files is not None:
        image_files.fetch(images)
    return RunPlan(
        experiment_id=experiment.experiment_id,
        experiment_dir=experiment_dir,
        max_tokens=experiment.max_tokens,
        models=tuple(state.summary() for state in states.values()),
        version_warnings=tuple(_version_warnings(experiment_dir)),
        images_to_send=len({image.sha256 for image in images}),
    )


def execute_run(
    config: RunConfig,
    experiment_dir: Path,
    *,
    model_configs: Mapping[str, Any],
    call_model: CallModel,
    registry: Mapping[str, Any],
    image_files: ImageFiles | None = None,
    on_artifact_write: ArtifactWriteHook | None = None,
) -> RunResult:
    """Run every planned call without a reusable result, checkpointing as it goes.

    Each model gets the experiment's stored inputs and its current ``max_tokens``,
    which every row records. Before any call, ``image_files`` fetches and checks
    the images of the calls to make; it is needed only when there are some. A
    model's results and record are written only by the job running it;
    ``on_artifact_write`` sees each file as it is written.
    """

    experiment = read_experiment(experiment_dir)
    states = _plan_models(config, experiment, experiment_dir, model_configs)
    images = _images_to_send(states)
    if images:
        if image_files is None:
            raise ConfigError(
                "This run sends images; pass image_files to fetch and check them"
            )
        image_files.fetch(images)
    warnings = tuple(_version_warnings(experiment_dir))
    environment = run_environment()

    def record_model(state: _ModelState, executed: int, written: bool) -> None:
        if not written:
            return
        path = records.model_record_path(experiment_dir, state.model)
        records.update_model_record(
            path,
            experiment=experiment,
            model_name=state.model,
            model_config=state.model_config,
            registry=registry,
            rows=state.ordered_rows(),
            run_entry={
                "dropped_stale_rows": state.stale_count,
                "environment": environment,
                "executed_calls": executed,
                "finished_utc": utc_now(),
                "force": config.force,
                "max_tokens": experiment.max_tokens,
                "retry_errors": config.retry_errors,
                "run_config": dict(config.raw),
                "run_config_source": config.source,
            },
        )
        if on_artifact_write is not None:
            on_artifact_write(path)

    executed = _run_models_in_parallel(
        config=config,
        experiment=experiment,
        states=states,
        call_model=call_model,
        image_files=image_files,
        on_artifact_write=on_artifact_write,
        after_model=record_model,
    )
    rows = [row for state in states.values() for row in state.ordered_rows()]
    return RunResult(
        experiment_id=experiment.experiment_id,
        experiment_dir=experiment_dir,
        results_paths={name: state.results_path for name, state in states.items()},
        planned=sum(len(state.calls) for state in states.values()),
        executed=sum(executed.values()),
        reused=sum(len(state.calls) - len(state.pending) for state in states.values()),
        errors=error_count(rows),
        version_warnings=warnings,
    )


def _row_input_fields(
    experiment: Experiment, model_name: str, call: _PlannedCall
) -> dict[str, Any]:
    """Return the columns every row starts with: IDs, and the input repeated."""

    model_input = call.model_input
    return {
        "benchmark": experiment.benchmark,
        ITEM_ID_KEY: model_input.benchmark_item_id,
        "experiment_id": experiment.experiment_id,
        "images": model_input.to_row().get("images", []),
        "max_tokens": experiment.max_tokens,
        "model": model_name,
        "model_call_id": call.model_call_id,
        "prompt": model_input.prompt,
        "result_id": call.result_id,
        "system_prompt": model_input.system_prompt,
    }


def _images_to_send(states: Mapping[str, _ModelState]) -> list[ImageRef]:
    """Return the images of every call to make, in model then item order."""

    return [
        image
        for state in states.values()
        for call in state.pending
        for image in call.model_input.images
    ]


def _version_warnings(experiment_dir: Path) -> list[str]:
    earlier = [
        run["environment"]
        for record in records.read_model_records(experiment_dir)
        for run in record["runs"]
        if isinstance(run, Mapping) and isinstance(run.get("environment"), Mapping)
    ]
    return version_differences(run_environment(), earlier)


def _plan_models(
    config: RunConfig,
    experiment: Experiment,
    experiment_dir: Path,
    model_configs: Mapping[str, Any],
) -> dict[str, _ModelState]:
    sanitized = sanitize_model_config(dict(model_configs))
    states = {
        model_name: _plan_model(
            config,
            experiment,
            experiment_dir,
            model_name=model_name,
            model_config=sanitized.get(model_name, {}),
        )
        for model_name in config.models
    }
    if config.retry_errors:
        _check_retry_caches(states)
    return states


def _plan_model(
    config: RunConfig,
    experiment: Experiment,
    experiment_dir: Path,
    *,
    model_name: str,
    model_config: dict[str, Any],
) -> _ModelState:
    results_path = records.results_path(experiment_dir, model_name)
    existing_rows = {
        str(row["result_id"]): row
        for row in load_results(results_path, force=config.force)
    }
    calls = []
    for model_input in experiment.inputs:
        model_call_id = ids.model_call_id(
            benchmark_item_id=model_input.benchmark_item_id,
            model=model_name,
            model_config=model_config,
            prompt=model_input.prompt,
            system_prompt=model_input.system_prompt,
            images=[
                {"detail": image.detail, "sha256": image.sha256}
                for image in model_input.images
            ],
        )
        calls.append(
            _PlannedCall(
                model_input=model_input,
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
    """With retry_errors, refuse before any call unless every call has a row."""

    missing_models = [
        name for name, state in states.items() if not state.results_file_exists
    ]
    if missing_models:
        raise ConfigError(
            "No results to retry for "
            + ", ".join(missing_models)
            + "; run without retry_errors first."
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
            "retry_errors needs a result for every planned call; missing calls by "
            f"model: {details}. Run without retry_errors first."
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
            "retry_errors found rows with invalid call_status values by model: "
            f"{details}."
        )


def _run_models_in_parallel(
    *,
    config: RunConfig,
    experiment: Experiment,
    states: Mapping[str, _ModelState],
    call_model: CallModel,
    image_files: ImageFiles | None,
    on_artifact_write: ArtifactWriteHook | None,
    after_model: Callable[[_ModelState, int, bool], None],
) -> dict[str, int]:
    """Run up to ``model_workers`` models at once, each model's items in order.

    Models start in config order. Once a model fails with an exception, no further
    model starts; the models already running finish before the error propagates.
    Returns the number of calls made per model.
    """

    def run(model_name: str) -> int:
        state = states[model_name]
        executed, written = _run_model(
            config=config,
            experiment=experiment,
            state=state,
            call_model=call_model,
            image_files=image_files,
            on_artifact_write=on_artifact_write,
        )
        after_model(state, executed, written)
        return executed

    waiting = list(config.models)
    workers = min(config.model_workers, len(waiting))
    executed: dict[str, int] = {}
    if workers <= 1:
        for model_name in waiting:
            executed[model_name] = run(model_name)
        return executed
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="model") as pool:
        running: dict[Future[int], str] = {}

        def start_next() -> None:
            model_name = waiting.pop(0)
            running[pool.submit(run, model_name)] = model_name

        while waiting and len(running) < workers:
            start_next()
        while running:
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                executed[running.pop(future)] = future.result()
                if waiting:
                    start_next()
    return executed


def _run_model(
    *,
    config: RunConfig,
    experiment: Experiment,
    state: _ModelState,
    call_model: CallModel,
    image_files: ImageFiles | None,
    on_artifact_write: ArtifactWriteHook | None,
) -> tuple[int, bool]:
    """Run one model's pending calls in item order, checkpointing each chunk.

    Returns how many calls were made and whether the results file was written.
    """

    def checkpoint() -> None:
        save_results(state.ordered_rows(), state.results_path)
        if on_artifact_write is not None:
            on_artifact_write(state.results_path)

    executed = 0
    written = False
    for chunk in _chunked(state.pending, config.chunk_size):
        for call in chunk:
            state.rows_by_id[call.result_id] = _evaluate_call(
                experiment=experiment,
                model_name=state.model,
                call=call,
                call_model=call_model,
                image_files=image_files,
            )
            executed += 1
        checkpoint()
        written = True
    if state.stale_count and not state.pending:
        checkpoint()
        written = True
    return executed, written


def _evaluate_call(
    *,
    experiment: Experiment,
    model_name: str,
    call: _PlannedCall,
    call_model: CallModel,
    image_files: ImageFiles | None,
) -> dict[str, Any]:
    input_fields = _row_input_fields(experiment, model_name, call)
    sent_input = call.model_input
    if image_files is not None:
        sent_input = image_files.local_input(sent_input)
    try:
        payload = call_model(model_name, sent_input, experiment.max_tokens)
    except PROVIDER_ERRORS as e:
        return _error_row(
            input_fields,
            error_type=type(e).__name__,
            message=str(e),
            stop_reason=getattr(e, "stop_reason", None),
        )
    if not payload.get("output"):
        return _error_row(
            input_fields,
            error_type="EmptyLLMResponse",
            message=f"LLM response empty for item {call.model_input.benchmark_item_id}",
        )
    return {
        **payload,
        **input_fields,
        "call_status": CALL_STATUS_SUCCESS,
        "error_message": "",
        "error_type": "",
        "stop_reason": None,
    }


def _error_row(
    input_fields: Mapping[str, Any],
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
        **input_fields,
        "call_status": CALL_STATUS_ERROR,
        "error_message": message,
        "error_type": error_type,
        "output": "",
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
