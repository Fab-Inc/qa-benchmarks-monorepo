"""Shared run machinery for FAB benchmarks.

A benchmark hands over its rendered inputs; the runner stores them once as an
experiment and runs any number of models on it, now or later, with resumable
results keyed by call IDs. ``fab-bench`` is its command line.
"""

from fab_benchmark_runner.config import (
    RunConfig,
    apply_execution_overrides,
    load_run_config,
    parse_run_config,
)
from fab_benchmark_runner.errors import ConfigError
from fab_benchmark_runner.experiment import (
    Experiment,
    ExperimentCreation,
    MaxTokensChange,
    create_experiment,
    default_experiment_id,
    raise_max_tokens,
    read_experiment,
)
from fab_benchmark_runner.ids import model_call_id, result_id
from fab_benchmark_runner.inputs import ImageRef, ModelInput, read_inputs, write_inputs
from fab_benchmark_runner.jobs import (
    fetch_results,
    plan_job,
    run_job,
    upload_experiment,
)
from fab_benchmark_runner.results import load_results, model_results_filename
from fab_benchmark_runner.runner import (
    CallModel,
    ModelPlan,
    RunPlan,
    RunResult,
    execute_run,
    plan_run,
)

__all__ = [
    "CallModel",
    "ConfigError",
    "Experiment",
    "ExperimentCreation",
    "ImageRef",
    "MaxTokensChange",
    "ModelInput",
    "ModelPlan",
    "RunConfig",
    "RunPlan",
    "RunResult",
    "apply_execution_overrides",
    "create_experiment",
    "default_experiment_id",
    "execute_run",
    "fetch_results",
    "load_results",
    "load_run_config",
    "model_call_id",
    "model_results_filename",
    "parse_run_config",
    "plan_job",
    "plan_run",
    "raise_max_tokens",
    "read_experiment",
    "read_inputs",
    "result_id",
    "run_job",
    "upload_experiment",
    "write_inputs",
]
