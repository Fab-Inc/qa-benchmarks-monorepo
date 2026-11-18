"""Shared run machinery for FAB benchmarks.

A benchmark describes itself with a ``Benchmark`` adapter and gets config loading,
resumable runs keyed by result IDs, model registry handling, GCS sync and a CLI.
"""

from fab_benchmark_runner.benchmark import CORE_CONFIG_KEYS, Benchmark
from fab_benchmark_runner.cli import build_app
from fab_benchmark_runner.cloud import (
    execute_with_optional_sync,
    load_cloud_run_config,
    load_local_run_config,
    plan_with_optional_sync,
)
from fab_benchmark_runner.config import (
    RunConfig,
    apply_execution_overrides,
    config_identity,
    config_identity_sha256,
    load_run_config,
    parse_run_config,
)
from fab_benchmark_runner.errors import ConfigError
from fab_benchmark_runner.ids import model_call_id, result_id
from fab_benchmark_runner.items import file_sha256, load_jsonl_items
from fab_benchmark_runner.registry import prepare_model_registry
from fab_benchmark_runner.request import ImageInput, ModelRequest
from fab_benchmark_runner.results import load_results, model_results_filename
from fab_benchmark_runner.runner import (
    ModelPlan,
    RunOutput,
    RunPlan,
    execute_run,
    plan_run,
)

__all__ = [
    "CORE_CONFIG_KEYS",
    "Benchmark",
    "ConfigError",
    "ImageInput",
    "ModelPlan",
    "ModelRequest",
    "RunConfig",
    "RunOutput",
    "RunPlan",
    "apply_execution_overrides",
    "build_app",
    "config_identity",
    "config_identity_sha256",
    "execute_run",
    "execute_with_optional_sync",
    "file_sha256",
    "load_cloud_run_config",
    "load_jsonl_items",
    "load_local_run_config",
    "load_results",
    "load_run_config",
    "model_call_id",
    "model_results_filename",
    "parse_run_config",
    "plan_run",
    "plan_with_optional_sync",
    "prepare_model_registry",
    "result_id",
]
