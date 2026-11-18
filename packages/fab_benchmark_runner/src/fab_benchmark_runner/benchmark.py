"""The adapter a benchmark supplies to plug into the shared runner."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fab_benchmark_runner.items import load_jsonl_items
from fab_benchmark_runner.request import ModelRequest

ItemLoader = Callable[[Path], tuple[list[dict[str, Any]], str]]
"""Load items from a local file; return them with the SHA-256 of the input bytes."""

PromptBuilder = Callable[[Mapping[str, Any], Any], str]
"""Render the prompt for one item, given the benchmark's parsed params."""

RequestBuilder = Callable[[Mapping[str, Any], Any], ModelRequest]
"""Build the complete user request for one item."""

IdentityBuilder = Callable[[Any], Mapping[str, Any]]
"""Return the params that define a run's semantics (config_id and result rows)."""

# Keys the shared runner reads from a run config. Benchmark params may not reuse them.
CORE_CONFIG_KEYS = frozenset(
    {
        "run_id",
        "output_dir",
        "models",
        "items_path",
        "inputs_dir",
        "env_path",
        "custom_models_path",
        "custom_models_uri",
        "custom_models_sha256",
        "max_tokens",
        "chunk_size",
        "model_workers",
        "force",
        "retry_errors",
        "results_dest",
        "metadata",
    }
)


@dataclass(frozen=True)
class Benchmark:
    """Everything benchmark-specific the shared runner needs.

    ``params_type`` is a dataclass whose fields are additional top-level keys of
    the run config, for example a language or prompt name. Fields without a default
    are required. Validate values in its ``__post_init__`` and raise ``ValueError``.

    ``identity`` selects the params that change what a run means. They go into the
    config ID and into every result row. By default all params are included.
    """

    name: str
    build_prompt: PromptBuilder | None = None
    params_type: type | None = None
    identity: IdentityBuilder | None = None
    load_items: ItemLoader = load_jsonl_items
    system_prompt: str | None = None
    default_env_path: Path | None = None
    default_custom_models_path: Path | None = None
    build_request: RequestBuilder | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Benchmark name must be a non-empty string")
        if (self.build_prompt is None) == (self.build_request is None):
            raise ValueError(
                "Benchmark must define exactly one of build_prompt or build_request"
            )
        if self.params_type is None:
            return
        if not dataclasses.is_dataclass(self.params_type):
            raise TypeError("Benchmark params_type must be a dataclass")
        clashes = sorted(set(self.param_keys) & CORE_CONFIG_KEYS)
        if clashes:
            raise ValueError(
                "Benchmark params reuse core run config keys: " + ", ".join(clashes)
            )

    @property
    def param_keys(self) -> tuple[str, ...]:
        """Return the run config keys owned by this benchmark."""

        if self.params_type is None:
            return ()
        return tuple(field.name for field in dataclasses.fields(self.params_type))

    def identity_fields(self, params: Any) -> dict[str, Any]:
        """Return the params that define the run's semantics."""

        if self.identity is not None:
            return dict(self.identity(params))
        if params is None:
            return {}
        return dataclasses.asdict(params)

    def params_dict(self, params: Any) -> dict[str, Any]:
        """Return every param for the resolved config snapshot."""

        if params is None:
            return {}
        return dataclasses.asdict(params)
