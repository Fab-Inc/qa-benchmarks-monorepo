"""What a run folder records besides its results.

A run folder can be written by several jobs, so no file is shared between jobs that
run different models:

- ``config.json``, the run record, holds what every job writing to the folder must
  agree on: the benchmark, the config ID with the payload behind it, and the hash of
  the items. The first job writes it; later jobs check it and never rewrite it, so
  jobs that agree write identical bytes and never conflict.
- ``models/<model>.json``, one record per model, is written only by the job running
  that model. It holds the model's registered settings, the registry they came from,
  and one entry per run that wrote the model's results.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fab_benchmark_runner.config import (
    RunConfig,
    config_identity,
    config_identity_sha256,
)
from fab_benchmark_runner.errors import ConfigError
from fab_benchmark_runner.ids import canonical_json
from fab_benchmark_runner.results import (
    error_count,
    model_record_filename,
    write_text_atomic,
)

RUN_RECORD_NAME = "config.json"
MODEL_RECORDS_DIR = "models"
RUN_RECORD_FORMAT = 1


def run_record(config: RunConfig, *, items_sha256: str) -> dict[str, Any]:
    """Return the run record for a config. Equal inputs give identical bytes."""

    return {
        "benchmark": config.benchmark.name,
        "config_id": config_identity_sha256(config),
        "identity": config_identity(config),
        "items_sha256": items_sha256,
        "record_format": RUN_RECORD_FORMAT,
    }


def check_run_record(path: Path, expected: Mapping[str, Any]) -> bool:
    """Check an existing run record against a run; return whether to write one.

    Returns True when no record exists yet, or when the folder still holds a record
    in the phonics runner's format that matches this run and should be replaced.
    Raises ConfigError when the folder belongs to a different run.
    """

    if not path.exists():
        return True
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ConfigError(f"Could not read the run record {path}: {e}") from e
    if not isinstance(existing, dict) or "config_id" not in existing:
        raise ConfigError(
            f"{path} is not a run record, so this folder can't hold this run's "
            "results. Keep run configs outside their run folder."
        )

    legacy = "items_sha256" not in existing and isinstance(
        existing.get("resolved_config"), dict
    )
    existing_items_sha256 = (
        existing["resolved_config"].get("items_sha256")
        if legacy
        else existing.get("items_sha256")
    )
    if "benchmark" in existing and existing["benchmark"] != expected["benchmark"]:
        raise ConfigError(
            f"This run folder belongs to the {existing['benchmark']!r} benchmark, "
            f"not {expected['benchmark']!r}"
        )
    if existing["config_id"] != expected["config_id"]:
        raise ConfigError(
            "Existing run config_id does not match this run config: "
            f"{existing['config_id']!r} != {expected['config_id']!r}"
        )
    if existing_items_sha256 != expected["items_sha256"]:
        raise ConfigError(
            "Existing run items_sha256 does not match the local items artifact: "
            f"{existing_items_sha256!r} != {expected['items_sha256']!r}"
        )
    return legacy


def model_record_path(run_dir: Path, model_name: str) -> Path:
    return run_dir / MODEL_RECORDS_DIR / model_record_filename(model_name)


def update_model_record(
    path: Path,
    *,
    model_name: str,
    config_id: str,
    model_config: Mapping[str, Any],
    registry: Mapping[str, Any],
    rows: Iterable[Mapping[str, Any]],
    run_entry: Mapping[str, Any],
) -> None:
    """Write a model's record, appending this run to the runs it lists."""

    runs: list[Any] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ConfigError(f"Could not read the model record {path}: {e}") from e
        if isinstance(existing, dict) and isinstance(existing.get("runs"), list):
            runs = existing["runs"]
    rows = list(rows)
    write_json(
        path,
        {
            "model": model_name,
            "config_id": config_id,
            "model_config": dict(model_config),
            "model_registry": dict(registry),
            "results": {"rows": len(rows), "errors": error_count(rows)},
            "runs": [*runs, dict(run_entry)],
        },
    )


def write_json(path: Path, payload: Any) -> None:
    """Write canonical, indented JSON, so equal payloads give identical bytes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, canonical_json(payload, indent=2) + "\n")


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
