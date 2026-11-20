"""What an experiment records about each model run on it.

Jobs running different models on one experiment never write the same file:

- ``responses/responses_<model>.jsonl`` holds the model's result rows;
- ``models/<model>.json`` holds the model's registered settings, the registry they
  came from, a count of its results, and one entry per run that wrote them, with
  the versions that run used.

Only the job running a model writes its two files.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from fab_benchmark_runner.errors import ConfigError
from fab_benchmark_runner.experiment import Experiment
from fab_benchmark_runner.ids import canonical_json
from fab_benchmark_runner.results import (
    error_count,
    model_record_filename,
    model_results_filename,
    write_text_atomic,
)

MODEL_RECORDS_DIR = "models"
RESPONSES_DIR = "responses"


def model_record_path(experiment_dir: Path, model_name: str) -> Path:
    return experiment_dir / MODEL_RECORDS_DIR / model_record_filename(model_name)


def results_path(experiment_dir: Path, model_name: str) -> Path:
    return experiment_dir / RESPONSES_DIR / model_results_filename(model_name)


def read_model_record(path: Path) -> dict[str, Any] | None:
    """Return a model's record, or None when the model hasn't run yet."""

    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ConfigError(f"Could not read the model record {path}: {e}") from e
    if not isinstance(record, dict) or not isinstance(record.get("runs"), list):
        raise ConfigError(f"{path} is not a model record")
    return record


def read_model_records(experiment_dir: Path) -> list[dict[str, Any]]:
    """Return the records of every model that has run on the experiment."""

    folder = experiment_dir / MODEL_RECORDS_DIR
    if not folder.is_dir():
        return []
    records = []
    for path in sorted(folder.glob("*.json")):
        record = read_model_record(path)
        if record is not None:
            records.append(record)
    return records


def update_model_record(
    path: Path,
    *,
    experiment: Experiment,
    model_name: str,
    model_config: Mapping[str, Any],
    registry: Mapping[str, Any],
    rows: Iterable[Mapping[str, Any]],
    run_entry: Mapping[str, Any],
) -> None:
    """Write a model's record, appending this run to the runs it lists."""

    existing = read_model_record(path)
    if existing is not None and (
        existing.get("experiment_id") != experiment.experiment_id
        or existing.get("inputs_sha256") != experiment.inputs_sha256
    ):
        raise ConfigError(
            f"{path} belongs to experiment {existing.get('experiment_id')!r} with "
            f"other inputs, not to {experiment.experiment_id!r}"
        )
    rows = list(rows)
    write_json(
        path,
        {
            "experiment_id": experiment.experiment_id,
            "inputs_sha256": experiment.inputs_sha256,
            "model": model_name,
            "model_config": dict(model_config),
            "model_registry": dict(registry),
            "results": {"errors": error_count(rows), "rows": len(rows)},
            "runs": [*(existing or {}).get("runs", []), dict(run_entry)],
        },
    )


def write_json(path: Path, payload: Any) -> None:
    """Write canonical, indented JSON, so equal payloads give identical bytes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, canonical_json(payload, indent=2) + "\n")
