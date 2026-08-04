"""Load and interactively inspect any phonics benchmark run bundle.

Run from the repository root:

uv run --frozen --package phonics-benchmark python \
    benchmarks/phonics_benchmark/scripts/inspect_run.py path/to/run-directory

The ``# %%`` markers can also be opened as notebook-style cells in editors that
support Python interactive windows. Set ``PHONICS_RUN_DIR`` to an absolute path
before starting a converted notebook.
"""

# %%
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def configured_run_dir() -> Path:
    """Select a run from PHONICS_RUN_DIR, a script argument, or the default output."""
    configured = os.environ.get("PHONICS_RUN_DIR")
    if configured:
        return Path(configured).expanduser()

    script_path = globals().get("__file__")
    if script_path is None:
        raise RuntimeError(
            "Set PHONICS_RUN_DIR to an absolute run directory before starting "
            "the notebook."
        )
    if len(sys.argv) > 2:
        raise SystemExit("Usage: inspect_run.py [RUN_DIR]")
    if len(sys.argv) == 2:
        return Path(sys.argv[1]).expanduser()
    return Path(script_path).resolve().parents[1] / "outputs" / "tusome-llm-trial"


# Change this value in an interactive session, set PHONICS_RUN_DIR, or pass a path.
RUN_DIR = configured_run_dir()


# %%
def load_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    """Load a JSON object, optionally returning an empty object when absent."""
    if not path.exists() and not required:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load non-empty JSONL records from one result file."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_run(run_dir: str | Path) -> dict[str, Any]:
    """Load config, log, result rows, summaries, and metadata for one run."""
    resolved_run_dir = Path(run_dir).expanduser().resolve()
    config_snapshot = load_json(resolved_run_dir / "config.json")
    run_log = load_json(resolved_run_dir / "log.json", required=False)

    raw_config = config_snapshot.get("raw_config", {})
    resolved_config = config_snapshot.get("resolved_config", {})
    model_configs = config_snapshot.get("model_configs", {})

    result_files = sorted((resolved_run_dir / "answers").glob("resps_*.jsonl"))
    result_frames = []
    for result_file in result_files:
        frame = pd.DataFrame(load_jsonl(result_file))
        if not frame.empty:
            frame["results_file"] = result_file.name
            result_frames.append(frame)

    results = (
        pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    )
    summary = (
        results.groupby(["model", "call_status"], dropna=False)
        .size()
        .unstack(fill_value=0)
        if not results.empty
        else pd.DataFrame()
    )
    errors = (
        results.loc[results["call_status"].eq("error")].copy()
        if "call_status" in results
        else pd.DataFrame()
    )
    run_metadata = {
        "run_dir": str(resolved_run_dir),
        "config_id": config_snapshot.get("config_id"),
        "code_git_hash": config_snapshot.get("code_git_hash"),
        "source_config_path": config_snapshot.get("source_config_path"),
        "result_files": [str(path) for path in result_files],
        "run_log": run_log,
    }
    return {
        "config_snapshot": config_snapshot,
        "raw_config": raw_config,
        "resolved_config": resolved_config,
        "model_configs": model_configs,
        "run_log": run_log,
        "run_metadata": run_metadata,
        "results": results,
        "summary": summary,
        "errors": errors,
    }


# %%
run_bundle = load_run(RUN_DIR)
config_snapshot = run_bundle["config_snapshot"]
raw_config = run_bundle["raw_config"]
resolved_config = run_bundle["resolved_config"]
model_configs = run_bundle["model_configs"]
run_log = run_bundle["run_log"]
run_metadata = run_bundle["run_metadata"]
results = run_bundle["results"]
summary = run_bundle["summary"]
errors = run_bundle["errors"]

print(
    f"Loaded {len(results)} result rows from "
    f"{len(run_metadata['result_files'])} files in {run_metadata['run_dir']}"
)
print(summary.to_string() if not summary.empty else "No result rows found.")


# %%
run_metadata


# %%
resolved_config


# %%
model_configs


# %%
results.head()


# %%
errors[
    [
        column
        for column in (
            "model",
            "benchmark_item_id",
            "error_type",
            "error_message",
        )
        if column in errors
    ]
]
