"""Package the 22 September leaderboard responses as two normal runner runs.

Each run folder holds what the runner writes: a config.json snapshot and one
responses/responses_<model>.jsonl file per model. The snapshot is composed with
the runner's own loader and helpers, as run-cloud writes it before its first
call, and no model is called. Every kept response must have the call ID that
the runner computes for this config, so a later run of the same config skips
it. The merge records (items, provenance, coverage, exclusions) stay beside the
responses, and source_config.json keeps the selection's config.

The sealed selection and snapshot are read-only inputs, and no source run is
contacted. The run's items must already be at gs://.../datasets/<run-id>.jsonl,
as launch-run.sh publishes items before a run starts.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from merge_local_benchmark_runs import (
    digest,
    seal_artifacts,
    verify_artifacts,
    write_analysis_run,
    write_json,
)
from phonics_benchmark import runner
from phonics_benchmark.cmd import _materialize_run_config
from phonics_benchmark.config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MODEL_WORKERS,
    BenchmarkRunConfig,
    config_identity_sha256,
)
from phonics_benchmark.items import load_benchmark_items
from phonics_benchmark.results import save_results

DEFAULT_SELECTION = Path(
    "outputs/phonics-reports-20260922-coverage90/selections/fixed-991-en-1000-sw"
)
DEFAULT_SOURCE_FULL = Path("outputs/phonics-sources-20260922-coverage90/full")
DEFAULT_OUTPUT = Path("outputs/phonics-unified-leaderboard-20260922")
DEFAULT_GCS_ROOT = "gs://phonics-benchmark/phonics"
DEFAULT_RUN_PREFIX = f"{DEFAULT_GCS_ROOT}/runs"
# Resolved runner paths are recorded in config.json, so keep them neutral.
DEFAULT_WORKDIR = Path("/tmp/phonics-gold-leaderboard-workspace")
RUN_IDS = {
    "english": "en-gold-leaderboard-20260922",
    "kiswahili": "sw-gold-leaderboard-20260922",
}
LANGUAGES = {"english": "en", "kiswahili": "sw"}
PROMPT_NAMES = {
    "en": "decodability_evaluation_schwa_rule",
    "sw": "decodability_evaluation_clusters",
}
MAX_TOKENS = 30_000
# These ran with a smaller max_tokens, so no 30,000-token config reproduces
# their call IDs. Both are below the coverage threshold in both languages.
DROPPED_MODELS = {
    "qwen-3-14b": "ran with max_tokens 16384",
    "qwen3-8b-alibaba": "ran with max_tokens 8192",
}
# Merge records kept beside the runner files, recomputed for the kept models.
RECORD_FILES = (
    "benchmark_items.jsonl",
    "provenance.jsonl",
    "item_provenance.jsonl",
    "coverage_by_model.csv",
    "coverage_by_model_and_ratio_band.csv",
    "excluded_responses.jsonl",
)
# Other selection metadata describes upstream steps. It can hold local paths or,
# as with the 22 September gold update, a record that applies to one language.
CARRIED_METADATA = (
    "source_manifest_sha256",
    "snapshot_started_at",
    "selection_name",
    "selection_spec_sha256",
)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def check_run(run: Path, language: str) -> dict:
    """Refuse missing, duplicate, or off-dataset calls before packaging."""
    items = list(read_jsonl(run / "benchmark_items.jsonl"))
    item_ids = {item["benchmark_item_id"] for item in items}
    if not items or len(item_ids) != len(items):
        raise ValueError(f"Missing or duplicate benchmark items: {run}")
    if any(item.get("language_iso_code") != language for item in items):
        raise ValueError(f"Wrong item language: {run}")
    config = json.loads((run / "config.json").read_text())["resolved_config"]
    if (
        config["language"] != language
        or digest(run / "benchmark_items.jsonl") != config["items_sha256"]
    ):
        raise ValueError(f"Config disagrees with selected items: {run}")
    models = set(config["models"])
    calls, pairs, origins, counts = set(), set(), set(), {}
    for row in read_jsonl(run / "responses/merged.jsonl"):
        pair = (row["model"], row["benchmark_item_id"])
        call_id = row["model_call_id"]
        if (
            pair in pairs
            or call_id in calls
            or pair[1] not in item_ids
            or pair[0] not in models
            or row["language"] != language
        ):
            raise ValueError(f"Invalid merged response: {pair} in {run}")
        pairs.add(pair)
        calls.add(call_id)
        origins.add(row["run_id"])
        counts[row["model"]] = counts.get(row["model"], 0) + 1
    provenance_ids = {
        row["model_call_id"] for row in read_jsonl(run / "provenance.jsonl")
    }
    if calls != provenance_ids:
        raise ValueError(f"Response provenance differs: {run}")
    item_origins = [
        row["benchmark_item_id"] for row in read_jsonl(run / "item_provenance.jsonl")
    ]
    if len(item_origins) != len(items) or set(item_origins) != item_ids:
        raise ValueError(f"Item provenance differs from items: {run}")
    with (run / "coverage_by_model.csv").open(newline="") as stream:
        coverage = {row["model"]: row for row in csv.DictReader(stream)}
    if set(coverage) != models or any(
        int(row["expected_items"]) != len(items)
        or int(row["response_rows"]) != counts.get(model, 0)
        for model, row in coverage.items()
    ):
        raise ValueError(f"Coverage differs from merged responses: {run}")
    return {
        "items": len(items),
        "response_rows": len(calls),
        "models_with_responses": sum(value > 0 for value in counts.values()),
        "source_run_ids": sorted(origins),
    }


def local_paths(value) -> list[str]:
    """Return absolute filesystem paths anywhere in a JSON value."""
    if isinstance(value, dict):
        return [path for item in value.values() for path in local_paths(item)]
    if isinstance(value, list):
        return [path for item in value for path in local_paths(item)]
    return [value] if isinstance(value, str) and Path(value).is_absolute() else []


def run_metadata(source: dict, run: Path, lineage: dict) -> dict:
    """Describe the run's data; source_config.json keeps the selection's config.

    Gold provenance comes from the run's own item_provenance.jsonl.
    """
    inherited = source["resolved_config"].get("metadata", {})
    statuses, gold_sources = Counter(), Counter()
    for row in read_jsonl(run / "item_provenance.jsonl"):
        statuses[row["gold_status"]] += 1
        gold_sources.update(row["gold_sources"])
    if inherited.get("gold_status_counts", statuses) != statuses:
        raise ValueError(f"Gold status counts disagree with item provenance: {run}")
    if inherited.get("expected_items", statuses.total()) != statuses.total():
        raise ValueError(f"Expected items disagree with item provenance: {run}")
    return {
        **{key: inherited[key] for key in CARRIED_METADATA if key in inherited},
        "dataset_kind": "unified_leaderboard",
        "expected_items": statuses.total(),
        "gold_status_counts": dict(sorted(statuses.items())),
        "gold_sources": dict(sorted(gold_sources.items())),
        "dropped_models": DROPPED_MODELS,
        **lineage,
    }


def run_config(
    run_id: str, language: str, models: list[str], metadata: dict, gcs_root: str
) -> dict:
    """Return the source config, as launch-run.sh stores it under run-configs/."""
    config = {
        "run_id": run_id,
        "language": language,
        "items_path": f"{gcs_root}/datasets/{run_id}.jsonl",
        "results_dest": f"{gcs_root}/runs/{run_id}/",
        "models": models,
        "prompt_name": PROMPT_NAMES[language],
        "max_tokens": MAX_TOKENS,
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "model_workers": DEFAULT_MODEL_WORKERS,
        "metadata": metadata,
    }
    if paths := local_paths(config):
        raise ValueError(f"Local paths in run config for {run_id}: {paths}")
    return config


def cloud_run_config(source: dict, workspace: Path) -> BenchmarkRunConfig:
    """Load a source config as run-cloud does, minus the config download.

    The items are downloaded from the config's items_path, which records their
    GCS generation. See cmd._load_cloud_run_config.
    """
    workspace = workspace.expanduser().resolve()
    inputs_dir = workspace / "inputs"
    inputs_dir.mkdir(parents=True)
    source_path = inputs_dir / "run-config-source.json"
    source_path.write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n")
    raw_config = {
        **source,
        "output_dir": str(workspace / "outputs"),
        "inputs_dir": str(inputs_dir),
        "custom_models_path": str(inputs_dir / runner.CUSTOM_MODELS_FILE_NAME),
    }
    return _materialize_run_config(
        raw_config,
        source_config_path=source_path,
        inputs_dir=inputs_dir / source["run_id"],
        use_metadata_fallback=True,
    )


def september_model_configs(rows: list, provenance: dict, runs: Path) -> dict:
    """Pick each model's config from the source runs that carried its rows."""
    configs, counts = {}, defaultdict(Counter)
    for row in rows:
        source = provenance[row["model_call_id"]]["selected_source"]
        run = source["destination"].rsplit("/", 1)[-1]
        if run not in configs:
            configs[run] = json.loads((runs / run / "config.json").read_text())
        model_config = configs[run]["model_configs"].get(row["model"], {})
        counts[row["model"]][json.dumps(model_config, sort_keys=True)] += 1
    return {
        model: json.loads(options.most_common(1)[0][0])
        for model, options in sorted(counts.items())
    }


def write_runner_run(
    config: BenchmarkRunConfig, model_configs: dict, rows: list, run_dir: Path
) -> dict:
    """Write per-model response files and the snapshot, as the runner would.

    A response is kept only if its result_id is one this config plans, which is
    the condition under which the runner skips it on a later run.
    """
    items, items_sha256 = load_benchmark_items(config.items_path)
    prepared = runner._prepare_items(
        items, language=config.language, prompt_name=config.prompt_name
    )
    by_model = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)
    if set(by_model) != set(config.models):
        raise ValueError(f"Configured models differ from responses: {config.run_id}")
    for model in config.models:
        planned = runner._planned_model_items(
            config=config,
            model_name=model,
            model_config=model_configs[model],
            items=prepared,
        )
        order = {str(item["result_id"]): n for n, item in enumerate(planned)}
        unplanned = [row for row in by_model[model] if row["result_id"] not in order]
        if unplanned:
            raise ValueError(
                f"{len(unplanned)} {model} responses are not calls this "
                f"config would make: {config.run_id}"
            )
        ordered = sorted(by_model[model], key=lambda row: order[row["result_id"]])
        save_results(
            pd.DataFrame(ordered),
            runner._model_results_path(run_dir / "responses", model),
        )
    # The snapshot that runner.run_benchmark writes before it calls any model.
    custom_models_path = runner._resolve_custom_models_path(config.custom_models_path)
    # A later run refuses models that its custom_models.yaml does not group.
    runner._provider_groups_for_models(custom_models_path, config.models)
    resolved_config = config.to_resolved_dict()
    resolved_config["custom_models_path"] = str(custom_models_path)
    resolved_config["items_sha256"] = items_sha256
    runner._write_json(
        run_dir / "config.json",
        {
            "config_id": config_identity_sha256(config),
            "source_config_path": (
                str(config.config_path) if config.config_path else None
            ),
            "raw_config": config.raw_config,
            "resolved_config": resolved_config,
            "code_git_hash": runner.git_hash(),
            "code_git_dirty": runner.git_dirty(),
            "model_configs": runner._sanitize_model_config(
                {model: model_configs[model] for model in config.models}
            ),
        },
    )
    return {
        "models": len(config.models),
        "response_rows": len(rows),
        "planned_calls": len(config.models) * len(prepared),
        "missing_calls": len(config.models) * len(prepared) - len(rows),
    }


def build(
    selection: Path = DEFAULT_SELECTION,
    source_full: Path = DEFAULT_SOURCE_FULL,
    output: Path = DEFAULT_OUTPUT,
    gcs_root: str = DEFAULT_GCS_ROOT,
    workdir: Path = DEFAULT_WORKDIR,
) -> dict:
    selection, source_full, output = (
        selection.resolve(),
        source_full.resolve(),
        output.resolve(),
    )
    gcs_root = gcs_root.rstrip("/")
    if output.exists():
        raise FileExistsError(f"Preserving existing unified runs: {output}")
    if workdir.exists():
        raise FileExistsError(f"Runner workspace already exists: {workdir}")
    selected_manifest = verify_artifacts(selection)
    source_manifest = verify_artifacts(source_full)
    report_full = selection.parent.parent / "full"
    report_manifest = verify_artifacts(report_full)
    if (
        selected_manifest.get("kind") != "subset"
        or report_manifest.get("kind") != "full"
        or source_manifest.get("kind") != "full"
        or selected_manifest["parent_manifest_sha256"]
        != digest(report_full / "artifact_manifest.json")
        or report_manifest["parent_full_manifest_sha256"]
        != digest(source_full / "artifact_manifest.json")
    ):
        raise ValueError(
            "Selection does not descend from the specified source snapshot"
        )
    snapshot = source_full.parent
    downloads = json.loads((snapshot / "download_manifest.json").read_text())
    downloaded = {entry["path"]: entry["sha256"] for entry in downloads["files"]}
    for path in (snapshot / "sources/runs").glob("*/config.json"):
        if downloaded.get(str(path.relative_to(snapshot))) != digest(path):
            raise ValueError(f"Source run config changed after download: {path}")
    spec = json.loads((selection / "selection_spec.json").read_text())
    summary = {
        "kind": "unified_leaderboard_runs",
        "source_selection": str(selection),
        "source_selection_manifest_sha256": digest(
            selection / "artifact_manifest.json"
        ),
        "source_full": str(source_full),
        "source_full_manifest_sha256": digest(source_full / "artifact_manifest.json"),
        "selection_spec_sha256": digest(selection / "selection_spec.json"),
        "dropped_models": DROPPED_MODELS,
        "runs": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(
            prefix=".unified-leaderboard-", dir=output.parent
        ) as tmp:
            staging = Path(tmp) / output.name
            staging.mkdir()
            shutil.copy2(
                selection / "selection_spec.json", staging / "selection_spec.json"
            )
            shutil.copy2(
                source_full / "source_manifest.json", staging / "source_manifest.json"
            )
            shutil.copytree(selection / "plot_metadata", staging / "plot_metadata")
            for name, language in LANGUAGES.items():
                summary["runs"][name] = build_run(
                    selection / name,
                    staging / name,
                    staging / "records" / name,
                    name,
                    language,
                    spec,
                    report_full,
                    source_full,
                    summary,
                    gcs_root,
                    workdir / RUN_IDS[name],
                )
            shutil.rmtree(staging / "records")
            write_json(staging / "bundle_info.json", summary)
            seal_artifacts(staging, {"kind": "unified_leaderboard_runs", **summary})
            if verify_artifacts(selection) != selected_manifest:
                raise ValueError("Selection changed during packaging")
            staging.rename(output)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return summary


def build_run(
    source: Path,
    target: Path,
    records: Path,
    name: str,
    language: str,
    spec: dict,
    report_full: Path,
    source_full: Path,
    summary: dict,
    gcs_root: str,
    workspace: Path,
) -> dict:
    stats = check_run(source, language)
    expected_ids = set(spec["languages"][language]["item_ids"])
    items = {
        row["benchmark_item_id"]: row
        for row in read_jsonl(source / "benchmark_items.jsonl")
    }
    if expected_ids != set(items):
        raise ValueError(f"Selected IDs disagree with selection spec: {name}")
    if digest(report_full / name / "responses/merged.jsonl") != digest(
        source_full / name / "responses/merged.jsonl"
    ):
        raise ValueError(f"Source responses changed before selection: {name}")
    rows = list(read_jsonl(source / "responses/merged.jsonl"))
    kept = [row for row in rows if row["model"] not in DROPPED_MODELS]
    provenance = {
        row["model_call_id"]: row for row in read_jsonl(source / "provenance.jsonl")
    }
    models = sorted({row["model"] for row in kept})
    write_analysis_run(
        records,
        records,
        language,
        items,
        [
            (
                row,
                provenance[row["model_call_id"]]["selected_source"],
                provenance[row["model_call_id"]]["all_sources"],
            )
            for row in kept
        ],
        set(models),
        {
            row["benchmark_item_id"]: row
            for row in read_jsonl(source / "item_provenance.jsonl")
        },
        {},
    )
    if digest(records / "benchmark_items.jsonl") != digest(
        source / "benchmark_items.jsonl"
    ):
        raise ValueError(f"Rewritten items differ from the selection: {name}")
    target.mkdir()
    for relative in RECORD_FILES:
        shutil.copy2(records / relative, target / relative)
    shutil.copy2(source / "config.json", target / "source_config.json")
    run_id = RUN_IDS[name]
    source_config = run_config(
        run_id,
        language,
        models,
        run_metadata(
            json.loads((source / "config.json").read_text()),
            target,
            {
                "source_selection_manifest_sha256": summary[
                    "source_selection_manifest_sha256"
                ],
                "source_responses_sha256": digest(source / "responses/merged.jsonl"),
            },
        ),
        gcs_root,
    )
    write_json(target.parent / "run-configs" / f"{run_id}.json", source_config)
    config = cloud_run_config(source_config, workspace)
    if digest(config.items_path) != digest(target / "benchmark_items.jsonl"):
        raise ValueError(f"Published dataset differs from the selection: {run_id}")
    runner.download_default_custom_models(config.custom_models_path)
    model_configs = september_model_configs(
        kept, provenance, source_full.parent / "sources/runs"
    )
    stats.update(write_runner_run(config, model_configs, kept, target))
    dropped = Counter(row["model"] for row in rows if row["model"] in DROPPED_MODELS)
    stats.update(
        run_id=run_id,
        gcs_prefix=source_config["results_dest"],
        items_uri=source_config["items_path"],
        run_config_uri=f"{gcs_root}/run-configs/{run_id}.json",
        dropped_rows=dict(sorted(dropped.items())),
        items_sha256=digest(target / "benchmark_items.jsonl"),
    )
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--source-full", type=Path, default=DEFAULT_SOURCE_FULL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gcs-root", default=DEFAULT_GCS_ROOT)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.selection,
                args.source_full,
                args.output,
                args.gcs_root,
                args.workdir,
            ),
            indent=2,
        )
    )
