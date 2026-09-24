"""Package the exact 22 September leaderboard inputs as two analysis-only runs.

The source snapshot and selected report are immutable inputs. Original GCS runs
are neither contacted nor modified. Each bundle keeps raw response provenance;
the new run ID identifies the scoring dataset, not the original model calls.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from merge_local_benchmark_runs import (
    digest,
    seal_artifacts,
    verify_artifacts,
    write_json,
)

DEFAULT_SELECTION = Path(
    "outputs/phonics-reports-20260922-coverage90/selections/fixed-991-en-1000-sw"
)
DEFAULT_SOURCE_FULL = Path("outputs/phonics-sources-20260922-coverage90/full")
DEFAULT_OUTPUT = Path("outputs/phonics-unified-leaderboard-20260922")
DEFAULT_RUN_PREFIX = "gs://phonics-benchmark/phonics/runs"
RUN_IDS = {
    "english": "en-gold-leaderboard-20260922",
    "kiswahili": "sw-gold-leaderboard-20260922",
}
LANGUAGES = {"english": "en", "kiswahili": "sw"}
RUN_FILES = (
    "benchmark_items.jsonl",
    "responses/merged.jsonl",
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


def unified_config(source: dict, run: Path, run_id: str, lineage: dict) -> dict:
    """Describe the analysis run itself; source_config.json keeps the rest.

    Gold provenance comes from the run's own item_provenance.jsonl.
    """
    resolved = source["resolved_config"]
    inherited = resolved.get("metadata", {})
    statuses, gold_sources = Counter(), Counter()
    for row in read_jsonl(run / "item_provenance.jsonl"):
        statuses[row["gold_status"]] += 1
        gold_sources.update(row["gold_sources"])
    if inherited.get("gold_status_counts", statuses) != statuses:
        raise ValueError(f"Gold status counts disagree with item provenance: {run}")
    if inherited.get("expected_items", statuses.total()) != statuses.total():
        raise ValueError(f"Expected items disagree with item provenance: {run}")
    language = resolved["language"]
    config = {
        "analysis_only": True,
        "resolved_config": {
            "run_id": run_id,
            "language": language,
            "task": "decodability",
            "prompt_name": (
                "decodability_evaluation_schwa_rule"
                if language == "en"
                else "decodability_evaluation_clusters"
            ),
            "items_path": "benchmark_items.jsonl",
            "items_sha256": resolved["items_sha256"],
            "models": resolved["models"],
            "metadata": {
                **{key: inherited[key] for key in CARRIED_METADATA if key in inherited},
                "dataset_kind": "unified_leaderboard",
                "expected_items": statuses.total(),
                "gold_status_counts": dict(sorted(statuses.items())),
                "gold_sources": dict(sorted(gold_sources.items())),
                **lineage,
            },
        },
    }
    if paths := local_paths(config):
        raise ValueError(f"Local paths in unified config for {run}: {paths}")
    return config


def build(
    selection: Path = DEFAULT_SELECTION,
    source_full: Path = DEFAULT_SOURCE_FULL,
    output: Path = DEFAULT_OUTPUT,
    run_prefix: str = DEFAULT_RUN_PREFIX,
) -> dict:
    selection, source_full, output = (
        selection.resolve(),
        source_full.resolve(),
        output.resolve(),
    )
    if output.exists():
        raise FileExistsError(f"Preserving existing unified runs: {output}")
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
        "runs": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".unified-leaderboard-", dir=output.parent
    ) as tmp:
        staging = Path(tmp) / output.name
        staging.mkdir()
        shutil.copy2(selection / "selection_spec.json", staging / "selection_spec.json")
        shutil.copy2(
            source_full / "source_manifest.json", staging / "source_manifest.json"
        )
        shutil.copytree(selection / "plot_metadata", staging / "plot_metadata")
        for name, language in LANGUAGES.items():
            source = selection / name
            stats = check_run(source, language)
            expected_ids = set(spec["languages"][language]["item_ids"])
            actual_ids = {
                row["benchmark_item_id"]
                for row in read_jsonl(source / "benchmark_items.jsonl")
            }
            if expected_ids != actual_ids:
                raise ValueError(f"Selected IDs disagree with selection spec: {name}")
            if digest(report_full / name / "responses/merged.jsonl") != digest(
                source_full / name / "responses/merged.jsonl"
            ):
                raise ValueError(f"Source responses changed before selection: {name}")
            target = staging / name
            target.mkdir()
            for relative in RUN_FILES:
                destination = target / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / relative, destination)
            shutil.copy2(source / "config.json", target / "source_config.json")
            run_id = RUN_IDS[name]
            uri = f"{run_prefix.rstrip('/')}/{run_id}/benchmark_items.jsonl"
            config = unified_config(
                json.loads((source / "config.json").read_text()),
                target,
                run_id,
                {
                    "items_source_uri": uri,
                    "source_selection_manifest_sha256": summary[
                        "source_selection_manifest_sha256"
                    ],
                    "source_responses_sha256": digest(
                        source / "responses/merged.jsonl"
                    ),
                },
            )
            write_json(target / "config.json", config)
            stats.update(
                run_id=run_id,
                gcs_prefix=f"{run_prefix.rstrip('/')}/{run_id}/",
                items_sha256=digest(target / "benchmark_items.jsonl"),
                responses_sha256=digest(target / "responses/merged.jsonl"),
            )
            summary["runs"][name] = stats
        write_json(staging / "bundle_info.json", summary)
        seal_artifacts(staging, {"kind": "unified_leaderboard_runs", **summary})
        if verify_artifacts(selection) != selected_manifest:
            raise ValueError("Selection changed during packaging")
        staging.rename(output)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--source-full", type=Path, default=DEFAULT_SOURCE_FULL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-prefix", default=DEFAULT_RUN_PREFIX)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.selection, args.source_full, args.output, args.run_prefix),
            indent=2,
        )
    )
