"""Download the published runs into the layout the report scripts read.

Each published run is a normal runner run: config.json and one
responses/responses_<model>.jsonl file per model, plus merge records such as
item_provenance.jsonl. The committed report scripts read one root instead, with
english/ and kiswahili/ analysis runs, an audit/ directory and an
artifact_manifest.json, as the local merge and selection steps wrote them.

Each object is downloaded at the generation listed here and checked against its
CRC32C. The items come from the run's items_path, at the generation the run
recorded. Each analysis run is then rebuilt with the merge code, so coverage is
recomputed from the responses and includes models added later. The downloaded
runs are kept under gcs/.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import tempfile
from pathlib import Path

from build_unified_leaderboard_runs import DEFAULT_RUN_PREFIX, LANGUAGES, RUN_IDS
from fab_benchmark_publisher.publisher import list_gcs_objects, parse_gs_uri
from fab_benchmark_publisher.run_sync import _file_crc32c
from google.cloud.storage import Client
from merge_local_benchmark_runs import (
    digest,
    read_jsonl,
    seal_artifacts,
    write_analysis_run,
    write_json,
)
from phonics_benchmark.runner import model_results_filename

REQUIRED = {"config.json", "item_provenance.jsonl"}
# Runner output, plus the merge records the builder keeps beside it.
ALLOWED = REQUIRED | {
    "log.json",
    "benchmark_items.jsonl",
    "provenance.jsonl",
    "coverage_by_model.csv",
    "coverage_by_model_and_ratio_band.csv",
    "excluded_responses.jsonl",
    "source_config.json",
}
RESPONSE_FILE = re.compile(r"responses/responses_[^/]+\.jsonl")


def fetch(client, bucket: str, name: str, generation: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    client.bucket(bucket).blob(name).download_to_filename(
        str(path), if_generation_match=generation
    )


def download_run(client, source: str, destination: Path) -> dict:
    """Download one run's objects; refuse missing, unknown or corrupt ones."""
    bucket, prefix = parse_gs_uri(source)
    prefix = prefix.rstrip("/") + "/"
    objects = {
        obj.name.removeprefix(prefix): obj
        for obj in list_gcs_objects(
            bucket_name=bucket, prefix=prefix, storage_client=client
        )
    }
    responses = {name for name in objects if name.startswith("responses/")}
    others = set(objects) - responses
    if missing := sorted(REQUIRED - others):
        raise ValueError(f"Missing objects under {source}: {missing}")
    if unknown := sorted(others - ALLOWED) + sorted(
        name for name in responses if not RESPONSE_FILE.fullmatch(name)
    ):
        raise ValueError(f"Unexpected objects under {source}: {unknown}")
    if not responses:
        raise ValueError(f"No response files under {source}")
    for relative, obj in objects.items():
        path = destination / relative
        if not path.resolve().is_relative_to(destination.resolve()):
            raise ValueError(f"Unsafe object name: {obj.name}")
        fetch(client, bucket, obj.name, obj.generation, path)
        if _file_crc32c(path) != obj.crc32c:
            raise ValueError(f"Checksum differs: gs://{bucket}/{obj.name}")
    return {
        relative: {"generation": obj.generation, "crc32c": obj.crc32c}
        for relative, obj in sorted(objects.items())
    }


def analysis_run(
    run: Path, items_path: Path, target: Path, published: Path, source: str
) -> dict:
    """Rebuild the merge layout the report scripts read from a runner run."""
    config = json.loads((run / "config.json").read_text())
    language = config["resolved_config"]["language"]
    selected = []
    for path in sorted((run / "responses").glob("*.jsonl")):
        relative = str(path.relative_to(run))
        for line, row in enumerate(read_jsonl(path), 1):
            if model_results_filename(row["model"]) != path.name:
                raise ValueError(f"Response for {row['model']} in {relative}")
            origin = {"destination": source, "file": relative, "line": line}
            selected.append((row, origin, [origin]))
    items = {row["benchmark_item_id"]: row for row in read_jsonl(items_path)}
    item_provenance = {
        row["benchmark_item_id"]: row
        for row in read_jsonl(run / "item_provenance.jsonl")
    }
    return write_analysis_run(
        target,
        published,
        language,
        items,
        selected,
        set(config["resolved_config"]["models"]),
        item_provenance,
        {"dataset_kind": "gcs_download", "source_run": source},
    )


def download(root: Path, run_prefix: str = DEFAULT_RUN_PREFIX, client=None) -> dict:
    root = root.resolve()
    if root.exists():
        raise FileExistsError(f"Preserving existing download: {root}")
    client = client or Client()
    record = {"kind": "unified_leaderboard_download", "runs": {}}
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".unified-download-", dir=root.parent
    ) as tmp:
        staging = Path(tmp) / root.name
        for name, language in LANGUAGES.items():
            run_id = RUN_IDS[name]
            source = f"{run_prefix.rstrip('/')}/{run_id}/"
            run = staging / "gcs" / run_id
            objects = download_run(client, source, run)
            config = json.loads((run / "config.json").read_text())
            resolved, raw = config["resolved_config"], config["raw_config"]
            if resolved["run_id"] != run_id or resolved["language"] != language:
                raise ValueError(f"Unexpected run config under {source}")
            items_uri = raw["items_path"]
            generation = raw["metadata"]["items_source_generation"]
            items_path = staging / "gcs/datasets" / f"{run_id}.jsonl"
            fetch(client, *parse_gs_uri(items_uri), generation, items_path)
            if digest(items_path) != resolved["items_sha256"]:
                raise ValueError(f"Items differ from the run's record: {items_uri}")
            stats = analysis_run(
                run, items_path, staging / name, root / name, source.rstrip("/")
            )
            record["runs"][name] = {
                "source": source,
                "config_id": config["config_id"],
                "items": {"uri": items_uri, "generation": generation},
                "objects": objects,
                "models": stats["expected_models"],
                "response_rows": stats["retained_rows"],
            }
        write_json(staging / "audit/download_summary.json", record)
        seal_artifacts(
            staging,
            {
                "kind": "unified_leaderboard_download",
                "runs": {name: run["source"] for name, run in record["runs"].items()},
            },
        )
        # Outside the seal, so the same generations give the same manifest.
        write_json(
            staging / "audit/downloaded_at.json",
            {"downloaded_at": datetime.datetime.now(datetime.UTC).isoformat()},
        )
        staging.rename(root)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-prefix", default=DEFAULT_RUN_PREFIX)
    args = parser.parse_args()
    summary = download(args.root, args.run_prefix)
    print(
        json.dumps(
            {
                name: {key: run[key] for key in ("source", "models", "response_rows")}
                for name, run in summary["runs"].items()
            },
            indent=2,
        )
    )
