"""Create the unified runs in GCS without replacing existing objects.

Only files listed in the bundle's seal are uploaded, and any other file in a run
directory stops the publish. Every new object is uploaded with an
absent-generation precondition. Existing objects are accepted only when their
CRC32C matches the local sealed bundle. A run-relative path named with
--replace may be overwritten, and one named with --delete may be removed, each
conditional on the generation checked here. Each run's source config goes to
run-configs/, and its items must already be at its items_path. Both runs are
checked before anything is written. No source run is changed or deleted.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from fab_benchmark_publisher.publisher import (
    GCSObject,
    list_gcs_objects,
    parse_gs_uri,
    upload_file_with_generation,
)
from fab_benchmark_publisher.run_sync import RunSync, _file_crc32c
from google.cloud.storage import Client
from merge_local_benchmark_runs import digest, verify_artifacts

DEFAULT_ROOT = Path("outputs/phonics-unified-leaderboard-20260922")


@dataclass
class RunPlan:
    """The writes one run needs, decided before any run is written."""

    run_id: str
    run: Path
    destination: str
    bucket: str
    prefix: str
    files: dict[str, Path]
    checksums: dict[str, str]
    create: list[str]
    replace: dict[str, GCSObject]
    delete: dict[str, GCSObject]
    run_config: tuple[str, str, Path] | None
    response_rows: int

    def report(self) -> dict:
        return {
            "run_id": self.run_id,
            "destination": self.destination,
            "objects": len(self.files),
            "response_rows": self.response_rows,
            "create": [f"gs://{self.bucket}/{name}" for name in self.create]
            + (
                [f"gs://{self.run_config[0]}/{self.run_config[1]}"]
                if self.run_config
                else []
            ),
            "replace": [
                {
                    "object": f"gs://{self.bucket}/{name}",
                    "previous_generation": remote.generation,
                    "previous_crc32c": remote.crc32c,
                    "crc32c": self.checksums[name],
                }
                for name, remote in self.replace.items()
            ],
            "delete": [
                {
                    "object": f"gs://{self.bucket}/{name}",
                    "previous_generation": remote.generation,
                    "previous_crc32c": remote.crc32c,
                }
                for name, remote in self.delete.items()
            ],
        }


def sealed_run_files(root: Path, sealed: dict, name: str) -> dict[str, Path]:
    """Map each sealed run-relative path to its file; refuse unsealed extras."""
    run = root / name
    sealed_paths = {
        path.removeprefix(f"{name}/")
        for path in sealed["files"]
        if path.startswith(f"{name}/")
    }
    on_disk = {
        path.relative_to(run).as_posix() for path in run.rglob("*") if path.is_file()
    }
    if on_disk != sealed_paths:
        raise ValueError(
            f"Files outside the seal in {run}: {sorted(on_disk ^ sealed_paths)}"
        )
    return {relative: run / relative for relative in sorted(sealed_paths)}


def remote_object(uri: str, client) -> tuple[str, str, GCSObject | None]:
    bucket, name = parse_gs_uri(uri)
    listed = list_gcs_objects(bucket_name=bucket, prefix=name, storage_client=client)
    return bucket, name, next((obj for obj in listed if obj.name == name), None)


def plan_run(
    root: Path,
    sealed: dict,
    language: str,
    details: dict,
    replace: set,
    delete: set,
    client,
) -> RunPlan:
    run = root / language
    config = json.loads((run / "config.json").read_text())
    destination = details["gcs_prefix"]
    if (
        config["resolved_config"]["run_id"] != details["run_id"]
        or config["resolved_config"]["results_dest"] != destination
        or config["raw_config"]["items_path"] != details["items_uri"]
    ):
        raise ValueError(f"Run config disagrees with destination: {language}")
    local = sealed_run_files(root, sealed, language)
    if unknown := sorted(replace - set(local)):
        raise ValueError(f"--replace names files not sealed in {language}: {unknown}")
    if sealed_names := sorted(delete & set(local)):
        raise ValueError(f"--delete names sealed files in {language}: {sealed_names}")
    _, _, items = remote_object(details["items_uri"], client)
    if items is None or items.crc32c != _file_crc32c(run / "benchmark_items.jsonl"):
        raise ValueError(f"Published items differ from the run: {details['items_uri']}")
    config_bucket, config_name, published_config = remote_object(
        details["run_config_uri"], client
    )
    config_path = root / "run-configs" / f"{details['run_id']}.json"
    if published_config and published_config.crc32c != _file_crc32c(config_path):
        raise ValueError(f"Existing run config differs: {details['run_config_uri']}")
    bucket, prefix = parse_gs_uri(destination)
    prefix = prefix.rstrip("/") + "/"
    files = {prefix + relative: path for relative, path in local.items()}
    existing = {
        obj.name: obj
        for obj in list_gcs_objects(
            bucket_name=bucket, prefix=prefix, storage_client=client
        )
    }
    removable = {prefix + relative for relative in delete}
    if unexpected := sorted(set(existing) - set(files) - removable):
        raise ValueError(f"Unexpected objects under {destination}: {unexpected}")
    if absent := sorted(removable - set(existing)):
        raise ValueError(f"--delete names objects that do not exist: {absent}")
    checksums = {name: _file_crc32c(path) for name, path in files.items()}
    changed = {
        name: obj
        for name, obj in existing.items()
        if name in files and obj.crc32c != checksums[name]
    }
    if blocked := sorted(
        name for name in changed if name.removeprefix(prefix) not in replace
    ):
        raise ValueError(f"Existing objects differ under {destination}: {blocked}")
    return RunPlan(
        run_id=details["run_id"],
        run=run,
        destination=destination,
        bucket=bucket,
        prefix=prefix,
        files=files,
        checksums=checksums,
        create=sorted(set(files) - set(existing)),
        replace=dict(sorted(changed.items())),
        delete={name: existing[name] for name in sorted(removable)},
        run_config=None
        if published_config
        else (config_bucket, config_name, config_path),
        response_rows=details["response_rows"],
    )


def execute(plan: RunPlan, client) -> None:
    """Create, then replace, then delete, so no response is ever missing."""
    sync = RunSync(plan.run, plan.destination, storage_client=client)
    for name in plan.create:
        sync.push_file(plan.files[name])
    for name, remote in plan.replace.items():
        upload_file_with_generation(
            source_path=plan.files[name],
            bucket_name=plan.bucket,
            file_name=name,
            if_generation_match=remote.generation,
            storage_client=client,
        )
    for name, remote in plan.delete.items():
        client.bucket(plan.bucket).blob(name).delete(
            if_generation_match=remote.generation
        )
    if plan.run_config:
        bucket, name, path = plan.run_config
        upload_file_with_generation(
            source_path=path,
            bucket_name=bucket,
            file_name=name,
            if_generation_match=0,
            storage_client=client,
        )
    remote = {
        obj.name: obj
        for obj in list_gcs_objects(
            bucket_name=plan.bucket, prefix=plan.prefix, storage_client=client
        )
    }
    if set(remote) != set(plan.files) or any(
        remote[name].crc32c != checksum for name, checksum in plan.checksums.items()
    ):
        raise ValueError(f"Published bundle does not match local files: {plan.run}")


def publish(
    root: Path = DEFAULT_ROOT,
    replace: tuple[str, ...] = (),
    dry_run: bool = False,
    client=None,
    delete: tuple[str, ...] = (),
) -> dict:
    root = root.resolve()
    sealed = verify_artifacts(root)
    if sealed.get("kind") != "unified_leaderboard_runs":
        raise ValueError("Expected sealed unified leaderboard runs")
    comparison = json.loads((root / "analysis/comparison.json").read_text())
    if comparison["source_bundle_manifest_sha256"] != digest(
        root / "artifact_manifest.json"
    ) or any(
        result["comparison"]["max_absolute_difference"]
        > result["comparison"]["matching_tolerance"]
        or result["combined_summary_comparison"]["max_absolute_difference"]
        > result["comparison"]["matching_tolerance"]
        for result in comparison["languages"].values()
    ):
        raise ValueError("Adjusted BA comparison has not passed")
    info = json.loads((root / "bundle_info.json").read_text())
    client = client or Client()
    plans = {
        language: plan_run(
            root, sealed, language, details, set(replace), set(delete), client
        )
        for language, details in info["runs"].items()
    }
    if not dry_run:
        for plan in plans.values():
            execute(plan, client)
    return {language: plan.report() for language, plan in plans.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--replace",
        action="append",
        default=[],
        metavar="PATH",
        help="run-relative path whose existing object may be overwritten; repeatable",
    )
    parser.add_argument(
        "--delete",
        action="append",
        default=[],
        metavar="PATH",
        help="run-relative path of an unsealed existing object to remove; repeatable",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="check everything and print the planned writes without uploading",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            publish(
                args.root,
                tuple(args.replace),
                args.dry_run,
                delete=tuple(args.delete),
            ),
            indent=2,
        )
    )
