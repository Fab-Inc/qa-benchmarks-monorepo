"""Download the manifest's GCS sources into a NEW local snapshot directory.

Only GCS-to-local copies are supported. Existing snapshots are never refreshed
in place, because the source runs can still be changing.
"""

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import subprocess
from pathlib import Path


def download(source: str, destination: Path) -> dict:
    if not source.startswith("gs://") or str(destination).startswith("gs:"):
        raise ValueError("Only GCS-to-local downloads are supported")
    destination.parent.mkdir(parents=True, exist_ok=True)
    target = destination if source.endswith(".jsonl") else destination.parent
    result = subprocess.run(
        ["gcloud", "storage", "cp", "--recursive", source, str(target)],
        capture_output=True,
        text=True,
    )
    entry = {
        "source": source,
        "destination": str(destination),
        "returncode": result.returncode,
        "finished_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    print(json.dumps(entry), flush=True)
    if result.returncode:
        entry["error"] = result.stderr
    return entry


def download_snapshot(root: Path, manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    root.mkdir(parents=True, exist_ok=False)
    started = datetime.datetime.now(datetime.UTC).isoformat()
    run_tasks = [
        (manifest["run_prefix"] + run, root / "sources/runs" / run)
        for runs in manifest["groups"].values()
        for run in runs
    ]
    tasks = run_tasks + [
        (uri, root / "sources/datasets" / uri.rsplit("/", 1)[-1])
        for uri in manifest["updated_gold"]
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda pair: download(*pair), tasks))

    dataset_uris = set()
    for _, destination in run_tasks:
        config_path = destination / "config.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text())
        uri = config["resolved_config"].get("metadata", {}).get(
            "items_source_uri"
        ) or config.get("raw_config", {}).get("items_path")
        if not uri or not uri.startswith("gs://"):
            raise ValueError(f"Cannot find source dataset for {destination}: {uri}")
        dataset_uris.add(uri)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results.extend(
            pool.map(
                lambda uri: download(
                    uri, root / "sources/datasets" / uri.rsplit("/", 1)[-1]
                ),
                sorted(dataset_uris),
            )
        )
    files = [
        {
            "path": str(path.relative_to(root)),
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted((root / "sources").rglob("*"))
        if path.is_file()
    ]
    (root / "download_manifest.json").write_text(
        json.dumps(
            {"started_at": started, "downloads": results, "files": files}, indent=2
        )
        + "\n"
    )
    unexpected_errors = [
        r
        for r in results
        if r["returncode"] and "matched no objects or files" not in r.get("error", "")
    ]
    if unexpected_errors:
        raise RuntimeError("Download failures recorded in download_manifest.json")
    print(f"Snapshot complete: {len(files)} files", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("merged_sources_20260918.json"),
    )
    args = parser.parse_args()
    download_snapshot(args.root, args.manifest)
