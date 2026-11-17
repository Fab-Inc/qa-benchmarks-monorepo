"""Validate optional sealed inputs used by historical report reproduction."""

import json
from collections import Counter
from hashlib import sha256
from pathlib import Path

import pandas as pd


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def verify_merge(root: Path) -> dict:
    """Verify the merger/selector manifest before scoring or writing reports."""
    manifest = json.loads((root / "artifact_manifest.json").read_text())
    for relative, expected in manifest["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"Manifest path leaves the input root: {relative}")
        if digest(path) != expected:
            raise ValueError(f"Merged input has changed: {relative}")
    return manifest


def inspect_dataset(run: Path) -> tuple[int, str]:
    items_path = run / "benchmark_items.jsonl"
    items = pd.read_json(items_path, lines=True)
    ids = items.benchmark_item_id
    if items.empty or not ids.is_unique:
        raise ValueError(f"Expected non-empty unique item IDs in {run}")
    config = json.loads((run / "config.json").read_text())["resolved_config"]
    configured_path = Path(config["items_path"])
    if not configured_path.is_absolute():
        configured_path = run / configured_path
    if not configured_path.is_file() or digest(configured_path) != digest(items_path):
        raise ValueError(
            f"Config must resolve to this merged dataset in {run}; "
            "rebuild the merge/selection locally if its old absolute "
            "path is unavailable."
        )
    if config.get("items_sha256") != digest(items_path):
        raise ValueError(f"Dataset hash disagrees with config in {run}")
    coverage = pd.read_csv(run / "coverage_by_model.csv")
    if not coverage.expected_items.eq(len(items)).all():
        raise ValueError(f"Coverage denominator disagrees with dataset size in {run}")
    provenance = pd.read_json(run / "item_provenance.jsonl", lines=True)
    if not provenance.benchmark_item_id.is_unique or set(
        provenance.benchmark_item_id
    ) != set(ids):
        raise ValueError(f"Gold provenance disagrees with dataset IDs in {run}")
    statuses = Counter(provenance.gold_status)
    details = "; ".join(
        f"{n:,} {status} items" for status, n in sorted(statuses.items())
    )
    return len(items), f"{run.name.title()} gold: {details}."
