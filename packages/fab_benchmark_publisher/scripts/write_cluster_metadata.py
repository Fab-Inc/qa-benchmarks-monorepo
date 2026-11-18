#!/usr/bin/env python3
"""
Write the cluster metadata file that sits above the per-benchmark results.

    <out-root>/<cluster_slug>_cluster_metadata.json
    {"cluster_slug": ..., "display_name": ..., "benchmark_ids": [...]}

cluster_slug and the ordered benchmark_ids come from leaderboard_pairs.json at the
repo root (its cluster_name and the benchmark_domain of each pair, in order), so the
order matches the leaderboard's columns. That file has no cluster display name, so
it is a flag here, defaulting to the slug in title case.

Run from the repo root:

    uv run python packages/fab_benchmark_publisher/scripts/write_cluster_metadata.py

Standalone on purpose: nothing imports it, so deleting it (and its test,
tests/test_write_cluster_metadata.py) removes it cleanly.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from fab_benchmark_publisher.benchmark_results import cluster_metadata_file_name
from fab_benchmark_publisher.result_metadata import ClusterMeta, write_cluster_meta

REPO_ROOT = Path(__file__).resolve().parents[3]


def cluster_meta_from_pairs(path: Path, display_name: str | None) -> ClusterMeta:
    raw = json.loads(path.read_text(encoding="utf-8"))
    slug = raw["cluster_name"]
    return ClusterMeta(
        cluster_slug=slug,
        display_name=display_name or slug.replace("-", " ").title(),
        benchmark_ids=tuple(p["benchmark_domain"] for p in raw["pairs"]),
    )


def main(argv: Sequence[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--pairs-file", type=Path, default=REPO_ROOT / "leaderboard_pairs.json"
    )
    parser.add_argument("--display-name", default=None)
    parser.add_argument("--out-root", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args(argv)

    meta = cluster_meta_from_pairs(args.pairs_file, args.display_name)
    path = write_cluster_meta(
        meta, args.out_root / cluster_metadata_file_name(meta.cluster_slug)
    )
    print(f"{meta.cluster_slug}: {', '.join(meta.benchmark_ids)} -> {path}")
    return path


if __name__ == "__main__":
    main()
