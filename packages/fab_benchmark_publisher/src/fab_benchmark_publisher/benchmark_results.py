"""One benchmark's result files, written and checked as a set.

Per benchmark directory:

  <benchmark>_<metric>_<experiment>.csv              metrics.py
  <benchmark>_<metric>_<experiment>_frontiers.json   benchmark_frontiers.py
  benchmark_metadata.json                            result_metadata.BenchmarkMeta,
                                                     metric metadata inside it

One level up, shared by every benchmark in the cluster:

  <cluster>_cluster_metadata.json                    result_metadata.ClusterMeta

Locally the caller picks the directories. On GCS the files belong under

  gs://fab-benchmarks-external/website/benchmark-clusters/<cluster>/
      <cluster>_cluster_metadata.json
      <benchmark group>/<benchmark>/<the three files above>

which gcs_object_name spells out. Nothing here uploads yet.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fab_benchmark_publisher.benchmark_frontiers import (
    frontiers_file_name,
    read_frontiers_json,
    validate_frontiers,
    write_frontiers_json,
)
from fab_benchmark_publisher.contracts import LeaderboardRecord
from fab_benchmark_publisher.metrics import (
    OVERALL,
    metrics_file_name,
    read_metrics_csv,
    validate_metrics,
    write_metrics_csv,
)
from fab_benchmark_publisher.result_metadata import (
    BenchmarkMeta,
    check_slug,
    read_benchmark_meta,
    write_benchmark_meta,
)

GCS_BUCKET = "fab-benchmarks-external"
GCS_CLUSTERS_PREFIX = "website/benchmark-clusters"
BENCHMARK_METADATA_FILE = "benchmark_metadata.json"


def cluster_metadata_file_name(cluster_slug: str) -> str:
    return f"{check_slug(cluster_slug, 'cluster_slug')}_cluster_metadata.json"


def gcs_object_name(
    file_name: str,
    *,
    cluster_slug: str,
    benchmark_group: str | None = None,
    benchmark_slug: str | None = None,
) -> str:
    """Object name in GCS_BUCKET. Leave benchmark_group/benchmark_slug out for the
    cluster metadata file, which sits at the cluster level."""
    parts = [GCS_CLUSTERS_PREFIX, cluster_slug]
    if (benchmark_group is None) != (benchmark_slug is None):
        raise ValueError("give both benchmark_group and benchmark_slug, or neither")
    if benchmark_group is not None:
        parts += [benchmark_group, benchmark_slug]
    return "/".join([*parts, file_name])


def validate_benchmark_results(
    metrics: pd.DataFrame, frontiers: dict, meta: BenchmarkMeta
) -> None:
    """Each file on its own, then that all three describe the same benchmark and
    metric."""
    slugs = set(metrics["benchmark_slug"])
    if slugs != {meta.benchmark_slug}:
        raise ValueError(
            f"metrics are for {sorted(slugs)}, metadata for '{meta.benchmark_slug}'"
        )
    metric = meta.metric(str(metrics["metric_slug"].iloc[0]))
    validate_metrics(metrics, metric=metric)
    validate_frontiers(frontiers, metrics)
    if frontiers["is_higher_better"] != metric.is_higher_better:
        raise ValueError("frontiers and metric metadata disagree on is_higher_better")


def write_benchmark_results(
    outdir: str | Path,
    *,
    metrics: pd.DataFrame,
    frontiers: dict,
    meta: BenchmarkMeta,
) -> dict[str, Path]:
    """Validate the set, then write all three files to `outdir`."""
    validate_benchmark_results(metrics, frontiers, meta)
    metrics_path = write_metrics_csv(metrics, outdir)
    frontiers_path = write_frontiers_json(frontiers, outdir)
    meta_path = write_benchmark_meta(meta, Path(outdir) / BENCHMARK_METADATA_FILE)
    return {p.name: p for p in (metrics_path, frontiers_path, meta_path)}


def read_benchmark_results(
    outdir: str | Path, *, metric_slug: str, experiment_slug: str
) -> tuple[pd.DataFrame, dict, BenchmarkMeta]:
    """Read and cross-validate one metric/experiment's files from `outdir`."""
    outdir = Path(outdir)
    meta = read_benchmark_meta(outdir / BENCHMARK_METADATA_FILE)
    key = (meta.benchmark_slug, metric_slug, experiment_slug)
    metrics = read_metrics_csv(
        outdir / metrics_file_name(*key), metric=meta.metric(metric_slug)
    )
    frontiers = read_frontiers_json(outdir / frontiers_file_name(*key))
    validate_benchmark_results(metrics, frontiers, meta)
    return metrics, frontiers, meta


def metrics_to_leaderboard_records(
    metrics: pd.DataFrame,
    meta: BenchmarkMeta,
    *,
    benchmark_domain: str | None = None,
) -> list[LeaderboardRecord]:
    """A metrics file's overall rows as LeaderboardRecords, so the existing
    `fab-leaderboard publish` can read a benchmark that only emits the new files.

    `bad_format` is the percentage of all items that were badly formatted and `n` is
    the total item count, matching what the records files carry today.
    """
    metric = meta.metric(str(metrics["metric_slug"].iloc[0]))
    if not metric.is_higher_better:
        raise ValueError("the leaderboard assumes higher is better")
    validate_metrics(metrics, metric=metric)
    rows = metrics[
        (metrics["benchmark_subset"].fillna(OVERALL) == OVERALL)
        & metrics["metric_value"].notna()
    ]
    records = []
    for r in rows.itertuples(index=False):
        n_scored, n_bad = int(r.n_items_scored), int(r.n_items_bad_format)
        n_total = n_scored if metric.is_bad_format_scored else n_scored + n_bad
        records.append(
            LeaderboardRecord(
                benchmark_domain=benchmark_domain or meta.benchmark_slug,
                model=str(r.model_slug),
                accuracy=float(r.metric_value),
                bad_format=round(100.0 * n_bad / n_total, 4) if n_total else 0.0,
                n=n_total,
            )
        )
    return records
