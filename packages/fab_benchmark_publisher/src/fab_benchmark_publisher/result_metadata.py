"""Metadata that sits next to the metrics files: benchmark, metric and cluster.

  benchmark metadata   one JSON file per benchmark, holding that benchmark's metric
                       metadata as an array (one entry per metric it publishes)
  cluster metadata     one JSON file per cluster, one level above the benchmarks it
                       lists, since a cluster spans several benchmarks

Slugs appear in filenames (<benchmark>_<metric>_<experiment>.csv), where "_" is the
separator, so a slug must not contain one.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]*$")


def check_slug(value: Any, name: str) -> str:
    """Return `value` if it is a valid slug, else raise naming the field."""
    if not isinstance(value, str) or not _SLUG_RE.match(value):
        raise ValueError(
            f"{name} must be letters, digits, '-' or '.' (no '_', which separates "
            f"slugs in filenames); got {value!r}"
        )
    return value


def _check_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string; got {value!r}")
    return value


def _check_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false; got {value!r}")
    return value


@dataclass(frozen=True)
class MetricMeta:
    """How to read one metric.

    `is_higher_better` sets the direction frontiers improve in.
    `is_bad_format_scored` decides what a badly formatted response counts as: True
    keeps it in the metric with the score the benchmark gave it (usually 0), False
    leaves it out of the metric and only counts it in n_items_bad_format.
    """

    metric_slug: str
    display_name: str
    display_description: str
    unit: str
    is_higher_better: bool
    is_bad_format_scored: bool

    def __post_init__(self) -> None:
        check_slug(self.metric_slug, "metric_slug")
        _check_text(self.display_name, "display_name")
        _check_text(self.display_description, "display_description")
        if not isinstance(self.unit, str):
            raise ValueError(
                f'unit must be a string ("" if unitless); got {self.unit!r}'
            )
        _check_bool(self.is_higher_better, "is_higher_better")
        _check_bool(self.is_bad_format_scored, "is_bad_format_scored")


@dataclass(frozen=True)
class BenchmarkMeta:
    benchmark_slug: str
    display_name: str
    metrics: tuple[MetricMeta, ...]

    def __post_init__(self) -> None:
        check_slug(self.benchmark_slug, "benchmark_slug")
        _check_text(self.display_name, "display_name")
        if not self.metrics:
            raise ValueError("a benchmark must declare at least one metric")
        slugs = [m.metric_slug for m in self.metrics]
        if len(set(slugs)) != len(slugs):
            raise ValueError(f"duplicate metric_slug in {slugs}")

    def metric(self, metric_slug: str) -> MetricMeta:
        for m in self.metrics:
            if m.metric_slug == metric_slug:
                return m
        raise ValueError(
            f"benchmark '{self.benchmark_slug}' declares no metric '{metric_slug}'"
        )


@dataclass(frozen=True)
class ClusterMeta:
    cluster_slug: str
    display_name: str
    benchmark_ids: tuple[str, ...]  # ordered, as the website shows them

    def __post_init__(self) -> None:
        check_slug(self.cluster_slug, "cluster_slug")
        _check_text(self.display_name, "display_name")
        if not self.benchmark_ids:
            raise ValueError("a cluster must list at least one benchmark")
        for b in self.benchmark_ids:
            check_slug(b, "benchmark_ids entry")
        if len(set(self.benchmark_ids)) != len(self.benchmark_ids):
            raise ValueError(f"duplicate entry in benchmark_ids {self.benchmark_ids}")


# --------------------------------------------------------------------------- #
# JSON I/O
# --------------------------------------------------------------------------- #
def _expect_keys(raw: Any, keys: set[str], what: str) -> dict:
    if not isinstance(raw, dict) or set(raw) != keys:
        got = sorted(raw) if isinstance(raw, dict) else type(raw).__name__
        raise ValueError(f"{what}: expected exactly the keys {sorted(keys)}, got {got}")
    return raw


def _write_json(payload: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def benchmark_meta_to_dict(meta: BenchmarkMeta) -> dict:
    return {
        "benchmark_slug": meta.benchmark_slug,
        "display_name": meta.display_name,
        "metrics": [asdict(m) for m in meta.metrics],
    }


def benchmark_meta_from_dict(raw: Any) -> BenchmarkMeta:
    raw = _expect_keys(
        raw, {"benchmark_slug", "display_name", "metrics"}, "benchmark metadata"
    )
    if not isinstance(raw["metrics"], list):
        raise ValueError("benchmark metadata: metrics must be an array")
    metric_keys = set(MetricMeta.__dataclass_fields__)
    metrics = tuple(
        MetricMeta(**_expect_keys(m, metric_keys, "metric metadata"))
        for m in raw["metrics"]
    )
    return BenchmarkMeta(raw["benchmark_slug"], raw["display_name"], metrics)


def write_benchmark_meta(meta: BenchmarkMeta, path: str | Path) -> Path:
    return _write_json(benchmark_meta_to_dict(meta), path)


def read_benchmark_meta(path: str | Path) -> BenchmarkMeta:
    return benchmark_meta_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def write_cluster_meta(meta: ClusterMeta, path: str | Path) -> Path:
    return _write_json(
        {
            "cluster_slug": meta.cluster_slug,
            "display_name": meta.display_name,
            "benchmark_ids": list(meta.benchmark_ids),
        },
        path,
    )


def read_cluster_meta(path: str | Path) -> ClusterMeta:
    raw = _expect_keys(
        json.loads(Path(path).read_text(encoding="utf-8")),
        {"cluster_slug", "display_name", "benchmark_ids"},
        "cluster metadata",
    )
    if not isinstance(raw["benchmark_ids"], list):
        raise ValueError("cluster metadata: benchmark_ids must be an array")
    return ClusterMeta(
        raw["cluster_slug"], raw["display_name"], tuple(raw["benchmark_ids"])
    )
