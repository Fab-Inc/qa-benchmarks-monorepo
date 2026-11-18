"""The frontiers file: one per metrics file, built from it.

    <benchmark_slug>_<metric_slug>_<experiment_slug>_frontiers.json

holding every frontier as a crosstab of

  axis               price (models.csv blended_price) or size (open-weight only)
  benchmark_subset   the overall result ("") and every subset in the metrics

Each frontier is the raw Pareto frontier -- the same line the leaderboard draws, not
the concave hull used for default-model selection -- stored as plain points. Turning
points into step segments, provider ids and so on is presentation, which belongs to
the bundle step.

Frontiers are computed once per metrics file because model prices and sizes are
fixed: a frontier saved today does not go stale.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from fab_benchmark_publisher.frontier import compute_frontier
from fab_benchmark_publisher.leaderboard import load_models
from fab_benchmark_publisher.metrics import (
    KEY_COLUMNS,
    OVERALL,
    metrics_file_name,
    validate_metrics,
)
from fab_benchmark_publisher.result_metadata import MetricMeta

# axis name -> models.csv-derived field it is plotted against.
FRONTIER_AXES = {"price": "blended_price", "size": "size"}

_TOP_LEVEL_KEYS = (
    *KEY_COLUMNS,
    "is_higher_better",
    "models_without_metadata",
    "frontiers",
)


def frontiers_file_name(
    benchmark_slug: str, metric_slug: str, experiment_slug: str
) -> str:
    stem = metrics_file_name(benchmark_slug, metric_slug, experiment_slug)
    return stem.removesuffix(".csv") + "_frontiers.json"


def _axis_population(axis: str, model_slugs: list[str], models_db: dict) -> list[dict]:
    """(model_slug, x) for every model that can be placed on `axis`.

    Matches the website's population: a positive price, or for size an open-weight
    model with a positive parameter count. Anything else would let a model with an
    unknown x land on the frontier at a made-up position.
    """
    field = FRONTIER_AXES[axis]
    population = []
    for model_slug in model_slugs:
        meta = models_db.get(model_slug)
        if meta is None:
            continue
        if axis == "size" and not meta["open"]:
            continue
        x = meta[field]
        if x is not None and x > 0:
            population.append({"model_slug": model_slug, "x": float(x)})
    return population


def _frontier_points(
    rows: pd.DataFrame, axis: str, models_db: dict, *, is_higher_better: bool
) -> list[dict]:
    values = dict(zip(rows["model_slug"], rows["metric_value"].astype(float)))
    population = _axis_population(axis, sorted(values), models_db)
    if not population:
        return []
    df = pd.DataFrame(population)
    df["y"] = df["model_slug"].map(values)
    # compute_frontier keeps whatever is largest so far; flip the sign for metrics
    # where smaller is better rather than teaching it a direction.
    df["_score"] = df["y"] if is_higher_better else -df["y"]
    frontier = compute_frontier(df, acc_col="_score", sorting_col="x", smooth=False)
    return [
        {"model_slug": r["model_slug"], "x": float(r["x"]), "y": float(r["y"])}
        for _, r in frontier.sort_values("x").iterrows()
    ]


def build_frontiers(
    metrics: pd.DataFrame, *, metric: MetricMeta, models_csv: str | Path
) -> dict[str, Any]:
    """Every (axis x benchmark_subset) frontier of one metrics file.

    Models absent from models.csv cannot be placed on any axis; they are listed in
    `models_without_metadata` rather than dropped silently.
    """
    validate_metrics(metrics, metric=metric)
    models_db, _ = load_models(models_csv)
    scored = metrics[metrics["metric_value"].notna()]
    subsets = sorted(
        set(scored["benchmark_subset"].fillna(OVERALL)),
        key=lambda s: (s != OVERALL, s),
    )

    frontiers = []
    for axis in FRONTIER_AXES:
        for subset in subsets:
            rows = scored[scored["benchmark_subset"].fillna(OVERALL) == subset]
            frontiers.append(
                {
                    "axis": axis,
                    "x": FRONTIER_AXES[axis],
                    "benchmark_subset": subset,
                    "points": _frontier_points(
                        rows, axis, models_db, is_higher_better=metric.is_higher_better
                    ),
                }
            )

    first = metrics.iloc[0]
    result = {
        **{c: str(first[c]) for c in KEY_COLUMNS},
        "is_higher_better": metric.is_higher_better,
        "models_without_metadata": sorted(set(metrics["model_slug"]) - set(models_db)),
        "frontiers": frontiers,
    }
    validate_frontiers(result, metrics)
    return result


def validate_frontiers(frontiers: dict[str, Any], metrics: pd.DataFrame) -> None:
    """Raise ValueError listing every way `frontiers` breaks the file's contract or
    disagrees with the metrics file it was built from."""
    missing = [k for k in _TOP_LEVEL_KEYS if k not in frontiers]
    if missing:
        raise ValueError(f"invalid frontiers: missing key(s) {', '.join(missing)}")

    problems: list[str] = []
    first = metrics.iloc[0]
    for col in KEY_COLUMNS:
        if frontiers[col] != first[col]:
            problems.append(
                f"{col} is {frontiers[col]!r} but the metrics say {first[col]!r}"
            )
    subsets = metrics["benchmark_subset"].fillna(OVERALL)
    values = dict(zip(zip(metrics["model_slug"], subsets), metrics["metric_value"]))
    better = 1.0 if frontiers["is_higher_better"] else -1.0

    seen: set[tuple[str, str]] = set()
    for f in frontiers["frontiers"]:
        label = f"{f.get('axis')}/{f.get('benchmark_subset')!r}"
        if f.get("axis") not in FRONTIER_AXES:
            problems.append(f"{label}: unknown axis")
        key = (f.get("axis"), f.get("benchmark_subset"))
        if key in seen:
            problems.append(f"{label}: appears more than once")
        seen.add(key)

        points = f.get("points", [])
        for prev, cur in zip(points, points[1:]):
            if not cur["x"] > prev["x"]:
                problems.append(f"{label}: x is not strictly increasing")
                break
            if not (cur["y"] - prev["y"]) * better > 0:
                problems.append(f"{label}: a point does not improve on the last one")
                break
        for p in points:
            expected = values.get((p["model_slug"], f.get("benchmark_subset")))
            if expected is None or pd.isna(expected):
                problems.append(f"{label}: '{p['model_slug']}' has no metric value")
            elif abs(float(expected) - p["y"]) > 1e-9:
                problems.append(f"{label}: '{p['model_slug']}' y differs from metrics")

    if problems:
        raise ValueError(
            "invalid frontiers:\n" + "\n".join(f"  - {p}" for p in problems)
        )


def write_frontiers_json(frontiers: dict[str, Any], outdir: str | Path) -> Path:
    """Write to <outdir>/<benchmark>_<metric>_<experiment>_frontiers.json."""
    path = Path(outdir) / frontiers_file_name(*(frontiers[c] for c in KEY_COLUMNS))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(frontiers, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def read_frontiers_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
