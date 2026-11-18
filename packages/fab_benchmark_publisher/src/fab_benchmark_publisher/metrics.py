"""The metrics file: item scores in, one CSV per (benchmark, metric, experiment).

    <benchmark_slug>_<metric_slug>_<experiment_slug>.csv

Those three slugs are the composite key: every row of a file carries the same three.
Within a file each model has at most one row per benchmark_subset. The overall result
is the row with an empty benchmark_subset; any other row is the metric restricted to
one subset, stored as canonical JSON ({"subject":"History"}).

Uncertainty comes from a non-parametric bootstrap over items: resample the scored
items with replacement, take the mean, repeat. metric_stderr is the standard
deviation of those means and metric_ci_low/high are their percentiles. Each row's
random stream is seeded from (seed, model_slug, benchmark_subset), so adding a model
or a subset never moves the intervals of rows that were already there.

The token and cost columns are part of the schema but aggregate_item_scores leaves
them empty: how to derive them consistently across providers is still open.

NOTHING HERE NAMES A BENCHMARK. Each benchmark's adapter turns its own outputs into
the item-scores frame described in aggregate_item_scores.
"""

from __future__ import annotations

import json
import zlib
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fab_benchmark_publisher.result_metadata import MetricMeta, check_slug

KEY_COLUMNS = ("benchmark_slug", "metric_slug", "experiment_slug")

# Left empty for now -- see the module docstring.
USAGE_COLUMNS = (
    "input_tokens_per_repeat_mean",
    "output_total_tokens_per_repeat_mean",
    "output_reason_tokens_per_repeat_mean",
    "output_non_reason_tokens_per_repeat_mean",
    "cost_usd_per_repeat_mean",
    "cost_usd_per_item_mean",
)

METRICS_COLUMNS = (
    *KEY_COLUMNS,
    "model_slug",
    "benchmark_subset",
    "metric_value",
    "metric_stderr",
    "metric_ci_low",
    "metric_ci_high",
    "n_items_scored",
    "n_items_bad_format",
    "n_repeats",
    *USAGE_COLUMNS,
    "timestamp_metric",
)

# At most one row per combination of these.
UNIQUE_COLUMNS = (*KEY_COLUMNS, "model_slug", "benchmark_subset")

_TEXT_COLUMNS = (*KEY_COLUMNS, "model_slug", "benchmark_subset", "timestamp_metric")
_COUNT_COLUMNS = ("n_items_scored", "n_items_bad_format", "n_repeats")
# Undefined (empty) exactly when no item was scored.
_ESTIMATE_COLUMNS = ("metric_value", "metric_stderr", "metric_ci_low", "metric_ci_high")

OVERALL = ""
DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_CI_LEVEL = 0.95

_TOLERANCE = 1e-9
_DECIMALS = 4


def metrics_file_name(
    benchmark_slug: str, metric_slug: str, experiment_slug: str
) -> str:
    for value, name in zip((benchmark_slug, metric_slug, experiment_slug), KEY_COLUMNS):
        check_slug(value, name)
    return f"{benchmark_slug}_{metric_slug}_{experiment_slug}.csv"


def canonical_subset(subset: dict) -> str:
    """The one string a subset may be stored as: "" for the overall result, else
    JSON with sorted keys, no whitespace and string values. A single form is what
    lets the uniqueness check catch {"a":"1","b":"2"} vs {"b": "2", "a": "1"}."""
    if not subset:
        return OVERALL
    return json.dumps(
        {str(k): str(v) for k, v in subset.items()},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def bootstrap_mean(
    scores: np.ndarray,
    *,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci_level: float = DEFAULT_CI_LEVEL,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """(stderr, ci_low, ci_high) of the mean of `scores`, resampling with
    replacement. Percentile interval."""
    n = len(scores)
    if n == 0:
        raise ValueError("cannot bootstrap an empty sample")
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    means = scores[idx].mean(axis=1)
    alpha = (1.0 - ci_level) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    stderr = float(means.std(ddof=1)) if n_bootstrap > 1 else 0.0
    return stderr, float(low), float(high)


def _row_rng(seed: int, model_slug: str, subset: str) -> np.random.Generator:
    key = zlib.crc32(f"{model_slug}\x00{subset}".encode())
    return np.random.default_rng(np.random.SeedSequence([seed, key]))


def aggregate_item_scores(
    item_scores: pd.DataFrame,
    *,
    benchmark_slug: str,
    metric: MetricMeta,
    experiment_slug: str,
    subset_groupings: Sequence[Sequence[str]] = (),
    scale: float = 100.0,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci_level: float = DEFAULT_CI_LEVEL,
    seed: int = 0,
    timestamp_metric: str | None = None,
) -> pd.DataFrame:
    """Item scores -> one metrics file's table (validated).

    `item_scores` has one row per (model_slug, item_id) -- one repeat per item, so
    n_repeats is 1 -- with columns:

      model_slug      model id as in models.csv
      item_id         unique per model
      item_score      the item's score (e.g. 0/1). May be empty for a bad-format
                      item when metric.is_bad_format_scored is False.
      is_bad_format   optional bool, default False
      <subset col>    any columns named in `subset_groupings`

    One overall row is produced per model, plus one row per model per distinct value
    combination of each grouping: [["domain"], ["grade"]] gives per-domain and
    per-grade rows, [["domain", "grade"]] gives per-(domain, grade) rows.

    metric_value and its interval are the mean item score times `scale` (100 turns a
    0/1 mean into a percentage). timestamp_metric defaults to now, in UTC.
    """
    df = item_scores.copy()
    missing = [c for c in ("model_slug", "item_id", "item_score") if c not in df]
    grouping_cols = sorted({c for g in subset_groupings for c in g})
    missing += [c for c in grouping_cols if c not in df]
    if missing:
        raise ValueError(f"item_scores is missing column(s): {', '.join(missing)}")
    if any(len(g) == 0 for g in subset_groupings):
        raise ValueError("an empty subset grouping would duplicate the overall row")
    if "is_bad_format" not in df:
        df["is_bad_format"] = False
    df["is_bad_format"] = df["is_bad_format"].fillna(False).astype(bool)
    df["model_slug"] = df["model_slug"].astype(str)

    dupes = df[df.duplicated(["model_slug", "item_id"], keep=False)]
    if not dupes.empty:
        pairs = sorted(set(zip(dupes["model_slug"], dupes["item_id"].astype(str))))
        raise ValueError(f"duplicate (model_slug, item_id) rows: {pairs[:10]}")

    must_have_score = (
        pd.Series(True, index=df.index)
        if metric.is_bad_format_scored
        else ~df["is_bad_format"]
    )
    unscored = df[must_have_score & df["item_score"].isna()]
    if not unscored.empty:
        first = unscored.iloc[0]
        raise ValueError(
            f"{len(unscored)} item(s) have no item_score but must be scored, e.g. "
            f"model '{first['model_slug']}' item '{first['item_id']}'"
        )

    timestamp_metric = timestamp_metric or datetime.now(UTC).isoformat()
    rows: list[dict] = []
    for grouping in [(), *(tuple(g) for g in subset_groupings)]:
        keys = ["model_slug", *grouping]
        for key, group in df.groupby(keys, sort=True, dropna=True):
            key = key if isinstance(key, tuple) else (key,)
            model_slug = str(key[0])
            subset = canonical_subset(dict(zip(grouping, key[1:])))
            rows.append(
                {
                    "benchmark_slug": benchmark_slug,
                    "metric_slug": metric.metric_slug,
                    "experiment_slug": experiment_slug,
                    "model_slug": model_slug,
                    "benchmark_subset": subset,
                    **_estimate(
                        group,
                        is_bad_format_scored=metric.is_bad_format_scored,
                        scale=scale,
                        n_bootstrap=n_bootstrap,
                        ci_level=ci_level,
                        rng=_row_rng(seed, model_slug, subset),
                    ),
                    "n_repeats": 1,
                    **dict.fromkeys(USAGE_COLUMNS),
                    "timestamp_metric": timestamp_metric,
                }
            )

    metrics = pd.DataFrame(rows, columns=list(METRICS_COLUMNS))
    metrics = metrics.sort_values(
        ["model_slug", "benchmark_subset"], kind="stable"
    ).reset_index(drop=True)
    validate_metrics(metrics, metric=metric)
    return metrics


def _estimate(
    group: pd.DataFrame,
    *,
    is_bad_format_scored: bool,
    scale: float,
    n_bootstrap: int,
    ci_level: float,
    rng: np.random.Generator,
) -> dict:
    n_bad = int(group["is_bad_format"].sum())
    scored = group if is_bad_format_scored else group[~group["is_bad_format"]]
    scores = scored["item_score"].to_numpy(dtype=float)

    value = stderr = low = high = None
    if len(scores):
        stderr, low, high = bootstrap_mean(
            scores, n_bootstrap=n_bootstrap, ci_level=ci_level, rng=rng
        )
        value = round(float(scores.mean()) * scale, _DECIMALS)
        stderr = round(stderr * scale, _DECIMALS)
        low = round(low * scale, _DECIMALS)
        high = round(high * scale, _DECIMALS)
    return {
        "metric_value": value,
        "metric_stderr": stderr,
        "metric_ci_low": low,
        "metric_ci_high": high,
        "n_items_scored": len(scores),
        "n_items_bad_format": n_bad,
    }


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_metrics(
    metrics: pd.DataFrame, *, metric: MetricMeta | None = None
) -> None:
    """Raise ValueError listing every way `metrics` breaks the file's contract.

    Pass `metric` to also check that the file is about that metric and that values
    fit its unit ("%" must be 0-100).
    """
    problems: list[str] = []

    missing = [c for c in METRICS_COLUMNS if c not in metrics.columns]
    extra = [c for c in metrics.columns if c not in METRICS_COLUMNS]
    if missing:
        problems.append(f"missing column(s): {', '.join(missing)}")
    if extra:
        problems.append(f"unexpected column(s): {', '.join(map(str, extra))}")
    if problems:
        _raise(problems)
    if metrics.empty:
        _raise(["no rows"])

    for col in (*KEY_COLUMNS, "model_slug", "timestamp_metric"):
        bad = metrics[col].isna() | (metrics[col].astype(str).str.strip() == "")
        if bad.any():
            problems.append(f"{col} is empty in {int(bad.sum())} row(s)")

    for col in KEY_COLUMNS:
        values = metrics[col].dropna().unique()
        if len(values) > 1:
            problems.append(
                f"one {col} per file (it is part of the file name), found: "
                + ", ".join(map(str, values))
            )
        for v in values:
            try:
                check_slug(v, col)
            except ValueError as e:
                problems.append(str(e))
    if metric is not None and set(metrics["metric_slug"]) != {metric.metric_slug}:
        problems.append(f"file is not about metric '{metric.metric_slug}'")

    subsets = metrics["benchmark_subset"].fillna(OVERALL)
    for raw in subsets.unique():
        if raw == OVERALL:
            continue
        try:
            parsed = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            problems.append(f"benchmark_subset is neither empty nor JSON: {raw!r}")
            continue
        if not isinstance(parsed, dict):
            problems.append(f"benchmark_subset is not a JSON object: {raw!r}")
        elif canonical_subset(parsed) != raw:
            problems.append(
                f"benchmark_subset {raw!r} is not canonical; expected "
                f"{canonical_subset(parsed)!r}"
            )

    keyed = metrics.assign(benchmark_subset=subsets)
    dupes = keyed[keyed.duplicated(list(UNIQUE_COLUMNS), keep=False)]
    if not dupes.empty:
        first = dupes.iloc[0]
        problems.append(
            f"{len(dupes)} row(s) repeat a model's result for one key, e.g. "
            f"model_slug={first['model_slug']!r} "
            f"benchmark_subset={first['benchmark_subset']!r}"
        )

    counts_ok = True
    for col in _COUNT_COLUMNS:
        values = pd.to_numeric(metrics[col], errors="coerce")
        if values.isna().any() or (values < 0).any() or (values % 1 != 0).any():
            problems.append(f"{col} must be a non-negative integer in every row")
            counts_ok = False
    if counts_ok and (pd.to_numeric(metrics["n_repeats"]) < 1).any():
        problems.append("n_repeats must be at least 1")

    numeric = {
        c: pd.to_numeric(metrics[c], errors="coerce")
        for c in (*_ESTIMATE_COLUMNS, *USAGE_COLUMNS)
    }
    for col, values in numeric.items():
        non_numeric = values.isna() & metrics[col].notna()
        if non_numeric.any():
            problems.append(f"{col} is not numeric in {int(non_numeric.sum())} row(s)")
    for col in USAGE_COLUMNS:
        if (numeric[col] < 0).any():
            problems.append(f"{col} is negative")
    reason = numeric["output_reason_tokens_per_repeat_mean"]
    non_reason = numeric["output_non_reason_tokens_per_repeat_mean"]
    total = numeric["output_total_tokens_per_repeat_mean"]
    split = total.notna() & reason.notna() & non_reason.notna()
    if ((reason + non_reason - total).abs()[split] > 1e-6).any():
        problems.append("output reason + non-reason tokens do not add up to total")

    if counts_ok:
        no_items = pd.to_numeric(metrics["n_items_scored"]) == 0
        for col in _ESTIMATE_COLUMNS:
            wrong = numeric[col].isna() != no_items
            if wrong.any():
                problems.append(
                    f"{col} must be empty exactly when n_items_scored is 0 "
                    f"({int(wrong.sum())} row(s) disagree)"
                )

    value = numeric["metric_value"]
    low, high = numeric["metric_ci_low"], numeric["metric_ci_high"]
    outside = (low > value + _TOLERANCE) | (value > high + _TOLERANCE)
    if outside.any():
        row = metrics[outside].iloc[0]
        problems.append(
            f"{int(outside.sum())} row(s) have metric_value outside "
            f"[metric_ci_low, metric_ci_high], e.g. model '{row['model_slug']}'"
        )
    if (numeric["metric_stderr"] < 0).any():
        problems.append("metric_stderr is negative")
    if metric is not None and metric.unit == "%":
        if ((value < -_TOLERANCE) | (value > 100 + _TOLERANCE)).any():
            problems.append("a '%' metric_value lies outside 0-100")

    for raw in metrics["timestamp_metric"].dropna().unique():
        try:
            parsed_ts = datetime.fromisoformat(str(raw))
        except ValueError:
            problems.append(f"timestamp_metric is not ISO 8601: {raw!r}")
            continue
        offset = parsed_ts.utcoffset()
        if offset is None or offset.total_seconds():
            problems.append(f"timestamp_metric is not UTC: {raw!r}")

    if problems:
        _raise(problems)


def _raise(problems: list[str]) -> None:
    raise ValueError(
        "invalid metrics table:\n" + "\n".join(f"  - {p}" for p in problems)
    )


# --------------------------------------------------------------------------- #
# CSV I/O
# --------------------------------------------------------------------------- #
def write_metrics_csv(
    metrics: pd.DataFrame, outdir: str | Path, *, metric: MetricMeta | None = None
) -> Path:
    """Validate, then write to <outdir>/<benchmark>_<metric>_<experiment>.csv."""
    validate_metrics(metrics, metric=metric)
    first = metrics.iloc[0]
    path = Path(outdir) / metrics_file_name(*(first[c] for c in KEY_COLUMNS))
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics[list(METRICS_COLUMNS)].to_csv(path, index=False, encoding="utf-8")
    return path


def read_metrics_csv(
    path: str | Path, *, metric: MetricMeta | None = None
) -> pd.DataFrame:
    """Read a metrics CSV back without pandas' type guessing mangling it ("1.0"
    stays a string, an empty subset stays ""), and check that its contents match
    its file name."""
    path = Path(path)
    numeric = [c for c in METRICS_COLUMNS if c not in _TEXT_COLUMNS]
    metrics = pd.read_csv(
        path,
        dtype={c: str for c in _TEXT_COLUMNS},
        keep_default_na=False,
        na_values={c: [""] for c in numeric},
        encoding="utf-8",
    )
    validate_metrics(metrics, metric=metric)
    expected = metrics_file_name(*(metrics.iloc[0][c] for c in KEY_COLUMNS))
    if path.name != expected:
        raise ValueError(f"{path.name}: contents say the file should be {expected}")
    return metrics
