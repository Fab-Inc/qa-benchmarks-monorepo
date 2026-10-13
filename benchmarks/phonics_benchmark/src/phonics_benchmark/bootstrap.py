"""Bootstrap confidence intervals for a run's per-model metrics.

To get a sense of the uncertainty in a model's performance, we can use
bootstrap resampling. We resample from the benchmark items with replacement
to create new pseudo-datasets.

Covers five of the metrics ``scoring.summarize_by_model`` reports:

* ``accuracy``
* ``pooled_precision``
* ``pooled_recall``
* ``pooled_f1``
* ``pooled_f2``
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from phonics_benchmark.scoring import (
    BASELINE_MODEL_NAMES,
    build_baseline_results,
    pooled_fbeta,
)
from phonics_benchmark.utils import require_columns

ITEM_METRIC_COLUMNS = ["perfect_match", "tp_count", "fp_count", "fn_count"]
ITEM_TOTAL_COLUMNS = ["n_correct", "n_items", "tp_count", "fp_count", "fn_count"]


def _item_matrices(
    scored: pd.DataFrame,
) -> tuple[pd.Index, list[str], dict[str, np.ndarray]]:
    """Pivot the per-item metrics to one (n_items, n_models) array per total.

    Returns ``(items, models, matrices)`` where
    - ``items``: the unique benchmark item IDs.
    - ``models``: the unique model names, sorted.
    - ``matrices``: a dictionary mapping each column in ``ITEM_TOTAL_COLUMNS``
      to a 2D array of shape (n_items, n_models), with rows corresponding to
      ``items`` and columns corresponding to ``models``.

    A NaN cell means that model has no row for that item, e.g. due to a failed
    provider call or an unparseable response. Callers sum with `nansum`, so a
    model is scored over the items it does have.
    """
    aggregated = scored.groupby(["benchmark_item_id", "model"], sort=True).agg(
        n_correct=("perfect_match", "sum"),
        n_items=("perfect_match", "count"),
        tp_count=("tp_count", "sum"),
        fp_count=("fp_count", "sum"),
        fn_count=("fn_count", "sum"),
    )

    items = aggregated.index.get_level_values("benchmark_item_id").unique()
    models = sorted(scored["model"].unique())
    matrices = {
        column: aggregated[column]
        .unstack("model")
        .reindex(index=items, columns=models)
        .to_numpy(dtype=float)
        for column in ITEM_TOTAL_COLUMNS
    }
    return items, models, matrices


def ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Elementwise numerator/denominator, NaN where the denominator is 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator > 0, numerator / denominator, np.nan)


def _metrics_from_counts(matrices: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Reduce the item-by-model matrices to one value per model per metric.

    Returns a dictionary of {metric names: arrays of per-model metric values}.

    Every metric is a ratio of two sums over the drawn items, so a model with no
    row for a drawn item contributes nothing to either side rather than skewing an
    average. A replicate that draws only items a model is missing leaves that
    model with a zero denominator, and its metric is NaN for that replicate;
    `np.nanpercentile` skips it when the interval is taken.
    """
    totals = {
        column: np.nansum(matrices[column], axis=0) for column in ITEM_TOTAL_COLUMNS
    }
    tp, fp, fn = totals["tp_count"], totals["fp_count"], totals["fn_count"]

    return {
        "accuracy": ratio(totals["n_correct"], totals["n_items"]),
        # Undefined rather than 0 when a model flagged no words at all, matching
        # summarize_by_model: it predicted nothing, so it was not wrong about
        # anything either.
        "pooled_precision": ratio(tp, tp + fp),
        # Undefined when no item in the draw had a non-decodable word to find.
        "pooled_recall": ratio(tp, tp + fn),
        "pooled_f1": np.asarray(pooled_fbeta(tp, fp, fn, beta=1), dtype=float),
        "pooled_f2": np.asarray(pooled_fbeta(tp, fp, fn, beta=2), dtype=float),
    }


@require_columns(["model", "benchmark_item_id", *ITEM_METRIC_COLUMNS])
def bootstrap_metric_cis(
    scored: pd.DataFrame,
    *,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 0,
    add_baseline: bool = False,
    language: str | None = None,
) -> pd.DataFrame:
    """Per-model accuracy, precision, recall, F1 and F2 with bootstrap CIs.

    Expects the output of ``scoring.add_per_item_confusion_matrix``. The point
    estimates reproduce ``scoring.summarize_by_model``.

    - ``n_bootstrap``: the number of bootstrap replicates to draw.
    - ``ci_level``: the confidence level for the bootstrap intervals.
    - ``seed``: a fixed integer to make the bootstrap draws reproducible.
    - ``add_baseline``: whether to add baseline rows for comparison.
    - ``language``: the language of the dataset, required if ``add_baseline`` is True.
    """
    if not 0 < ci_level < 1:
        raise ValueError(f"ci_level must be between 0 and 1, got {ci_level}")
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be at least 1, got {n_bootstrap}")

    if add_baseline:
        if language is None:
            raise ValueError("language must be specified when add_baseline is True")
        scored = pd.concat(
            [scored, build_baseline_results(scored, language)], ignore_index=True
        )

    item_ids, model_names, item_model_matrices = _item_matrices(scored)
    n_items = len(item_ids)
    point = _metrics_from_counts(item_model_matrices)
    metric_names = list(point)

    rng = np.random.default_rng(seed)
    # One index array per replicate, applied to every model's column. Item
    # difficulty correlates across models (a badly worded item is hard for all of
    # them), so a shared draw preserves that correlation instead of pretending
    # each model saw its own item pool. It leaves any single model's own interval
    # unchanged, and it keeps the replicate rows comparable across models, which
    # is what a later paired test between two models needs.
    replicates = {
        name: np.empty((n_bootstrap, len(model_names))) for name in metric_names
    }
    for replicate in range(n_bootstrap):
        index = rng.integers(0, n_items, size=n_items)  # with replacement, same n
        resampled = {
            name: matrix[index] for name, matrix in item_model_matrices.items()
        }
        for name, values in _metrics_from_counts(resampled).items():
            replicates[name][replicate] = values

    alpha = 1.0 - ci_level
    report = pd.DataFrame({"model": model_names})
    with warnings.catch_warnings():  # all-NaN columns, see _metrics_from_counts
        warnings.simplefilter("ignore", RuntimeWarning)
        for name in metric_names:
            report[name] = point[name]
            report[f"{name}_ci_low"] = np.nanpercentile(
                replicates[name], 100 * alpha / 2, axis=0
            )
            report[f"{name}_ci_high"] = np.nanpercentile(
                replicates[name], 100 * (1 - alpha / 2), axis=0
            )

    report["n_items_resampled"] = n_items
    report["n_bootstrap"] = n_bootstrap
    report["ci_level"] = ci_level

    if add_baseline:
        report.insert(1, "is_baseline", report["model"].isin(BASELINE_MODEL_NAMES))

    return report.reset_index(drop=True)
