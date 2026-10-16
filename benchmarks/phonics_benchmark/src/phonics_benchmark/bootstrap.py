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

from phonics_benchmark.balanced_accuracy import (
    COUNTS,
    METRICS,
    metrics_from_totals,
    vocabulary_counts,
)
from phonics_benchmark.scoring import pooled_fbeta
from phonics_benchmark.utils import require_columns

# The per-item columns a bootstrap replicate needs, all written by
# `scoring.add_per_item_confusion_matrix`.
ITEM_METRIC_COLUMNS = ["perfect_match", "tp_count", "fp_count", "fn_count"]

# What each item contributes, as totals rather than rates, so every metric below
# is a ratio of two sums and a model missing an item contributes to neither side.
ITEM_TOTAL_COLUMNS = ["n_correct", "n_items", "tp_count", "fp_count", "fn_count"]


def _item_matrices(
    scored: pd.DataFrame,
    item_ids: pd.Index | None = None,
) -> tuple[pd.Index, list[str], dict[str, np.ndarray]]:
    """Pivot the per-item metrics to one (n_items, n_models) array per total.

    Returns (items, models, matrices) where
    - Each matrix corresponds to one of the ITEM_TOTAL_COLUMNS.
    - Every matrix has rows corresponding to ``items`` and columns corresponding
        to ``models``.

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
    if item_ids is not None:
        target = pd.Index(item_ids)
        if target.empty or target.hasnans or not target.is_unique:
            raise ValueError("Target item IDs must be nonempty, unique and non-null")
        if not items.isin(target).all():
            raise ValueError("Scored item IDs are outside the target dataset")
        items = target.sort_values()
    models = sorted(scored["model"].unique())
    matrices = {
        column: aggregated[column]
        .unstack("model")
        .reindex(index=items, columns=models)
        .to_numpy(dtype=float)
        for column in ITEM_TOTAL_COLUMNS
    }
    return items, models, matrices


def _metrics_from_counts(matrices: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Reduce the item-by-model matrices to one value per model per metric.

    Returns a dictionary mapping metric names to arrays of per-model metric values.

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

    def ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        """Elementwise numerator/denominator, NaN where the denominator is 0."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(denominator > 0, numerator / denominator, np.nan)

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
    item_ids: pd.Index | None = None,
) -> pd.DataFrame:
    """Per-model accuracy, precision, recall, F1 and F2 with bootstrap CIs.

    Expects the output of ``scoring.add_per_item_confusion_matrix``. The point
    estimates reproduce ``scoring.summarize_by_model``.

    Set ``seed`` to a fixed integer to make the bootstrap draws reproducible.
    Reporting supplies all target ``item_ids``, including items with no scorable
    responses. If omitted, only the observed item IDs are available to resample.
    """
    if not 0 < ci_level < 1:
        raise ValueError(f"ci_level must be between 0 and 1, got {ci_level}")
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be at least 1, got {n_bootstrap}")

    items, models, matrices = _item_matrices(scored, item_ids)
    n_items = len(items)
    point = _metrics_from_counts(matrices)
    metric_names = list(point)

    # One index array per replicate, applied to every model's column. Item
    # difficulty correlates across models (a badly worded item is hard for all of
    # them), so a shared draw preserves that correlation instead of pretending
    # each model saw its own item pool. It leaves any single model's own interval
    # unchanged, and it keeps the replicate rows comparable across models, which
    # is what a later paired test between two models needs.
    replicates = {name: np.empty((n_bootstrap, len(models))) for name in metric_names}
    for replicate, index in enumerate(item_bootstrap_draws(n_items, n_bootstrap, seed)):
        resampled = {name: matrix[index] for name, matrix in matrices.items()}
        for name, values in _metrics_from_counts(resampled).items():
            replicates[name][replicate] = values

    alpha = 1.0 - ci_level
    report = pd.DataFrame({"model": models})
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
    return report.sort_values("accuracy", ascending=False).reset_index(drop=True)


def item_bootstrap_draws(n_items: int, n_bootstrap: int, seed: int):
    """Shared item resampling for every metric; callers use sorted target IDs."""
    if n_items < 1 or n_bootstrap < 0:
        raise ValueError("Invalid item or bootstrap count")
    rng = np.random.default_rng(seed)
    for _ in range(n_bootstrap):
        yield rng.integers(0, n_items, size=n_items)


def summarize_balanced_accuracy(scored, items, n_bootstrap=1000, seed=0, ci_level=0.95):
    if n_bootstrap < 0 or not 0 < ci_level < 1:
        raise ValueError("Invalid bootstrap count or confidence level")
    scored = vocabulary_counts(scored, items)
    models = sorted(scored.model.unique())
    ids = items.sort_index().index
    matrices = np.stack(
        [
            scored.pivot(index="benchmark_item_id", columns="model", values=column)
            .reindex(index=ids, columns=models)
            .to_numpy(dtype=float)
            for column in COUNTS
        ]
    )
    totals = np.nansum(matrices, axis=1)
    point = metrics_from_totals(totals)
    replicates = np.empty((n_bootstrap, len(METRICS), len(models)))
    for i, draw in enumerate(item_bootstrap_draws(len(ids), n_bootstrap, seed)):
        replicates[i] = metrics_from_totals(np.nansum(matrices[:, draw, :], axis=1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bounds = (
            np.nanpercentile(
                replicates, [100 * (1 - ci_level) / 2, 100 * (1 + ci_level) / 2], axis=0
            )
            if n_bootstrap
            else np.full((2, len(METRICS), len(models)), np.nan)
        )
    result = pd.DataFrame({"model": models})
    result["n_scorable"] = np.isfinite(matrices[0]).sum(axis=0)
    result["expected_items"] = len(ids)
    result["coverage"] = result.n_scorable / len(ids)
    result["all_words_balanced_accuracy"] = 0.5
    for i, name in enumerate(COUNTS):
        result[name] = totals[i]
    for i, name in enumerate(METRICS):
        result[name] = point[i]
        result[f"{name}_ci_low"] = bounds[0, i]
        result[f"{name}_ci_high"] = bounds[1, i]
    result["adjusted_balanced_accuracy_scale"] = 100
    result["n_bootstrap"], result["seed"], result["ci_level"] = (
        n_bootstrap,
        seed,
        ci_level,
    )
    result["n_valid_bootstrap"] = np.isfinite(replicates[:, 2]).sum(axis=0)
    result["vocabulary_evaluated"] = totals[:4].sum(axis=0)
    return result.sort_values("balanced_accuracy", ascending=False), scored
