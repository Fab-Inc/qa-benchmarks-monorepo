"""Bootstrap confidence intervals for a run's per-model metrics.

Each model answered the same fixed set of benchmark items, so its score is an
estimate over one sample of items rather than a fixed property of the model.
Resampling those items with replacement and re-scoring gives the spread that
estimate would show across other samples of the same size, and the percentiles
of that spread are the confidence interval.

Covers three of the metrics ``scoring.summarize_by_model`` reports:

* ``accuracy`` -- the share of items the model got exactly right
* ``pooled_f1`` -- word-level micro-averaged F1 over all items
* ``pooled_f2`` -- the same weighting a missed word four times a false alarm

Two limits to state wherever these numbers are reported:

1. The interval covers item-sampling error alone. It says nothing about
   run-to-run variation in the model's own output, and nothing about whether the
   item set represents Kiswahili texts in general.
2. Each ``benchmark_item_id`` counts as one independent draw. One text backs
   several items, one per student-knowledge profile scored against it, and in the
   20260905 run a single text backs 142 of the 1,665 items. Where a model tends
   to get all of a text's items right or all of them wrong, those items carry
   less information than their count suggests and the interval comes out too
   narrow. Drawing whole texts instead would answer that, and is not implemented
   here.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

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
    n_items = totals["n_items"]
    with np.errstate(divide="ignore", invalid="ignore"):
        accuracy = np.where(n_items > 0, totals["n_correct"] / n_items, np.nan)
    f1 = pooled_fbeta(
        totals["tp_count"], totals["fp_count"], totals["fn_count"], beta=1
    )
    f2 = pooled_fbeta(
        totals["tp_count"], totals["fp_count"], totals["fn_count"], beta=2
    )
    return {
        "accuracy": accuracy,
        "pooled_f1": np.asarray(f1, dtype=float),
        "pooled_f2": np.asarray(f2, dtype=float),
    }


@require_columns(["model", "benchmark_item_id", *ITEM_METRIC_COLUMNS])
def bootstrap_metric_cis(
    scored: pd.DataFrame,
    *,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """Per-model accuracy, pooled F1 and pooled F2 with percentile bootstrap CIs.

    Expects the output of ``scoring.add_per_item_confusion_matrix``. The point
    estimates reproduce ``scoring.summarize_by_model``.

    Set ``seed`` to a fixed integer to make the bootstrap draws reproducible.
    """
    if not 0 < ci_level < 1:
        raise ValueError(f"ci_level must be between 0 and 1, got {ci_level}")
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be at least 1, got {n_bootstrap}")

    items, models, matrices = _item_matrices(scored)
    n_items = len(items)
    point = _metrics_from_counts(matrices)
    metric_names = list(point)

    # One index array per replicate, applied to every model's column. Item
    # difficulty correlates across models (a badly worded item is hard for all of
    # them), so a shared draw preserves that correlation instead of pretending
    # each model saw its own item pool. It leaves any single model's own interval
    # unchanged, and it keeps the replicate rows comparable across models, which
    # is what a later paired test between two models needs.
    rng = np.random.default_rng(seed)
    replicates = {name: np.empty((n_bootstrap, len(models))) for name in metric_names}
    for replicate in range(n_bootstrap):
        index = rng.integers(0, n_items, size=n_items)  # with replacement, same n
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
