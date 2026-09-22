"""Pooled word-level balanced accuracy on local phonics snapshots.

Evaluate a fixed universe: each item's unique vocabulary words. The existing
F-score reports also count predicted words absent from the text as FP; record
those separately and include the original-count variant as a sensitivity check.
Adjusted balanced accuracy is reported in points: 100 * (2 * BA - 1),
with chance at 0, perfect at 100, and negative values retained.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from phonics_benchmark.labeling import get_vocabulary_set

COUNTS = ["tp_count", "fn_count", "tn_count", "fp_in_vocabulary", "extra_predictions"]
METRICS = [
    "recall",
    "specificity",
    "balanced_accuracy",
    "adjusted_balanced_accuracy",
    "false_positive_rate",
    "balanced_accuracy_using_original_fp",
]


def load_items(path: Path, language: str) -> pd.DataFrame:
    items = pd.read_json(path, lines=True).sort_values("benchmark_item_id")
    if items.benchmark_item_id.isna().any() or not items.benchmark_item_id.is_unique:
        raise ValueError(f"Non-unique or missing item IDs: {path}")
    items = items.set_index("benchmark_item_id", drop=False)
    items["vocab_set"] = get_vocabulary_set(items, language)
    items["gold_set"] = items.nondecodable_words.map(
        lambda words: {word.lower() for word in words}
    )
    if any(gold - vocab for gold, vocab in zip(items.gold_set, items.vocab_set)):
        raise ValueError(f"Gold words absent from item vocabulary: {path}")
    items["n_vocab"] = items.vocab_set.map(len)
    items["n_gold"] = items.gold_set.map(len)
    if (items.n_vocab <= 0).any():
        raise ValueError(f"Empty vocabulary: {path}")
    return items


def divide(a, b):
    return np.divide(a, b, out=np.full(np.shape(a), np.nan), where=b > 0)


def metrics_from_totals(totals):
    tp, fn, tn, fp, extra = np.asarray(totals, dtype=float)
    recall = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    balanced = (recall + specificity) / 2
    return np.array(
        [
            recall,
            specificity,
            balanced,
            100 * (2 * balanced - 1),
            1 - specificity,
            (recall + divide(tn, tn + fp + extra)) / 2,
        ]
    )


def vocabulary_counts(scored, items):
    if scored.duplicated(["model", "benchmark_item_id"]).any():
        raise ValueError("Duplicate model/item pairs")
    if set(scored.benchmark_item_id) - set(items.index):
        raise ValueError("Unknown scored items")
    result = scored.merge(
        items[["n_vocab", "n_gold"]],
        left_on="benchmark_item_id",
        right_index=True,
        validate="many_to_one",
    )
    np.testing.assert_array_equal(result.tp_count + result.fn_count, result.n_gold)
    result["fp_in_vocabulary"] = result.n_vocab - result.n_gold - result.tn_count
    result["extra_predictions"] = result.fp_count - result.fp_in_vocabulary
    if result[COUNTS].isna().any().any() or (result[COUNTS] < 0).any().any():
        raise ValueError("Invalid vocabulary confusion counts")
    return result


def summarize(scored, items, n_bootstrap=1000, seed=0):
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
    rng = np.random.default_rng(seed)
    replicates = np.empty((n_bootstrap, len(METRICS), len(models)))
    for i in range(n_bootstrap):
        draw = rng.integers(0, len(ids), size=len(ids))
        replicates[i] = metrics_from_totals(np.nansum(matrices[:, draw, :], axis=1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bounds = np.nanpercentile(replicates, [2.5, 97.5], axis=0)
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
    result["n_bootstrap"], result["seed"], result["ci_level"] = n_bootstrap, seed, 0.95
    result["n_valid_bootstrap"] = np.isfinite(replicates[:, 2]).sum(axis=0)
    result["vocabulary_evaluated"] = totals[:4].sum(axis=0)
    return result.sort_values("balanced_accuracy", ascending=False), scored
