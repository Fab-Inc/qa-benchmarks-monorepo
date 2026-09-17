"""Score model results against the benchmark items' gold non-decodable words."""

import math
from typing import Any

import pandas as pd

from phonics_benchmark.labeling import get_vocabulary_set
from phonics_benchmark.utils import require_columns


def pooled_fbeta(
    tp: pd.Series,
    fp: pd.Series,
    fn: pd.Series,
    beta: float,
) -> pd.Series:
    """F-beta from pooled word counts, only defined where the denominator is non-zero.

    Takes vectors, not single counts.
    """
    denominator = (1 + beta**2) * tp + beta**2 * fn + fp
    numerator = (1 + beta**2) * tp

    return (numerator / denominator).where(denominator > 0)


@require_columns(["response_parsed", "nondecodable_words"])
def add_per_item_confusion_matrix(results: pd.DataFrame, language: str) -> pd.DataFrame:
    """Add per-item true/false positive/negative word counts.

    A "positive" is a word the model listed as non-decodable.

    Expects ``response_parsed`` — the model's raw ``output`` text already parsed
    into a word list by ``parsing.parse_word_list``. Scoring the raw text instead
    would build a set of its characters.

    Words are compared lowercased: `labeling.label_nondecodable_words` lowercases
    the gold ``nondecodable_words``, so a model answering "Soko" for gold "soko"
    would otherwise be charged both a false positive and a false negative.
    """
    results = results.copy()

    if "vocab_set" not in results.columns:
        results["vocab_set"] = get_vocabulary_set(results, language)

    def get_confusion_matrix(row: pd.Series) -> pd.Series:
        predicted = {word.lower() for word in row["response_parsed"]}
        actual = {word.lower() for word in row["nondecodable_words"]}
        perfect_match = predicted == actual
        tp = predicted & actual
        fp = predicted - actual
        fn = actual - predicted
        tn = row["vocab_set"] - predicted - actual

        n_tp, n_fp, n_fn = len(tp), len(fp), len(fn)

        precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else math.nan
        recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else math.nan

        f1_denominator = 2 * n_tp + n_fp + n_fn
        f1 = 2 * n_tp / f1_denominator if f1_denominator > 0 else math.nan

        return pd.Series(
            {
                "perfect_match": perfect_match,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "tp_count": n_tp,
                "fp_count": n_fp,
                "fn_count": n_fn,
                "tn_count": len(tn),
                "has_actual_nondecodable": len(actual) > 0,
                "has_predicted_nondecodable": len(predicted) > 0,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    return pd.concat([results, results.apply(get_confusion_matrix, axis=1)], axis=1)


@require_columns(
    [
        "model",
        "benchmark_item_id",
        "perfect_match",
        "tp_count",
        "fp_count",
        "fn_count",
        "has_actual_nondecodable",
        "has_predicted_nondecodable",
        "precision",
        "recall",
        "f1",
    ]
)
def summarize_by_model(results: pd.DataFrame) -> pd.DataFrame:
    """Per-model macro (item-averaged) and pooled (word-level) metrics.

    A *fully decodable* item is one whose actual ``nondecodable_words`` list is
    empty — every word in it is decodable, so the correct prediction is an empty
    list.

    Each metric is computed over only the items where it is defined:

    * Items with >=1 actual non-decodable word: ``macro_recall``, ``macro_f1``,
      ``accuracy_for_not_fully_decodable``
    * Items with >=1 predicted non-decodable word: ``macro_precision``
    * Fully decodable items: ``false_alarm_rate_fully_decodable``,
      ``mean_fp_per_fully_decodable_item``
    * All items: ``accuracy``, ``pooled_*``

    Expects the output of ``add_per_item_confusion_matrix``.
    """
    grouped = results.groupby("model")
    models = grouped.size().index

    not_fully_decodable_items = results[results["has_actual_nondecodable"]].groupby(
        "model"
    )
    fully_decodable_items = results[~results["has_actual_nondecodable"]].groupby(
        "model"
    )
    items_with_predicted_nondecodable = results[
        results["has_predicted_nondecodable"]
    ].groupby("model")

    def count(subset: Any) -> pd.Series:
        """Item count per model, 0 (not NaN) for models absent from the subset."""
        counts = subset["benchmark_item_id"].count()
        return counts.reindex(models, fill_value=0).astype(int)

    summary = pd.DataFrame(
        {
            "n_items": count(grouped),
            "n_items_not_fully_decodable": count(not_fully_decodable_items),
            "n_items_fully_decodable": count(fully_decodable_items),
            "accuracy": grouped["perfect_match"].mean(),
            "accuracy_for_not_fully_decodable": not_fully_decodable_items[
                "perfect_match"
            ].mean(),
            "macro_precision": items_with_predicted_nondecodable["precision"].mean(),
            "macro_recall": not_fully_decodable_items["recall"].mean(),
            "macro_f1": not_fully_decodable_items["f1"].mean(),
            "false_alarm_rate_fully_decodable": fully_decodable_items[
                "has_predicted_nondecodable"
            ].mean(),
            "mean_fp_per_fully_decodable_item": fully_decodable_items[
                "fp_count"
            ].mean(),
            "pooled_tp": grouped["tp_count"].sum(),
            "pooled_fp": grouped["fp_count"].sum(),
            "pooled_fn": grouped["fn_count"].sum(),
        }
    )

    predicted_words = summary["pooled_tp"] + summary["pooled_fp"]
    actual_words = summary["pooled_tp"] + summary["pooled_fn"]

    summary["pooled_precision"] = (summary["pooled_tp"] / predicted_words).where(
        predicted_words > 0
    )
    summary["pooled_recall"] = (summary["pooled_tp"] / actual_words).where(
        actual_words > 0
    )
    for beta in (1, 2):
        summary[f"pooled_f{beta}"] = pooled_fbeta(
            summary["pooled_tp"],
            summary["pooled_fp"],
            summary["pooled_fn"],
            beta=beta,
        )

    return summary.rename_axis("model_name").reset_index()


@require_columns(["model", "num_nondecodable_words", "perfect_match"])
def accuracy_by_nondecodable_count(results: pd.DataFrame) -> pd.DataFrame:
    """Per-model, per-num_nondecodable_words accuracy (mean perfect_match).

    Reveals whether accuracy degrades as the number of non-decodable
    words increases.

    Expects a ``num_nondecodable_words`` column (see
    ``phonics_benchmark.curation.add_labels_and_analysis``).
    """
    return (
        results.groupby(["model", "num_nondecodable_words"])["perfect_match"]
        .agg(accuracy="mean", n_items="count")
        .reset_index()
    )
