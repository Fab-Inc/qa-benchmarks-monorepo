"""Result persistence for phonics benchmark runs.
Score and aggregate model results against benchmark item metadata."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from phonics_benchmark.config import ConfigError
from phonics_benchmark.labeling import get_vocabulary_set
from phonics_benchmark.parsing import parse_word_list
from phonics_benchmark.utils import require_columns


def load_results(
    file_name: str | Path,
    *,
    force: bool = False,
    require_result_id: bool = True,
) -> pd.DataFrame:
    """Load existing JSONL results, or return an empty frame.

    ``result_id`` is what lets the runner match cached rows to planned work, so
    it is required by default. Pass ``require_result_id=False`` for read-only
    analysis, which keys on ``benchmark_item_id`` instead and so can still read
    runs written before ``result_id`` existed.
    """

    path = Path(file_name)
    if force or not path.exists():
        return pd.DataFrame()

    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        return pd.DataFrame()

    results = pd.DataFrame(rows)
    if require_result_id and "result_id" not in results.columns:
        raise ConfigError(f"Existing results at {path} lack result_id")
    return results


def save_results(
    results: pd.DataFrame,
    file_name: str | Path,
) -> Path:
    """Persist benchmark results as local JSONL."""

    path = Path(file_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            key: value
            for key, value in row.items()
            if not (isinstance(value, float) and math.isnan(value))
        }
        for row in results.to_dict(orient="records")
    ]
    content = "\n".join(_json_dumps(row) for row in records)
    _write_text_atomic(path, content + ("\n" if content else ""))
    return path


def _write_text_atomic(path: Path, content: str) -> None:
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    tmp_path = Path(tmp_name)

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            tmp_file.write(content)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


@require_columns(["output", "nondecodable_words"])
def add_per_item_confusion_matrix(results: pd.DataFrame, language: str) -> pd.DataFrame:
    """Add per-item true/false positive/negative word counts.

    A "positive" is a word the model listed as non-decodable.

    ``output`` holds the model's raw response *text*, so it is parsed into a word
    list here rather than compared directly. ``parse_ok`` records whether the
    word list was read from a well-formed JSON array or scraped from prose, so a
    model that never formats its answer correctly is visible in the results.

    Words are compared lowercased: `labeling.label_nondecodable_words` lowercases
    the gold ``nondecodable_words``, so a model answering "Soko" for gold "soko"
    would otherwise be charged both a false positive and a false negative.
    """
    results = results.copy()

    if "vocab_set" not in results.columns:
        results["vocab_set"] = get_vocabulary_set(results, language)

    parsed = results["output"].apply(parse_word_list)
    results["response_parsed"] = [result.words for result in parsed]
    results["parse_ok"] = [result.parse_ok for result in parsed]

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
      ``accuracy_with_actual_nondecodable``
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
    f1_denominator = (
        2 * summary["pooled_tp"] + summary["pooled_fp"] + summary["pooled_fn"]
    )

    summary["pooled_precision"] = (summary["pooled_tp"] / predicted_words).where(
        predicted_words > 0
    )
    summary["pooled_recall"] = (summary["pooled_tp"] / actual_words).where(
        actual_words > 0
    )
    summary["pooled_f1"] = (2 * summary["pooled_tp"] / f1_denominator).where(
        f1_denominator > 0
    )

    return summary.rename_axis("model_name").reset_index()


@require_columns(["model", "num_nondecodable_words", "perfect_match"])
def accuracy_by_nondecodable_count(results: pd.DataFrame) -> pd.DataFrame:
    """Per-model, per-num_nondecodable_words accuracy (mean perfect_match).

    Reveals whether accuracy degrades as the number of non-decodable
    words increases.

    Expects a ``num_nondecodable_words`` column (see
    ``phonics_benchmark.dataset.add_analysis_columns``).
    """
    return (
        results.groupby(["model", "num_nondecodable_words"])["perfect_match"]
        .agg(accuracy="mean", n_items="count")
        .reset_index()
    )


def _sample(group, n_per_bin: int, random_state: int) -> pd.DataFrame:
    """Sample ``n_per_bin`` rows from each group, or all if fewer than that."""
    return group.sample(min(n_per_bin, len(group)), random_state=random_state)


@require_columns(["nondecodable_ratio"])
def sample_items_for_inspection(
    items: pd.DataFrame,
    *,
    bin_edges: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2),
    n_per_bin: int = 2,
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample ``n_per_bin`` items from each ``nondecodable_ratio`` bin.

    Bins are ``(edge_i, edge_i+1]``, except the first which also includes
    0.0. Intended for manual spot-checking across item difficulty — join the
    result on ``benchmark_item_id`` against a results DataFrame to pull every
    model's response for the sampled items.
    """
    items = items.copy()
    items["ratio_bin"] = pd.cut(
        items["nondecodable_ratio"], bins=bin_edges, include_lowest=True
    )
    return (
        items.groupby("ratio_bin", observed=True)
        .apply(
            _sample,
            n_per_bin=n_per_bin,
            random_state=random_state,
        )
        .reset_index(level=0)
        .reset_index(drop=True)
    )
