"""Result persistence for phonics benchmark runs."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from phonics_benchmark.config import ConfigError


def load_results(file_name: str | Path, *, force: bool = False) -> pd.DataFrame:
    """Load existing JSONL results, or return an empty frame."""

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
    if "result_id" not in results.columns:
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
"""Score and aggregate model results against benchmark item metadata.

Pairs each result row (one model's response to one benchmark item) with the
item's non-decodable-word ratio and text vocabulary, so accuracy and
word-level confusion counts can be sliced by item difficulty. Vocabulary is
recomputed from ``content`` via the language ``extract_words`` rather than trusting
``num_total_words`` — that column counts word *occurrences* (repeats
included), whereas confusion counts need the number of distinct word types in
the text.
"""
"""Score and aggregate model results against benchmark item metadata."""

from pathlib import Path

import pandas as pd

from phonics_benchmark.utils import require_columns


@require_columns(["response_parsed", "non_decodable_words", "vocab_set"])
def add_word_level_confusion_matrix(results: pd.DataFrame) -> pd.DataFrame:
    """Add per-row true/false positive/negative word counts.

    A "positive" is a word the model listed as non-decodable.

    Expects ``response_parsed``, ``non_decodable_words`` and ``vocab_set`` columns
    """

    def get_confusion_matrix(row: pd.Series) -> pd.Series:
        predicted = set(row["response_parsed"])
        actual = set(row["non_decodable_words"])
        perfect_match = predicted == actual
        tp = predicted & actual
        fp = predicted - actual
        fn = actual - predicted
        tn = row["vocab_set"] - predicted - actual
        return pd.Series(
            {
                "perfect_match": perfect_match,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "tp_count": len(tp),
                "fp_count": len(fp),
                "fn_count": len(fn),
                "tn_count": len(tn),
            }
        )

    return pd.concat([results, results.apply(get_confusion_matrix, axis=1)], axis=1)


@require_columns(
    [
        "model_name",
        "benchmark_item_id",
        "perfect_match",
        "tp_count",
        "fp_count",
        "fn_count",
        "tn_count",
    ]
)
def summarize_by_model(results: pd.DataFrame) -> pd.DataFrame:
    """Per-model item-level accuracy and pooled word-level confusion metrics.

    ``accuracy`` is the mean of ``perfect_match`` (exact non-decodable-set
    match per item). The tp/fp/fn/tn sums are pooled across every item for
    that model, then used to derive word-level precision/recall/f1 — a model
    can have low item-level accuracy but still show whether it over- or
    under-flags words (e.g. high recall, low precision means it flags too
    many words as non-decodable). Expects the output of
    ``add_word_level_confusion_matrix``.
    """
    summary = results.groupby("model_name").agg(
        n_items=("benchmark_item_id", "count"),
        accuracy=("perfect_match", "mean"),
        tp_count=("tp_count", "sum"),
        fp_count=("fp_count", "sum"),
        fn_count=("fn_count", "sum"),
        tn_count=("tn_count", "sum"),
    )
    summary["precision"] = summary["tp_count"] / (
        summary["tp_count"] + summary["fp_count"]
    )
    summary["recall"] = summary["tp_count"] / (
        summary["tp_count"] + summary["fn_count"]
    )
    summary["f1"] = (
        2
        * summary["precision"]
        * summary["recall"]
        / (summary["precision"] + summary["recall"])
    )
    return summary.reset_index()


@require_columns(["model_name", "num_nondecodable_words", "perfect_match"])
def accuracy_by_nondecodable_count(results: pd.DataFrame) -> pd.DataFrame:
    """Per-model, per-num_nondecodable_words accuracy (mean perfect_match).

    Reveals whether accuracy degrades as the number of non-decodable
    words increases.

    Expects a ``num_nondecodable_words`` column (see
    ``phonics_benchmark.dataset.add_analysis_columns``).
    """
    return (
        results.groupby(["model_name", "num_nondecodable_words"])["perfect_match"]
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
