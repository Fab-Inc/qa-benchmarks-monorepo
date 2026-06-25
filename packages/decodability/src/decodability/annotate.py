"""Pandas-based helpers for language-specific decodability annotators."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import pandas as pd

Measure = Callable[[str, Any], float]
Aggregation = Callable[[pd.DataFrame, Sequence[str]], pd.Series]
RowAggregation = Callable[[Mapping[str, float]], float]


def annotate_words(
    words: Iterable[str],
    *,
    scoring_methods: Sequence[Measure],
    student_knowledge: Any,
    aggregations: Mapping[str, Aggregation] | None = None,
) -> pd.DataFrame:
    """Score each word with each method and return a dataframe."""

    df = pd.DataFrame({"word": list(words)})
    score_columns: list[str] = []

    for scoring_method in scoring_methods:
        column = _function_name(scoring_method)
        df[column] = [scoring_method(word, student_knowledge) for word in df["word"]]
        score_columns.append(column)

    if aggregations is None:
        aggregations = {"mean_score": mean_score}

    for column, aggregation in aggregations.items():
        df[column] = aggregation(df, score_columns)

    return df


def rowwise(row_func: RowAggregation) -> Aggregation:
    """Turn a ``scores -> float`` decision function into an `Aggregation`.

    Using row-wise functions will be slower than vectorized functions, but
    they make the aggregation logic easier to understand.
    """

    def aggregation(df: pd.DataFrame, score_columns: Sequence[str]) -> pd.Series:
        columns = list(score_columns)
        return df[columns].apply(lambda row: row_func(row.to_dict()), axis=1)

    return aggregation


def mean_score(df: pd.DataFrame, score_columns: Sequence[str]) -> pd.Series:
    """Default equal-weight row-wise mean across scoring columns."""

    return df[list(score_columns)].mean(axis=1)


def weighted_average(
    df: pd.DataFrame,
    score_columns: Sequence[str],
    *,
    weights: Mapping[str, float],
) -> pd.Series:
    """Row-wise weighted average across scoring columns.

    Weights are keyed by score column name, usually the scoring function name.
    """

    total_weight = sum(weights[column] for column in score_columns)
    weighted_sum = pd.concat(
        [df[column] * weights[column] for column in score_columns], axis=1
    ).sum(axis=1)
    return weighted_sum / total_weight


def _function_name(function: Callable[..., Any]) -> str:
    return getattr(function, "__name__", function.__class__.__name__)
