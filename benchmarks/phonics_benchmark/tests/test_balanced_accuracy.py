"""Verify fixed-vocabulary scoring and constant-prediction baselines."""

import numpy as np
import pandas as pd
import pytest
from phonics_benchmark import balanced_accuracy as ba
from phonics_benchmark.bootstrap import summarize_balanced_accuracy


@pytest.mark.parametrize("totals", [[7, 0, 0, 13, 0], [0, 7, 13, 0, 0]])
def test_both_constant_predictors_score_half_regardless_of_prevalence(totals):
    scores = ba.metrics_from_totals(totals)
    assert scores[2] == 0.5
    assert scores[3] == 0


def test_out_of_text_predictions_are_audited_separately():
    items = pd.DataFrame({"n_vocab": [4], "n_gold": [1]}, index=["a"])
    scored = pd.DataFrame(
        {
            "model": ["m"],
            "benchmark_item_id": ["a"],
            "tp_count": [1],
            "fn_count": [0],
            "tn_count": [2],
            "fp_count": [2],
        }
    )
    result = ba.vocabulary_counts(scored, items)
    assert result.fp_in_vocabulary.iloc[0] == 1
    assert result.extra_predictions.iloc[0] == 1
    metrics = ba.metrics_from_totals(result[ba.COUNTS].iloc[0].to_numpy())
    assert metrics[2] == pytest.approx((1 + 2 / 3) / 2)
    assert metrics[5] == 0.75


def test_all_words_bootstrap_stays_at_half_with_missing_responses():
    items = pd.DataFrame(
        {"n_vocab": [10, 20, 10], "n_gold": [2, 10, 7]}, index=["a", "b", "c"]
    )
    scored = pd.DataFrame(
        {
            "model": ["m", "m"],
            "benchmark_item_id": ["a", "b"],
            "tp_count": [2, 10],
            "fn_count": [0, 0],
            "tn_count": [0, 0],
            "fp_count": [8, 10],
        }
    )
    summary, _ = summarize_balanced_accuracy(scored, items, n_bootstrap=100)
    row = summary.iloc[0]
    assert row.n_scorable == 2
    assert row.balanced_accuracy == 0.5
    assert row.balanced_accuracy_ci_low == 0.5
    assert row.balanced_accuracy_ci_high == 0.5
    assert np.isfinite(row.recall)


@pytest.mark.parametrize(
    "totals, expected_ba, expected_adjusted",
    [
        ([7, 0, 13, 0, 0], 1.0, 100.0),
        ([0, 7, 0, 13, 0], 0.0, -100.0),
        ([10, 10, 80, 0, 0], 0.75, 50.0),
        ([18, 2, 48, 32, 0], 0.75, 50.0),
        ([4, 16, 64, 16, 0], 0.5, 0.0),
    ],
)
def test_adjusted_ba_points_match_youden_j(totals, expected_ba, expected_adjusted):
    metrics = ba.metrics_from_totals(totals)
    assert metrics[2] == pytest.approx(expected_ba)
    assert metrics[3] == pytest.approx(expected_adjusted)
    tp, fn, tn, fp, _ = totals
    assert metrics[3] == pytest.approx(100 * (tp / (tp + fn) - fp / (fp + tn)))


def test_adjusted_bootstrap_bounds_are_in_points_without_clipping():
    items = pd.DataFrame(
        {"n_vocab": [20, 20, 20], "n_gold": [2, 5, 10]},
        index=["a", "b", "c"],
    )
    scored = pd.DataFrame(
        {
            "model": ["m", "m", "m"],
            "benchmark_item_id": ["a", "b", "c"],
            "tp_count": [2, 0, 0],
            "fn_count": [0, 5, 10],
            "tn_count": [18, 0, 0],
            "fp_count": [0, 15, 10],
        }
    )
    summary, _ = summarize_balanced_accuracy(scored, items, n_bootstrap=100, seed=0)
    row = summary.iloc[0]
    assert row.adjusted_balanced_accuracy_scale == 100
    assert row.adjusted_balanced_accuracy < 0
    assert row.adjusted_balanced_accuracy_ci_low < 0
    for suffix in ("", "_ci_low", "_ci_high"):
        assert row["adjusted_balanced_accuracy" + suffix] == pytest.approx(
            100 * (2 * row["balanced_accuracy" + suffix] - 1)
        )
