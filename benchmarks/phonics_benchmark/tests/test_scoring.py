"""Tests for scoring model results against the benchmark items."""

import numpy as np
import pandas as pd
import pytest
from phonics_benchmark.scoring import (
    accuracy_by_nondecodable_count,
    add_per_item_confusion_matrix,
    pooled_fbeta,
    summarize_by_model,
)

ITEMS = pd.DataFrame(
    {
        "benchmark_item_id": ["p1_t1", "p1_t2"],
        "content": ["Mama na paka soko.", "Kuku soko kaka."],
        "nondecodable_ratio": [0.25, 0.33],
    }
)


@pytest.mark.parametrize(
    ("beta", "expected"),
    [
        # Given: tp=3, fp=1, fn=2
        # F-beta = (1 + b**2) * tp / ((1 + b**2) * tp + b**2 * fn + fp)
        (0, 3 / 4),  # b=0 collapses to precision
        (1, 6 / 9),  # b=1, so f1
        (2, 15 / 24),  # b=2, so f2
    ],
)
def test_pooled_fbeta_matches_the_definition(beta: float, expected: float) -> None:
    """Beta shifts the score between precision (b=0) and recall (b -> inf)."""
    scores = pooled_fbeta(pd.Series([3]), pd.Series([1]), pd.Series([2]), beta=beta)

    assert scores.iloc[0] == pytest.approx(expected)


def test_pooled_fbeta_leaves_empty_denominators_undefined() -> None:
    """If tp, fp, and fn are all zero, the row should be NaN."""
    scores = pooled_fbeta(
        pd.Series([0]),
        pd.Series([0]),
        pd.Series([0]),
        beta=2,
    )

    assert pd.isna(scores.iloc[0])


def test_pooled_fbeta_accepts_numpy_arrays() -> None:
    """The function should work with numpy arrays as well as pandas Series."""
    tp = np.array([3, 0])
    fp = np.array([1, 0])
    fn = np.array([2, 0])
    beta = 1

    scores = pooled_fbeta(tp, fp, fn, beta=beta)

    assert scores[0] == pytest.approx(6 / 9)
    assert np.isnan(scores[1])


def test_add_word_level_confusion_matrix_derives_confusion_counts() -> None:
    """tp/fp/fn/tn are derived from the predicted vs. actual word sets."""
    results = pd.DataFrame(
        {
            # The model's output, already parsed into a word list by the caller.
            # "Soko" is deliberately capitalised: gold words are lowercased, so
            # this checks the comparison is case-insensitive.
            "response_parsed": [["Soko", "kaka"]],  # predicted non-decodable
            "nondecodable_words": [["soko"]],  # actually non-decodable
            "vocab_set": [{"mama", "na", "paka", "soko", "kaka"}],
        }
    )

    scored = add_per_item_confusion_matrix(results, "sw")

    assert not scored.loc[0, "perfect_match"]
    assert scored.loc[0, "tp"] == {"soko"}
    assert scored.loc[0, "fp"] == {"kaka"}
    assert scored.loc[0, "fn"] == set()
    assert scored.loc[0, "tn"] == {"mama", "na", "paka"}
    assert scored.loc[0, "tp_count"] == 1
    assert scored.loc[0, "fp_count"] == 1
    assert scored.loc[0, "fn_count"] == 0
    assert scored.loc[0, "tn_count"] == 3
    assert scored.loc[0, "has_actual_nondecodable"]
    assert scored.loc[0, "has_predicted_nondecodable"]
    assert scored.loc[0, "precision"] == 0.5
    assert scored.loc[0, "recall"] == 1.0
    assert scored.loc[0, "f1"] == 2 / 3


def test_add_per_item_confusion_matrix_leaves_empty_denominators_undefined() -> None:
    """Precision/recall are NaN, not 1.0, where their denominator is empty."""
    results = pd.DataFrame(
        {
            "response_parsed": [[], [], ["mama"]],
            "nondecodable_words": [[], ["soko"], []],
            "vocab_set": [{"mama", "soko"}] * 3,
        }
    )

    scored = add_per_item_confusion_matrix(results, "sw")

    # No actual non-decodable words and nothing predicted: both metrics
    # undefined, and F1 too, because the item shows nothing about the
    # model's detection ability.
    assert pd.isna(scored.loc[0, "precision"])
    assert pd.isna(scored.loc[0, "recall"])
    assert pd.isna(scored.loc[0, "f1"])
    assert scored.loc[0, "perfect_match"]  # still credited as an exact match

    # Total miss: precision undefined, but recall and F1 score the failure.
    assert pd.isna(scored.loc[1, "precision"])
    assert scored.loc[1, "recall"] == 0.0
    assert scored.loc[1, "f1"] == 0.0

    # False alarm on a fully decodable item: recall undefined, precision 0.
    assert scored.loc[2, "precision"] == 0.0
    assert pd.isna(scored.loc[2, "recall"])
    assert scored.loc[2, "f1"] == 0.0


def test_summarize_by_model_pools_counts_and_derives_rates() -> None:
    """Per-model accuracy plus macro and pooled precision/recall/f1."""
    results = pd.DataFrame(
        {
            "model": ["m1", "m1"],
            "benchmark_item_id": ["p1_t1", "p1_t2"],
            "perfect_match": [True, False],
            "tp_count": [1, 1],
            "fp_count": [0, 1],
            "fn_count": [0, 1],
            "has_actual_nondecodable": [True, True],
            "has_predicted_nondecodable": [True, True],
            "precision": [1.0, 0.5],
            "recall": [1.0, 0.5],
            "f1": [1.0, 0.5],
        }
    )

    summary = summarize_by_model(results).set_index("model_name")

    assert summary.loc["m1", "n_items"] == 2
    assert summary.loc["m1", "accuracy"] == 0.5
    assert summary.loc["m1", "macro_precision"] == 0.75  # (1.0 + 0.5) / 2
    assert summary.loc["m1", "macro_recall"] == 0.75  # (1.0 + 0.5) / 2
    assert summary.loc["m1", "macro_f1"] == 0.75  # (1.0 + 0.5) / 2
    assert summary.loc["m1", "pooled_precision"] == 2 / 3  # tp=2, fp=1
    assert summary.loc["m1", "pooled_recall"] == 2 / 3  # tp=2, fn=1
    assert summary.loc["m1", "pooled_f1"] == 2 / 3  # tp=2, fp=1, fn=1
    assert summary.loc["m1", "pooled_f2"] == 2 / 3  # tp=2, fp=1, fn=1, same as f1


def test_summarize_by_model_restricts_each_macro_metric_to_its_support() -> None:
    """Macro metrics average only over items where they are defined.

    Test with benchmark results with
    - `i1`: item with actual and predicted non-decodable words + a miss
    - `i2`: total miss on a item with an actual non-decodable word
    - `i3`: false alarm on a fully decodable item
    - `i4`: fully decodable item with nothing predicted (perfect match)
    """
    test_results = pd.DataFrame(
        {
            "model": ["m1"] * 4,
            "benchmark_item_id": ["i1", "i2", "i3", "i4"],
            "perfect_match": [False, False, False, True],
            "tp_count": [1, 0, 0, 0],
            "fp_count": [1, 0, 2, 0],
            "fn_count": [1, 1, 0, 0],
            "has_actual_nondecodable": [True, True, False, False],
            "has_predicted_nondecodable": [True, False, True, False],
            "precision": [0.5, float("nan"), 0.0, float("nan")],
            "recall": [0.5, 0.0, float("nan"), float("nan")],
            "f1": [0.5, 0.0, 0.0, float("nan")],
        }
    )

    summary = summarize_by_model(test_results).set_index("model_name")

    assert summary.loc["m1", "n_items"] == 4
    assert summary.loc["m1", "n_items_not_fully_decodable"] == 2  # i1, i2
    assert summary.loc["m1", "n_items_fully_decodable"] == 2  # i3, i4
    # Precision over i1, i3 only — the total miss is not forgiven, it is scored
    # by recall instead.
    assert summary.loc["m1", "macro_precision"] == 0.25  # (0.5 + 0.0) / 2
    assert summary.loc["m1", "macro_recall"] == 0.25  # (0.5 + 0.0) / 2
    # F1 covers only items with actual non-decodable words, so the false alarm
    # on i3 does not double-count here; it lands in precision and in the
    # false-alarm rate.
    assert summary.loc["m1", "macro_f1"] == 0.25  # (0.5 + 0.0) / 2
    # i3 of {i3, i4} drew a false alarm.
    assert summary.loc["m1", "false_alarm_rate_fully_decodable"] == 0.5
    assert summary.loc["m1", "mean_fp_per_fully_decodable_item"] == 1.0  # (2 + 0) / 2
    assert summary.loc["m1", "accuracy"] == 0.25  # i4 only
    assert summary.loc["m1", "accuracy_for_not_fully_decodable"] == 0.0


def test_summarize_by_model_reports_undefined_rather_than_perfect_precision() -> None:
    """A model that never predicts anything gets NaN precision, not 1.0."""
    results = pd.DataFrame(
        {
            "model": ["null_model"] * 2,
            "benchmark_item_id": ["i1", "i2"],
            "perfect_match": [False, True],
            "tp_count": [0, 0],
            "fp_count": [0, 0],
            "fn_count": [1, 0],
            "has_actual_nondecodable": [True, False],
            "has_predicted_nondecodable": [False, False],
            "precision": [float("nan"), float("nan")],
            "recall": [0.0, float("nan")],
            "f1": [0.0, float("nan")],
        }
    )

    summary = summarize_by_model(results).set_index("model_name")

    assert pd.isna(summary.loc["null_model", "macro_precision"])
    assert pd.isna(summary.loc["null_model", "pooled_precision"])
    assert summary.loc["null_model", "macro_recall"] == 0.0
    assert summary.loc["null_model", "macro_f1"] == 0.0


def test_summarize_by_model_leaves_f_scores_undefined_with_no_pooled_words() -> None:
    """A model that predicts nothing on fully decodable items scores NaN, not 0.0.

    Given
    - fully decodable items only
    - model predicted nothing
    then, tp, fp and fn all pool to 0
    """
    results = pd.DataFrame(
        {
            "model": ["silent"] * 2,
            "benchmark_item_id": ["i1", "i2"],
            "perfect_match": [True, True],
            "tp_count": [0, 0],
            "fp_count": [0, 0],
            "fn_count": [0, 0],
            "has_actual_nondecodable": [False, False],
            "has_predicted_nondecodable": [False, False],
            "precision": [float("nan"), float("nan")],
            "recall": [float("nan"), float("nan")],
            "f1": [float("nan"), float("nan")],
        }
    )

    summary = summarize_by_model(results).set_index("model_name")

    assert pd.isna(summary.loc["silent", "pooled_f1"])
    assert pd.isna(summary.loc["silent", "pooled_f2"])
    assert summary.loc["silent", "accuracy"] == 1.0


def test_accuracy_by_nondecodable_count_groups_by_model_and_count() -> None:
    """Accuracy is broken out per model and per item difficulty level."""
    results = pd.DataFrame(
        {
            "model": ["m1", "m1", "m1"],
            "num_nondecodable_words": [0, 0, 1],
            "perfect_match": [True, False, True],
        }
    )

    breakdown = accuracy_by_nondecodable_count(results).set_index(
        ["model", "num_nondecodable_words"]
    )

    assert breakdown.loc[("m1", 0), "accuracy"] == 0.5
    assert breakdown.loc[("m1", 0), "n_items"] == 2
    assert breakdown.loc[("m1", 1), "accuracy"] == 1.0


def test_add_per_item_confusion_matrix_names_the_column_it_needs() -> None:
    """A frame with only the raw `output` fails on the column contract.

    Scoring consumes `response_parsed`, not `output`. When the declared columns
    disagreed with the ones actually read, this case passed the guard and then
    died on a KeyError inside the per-row apply.
    """
    results = pd.DataFrame(
        {
            "output": ['["soko"]'],
            "nondecodable_words": [["soko"]],
            "vocab_set": [{"soko"}],
        }
    )

    with pytest.raises(ValueError, match="response_parsed"):
        add_per_item_confusion_matrix(results, "sw")
