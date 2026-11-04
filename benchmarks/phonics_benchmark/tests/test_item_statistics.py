"""Tests for classical item difficulty and item-rest discrimination."""

import numpy as np
import pandas as pd
import pytest
from phonics_benchmark.irt import IRTFitError
from phonics_benchmark.item_statistics import (
    STATUS_ALL_CORRECT,
    STATUS_ALL_INCORRECT,
    STATUS_INTERIOR,
    simple_difficulty_and_discrimination,
)


def build_responses(scores: dict[str, list[bool | None]]) -> pd.DataFrame:
    """One row per model and item; a None score means the model skipped it."""
    return pd.DataFrame(
        [
            {
                "model": model,
                "benchmark_item_id": f"item-{index + 1}",
                "perfect_match": score,
            }
            for model, model_scores in scores.items()
            for index, score in enumerate(model_scores)
            if score is not None
        ]
    )


# Five models from strongest to weakest on items 2-6, which act as the rest of
# the test. Item 1 varies per test.
REST_OF_TEST = {
    "model-a": [True, True, True, True, True],
    "model-b": [True, True, True, True, False],
    "model-c": [True, True, True, False, False],
    "model-d": [True, True, False, False, False],
    "model-e": [True, False, False, False, False],
}


def with_first_item(first_item: list[bool | None]) -> pd.DataFrame:
    """Responses for REST_OF_TEST with item-1 prepended, one score per model."""
    return build_responses(
        {
            model: [score, *rest]
            for (model, rest), score in zip(REST_OF_TEST.items(), first_item)
        }
    )


def item_row(items: pd.DataFrame, item_id: str = "item-1") -> pd.Series:
    return items.set_index("benchmark_item_id").loc[item_id]


def test_item_rest_correlation_matches_the_point_biserial_formula() -> None:
    """The correlation equals (M1 - M0) / s_n * sqrt(n1 * n0 / n^2)."""
    responses = with_first_item([True, False, True, False, False])

    row = item_row(simple_difficulty_and_discrimination(responses, n_permutations=10))

    # Rest scores exclude item-1: each model's rate on items 2-6.
    rest = np.array([5, 4, 3, 2, 1]) / 5
    correct = np.array([1, 0, 1, 0, 0], dtype=bool)
    n, n1 = len(rest), correct.sum()
    expected = (
        (rest[correct].mean() - rest[~correct].mean())
        / rest.std()  # population SD, to match sqrt(n1 * n0 / n^2)
        * np.sqrt(n1 * (n - n1) / n**2)
    )
    assert row["item_rest_correlation"] == pytest.approx(expected)


def test_an_item_solved_by_the_strongest_models_reaches_its_maximum() -> None:
    """When the top models are the ones correct, normalized discrimination is 1."""
    responses = with_first_item([True, True, False, False, False])

    row = item_row(simple_difficulty_and_discrimination(responses, n_permutations=10))

    assert row["item_rest_correlation"] == pytest.approx(
        row["max_item_rest_correlation"]
    )
    assert row["normalized_item_rest_correlation"] == pytest.approx(1.0)
    assert row["mean_rest_percentile_of_correct"] == pytest.approx(0.9)


def test_an_item_solved_only_by_the_weakest_model_is_flagged() -> None:
    """A hard item only the weakest model got right discriminates negatively.

    Its permutation p-value is the chance a random model scores as low as the
    weakest, 1 in 5, so it lands near 0.2 rather than near 1.
    """
    responses = with_first_item([False, False, False, False, True])

    row = item_row(simple_difficulty_and_discrimination(responses, n_permutations=2000))

    assert row["proportion_correct"] == pytest.approx(0.2)
    assert row["item_rest_correlation"] < 0
    assert row["normalized_item_rest_correlation"] == pytest.approx(-1.0)
    assert row["mean_rest_percentile_of_correct"] == pytest.approx(0.2)
    assert row["p_value_low_discrimination"] == pytest.approx(0.2, abs=0.03)


def test_items_answered_identically_have_no_discrimination() -> None:
    """All-correct and all-incorrect items carry a status and no correlation."""
    all_correct = simple_difficulty_and_discrimination(with_first_item([True] * 5))
    all_incorrect = simple_difficulty_and_discrimination(with_first_item([False] * 5))

    assert item_row(all_correct)["status"] == STATUS_ALL_CORRECT
    assert item_row(all_incorrect)["status"] == STATUS_ALL_INCORRECT
    for items in (all_correct, all_incorrect):
        row = item_row(items)
        assert np.isnan(row["item_rest_correlation"])
        assert np.isnan(row["p_value_low_discrimination"])


def test_missing_responses_are_left_out_rather_than_scored_wrong() -> None:
    """A model that skipped an item neither counts against it nor in its rest."""
    responses = with_first_item([True, True, None, False, False])

    row = item_row(simple_difficulty_and_discrimination(responses, n_permutations=10))

    assert row["models_answered"] == 4
    assert row["proportion_correct"] == pytest.approx(0.5)
    assert row["status"] == STATUS_INTERIOR

    # Only the four answering models enter the correlation.
    expected = np.corrcoef([1, 1, 0, 0], [1.0, 0.8, 0.4, 0.2])[0, 1]
    assert row["item_rest_correlation"] == pytest.approx(expected)


def test_the_permutation_p_value_is_reproducible_for_a_seed() -> None:
    responses = with_first_item([False, True, False, True, False])

    first = simple_difficulty_and_discrimination(responses, seed=7)
    second = simple_difficulty_and_discrimination(responses, seed=7)

    pd.testing.assert_frame_equal(first, second)


def test_item_discrimination_requires_at_least_three_models() -> None:
    responses = build_responses({"model-a": [True, False], "model-b": [False, True]})

    with pytest.raises(IRTFitError, match="at least three models"):
        simple_difficulty_and_discrimination(responses)
