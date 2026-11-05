"""Tests for simple item difficulty and rank-based discrimination."""

import numpy as np
import pandas as pd
import pytest
from phonics_benchmark.irt import IRTFitError
from phonics_benchmark.item_statistics import (
    STATUS_ALL_CORRECT,
    STATUS_ALL_INCORRECT,
    STATUS_INTERIOR,
    scaled_ranks,
    simple_difficulty_and_discrimination,
)

MODELS = ["model-a", "model-b", "model-c", "model-d", "model-e"]

# Five models from strongest to weakest, so their scaled ranks are 1, 0.75,
# 0.5, 0.25 and 0.
SCORES = pd.Series([0.9, 0.8, 0.7, 0.6, 0.5], index=MODELS)


def build_responses(item: list[bool | None], models=MODELS) -> pd.DataFrame:
    """One row per model for a single item; None means the model skipped it."""
    return pd.DataFrame(
        [
            {"model": model, "benchmark_item_id": "item-1", "perfect_match": score}
            for model, score in zip(models, item)
            if score is not None
        ]
    )


def item_row(items: pd.DataFrame, item_id: str = "item-1") -> pd.Series:
    return items.set_index("benchmark_item_id").loc[item_id]


def test_scaled_ranks_run_from_weakest_to_strongest() -> None:
    """0 is the lowest score, 1 the highest, and ties share their rank."""
    ranks = scaled_ranks(np.array([0.2, 0.9, 0.5, 0.5]))

    assert ranks.tolist() == pytest.approx([0.0, 1.0, 0.5, 0.5])


def test_an_item_solved_by_the_strongest_models_ranks_high() -> None:
    """The two strongest models got it right: mean rank (1 + 0.75) / 2."""
    responses = build_responses([True, True, False, False, False])

    row = item_row(simple_difficulty_and_discrimination(responses, model_scores=SCORES))

    assert row["mean_rank_of_correct"] == pytest.approx(0.875)


def test_an_item_solved_only_by_the_weakest_model_ranks_zero() -> None:
    """Only the weakest model got it right, so its mean rank is 0."""
    responses = build_responses([False, False, False, False, True])

    row = item_row(simple_difficulty_and_discrimination(responses, model_scores=SCORES))

    assert row["proportion_correct"] == pytest.approx(0.2)
    assert row["mean_rank_of_correct"] == pytest.approx(0.0)


def test_the_chance_band_brackets_what_random_models_would_average() -> None:
    """Random models average a rank of 0.5, inside the chance band."""
    responses = build_responses([True, False, True, False, False])

    row = item_row(simple_difficulty_and_discrimination(responses, model_scores=SCORES))

    assert 0.0 <= row["chance_rank_low"] < 0.5 < row["chance_rank_high"] <= 1.0


def test_the_ranking_follows_model_scores() -> None:
    """Reversing the scores makes model-a the weakest."""
    responses = build_responses([True, True, False, False, False])
    reversed_scores = pd.Series(SCORES.to_numpy()[::-1], index=MODELS)

    row = item_row(
        simple_difficulty_and_discrimination(responses, model_scores=reversed_scores)
    )

    assert row["mean_rank_of_correct"] == pytest.approx(0.125)


def test_models_without_a_score_are_left_out() -> None:
    """A model missing from model_scores cannot be ranked, so it is dropped."""
    responses = build_responses([True, True, False, False, False])

    row = item_row(
        simple_difficulty_and_discrimination(
            responses, model_scores=SCORES.drop("model-e")
        )
    )

    assert row["models_answered"] == 4
    # Ranks among four models are 1, 2/3, 1/3, 0; the top two got it right.
    assert row["mean_rank_of_correct"] == pytest.approx((1 + 2 / 3) / 2)


def test_items_answered_identically_have_no_discrimination() -> None:
    """All-correct and all-incorrect items carry a status and no rank."""
    all_correct = simple_difficulty_and_discrimination(
        build_responses([True] * 5), model_scores=SCORES
    )
    all_incorrect = simple_difficulty_and_discrimination(
        build_responses([False] * 5), model_scores=SCORES
    )

    assert item_row(all_correct)["status"] == STATUS_ALL_CORRECT
    assert item_row(all_incorrect)["status"] == STATUS_ALL_INCORRECT
    for items in (all_correct, all_incorrect):
        row = item_row(items)
        assert np.isnan(row["mean_rank_of_correct"])
        assert np.isnan(row["chance_rank_low"])


def test_missing_responses_are_left_out_rather_than_scored_wrong() -> None:
    """A model that skipped an item neither counts against it nor is ranked."""
    responses = build_responses([True, True, None, False, False])

    row = item_row(simple_difficulty_and_discrimination(responses, model_scores=SCORES))

    assert row["models_answered"] == 4
    assert row["proportion_correct"] == pytest.approx(0.5)
    assert row["status"] == STATUS_INTERIOR
    # Only the four answering models are ranked: 1, 2/3, 1/3, 0.
    assert row["mean_rank_of_correct"] == pytest.approx((1 + 2 / 3) / 2)


def test_the_chance_band_is_reproducible_for_a_seed() -> None:
    responses = build_responses([False, True, False, True, False])

    first = simple_difficulty_and_discrimination(responses, model_scores=SCORES, seed=7)
    second = simple_difficulty_and_discrimination(
        responses, model_scores=SCORES, seed=7
    )

    pd.testing.assert_frame_equal(first, second)


def test_item_discrimination_requires_at_least_three_models() -> None:
    responses = build_responses([True, False], models=["model-a", "model-b"])

    with pytest.raises(IRTFitError, match="at least three models"):
        simple_difficulty_and_discrimination(responses, model_scores=SCORES)
