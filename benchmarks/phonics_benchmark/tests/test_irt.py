"""Tests for the Rasch/1PL fit over model responses."""

import numpy as np
import pandas as pd
import pytest
from phonics_benchmark.irt import (
    STATUS_ALL_CORRECT,
    STATUS_ALL_INCORRECT,
    STATUS_INTERIOR,
    IRTFitError,
    fit_rasch,
    fit_warnings,
    point_biserial_discrimination,
    to_binary_responses,
)

MODELS = ["model-a", "model-b", "model-c"]
ITEMS = ["item-1", "item-2", "item-3"]


def build_responses(scores: dict[str, list[bool]], **columns: list) -> pd.DataFrame:
    """One row per model and item, from a model -> per-item scores mapping.

    ``scores`` maps a model name to its ``perfect_match`` for each item in order,
    so every model needs a score per item. A None entry means that model never
    answered the item, and is left out of the frame entirely.
    """
    item_counts = {len(model_scores) for model_scores in scores.values()}
    if len(item_counts) > 1:
        raise ValueError(f"Every model needs one score per item; got {item_counts}")

    return pd.DataFrame(
        [
            {
                "model": model,
                "benchmark_item_id": f"item-{index + 1}",
                "perfect_match": score,
                **columns,
            }
            for model, model_scores in scores.items()
            for index, score in enumerate(model_scores)
            if score is not None
        ]
    )


def test_fit_rasch_returns_one_row_per_item_and_model() -> None:
    """Items and models are each estimated once, in first-seen order."""
    responses = build_responses(
        {
            "model-a": [True, True, False],
            "model-b": [True, False, False],
            "model-c": [False, False, False],
        }
    )

    estimates = fit_rasch(responses)

    assert estimates.items["benchmark_item_id"].tolist() == ITEMS
    assert estimates.models["model"].tolist() == MODELS
    assert estimates.items["difficulty_logit_mml"].notna().all()
    assert estimates.models["capability_logit_eap"].notna().all()


def test_fit_rasch_ranks_models_by_capability() -> None:
    """The model that answers most items correctly ranks first."""
    responses = build_responses(
        {
            "model-a": [True, True, True],
            "model-b": [True, True, False],
            "model-c": [False, False, False],
        }
    )

    estimates = fit_rasch(responses)

    ranks = estimates.models.set_index("model")["capability_rank"]
    assert ranks["model-a"] == 1
    assert ranks["model-b"] == 2
    assert ranks["model-c"] == 3


def test_fit_rasch_flags_items_answered_identically_by_every_model() -> None:
    """All-correct and all-incorrect items are marked as optimizer bounds.

    Their difficulty is not a finite point estimate, so anything downstream
    (plots, comparisons) has to be able to tell them apart.
    """
    responses = build_responses(
        {
            "model-a": [True, True, False],
            "model-b": [True, False, False],
            "model-c": [True, False, False],
        }
    )

    estimates = fit_rasch(responses)

    statuses = estimates.items.set_index("benchmark_item_id")[
        "difficulty_estimate_status"
    ]
    assert statuses["item-1"] == STATUS_ALL_CORRECT
    assert statuses["item-2"] == STATUS_INTERIOR
    assert statuses["item-3"] == STATUS_ALL_INCORRECT


def test_fit_rasch_computes_rates_over_answered_items_only() -> None:
    """A model that never answered an item is not scored as having failed it."""
    responses = build_responses(
        {
            "model-a": [True, True, None],
            "model-b": [True, False, False],
            "model-c": [False, False, False],
        }
    )

    estimates = fit_rasch(responses)

    model_a = estimates.models.set_index("model").loc["model-a"]
    assert model_a["items_answered"] == 2
    assert model_a["items_total"] == 3
    assert model_a["raw_accuracy"] == 1.0

    item_3 = estimates.items.set_index("benchmark_item_id").loc["item-3"]
    assert item_3["models_answered"] == 2
    assert item_3["models_total"] == 3
    assert item_3["response_rate"] == 0.0


def test_fit_rasch_carries_item_metadata_onto_the_estimates() -> None:
    """Item metadata present in the responses is joined onto the difficulties."""
    responses = build_responses(
        {
            "model-a": [True, True, False],
            "model-b": [True, False, False],
        },
    )
    # Constant per item, as a real item join would be.
    responses["num_nondecodable_words"] = responses["benchmark_item_id"].map(
        {"item-1": 0, "item-2": 2, "item-3": 5}
    )

    estimates = fit_rasch(responses)

    assert estimates.items["num_nondecodable_words"].tolist() == [0, 2, 5]


def test_fit_rasch_ignores_item_metadata_columns_that_are_absent() -> None:
    """Metadata columns are optional; a bare response frame still fits."""
    responses = build_responses(
        {"model-a": [True, True, False], "model-b": [True, False, False]}
    )

    estimates = fit_rasch(responses)

    assert "num_nondecodable_words" not in estimates.items.columns


def test_fit_rasch_rejects_conflicting_item_metadata() -> None:
    """An item whose metadata differs by row means the join went wrong."""
    responses = build_responses(
        {"model-a": [True, True, False], "model-b": [True, False, False]}
    )
    responses["nondecodable_ratio"] = [0.1, 0.2, 0.3, 0.9, 0.2, 0.3]

    with pytest.raises(IRTFitError, match="conflicting values"):
        fit_rasch(responses)


def test_fit_rasch_rejects_duplicate_model_item_pairs() -> None:
    """Two responses from one model to one item would double-count it."""
    responses = build_responses(
        {"model-a": [True, True, False], "model-b": [True, False, False]}
    )
    duplicated = pd.concat([responses, responses.head(1)], ignore_index=True)

    with pytest.raises(IRTFitError, match="one response per model and item"):
        fit_rasch(duplicated)


def test_fit_rasch_requires_at_least_two_models() -> None:
    """A single respondent cannot separate item difficulty from capability."""
    responses = build_responses({"model-a": [True, True, False]})

    with pytest.raises(IRTFitError, match="at least two models"):
        fit_rasch(responses)


def test_fit_rasch_rejects_an_empty_frame() -> None:
    """An empty run is reported as such rather than reaching the estimator."""
    empty = build_responses({}).reindex(
        columns=["model", "benchmark_item_id", "perfect_match"]
    )

    with pytest.raises(IRTFitError, match="No responses"):
        fit_rasch(empty)


def test_fit_rasch_requires_the_response_columns() -> None:
    """The frame must carry the respondent, item, and score columns."""
    with pytest.raises(ValueError, match="missing required column"):
        fit_rasch(pd.DataFrame({"model": ["model-a"]}))


def build_response_frame(rows: list[list[float | None]]) -> pd.DataFrame:
    """An items-by-models frame, as ``fit_rasch`` pivots; None means unanswered."""
    return pd.DataFrame(
        rows,
        index=[f"item-{number}" for number in range(1, len(rows) + 1)],
        columns=[f"model-{letter}" for letter in "abcde"[: len(rows[0])]],
        dtype=float,
    )


def test_point_biserial_is_positive_when_stronger_models_do_better() -> None:
    """An item the panel's stronger models get right discriminates positively."""
    # model-a is strongest and model-d weakest over the other four items, and
    # item-1 follows exactly that ordering.
    response_frame = build_response_frame(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 1, 0],
        ]
    )

    discrimination = point_biserial_discrimination(response_frame)

    assert discrimination[0] > 0.8


def test_point_biserial_is_negative_when_weaker_models_do_better() -> None:
    """An item the panel's weakest models get right is flagged, not rewarded.

    A negative corrected item-total correlation usually means the item's gold
    labels are wrong rather than that the item is hard.
    """
    response_frame = build_response_frame(
        [
            [0, 0, 1, 1],
            [1, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 1, 0],
        ]
    )

    discrimination = point_biserial_discrimination(response_frame)

    assert discrimination[0] < 0


def test_point_biserial_is_undefined_for_boundary_items() -> None:
    """An item every model scored alike has no variance to correlate."""
    response_frame = build_response_frame(
        [
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 0],
        ]
    )

    discrimination = point_biserial_discrimination(response_frame)

    assert np.isnan(discrimination[0])
    assert np.isnan(discrimination[1])
    assert np.isfinite(discrimination[2])


def test_point_biserial_is_undefined_below_three_answering_models() -> None:
    """Pearson's r on two points is +/-1 by construction, so it is withheld."""
    response_frame = build_response_frame(
        [
            [1, 0, None, None],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
        ]
    )

    discrimination = point_biserial_discrimination(response_frame)

    assert np.isnan(discrimination[0])
    assert np.isfinite(discrimination[1])


def test_point_biserial_ignores_models_that_skipped_the_item() -> None:
    """A skipped response drops that model from the item's correlation.

    The unanswered cell is excluded rather than read as 0, which would otherwise
    make the skipping model look weak on precisely this item.
    """
    complete = build_response_frame(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 0],
        ]
    )
    # model-d skipped item-1; it answered 0 there, so dropping its cell should
    # leave the correlation over the remaining three models unchanged.
    with_skip = complete.copy()
    with_skip.iloc[0, 3] = np.nan

    expected = point_biserial_discrimination(complete.iloc[:, :3])[0]
    assert point_biserial_discrimination(with_skip)[0] == pytest.approx(expected)


def test_point_biserial_is_scaled_out_of_uneven_answer_counts() -> None:
    """Rest scores are accuracies, so a model answering fewer items is comparable.

    Correlating against rest *totals* would rank a model that answered half the
    panel as weak purely for having fewer chances to be correct.
    """
    response_frame = build_response_frame(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, None, 0, 0],
            [1, None, 0, 0],
            [1, None, 1, 0],
        ]
    )

    discrimination = point_biserial_discrimination(response_frame)

    # model-b answered 2 of 5 items but got both right, so it must not read as
    # weaker than model-c (2 of 5 correct over all five).
    assert discrimination[0] > 0.8


def test_fit_rasch_reports_discrimination_alongside_difficulty() -> None:
    """The item estimates carry a discrimination column per item."""
    responses = build_responses(
        {
            "model-a": [True, True, False],
            "model-b": [True, False, False],
            "model-c": [False, False, True],
        }
    )

    estimates = fit_rasch(responses)

    assert "discrimination_point_biserial" in estimates.items.columns
    assert len(estimates.items["discrimination_point_biserial"]) == len(ITEMS)


def test_fit_warnings_flags_a_panel_too_small_for_discrimination() -> None:
    """Two models cannot produce a discrimination estimate for any item."""
    responses = build_responses(
        {"model-a": [True, True, False], "model-b": [True, False, False]}
    )

    warnings = fit_warnings(fit_rasch(responses))

    assert any("too few for item discrimination" in text for text in warnings)


@pytest.mark.parametrize(
    "values",
    [
        [True, False],
        ["True", "False"],
        ["true", "false"],
        [1, 0],
        ["1", "0"],
    ],
)
def test_to_binary_responses_accepts_boolean_like_values(values: list) -> None:
    """Native booleans and the strings a CSV round-trip produces both convert."""
    assert to_binary_responses(pd.Series(values)).tolist() == [1, 0]


def test_to_binary_responses_rejects_other_values() -> None:
    """Anything else is a scoring bug, not a response to silently drop."""
    with pytest.raises(IRTFitError, match="perfect_match"):
        to_binary_responses(pd.Series(["yes", "no"]))


def test_fit_warnings_reports_missing_responses_and_bounded_items() -> None:
    """Caveats a reader needs to interpret the numbers are surfaced."""
    responses = build_responses(
        {
            "model-a": [True, True, None],
            "model-b": [True, False, False],
            "model-c": [True, False, False],
        }
    )

    warnings = fit_warnings(fit_rasch(responses))

    assert any("1 model x item responses were missing" in text for text in warnings)
    # item-1 (all correct) and item-3 (all incorrect, among answering models).
    assert any("2 items were answered identically" in text for text in warnings)
    assert not any("two model respondents" in text for text in warnings)


def test_fit_warnings_flags_a_two_model_fit() -> None:
    """With two respondents the difficulty scale has only three levels."""
    responses = build_responses(
        {"model-a": [True, True, False], "model-b": [True, False, False]}
    )

    warnings = fit_warnings(fit_rasch(responses))

    assert any("two model respondents" in text for text in warnings)


def test_fit_warnings_is_empty_for_a_clean_fit() -> None:
    """A complete panel with no bounded items has nothing to caveat.

    Deliberately wider than the other fixtures: with only three items, the rest
    scores two models are compared on can tie, which legitimately leaves an item
    without a discrimination estimate.
    """
    responses = build_responses(
        {
            "model-a": [True, True, True, True, False],
            "model-b": [True, True, False, False, True],
            "model-c": [False, True, True, False, True],
            "model-d": [False, False, False, True, True],
        }
    )

    assert fit_warnings(fit_rasch(responses)) == []
