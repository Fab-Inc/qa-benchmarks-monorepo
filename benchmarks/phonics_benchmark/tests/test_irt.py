"""Tests for the Rasch/1PL fit over model responses."""

import pandas as pd
import pytest
from phonics_benchmark.irt import (
    STATUS_ALL_CORRECT,
    STATUS_ALL_INCORRECT,
    STATUS_INTERIOR,
    IRTFitError,
    fit_rasch,
    fit_warnings,
    to_binary_responses,
)

MODELS = ["model-a", "model-b", "model-c"]
ITEMS = ["item-1", "item-2", "item-3"]


def build_responses(
    scores: dict[str, list[bool | None]], **columns: list
) -> pd.DataFrame:
    """One row per model and item, from a model -> per-item scores mapping.

    Example ``scores``:

        {
            "model-a": [True, True, False],
            "model-b": [True, False, False],
            "model-c": [False, False, None]
        }

    Every model needs a score per item. A None entry means that model never
    answered the item, and is left out of the frame entirely.
    """
    item_counts = {len(model_scores) for model_scores in scores.values()}
    if len(item_counts) > 1:
        raise ValueError("Every model must have the same number of scores")

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


def test_fit_rasch_requires_at_least_two_models() -> None:
    """A single respondent cannot separate item difficulty from capability."""
    responses = build_responses({"model-a": [True, True, False]})

    with pytest.raises(IRTFitError, match="at least two models"):
        fit_rasch(responses)


def test_fit_rasch_rejects_an_empty_frame() -> None:
    """An empty run raises IRTFitError rather than reaching the estimator.

    It trips the same too-few-models check as a single-model run, which is what
    the caller needs: an error it can catch and skip the fit on.
    """
    empty = build_responses({}).reindex(
        columns=["model", "benchmark_item_id", "perfect_match"]
    )

    with pytest.raises(IRTFitError, match="at least two models"):
        fit_rasch(empty)


def test_fit_rasch_requires_the_response_columns() -> None:
    """The frame must carry the respondent, item, and score columns."""
    with pytest.raises(ValueError, match="missing required column"):
        fit_rasch(pd.DataFrame({"model": ["model-a"]}))


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
    """A complete panel with no bounded items has nothing to caveat."""
    responses = build_responses(
        {
            "model-a": [True, True, False],
            "model-b": [True, False, False],
            "model-c": [False, False, True],
        }
    )

    assert fit_warnings(fit_rasch(responses)) == []
