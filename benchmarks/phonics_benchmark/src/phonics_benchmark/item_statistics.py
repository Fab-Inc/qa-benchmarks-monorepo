"""Classical item statistics: difficulty and item-rest discrimination.

Each ``model`` is a respondent and ``perfect_match`` its binary response to a
``benchmark_item_id``, as in the Rasch fit in ``irt``. With a few dozen model
respondents a 2PL discrimination per item is not estimable, so discrimination
here is the classical item-rest correlation instead: the point-biserial between
the item's 0/1 responses and each model's perfect-match rate on the *other*
items.

The raw item-rest correlation is capped by how hard an item is: when only a few
models get an item right, it cannot reach 1 even if those models are the
strongest ones. So each item also gets

- the largest correlation its difficulty allows, and the raw value as a share
  of it,
- the mean rest-score percentile of the models that got it right (1 means the
  strongest models solved it), and
- a one-sided permutation p-value for discrimination this low arising by
  chance, which accounts for how few models answered correctly.

Items every answering model got right, or none did, have no correlation.

Missing model x item responses are left out rather than scored as wrong: rest
scores are rates over the other items a model answered, and each item's
statistics use only the models that answered it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from phonics_benchmark.irt import (
    DEFAULT_ITEM_METADATA_COLUMNS,
    IRTFitError,
    to_binary_responses,
)
from phonics_benchmark.utils import require_columns

DEFAULT_N_PERMUTATIONS = 1000

STATUS_ALL_CORRECT = "all_correct"
STATUS_ALL_INCORRECT = "all_incorrect"
STATUS_INTERIOR = "interior"


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation, or NaN when either vector is constant."""
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _permutation_p_value(
    responses: np.ndarray,
    rest: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> float:
    """One-sided p-value for an item-rest correlation at least this low.

    Shuffling the item's responses across models keeps the number correct
    fixed, so the correlation is monotone in the summed rest score of the
    models marked correct. Comparing those sums is equivalent and cheaper.
    """
    observed = rest[responses == 1].sum()
    shuffled = rng.permuted(np.tile(responses, (n_permutations, 1)), axis=1)
    permuted_sums = shuffled @ rest
    # Small tolerance so ties with the observed sum count as "as low".
    as_low = np.count_nonzero(permuted_sums <= observed + 1e-9)
    # The +1s count the observed arrangement itself, so p is never exactly 0.
    return (as_low + 1) / (n_permutations + 1)


@require_columns(["model", "benchmark_item_id", "perfect_match"])
def simple_difficulty_and_discrimination(
    results: pd.DataFrame,
    *,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    seed: int = 0,
    item_metadata_columns: Sequence[str] = DEFAULT_ITEM_METADATA_COLUMNS,
) -> pd.DataFrame:
    """Per-item difficulty and item-rest discrimination over model responses.

    ``results`` is one row per model and benchmark item. Rows for calls that
    never returned should be dropped by the caller rather than passed in as
    zeros: a failed call is a missing response, not a wrong answer.

    Returns one row per item, in first-seen order.
    """
    metadata_columns = [
        column for column in item_metadata_columns if column in results.columns
    ]

    item_order = results["benchmark_item_id"].drop_duplicates().tolist()
    model_order = results["model"].drop_duplicates().tolist()
    # Three models is the fewest for which a rest score can vary once one
    # model's response is fixed.
    if len(model_order) < 3:
        raise IRTFitError(
            "Item discrimination requires responses from at least three models; "
            f"found {len(model_order)}"
        )

    results = results.assign(
        perfect_match=to_binary_responses(results["perfect_match"])
    )
    # items-by-models, with NaN for any model that never answered an item.
    responses = (
        results.pivot(
            index="benchmark_item_id", columns="model", values="perfect_match"
        )
        .reindex(index=item_order, columns=model_order)
        .to_numpy(dtype=float)
    )
    answered = ~np.isnan(responses)
    filled = np.nan_to_num(responses)

    # Each model's accuracy on every other item.
    correct_by_model = filled.sum(axis=0)
    answered_by_model = answered.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rest_scores = (correct_by_model - filled) / (answered_by_model - 1)
    rest_scores[~answered] = np.nan

    rng = np.random.default_rng(seed)
    rows = []
    for item_index in range(len(item_order)):
        rest = rest_scores[item_index]
        # A model with no other answered items has no rest score.
        usable = answered[item_index] & np.isfinite(rest)
        item_responses = responses[item_index, usable].astype(int)
        item_rest = rest[usable]
        n_correct = int(item_responses.sum())
        n_answered = int(usable.sum())

        correlation = max_correlation = p_value = mean_percentile = np.nan
        if 0 < n_correct < n_answered:
            correlation = _correlation(item_responses, item_rest)

            # The best case for this difficulty: the n_correct strongest
            # models are the ones that got it right.
            best_case = np.zeros(n_answered, dtype=int)
            best_case[np.argsort(-item_rest, kind="stable")[:n_correct]] = 1
            max_correlation = _correlation(best_case, item_rest)

            # Percentile 1.0 is the strongest model among those answering.
            percentiles = pd.Series(item_rest).rank(pct=True).to_numpy()
            mean_percentile = float(percentiles[item_responses == 1].mean())

            if not np.isnan(correlation):
                p_value = _permutation_p_value(
                    item_responses, item_rest, n_permutations, rng
                )

        rows.append(
            {
                "benchmark_item_id": item_order[item_index],
                "models_correct": n_correct,
                "models_answered": n_answered,
                "proportion_correct": n_correct / n_answered if n_answered else np.nan,
                "item_rest_correlation": correlation,
                "max_item_rest_correlation": max_correlation,
                "mean_rest_percentile_of_correct": mean_percentile,
                "p_value_low_discrimination": p_value,
            }
        )

    item_results = pd.DataFrame(rows)
    with np.errstate(divide="ignore", invalid="ignore"):
        item_results["normalized_item_rest_correlation"] = (
            item_results["item_rest_correlation"]
            / item_results["max_item_rest_correlation"]
        )
    item_results["status"] = np.select(
        [
            item_results["models_correct"] == item_results["models_answered"],
            item_results["models_correct"] == 0,
        ],
        [STATUS_ALL_CORRECT, STATUS_ALL_INCORRECT],
        default=STATUS_INTERIOR,
    )

    if metadata_columns:
        item_metadata = (
            results[["benchmark_item_id", *metadata_columns]]
            .drop_duplicates("benchmark_item_id")
            .set_index("benchmark_item_id")
            .reindex(item_order)
            .reset_index()
        )
        item_results = item_metadata.merge(
            item_results, on="benchmark_item_id", validate="one_to_one"
        )

    return item_results
