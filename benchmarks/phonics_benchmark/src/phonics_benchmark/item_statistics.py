"""Simple item statistics: difficulty and rank-based discrimination.

Each ``model`` is a respondent and ``perfect_match`` its binary response to a
``benchmark_item_id``, as in the Rasch fit in ``irt``.

Difficulty is the proportion of answering models that got the item right.

Discrimination asks whether the models that got it right are the strong ones:
it is the mean rank of those models by a per-model score, scaled so 0 is the
weakest answering model, 1 the strongest, and 0.5 what random models would
average.

We also compute the 95% interval for the mean rank, representing the range
of mean rank of random models.

Items every answering model got right, or none did, have no rank statistics.
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
DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_INTERVAL_LEVEL = 0.95

STATUS_ALL_CORRECT = "all_correct"
STATUS_ALL_INCORRECT = "all_incorrect"
STATUS_INTERIOR = "interior"


def scaled_ranks(scores: np.ndarray) -> np.ndarray:
    """Ranks scaled to [0, 1]: 0 for the lowest score, 1 for the highest.

    Tied scores share their average rank.
    """
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    return (ranks - 1) / (len(scores) - 1)


def _permutation_means(
    responses: np.ndarray,
    ranks: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Mean rank of the correct models after shuffling responses across models.

    Shuffling keeps the number models that got the item right fixed, so each
    draw is the mean rank of that many models picked at random.
    """
    shuffled = rng.permuted(np.tile(responses, (n_permutations, 1)), axis=1)
    return (shuffled @ ranks) / responses.sum()


@require_columns(["model", "benchmark_item_id", "perfect_match"])
def simple_difficulty_and_discrimination(
    results: pd.DataFrame,
    *,
    model_scores: pd.Series | None = None,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    interval_level: float = DEFAULT_INTERVAL_LEVEL,
    seed: int = 0,
    item_metadata_columns: Sequence[str] = DEFAULT_ITEM_METADATA_COLUMNS,
) -> pd.DataFrame:
    """Per-item difficulty and rank-based discrimination over model responses.
    - difficulty: low proportion of models that got it right
    - discrimination: low mean rank among the models that got it right

    ``results`` is one row per model and benchmark item. Rows for calls that
    never returned should be dropped by the caller rather than passed in as
    zeros: a failed call is a missing response, not a wrong answer.

    ``model_scores`` ranks the models, indexed by model, higher meaning
    stronger. You can feed in any metric here as long as higher means better.
    If not provided, defaults to accuracy.

    Returns one row per item, in first-seen order.
    """
    metadata_columns = [
        column for column in item_metadata_columns if column in results.columns
    ]

    item_order = results["benchmark_item_id"].drop_duplicates().tolist()
    model_order = results["model"].drop_duplicates().tolist()

    # We're only concerned with items with mixed responses. If there are only
    # two models, the mean rank can't meaningfully discriminate between them.
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

    # One score per model, aligned to the response columns. Models that
    # model_scores does not cover get NaN and cannot be ranked.
    if model_scores is None:
        model_scores = results.groupby("model")["perfect_match"].mean()
    scores = model_scores.reindex(model_order).to_numpy(dtype=float)
    has_score = np.isfinite(scores)

    rng = np.random.default_rng(seed)
    tail = (1 - interval_level) / 2
    rows = []
    for item_index in range(len(item_order)):
        # Rank only the models that answered this item and have a score.
        usable = answered[item_index] & has_score
        item_responses = responses[item_index, usable].astype(int)
        n_correct = int(item_responses.sum())
        n_answered = int(usable.sum())

        mean_rank = np.nan
        chance_low = chance_high = np.nan
        if 0 < n_correct < n_answered:
            ranks = scaled_ranks(scores[usable])
            ranks_of_correct_models = ranks[item_responses == 1]
            mean_rank = float(ranks_of_correct_models.mean())

            chance = _permutation_means(item_responses, ranks, n_permutations, rng)
            chance_low, chance_high = np.quantile(chance, [tail, 1 - tail])

        rows.append(
            {
                "benchmark_item_id": item_order[item_index],
                "models_correct": n_correct,
                "models_answered": n_answered,
                "proportion_correct": n_correct / n_answered if n_answered else np.nan,
                "mean_rank_of_correct": mean_rank,
                "chance_rank_low": chance_low,
                "chance_rank_high": chance_high,
            }
        )

    item_results = pd.DataFrame(rows)
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
