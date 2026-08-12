"""Rasch/1PL item difficulty and model capability estimates.

Each ``model`` is treated as an IRT respondent and ``perfect_match`` as its
binary response to a ``benchmark_item_id``. The fit uses GIRTH: marginal maximum
likelihood (MML) for item difficulty and expected a posteriori (EAP) scoring for
model capability.

Models need not have answered every item. Missing model x item responses are
tagged for GIRTH rather than rejected, so a run with a few dropped items (parse
failures, API errors) still fits, and per-item and per-model rates are computed
over answered items only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd
from girth import ability_eap, rasch_mml, tag_missing_data

from phonics_benchmark.utils import require_columns

# A Rasch/1PL model holds discrimination equal across items, so every item
# differs only in difficulty. With a handful of model respondents there is not
# enough data to estimate a discrimination per item (2PL) as well.
DISCRIMINATION = 1

VALID_RESPONSES = [0, 1]

DEFAULT_ITEM_METADATA_COLUMNS = (
    "content",
    "student_knowledge_id",
    "num_nondecodable_words",
    "nondecodable_ratio",
)

STATUS_ALL_CORRECT = "lower_bound_all_correct"
STATUS_ALL_INCORRECT = "upper_bound_all_incorrect"
STATUS_INTERIOR = "interior"


class IRTFitError(ValueError):
    """Raised when responses cannot support a Rasch fit."""


class RaschEstimates(NamedTuple):
    """Per-item difficulties and per-model capabilities from one Rasch fit."""

    items: pd.DataFrame
    models: pd.DataFrame


def to_binary_responses(values: pd.Series) -> pd.Series:
    """Convert boolean-like perfect-match values to integer IRT responses."""
    normalized = values.astype("string").str.strip().str.lower()
    mapping = {"true": 1, "false": 0, "1": 1, "0": 0}
    scores = normalized.map(mapping)
    if scores.isna().any():
        invalid = sorted(normalized[scores.isna()].dropna().unique().tolist())
        raise IRTFitError(
            "perfect_match must contain only True/False or 1/0 values; "
            f"found: {invalid}"
        )
    return scores.astype(int)


@require_columns(["model", "benchmark_item_id", "perfect_match"])
def fit_rasch(
    results: pd.DataFrame,
    *,
    item_metadata_columns: Sequence[str] = DEFAULT_ITEM_METADATA_COLUMNS,
) -> RaschEstimates:
    """Estimate item difficulties and model capabilities from scored results.

    ``results`` is one row per model and benchmark item, as produced by
    ``scoring.add_per_item_confusion_matrix``. Rows for calls that never
    returned should be dropped by the caller rather than passed in as zeros: a
    failed call is a missing response, not a wrong answer.
    """
    metadata_columns = [
        column for column in item_metadata_columns if column in results.columns
    ]

    item_order = results["benchmark_item_id"].drop_duplicates().tolist()
    model_order = results["model"].drop_duplicates().tolist()
    if len(model_order) < 2:
        raise IRTFitError(
            "IRT fitting requires responses from at least two models; found "
            f"{len(model_order)}"
        )

    results = results.assign(
        perfect_match=to_binary_responses(results["perfect_match"])
    )

    # items-by-respondents, with NaN for any model that never answered an item.
    response_frame = results.pivot(
        index="benchmark_item_id",
        columns="model",
        values="perfect_match",
    ).reindex(index=item_order, columns=model_order)

    response_matrix = tag_missing_data(
        response_frame.to_numpy(dtype=float), VALID_RESPONSES
    ).astype(int)

    estimates = rasch_mml(response_matrix, discrimination=DISCRIMINATION)

    difficulties = np.asarray(estimates["Difficulty"], dtype=float)
    capabilities = np.asarray(
        ability_eap(response_matrix, difficulties, DISCRIMINATION), dtype=float
    )

    answered_by_item = response_frame.notna().sum(axis=1).to_numpy(dtype=int)
    correct_by_item = response_frame.sum(axis=1).to_numpy(dtype=int)
    item_results = pd.DataFrame(
        {
            "benchmark_item_id": item_order,
            "difficulty_logit_mml": difficulties,
            "models_correct": correct_by_item,
            "models_answered": answered_by_item,
            "models_total": len(model_order),
            "response_rate": correct_by_item / answered_by_item,
        }
    )
    item_results["difficulty_estimate_status"] = np.select(
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

    answered_by_model = response_frame.notna().sum(axis=0).to_numpy(dtype=int)
    correct_by_model = response_frame.sum(axis=0).to_numpy(dtype=int)
    model_results = pd.DataFrame(
        {
            "model": model_order,
            "capability_logit_eap": capabilities,
            "items_correct": correct_by_model,
            "items_answered": answered_by_model,
            "items_total": len(item_order),
            "raw_accuracy": correct_by_model / answered_by_model,
        }
    )
    model_results["capability_rank"] = (
        model_results["capability_logit_eap"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    return RaschEstimates(items=item_results, models=model_results)


def fit_warnings(estimates: RaschEstimates) -> list[str]:
    """Caveats about a fit that a caller should surface alongside the numbers."""
    warnings = []

    missing_responses = int(
        (estimates.models["items_total"] - estimates.models["items_answered"]).sum()
    )
    if missing_responses:
        warnings.append(
            f"{missing_responses} model x item responses were missing and tagged "
            "as such; rates are computed over answered items only."
        )

    if len(estimates.models) == 2:
        warnings.append(
            "With only two model respondents, item response rates and difficulty "
            "estimates can take only three distinct levels."
        )

    boundary_items = estimates.items["difficulty_estimate_status"] != STATUS_INTERIOR
    if boundary_items.any():
        warnings.append(
            f"{int(boundary_items.sum())} items were answered identically by all "
            "answering models; their difficulty estimates are optimizer bounds, "
            "not finite point estimates."
        )

    return warnings
