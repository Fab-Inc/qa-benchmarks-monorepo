"""Inspection plots for the benchmark item distribution and model results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from phonics_benchmark.irt import STATUS_INTERIOR
from phonics_benchmark.utils import require_columns


@require_columns(["nondecodable_ratio"])
def save_nondecodable_ratio_histogram(
    benchmark_items: pd.DataFrame,
    output_path: Path,
    *,
    column: str = "nondecodable_ratio",
    title: str = "Distribution of Non-Decodable Word Ratios",
    bins: int = 20,
) -> None:
    """Save a histogram of ``column`` to ``output_path``.

    The caller passes the full output path, so the same function serves the
    pre-filter, post-filter, or any other view by varying path and title.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    benchmark_items[column].hist(bins=bins)
    plt.title(title)
    plt.xlabel("Non-Decodable Word Ratio")
    plt.ylabel("Frequency")
    plt.savefig(output_path)
    plt.close()


@require_columns(["model", "num_nondecodable_words", "accuracy"])
def save_accuracy_by_nondecodable_count_plot(
    accuracy_by_count: pd.DataFrame,
    output_path: Path,
    *,
    title: str = "Accuracy vs. Number of Non-Decodable Words in Item",
) -> None:
    """Save a per-model line plot of accuracy across item difficulty.

    ``accuracy_by_count`` is the long-format frame from
    ``results.accuracy_by_nondecodable_count`` (model,
    num_nondecodable_words, accuracy). A plain histogram would only show how
    many items fall at each difficulty level; plotting accuracy against it
    directly answers whether a model's exact-match rate drops off as items
    get harder.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for model_name, group in accuracy_by_count.groupby("model"):
        group = group.sort_values("num_nondecodable_words")
        plt.plot(
            group["num_nondecodable_words"],
            group["accuracy"],
            marker="o",
            label=model_name,
        )
    plt.title(title)
    plt.xlabel("Number of Non-Decodable Words in Item")
    plt.ylabel("Accuracy (Exact Match Rate)")
    plt.legend()
    plt.savefig(output_path)
    plt.close()


@require_columns(["difficulty_logit_mml", "discrimination_point_biserial"])
def save_item_map_plot(
    item_estimates: pd.DataFrame,
    output_path: Path,
    *,
    title: str = "Benchmark Item Map: Difficulty vs. Discrimination",
) -> None:
    """Save a difficulty-against-discrimination scatter of the benchmark items.

    ``item_estimates`` is the ``items`` frame from ``irt.fit_rasch``. The two
    axes are what an item is worth keeping for: where it sits on the ability
    scale, and whether it separates strong models from weak ones. Items in the
    lower band discriminate poorly — either too easy, too hard, or measuring
    something other than the rest of the panel.

    Items with no discrimination estimate (boundary items, and items too few
    models answered) cannot be placed and are counted in the axis label instead.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    placeable = item_estimates.dropna(
        subset=["difficulty_logit_mml", "discrimination_point_biserial"]
    )
    dropped = len(item_estimates) - len(placeable)

    plt.figure(figsize=(10, 6))
    plt.scatter(
        placeable["difficulty_logit_mml"],
        placeable["discrimination_point_biserial"],
        alpha=0.5,
    )
    # Below zero, models the rest of the panel scores as weaker did better on the
    # item — usually a mislabelled item rather than a hard one.
    plt.axhline(0, color="grey", linestyle="--", linewidth=1)

    plt.title(title)
    plt.xlabel(
        f"Estimated Difficulty (logits, higher = harder); "
        f"{len(placeable)} items shown, {dropped} without an estimate"
    )
    plt.ylabel("Discrimination (corrected item-total correlation)")
    plt.savefig(output_path)
    plt.close()


@require_columns(
    [
        "num_nondecodable_words",
        "difficulty_logit_mml",
        "difficulty_estimate_status",
    ]
)
def save_item_difficulty_by_nondecodable_count_plot(
    item_estimates: pd.DataFrame,
    output_path: Path,
    *,
    title: str = "Estimated Item Difficulty vs. Number of Non-Decodable Words",
) -> None:
    """Save a scatter of Rasch item difficulty against item composition.

    ``item_estimates`` is the ``items`` frame from ``irt.fit_rasch``. Where
    ``accuracy_by_nondecodable_count`` shows how each model fares as items get
    harder, this checks the construct itself: whether the difficulty the models
    reveal actually tracks the number of non-decodable words the items were
    built around.

    Items every answering model scored identically are drawn as open markers.
    Their difficulty is an optimizer bound rather than a point estimate, so a
    trend running through them is an artefact of the bound, not a measurement.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    at_bound = item_estimates["difficulty_estimate_status"] != STATUS_INTERIOR
    interior_items = item_estimates[~at_bound]
    bounded_items = item_estimates[at_bound]

    plt.figure(figsize=(10, 6))
    plt.scatter(
        interior_items["num_nondecodable_words"],
        interior_items["difficulty_logit_mml"],
        alpha=0.6,
        label=f"Point estimate (n={len(interior_items)})",
    )
    if not bounded_items.empty:
        plt.scatter(
            bounded_items["num_nondecodable_words"],
            bounded_items["difficulty_logit_mml"],
            facecolors="none",
            edgecolors="grey",
            alpha=0.6,
            label=f"At optimizer bound (n={len(bounded_items)})",
        )

    # The mean line is over point estimates only, for the same reason.
    if not interior_items.empty:
        mean_difficulty = interior_items.groupby("num_nondecodable_words")[
            "difficulty_logit_mml"
        ].mean()
        plt.plot(
            mean_difficulty.index,
            mean_difficulty.to_numpy(),
            color="black",
            marker="o",
            label="Mean point estimate",
        )

    plt.title(title)
    plt.xlabel("Number of Non-Decodable Words in Item")
    plt.ylabel("Estimated Difficulty (logits, higher = harder)")
    plt.legend()
    plt.savefig(output_path)
    plt.close()
