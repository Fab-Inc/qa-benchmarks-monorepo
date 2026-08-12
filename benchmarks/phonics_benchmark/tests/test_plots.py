"""Tests for the plots module."""

import pandas as pd
import pytest
from phonics_benchmark.irt import STATUS_ALL_CORRECT, STATUS_INTERIOR
from phonics_benchmark.plots import (
    save_item_difficulty_by_nondecodable_count_plot,
    save_nondecodable_ratio_histogram,
)

ITEM_ESTIMATES = pd.DataFrame(
    {
        "num_nondecodable_words": [0, 2, 2, 5],
        "difficulty_logit_mml": [-1.2, 0.3, 0.8, 2.1],
        "difficulty_estimate_status": [
            STATUS_ALL_CORRECT,
            STATUS_INTERIOR,
            STATUS_INTERIOR,
            STATUS_INTERIOR,
        ],
    }
)


def test_save_histogram_writes_file(tmp_path) -> None:
    """The histogram is written to the given path (creating parent dirs)."""
    items = pd.DataFrame({"nondecodable_ratio": [0.0, 0.1, 0.1, 0.3]})
    output_path = tmp_path / "plots" / "hist.png"

    save_nondecodable_ratio_histogram(items, output_path, title="Test")

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_save_item_difficulty_plot_writes_file(tmp_path) -> None:
    """The difficulty scatter is written to the given path."""
    output_path = tmp_path / "plots" / "irt_item_difficulty.png"

    save_item_difficulty_by_nondecodable_count_plot(ITEM_ESTIMATES, output_path)

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_save_item_difficulty_plot_handles_only_bounded_items(tmp_path) -> None:
    """A fit with no point estimates still plots, rather than failing the report."""
    bounded_only = ITEM_ESTIMATES.assign(difficulty_estimate_status=STATUS_ALL_CORRECT)
    output_path = tmp_path / "irt_item_difficulty.png"

    save_item_difficulty_by_nondecodable_count_plot(bounded_only, output_path)

    assert output_path.is_file()


def test_save_item_difficulty_plot_requires_the_estimate_columns(tmp_path) -> None:
    """A frame that is not the fit's item estimates is rejected up front."""
    with pytest.raises(ValueError, match="missing required column"):
        save_item_difficulty_by_nondecodable_count_plot(
            ITEM_ESTIMATES.drop(columns="difficulty_estimate_status"),
            tmp_path / "plot.png",
        )
