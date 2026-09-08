"""Tests for the report's inspection plots."""

from pathlib import Path

import pandas as pd
import pytest
from phonics_benchmark.plots import (
    save_item_phase_histogram,
    save_score_by_phase_plot,
    save_score_by_scenario_plot,
)


@pytest.fixture
def by_phase() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"model": "m1", "curriculum_phase": "very-early", "graded_score": 0.1},
            {"model": "m1", "curriculum_phase": "late", "graded_score": 0.8},
            {"model": "m2", "curriculum_phase": "very-early", "graded_score": 0.2},
            {"model": "m2", "curriculum_phase": "late", "graded_score": 0.9},
        ]
    )


@pytest.fixture
def by_scenario() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"model": "m1", "scenario_id": "story", "graded_score": 0.5},
            {"model": "m1", "scenario_id": "dialogue", "graded_score": 0.6},
        ]
    )


def test_saves_score_by_phase(by_phase: pd.DataFrame, tmp_path: Path) -> None:
    """The plot file is written, creating its parent directory."""
    output = tmp_path / "report" / "score_by_phase.png"
    save_score_by_phase_plot(by_phase, output)
    assert output.is_file()


def test_saves_score_by_scenario(by_scenario: pd.DataFrame, tmp_path: Path) -> None:
    """The scenario bar chart is written too."""
    output = tmp_path / "score_by_scenario.png"
    save_score_by_scenario_plot(by_scenario, output)
    assert output.is_file()


def test_saves_item_phase_histogram(tmp_path: Path) -> None:
    """The item-set composition plot is written at build time."""
    items = pd.DataFrame({"curriculum_phase": ["early", "early", "late"]})
    output = tmp_path / "items.png"
    save_item_phase_histogram(items, output)
    assert output.is_file()


def test_requires_its_columns(tmp_path: Path) -> None:
    """A frame without the score column fails rather than writing a blank plot."""
    with pytest.raises(ValueError, match="missing required column"):
        save_score_by_phase_plot(pd.DataFrame({"model": ["m1"]}), tmp_path / "x.png")
