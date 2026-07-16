"""Shared fixtures for phonics benchmark tests."""

import csv
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def tusome_stage_11_item() -> dict[str, str]:
    """Load a representative item from the generated Tusome trial data."""
    data_path = (
        Path(__file__).parents[1]
        / "data"
        / "kiswahili"
        / "tusome_kenya_llm_trial_cases.csv"
    )
    with data_path.open(encoding="utf-8", newline="") as file:
        rows = csv.DictReader(file)
        return next(row for row in rows if row["benchmark_item_id"] == "SW-KE-S11")
