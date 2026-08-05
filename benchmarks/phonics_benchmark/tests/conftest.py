"""Shared fixtures for phonics benchmark tests."""

import json
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv


@pytest.fixture(scope="session", autouse=True)
def load_env() -> None:
    """Load repo-root .env then benchmark-specific .env (bucket config)."""
    repo_root = Path(__file__).parents[3]
    benchmark_root = Path(__file__).parents[1]
    load_dotenv(repo_root / ".env")
    load_dotenv(benchmark_root / ".env")


@pytest.fixture(scope="session")
def tusome_stage_11_item() -> dict[str, Any]:
    """Load a representative item from the generated Tusome trial data."""
    data_path = (
        Path(__file__).parents[1]
        / "data"
        / "kiswahili"
        / "tusome_kenya_llm_trial_cases.jsonl"
    )
    with data_path.open(encoding="utf-8") as file:
        rows = (json.loads(line) for line in file)
        return next(row for row in rows if row["benchmark_item_id"] == "SW-KE-S11")
