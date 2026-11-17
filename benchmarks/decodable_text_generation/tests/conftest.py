"""Shared fixtures for decodable text generation benchmark tests."""

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
def generation_benchmark_item() -> dict[str, Any]:
    """Load the first item from the generation items contract fixture."""
    fixture_path = (
        Path(__file__).parent / "fixtures" / "generation_benchmark_items.jsonl"
    )
    with fixture_path.open(encoding="utf-8") as file:
        return json.loads(file.readline())
