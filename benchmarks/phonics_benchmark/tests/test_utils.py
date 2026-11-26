"""Tests for the utils module."""

from pathlib import Path

import pandas as pd
import pytest
from phonics_benchmark.utils import load_jsonl, require_columns


def test_require_column_decorator_raises_if_missing() -> None:
    """The decorator raises a ValueError if the required column is missing."""

    @require_columns(["foo"])
    def dummy_func(x: pd.DataFrame) -> None:
        return None

    with pytest.raises(ValueError):
        dummy_func(pd.DataFrame({"bar": [1, 2, 3]}))


@pytest.fixture
def tmp_jsonl_file(tmp_path) -> Path:
    file_path = tmp_path / "test.jsonl"
    file_path.write_text('{"a": 1}\n{"b": 2}\n\n')
    return file_path


def test_load_jsonl(tmp_jsonl_file) -> None:
    """load_jsonl correctly loads non-empty JSONL records."""
    result = load_jsonl(tmp_jsonl_file)
    assert result == [{"a": 1}, {"b": 2}]
