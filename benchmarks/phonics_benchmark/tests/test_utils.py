"""Tests for the utils module."""

import pandas as pd
import pytest
from phonics_benchmark.utils import require_columns, warn_fully_decodable_items


def test_require_column_decorator_raises_if_missing() -> None:
    """The decorator raises a ValueError if the required column is missing."""

    @require_columns(["foo"])
    def dummy_func(x: pd.DataFrame) -> None:
        return None

    with pytest.raises(ValueError):
        dummy_func(pd.DataFrame({"bar": [1, 2, 3]}))


def test_warn_fully_decodable_items_logs_percent(caplog) -> None:
    """The decorator logs the fraction of items that are fully decodable."""

    @warn_fully_decodable_items(threshold=0.5)
    def dummy_func(x: pd.DataFrame) -> None:
        return None

    df = pd.DataFrame({"nondecodable_words": [[], ["b"], ["a"]]})
    with caplog.at_level("INFO"):
        dummy_func(df)
    assert "% of the items are fully decodable" in caplog.text


def test_warn_fully_decodable_items_raises_warning() -> None:
    """The decorator raises a warning if a significant fraction of items are fully
    decodable."""

    @warn_fully_decodable_items(threshold=0.5)
    def dummy_func(x: pd.DataFrame) -> None:
        return None

    df = pd.DataFrame({"nondecodable_words": [[], [], ["a"]]})
    with pytest.warns(UserWarning) as warning:
        dummy_func(df)
    assert "% of the items are fully decodable" in str(warning[0].message)
