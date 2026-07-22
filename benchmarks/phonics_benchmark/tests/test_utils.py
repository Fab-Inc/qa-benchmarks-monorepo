import pandas as pd
import pytest
from phonics_benchmark.results import require_columns


def test_require_column_decorator_raises_if_missing() -> None:
    """The decorator raises a ValueError if the required column is missing."""

    @require_columns(["foo"])
    def dummy_func(x: pd.DataFrame) -> None:
        return None

    with pytest.raises(ValueError):
        dummy_func(pd.DataFrame({"bar": [1, 2, 3]}))
