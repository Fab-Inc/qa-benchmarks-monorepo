"""Tests for the plots module."""

import pandas as pd
from phonics_benchmark.plots import (
    save_nondecodable_ratio_histogram,
)


def test_save_histogram_writes_file(tmp_path) -> None:
    """The histogram is written to the given path (creating parent dirs)."""
    items = pd.DataFrame({"nondecodable_ratio": [0.0, 0.1, 0.1, 0.3]})
    output_path = tmp_path / "plots" / "hist.png"

    save_nondecodable_ratio_histogram(items, output_path, title="Test")

    assert output_path.is_file()
    assert output_path.stat().st_size > 0
