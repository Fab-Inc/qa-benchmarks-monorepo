"""Tests for phonics benchmark result persistence."""

import json
from pathlib import Path

import pandas as pd
from phonics_benchmark.results import save_results


def test_save_results_writes_jsonl_atomically(tmp_path: Path) -> None:
    """The canonical JSONL file is replaced via a temporary sibling file."""

    results_path = tmp_path / "responses" / "responses_model-a.jsonl"
    results = pd.DataFrame(
        [
            {"result_id": "r1", "output": "first"},
            {"result_id": "r2", "output": "second"},
        ]
    )

    saved_path = save_results(results, results_path)

    assert saved_path == results_path
    rows = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["result_id"] for row in rows] == ["r1", "r2"]
    assert list(results_path.parent.glob("*.tmp")) == []
    assert list(results_path.parent.glob(".*.tmp")) == []
