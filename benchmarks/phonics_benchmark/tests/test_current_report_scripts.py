"""Regression checks for the current report and snapshot-comparison drivers."""

import base64
import datetime
import hashlib
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
finalize = importlib.import_module("finalize_phonics_refresh")
ba = importlib.import_module("analyze_balanced_accuracy")


@pytest.mark.parametrize(
    "scale, explicit_scale", [(1, False), (100, False), (1, True), (100, True)]
)
def test_comparisons_accept_fractional_and_point_adjusted_scores(scale, explicit_scale):
    # All point values below one: the scale cannot safely be inferred from range.
    raw = np.array([0.501, 0.502])
    frame = pd.DataFrame(
        {"balanced_accuracy": raw, "adjusted_balanced_accuracy": scale * (2 * raw - 1)}
    )
    for suffix, offset in (("_ci_low", -0.01), ("_ci_high", 0.01)):
        frame["balanced_accuracy" + suffix] = raw + offset
        frame["adjusted_balanced_accuracy" + suffix] = scale * (2 * (raw + offset) - 1)
    if explicit_scale:
        frame["adjusted_balanced_accuracy_scale"] = scale
    result = finalize.adjusted_points(frame)
    np.testing.assert_allclose(result.adjusted_balanced_accuracy, [0.2, 0.4])
    assert result.adjusted_balanced_accuracy_ci_low.lt(0).all()
    assert result.adjusted_balanced_accuracy_scale.eq(100).all()


def test_gold_metadata_is_checked_against_local_bytes_and_date(tmp_path):
    gold = tmp_path / "gold.jsonl"
    gold.write_text("{}\n")
    md5 = base64.b64encode(hashlib.md5(gold.read_bytes()).digest()).decode()
    metadata = tmp_path / "metadata.txt"
    metadata.write_text(
        "gs://bucket/gold.jsonl:\n"
        "  Update Time: 2026-09-22T08:18:07Z\n"
        f"  Generation: 123\n  Hash (MD5): {md5}\n"
    )
    today = datetime.datetime(2026, 9, 22, 12, tzinfo=datetime.UTC)
    result = finalize.inspect_gold_metadata(metadata, gold, now=today)
    assert result["updated_today_or_yesterday"]
    assert result["generation"] == "123"
    assert not result["remote_latest_checked"]
    assert not finalize.inspect_gold_metadata(
        metadata, gold, now=today + datetime.timedelta(days=2)
    )["updated_today_or_yesterday"]
    gold.write_text('{"changed": true}\n')
    with pytest.raises(ValueError, match="does not match"):
        finalize.inspect_gold_metadata(metadata, gold, now=today)


def test_vocabulary_loader_preserves_unique_words_and_normalizes_gold(tmp_path):
    path = tmp_path / "items.jsonl"
    path.write_text(
        '{"benchmark_item_id":"a","content":"Cat cat sat.",'
        '"nondecodable_words":["CAT"]}\n'
    )
    items = ba.load_items(path, "en")
    assert items.loc["a", "n_vocab"] == 2
    assert items.loc["a", "n_gold"] == 1
    assert items.loc["a", "gold_set"] == {"cat"}


def test_importing_drivers_does_not_run_an_analysis():
    for name in (
        "apply_refreshed_gold",
        "run_balanced_accuracy_reports",
        "plot_adjusted_balanced_accuracy",
        "finalize_phonics_refresh",
    ):
        module = importlib.import_module(name)
        assert callable(module.main)


def test_unknown_adjusted_scale_is_rejected():
    with pytest.raises(ValueError, match="Unknown adjusted"):
        finalize.adjusted_points(
            pd.DataFrame(
                {
                    "balanced_accuracy": [0.6],
                    "adjusted_balanced_accuracy": [2],
                    "adjusted_balanced_accuracy_scale": [10],
                }
            )
        )


def test_gold_refresh_accepts_new_gate_column_but_rejects_changed_counts(tmp_path):
    refresh = importlib.import_module("apply_refreshed_gold")
    previous, current = tmp_path / "previous.csv", tmp_path / "current.csv"
    frame = pd.DataFrame(
        {
            "model": ["m"],
            "expected_items": [1000],
            "scorable_items": [900],
            "missing_responses": [100],
            "passes_80_percent_gate": [True],
        }
    )
    frame.to_csv(previous, index=False)
    frame["passes_90_percent_gate"] = False
    frame.to_csv(current, index=False)
    refresh.verify_coverage_unchanged(previous, current)
    frame["scorable_items"] = 901
    frame.to_csv(current, index=False)
    with pytest.raises(AssertionError):
        refresh.verify_coverage_unchanged(previous, current)


@pytest.mark.parametrize(
    "total,counts,expected",
    [(991, [891, 892], [False, True]), (1000, [900, 901], [False, True])],
)
def test_balanced_report_uses_strict_90_percent_gate(
    tmp_path, monkeypatch, total, counts, expected
):
    driver = importlib.import_module("run_balanced_accuracy_reports")
    models = ["below", "eligible"]
    metadata = pd.DataFrame(
        {
            "model": models,
            "n_items": counts,
            "expected_items": total,
            "score": 0.8,
            "ci_low": 0.7,
            "ci_high": 0.9,
            "n_bootstrap": 1000,
            "ci_level": 0.95,
        }
    )
    saved = pd.DataFrame(
        {
            "model": models,
            "n_scorable": counts,
            "expected_items": total,
            "recall": 0.8,
            "specificity": 0.8,
            "balanced_accuracy": 0.8,
            "balanced_accuracy_ci_low": 0.7,
            "balanced_accuracy_ci_high": 0.9,
            "n_bootstrap": 1000,
            "ci_level": 0.95,
            "passes_90_percent_gate": expected,
        }
    )
    monkeypatch.setattr(driver, "load_plot_table", lambda *_: (metadata, {}))
    monkeypatch.setattr(driver, "read_csv", lambda *_: saved)
    table, _ = driver.load_balanced_table(tmp_path, "english")
    assert (
        table.set_index("model").loc[models, "passes_90_percent_gate"].tolist()
        == expected
    )
    saved["passes_90_percent_gate"] = True
    with pytest.raises(ValueError, match="Coverage gate"):
        driver.load_balanced_table(tmp_path, "english")
