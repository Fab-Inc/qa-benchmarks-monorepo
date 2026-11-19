"""Tests for building the metrics table, naming result files, reading a metric's
definition from the config, and building the frontier."""

import json
from pathlib import Path

import pandas as pd
import pytest
from fab_benchmark_publisher.benchmark_frontiers import build_frontier
from fab_benchmark_publisher.leaderboard_cli import DEFAULT_CONFIG_FILE
from fab_benchmark_publisher.metrics import (
    METRICS_COLUMNS,
    PLACEHOLDER_COLUMNS,
    build_metrics,
    load_metric_definition,
    result_file_name,
)

ACCURACY = {
    "metric_slug": "accuracy",
    "display_name": "Accuracy",
    "display_description": "Percentage of questions answered correctly.",
    "unit": "%",
    "is_higher_better": True,
    "is_bad_format_scored": False,
}
# cheap/small -> pricey/big; "mid" is beaten by "cheap" on the higher-is-better
# frontiers; "unknown" is not in the models CSV
METRIC_VALUE_BY_MODEL = {"cheap": 60.0, "mid": 50.0, "pricey": 90.0, "unknown": 99.0}


def metrics() -> pd.DataFrame:
    """A metrics table for LK-en accuracy built from METRIC_VALUE_BY_MODEL."""
    return build_metrics(
        METRIC_VALUE_BY_MODEL,
        benchmark_slug="LK-en",
        metric_slug="accuracy",
        experiment_slug="final-137-items",
    )


@pytest.fixture
def models_csv(tmp_path):
    """A models CSV with the price, size and open weights of the test models."""
    path = tmp_path / "models.csv"
    path.write_text(
        "model_id,provider,open,size,input_price_usd_per_mtok,"
        "output_price_usd_per_mtok,release_date\n"
        "cheap,P,True,7,1.0,1.0,01/2026\n"
        "mid,P,True,13,2.0,2.0,01/2026\n"
        "pricey,P,False,,10.0,10.0,01/2026\n"
    )
    return path


# --------------------------------------------------------------------------- #
# metrics table and file names
# --------------------------------------------------------------------------- #
def test_build_metrics_fills_only_key_and_value_columns() -> None:
    """One overall row per model with the full schema; placeholders stay empty."""
    built = metrics()

    assert list(built.columns) == list(METRICS_COLUMNS)
    assert list(built["model_slug"]) == sorted(METRIC_VALUE_BY_MODEL)
    assert (built["benchmark_subset"] == "").all()
    assert set(built["experiment_slug"]) == {"final-137-items"}
    assert built[list(PLACEHOLDER_COLUMNS)].isna().all().all()
    assert not any("cost" in c for c in METRICS_COLUMNS)


def test_file_names_have_no_experiment_slug() -> None:
    """Names are <benchmark>_<metric><suffix>, and a slug with '_' is rejected."""
    assert result_file_name("LK-en", "accuracy", ".csv") == "LK-en_accuracy.csv"
    with pytest.raises(ValueError, match="letters, digits"):
        result_file_name("LK_en", "accuracy", ".csv")


# --------------------------------------------------------------------------- #
# metric definition
# --------------------------------------------------------------------------- #
def test_metric_definition_comes_from_the_config(tmp_path) -> None:
    """The matching entry is returned as-is; a missing benchmark or metric raises."""
    config = tmp_path / "leaderboard_config.json"
    config.write_text(
        json.dumps(
            {
                "benchmarks": [
                    {"benchmark_domain": "DA-en", "metrics": []},
                    {"benchmark_domain": "LK-en", "metrics": [ACCURACY]},
                ]
            }
        )
    )

    assert load_metric_definition(config, "LK-en", "accuracy") == ACCURACY
    for benchmark_slug in ("DA-en", "XX-zz"):
        with pytest.raises(ValueError, match="defines no metric 'accuracy'"):
            load_metric_definition(config, benchmark_slug, "accuracy")


def test_repo_config_defines_lk_accuracy() -> None:
    """The real repo-root config defines accuracy for both LK languages."""
    config = Path(__file__).resolve().parents[3] / DEFAULT_CONFIG_FILE
    for benchmark_slug in ("LK-en", "LK-sw"):
        assert load_metric_definition(config, benchmark_slug, "accuracy")["unit"] == "%"


# --------------------------------------------------------------------------- #
# frontier
# --------------------------------------------------------------------------- #
def frontier_models(frontier: dict) -> dict[str, list[str]]:
    """The models on each axis's frontier, {axis: [model_slug, ...]}."""
    return {
        axis_frontier["axis"]: [p["model_slug"] for p in axis_frontier["points"]]
        for axis_frontier in frontier["frontiers"]
    }


def test_price_and_size_frontiers(models_csv) -> None:
    """Size keeps only open-weight models; unknown models are listed, not placed."""
    frontier = build_frontier(metrics(), is_higher_better=True, models_csv=models_csv)

    assert frontier_models(frontier) == {
        "price": ["cheap", "pricey"],
        "size": ["cheap"],  # pricey is closed-weight, mid is beaten by cheap
    }
    assert frontier["models_without_metadata"] == ["unknown"]
    assert frontier["frontiers"][0]["x"] == "blended_price_usd_per_mtok"
    assert frontier["experiment_slug"] == "final-137-items"


def test_frontier_from_a_metrics_csv(tmp_path, models_csv) -> None:
    """A metrics CSV read back with plain pd.read_csv (empty subset -> NaN) still
    counts as overall rows."""
    path = tmp_path / "metrics.csv"
    metrics().to_csv(path, index=False)

    frontier = build_frontier(
        pd.read_csv(path), is_higher_better=True, models_csv=models_csv
    )
    assert frontier_models(frontier)["price"] == ["cheap", "pricey"]


def test_lower_is_better_flips_the_frontier(models_csv) -> None:
    """With is_higher_better False the frontier follows the lowest values."""
    frontier = build_frontier(metrics(), is_higher_better=False, models_csv=models_csv)

    # by price: cheap (60), then mid (50) is lower so it improves; pricey (90) does not
    assert frontier_models(frontier)["price"] == ["cheap", "mid"]
