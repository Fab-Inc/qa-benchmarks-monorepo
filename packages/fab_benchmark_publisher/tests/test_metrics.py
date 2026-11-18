"""Tests for the metrics file: aggregation, bootstrap, validation, CSV round-trip."""

import numpy as np
import pandas as pd
import pytest
from fab_benchmark_publisher.metrics import (
    METRICS_COLUMNS,
    USAGE_COLUMNS,
    aggregate_item_scores,
    bootstrap_mean,
    canonical_subset,
    metrics_file_name,
    read_metrics_csv,
    validate_metrics,
    write_metrics_csv,
)
from fab_benchmark_publisher.result_metadata import MetricMeta

TS = "2026-11-18T12:00:00+00:00"


def metric(**overrides) -> MetricMeta:
    fields = {
        "metric_slug": "accuracy",
        "display_name": "Accuracy",
        "display_description": "Share of items answered correctly.",
        "unit": "%",
        "is_higher_better": True,
        "is_bad_format_scored": False,
    }
    return MetricMeta(**{**fields, **overrides})


def items() -> pd.DataFrame:
    """Two models x four items in two domains; model b has one bad-format item."""
    return pd.DataFrame(
        [
            ("a", 1, 1.0, False, "D1"),
            ("a", 2, 0.0, False, "D1"),
            ("a", 3, 1.0, False, "D2"),
            ("a", 4, 1.0, False, "D2"),
            ("b", 1, 1.0, False, "D1"),
            ("b", 2, None, True, "D1"),
            ("b", 3, 0.0, False, "D2"),
            ("b", 4, 0.0, False, "D2"),
        ],
        columns=["model_slug", "item_id", "item_score", "is_bad_format", "domain"],
    )


def aggregate(df=None, **kwargs) -> pd.DataFrame:
    params = {
        "benchmark_slug": "LK-en",
        "metric": metric(),
        "experiment_slug": "final-137-items",
        "timestamp_metric": TS,
        "subset_groupings": [["domain"]],
    }
    return aggregate_item_scores(items() if df is None else df, **{**params, **kwargs})


def row(metrics, model_slug, subset=""):
    match = metrics[
        (metrics["model_slug"] == model_slug)
        & (metrics["benchmark_subset"].fillna("") == subset)
    ]
    assert len(match) == 1
    return match.iloc[0]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def test_canonical_subset_is_order_independent_and_empty_for_overall() -> None:
    assert canonical_subset({"b": 2, "a": "x"}) == '{"a":"x","b":"2"}'
    assert canonical_subset({}) == ""


def test_metrics_file_name_joins_the_key_slugs() -> None:
    assert (
        metrics_file_name("LK-en", "accuracy", "final-137-items")
        == "LK-en_accuracy_final-137-items.csv"
    )
    with pytest.raises(ValueError, match="no '_'"):
        metrics_file_name("LK_en", "accuracy", "final")


def test_bootstrap_of_constant_scores_has_zero_width() -> None:
    stderr, low, high = bootstrap_mean(np.ones(10), rng=np.random.default_rng(0))
    assert (stderr, low, high) == (0.0, 1.0, 1.0)


def test_bootstrap_interval_brackets_the_mean_and_matches_binomial_se() -> None:
    scores = np.array([1.0] * 60 + [0.0] * 40)
    stderr, low, high = bootstrap_mean(
        scores, n_bootstrap=4000, rng=np.random.default_rng(0)
    )
    assert low < 0.6 < high
    # sqrt(p(1-p)/n) = 0.049
    assert stderr == pytest.approx(0.049, abs=0.004)


# --------------------------------------------------------------------------- #
# aggregate_item_scores
# --------------------------------------------------------------------------- #
def test_aggregate_emits_overall_and_subset_rows() -> None:
    metrics = aggregate()

    assert list(metrics.columns) == list(METRICS_COLUMNS)
    assert set(metrics["benchmark_subset"]) == {
        "",
        '{"domain":"D1"}',
        '{"domain":"D2"}',
    }
    assert len(metrics) == 6
    a = row(metrics, "a")
    assert a["metric_value"] == 75.0
    assert (a["n_items_scored"], a["n_repeats"]) == (4, 1)
    assert a["metric_ci_low"] <= a["metric_value"] <= a["metric_ci_high"]
    assert a["timestamp_metric"] == TS
    assert a[list(USAGE_COLUMNS)].isna().all()


def test_without_groupings_only_overall_rows() -> None:
    metrics = aggregate(subset_groupings=())

    assert set(metrics["benchmark_subset"]) == {""}
    assert len(metrics) == 2


def test_bad_format_excluded_when_not_scored() -> None:
    b = row(aggregate(), "b")

    assert b["metric_value"] == pytest.approx(100 / 3, abs=1e-4)
    assert (b["n_items_scored"], b["n_items_bad_format"]) == (3, 1)


def test_bad_format_counted_when_scored() -> None:
    df = items()
    df.loc[df["item_score"].isna(), "item_score"] = 0.0

    b = row(aggregate(df, metric=metric(is_bad_format_scored=True)), "b")
    assert b["metric_value"] == 25.0
    assert (b["n_items_scored"], b["n_items_bad_format"]) == (4, 1)


def test_scored_bad_format_item_without_score_is_an_error() -> None:
    with pytest.raises(ValueError, match="no item_score"):
        aggregate(metric=metric(is_bad_format_scored=True))


def test_subset_with_only_bad_format_items_has_empty_estimates() -> None:
    df = items()
    first_b = (df["model_slug"] == "b") & (df["item_id"] == 1)
    df.loc[first_b, "item_score"] = None
    df.loc[first_b, "is_bad_format"] = True

    b_d1 = row(aggregate(df), "b", '{"domain":"D1"}')
    assert b_d1["n_items_scored"] == 0
    estimates = ["metric_value", "metric_stderr", "metric_ci_low", "metric_ci_high"]
    assert b_d1[estimates].isna().all()


def test_intervals_are_reproducible_and_independent_of_other_models() -> None:
    both = aggregate(n_bootstrap=200)
    only_a = aggregate(items()[items()["model_slug"] == "a"], n_bootstrap=200)

    cols = ["metric_stderr", "metric_ci_low", "metric_ci_high"]
    assert row(both, "a")[cols].tolist() == row(only_a, "a")[cols].tolist()
    assert both.equals(aggregate(n_bootstrap=200))


def test_timestamp_defaults_to_now_in_utc() -> None:
    metrics = aggregate(timestamp_metric=None)
    assert metrics["timestamp_metric"].iloc[0].endswith("+00:00")


def test_duplicate_items_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        aggregate(pd.concat([items(), items().head(1)]))


def test_missing_subset_column_is_rejected() -> None:
    with pytest.raises(ValueError, match="grade"):
        aggregate(subset_groupings=[["grade"]])


# --------------------------------------------------------------------------- #
# validate_metrics
# --------------------------------------------------------------------------- #
def test_a_model_with_two_results_for_one_key_is_rejected() -> None:
    metrics = aggregate()
    with pytest.raises(ValueError, match="repeat a model's result"):
        validate_metrics(pd.concat([metrics, metrics.head(1)]))


def test_two_experiments_in_one_file_are_rejected() -> None:
    metrics = aggregate()
    metrics.loc[0, "experiment_slug"] = "other-run"
    with pytest.raises(ValueError, match="one experiment_slug per file"):
        validate_metrics(metrics)


def test_slug_with_underscore_is_rejected() -> None:
    with pytest.raises(ValueError, match="no '_'"):
        aggregate(experiment_slug="final_137")


def test_non_canonical_subset_is_rejected() -> None:
    metrics = aggregate()
    metrics.loc[metrics["benchmark_subset"] != "", "benchmark_subset"] = (
        '{"domain": "D1"}'
    )
    with pytest.raises(ValueError, match="not canonical"):
        validate_metrics(metrics)


def test_empty_json_object_is_not_the_overall_subset() -> None:
    metrics = aggregate(subset_groupings=())
    metrics["benchmark_subset"] = "{}"
    with pytest.raises(ValueError, match="not canonical"):
        validate_metrics(metrics)


def test_value_outside_interval_is_rejected() -> None:
    metrics = aggregate()
    metrics.loc[0, "metric_ci_high"] = metrics.loc[0, "metric_value"] - 1
    with pytest.raises(ValueError, match="outside"):
        validate_metrics(metrics)


def test_percent_value_above_100_is_rejected_given_the_metric() -> None:
    metrics = aggregate()
    metrics.loc[0, ["metric_value", "metric_ci_high"]] = 101.0
    validate_metrics(metrics)  # no unit known: fine
    with pytest.raises(ValueError, match="outside 0-100"):
        validate_metrics(metrics, metric=metric())


@pytest.mark.parametrize("timestamp", ["2026-11-18T12:00:00+02:00", "2026-11-18"])
def test_non_utc_timestamp_is_rejected(timestamp) -> None:
    with pytest.raises(ValueError, match="not UTC"):
        aggregate(timestamp_metric=timestamp)


def test_usage_columns_are_checked_when_filled() -> None:
    metrics = aggregate()
    metrics["output_total_tokens_per_repeat_mean"] = 10.0
    metrics["output_reason_tokens_per_repeat_mean"] = 4.0
    metrics["output_non_reason_tokens_per_repeat_mean"] = 6.0
    validate_metrics(metrics)

    metrics["output_non_reason_tokens_per_repeat_mean"] = 7.0
    with pytest.raises(ValueError, match="do not add up"):
        validate_metrics(metrics)


def test_validation_reports_every_problem_at_once() -> None:
    metrics = aggregate()
    metrics.loc[0, "metric_stderr"] = -1.0
    metrics.loc[1, "model_slug"] = ""
    with pytest.raises(ValueError) as excinfo:
        validate_metrics(metrics)
    assert "metric_stderr is negative" in str(excinfo.value)
    assert "model_slug is empty" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# CSV round-trip
# --------------------------------------------------------------------------- #
def test_csv_is_named_after_its_key_and_round_trips(tmp_path) -> None:
    metrics = aggregate(experiment_slug="1.0")

    path = write_metrics_csv(metrics, tmp_path)
    back = read_metrics_csv(path, metric=metric())

    assert path.name == "LK-en_accuracy_1.0.csv"
    assert back["experiment_slug"].eq("1.0").all()  # not the float 1.0
    assert set(back["benchmark_subset"]) == {"", '{"domain":"D1"}', '{"domain":"D2"}'}
    assert back[list(USAGE_COLUMNS)].isna().all().all()


def test_renamed_csv_is_rejected(tmp_path) -> None:
    path = write_metrics_csv(aggregate(), tmp_path)
    renamed = path.rename(tmp_path / "LK-sw_accuracy_final-137-items.csv")

    with pytest.raises(ValueError, match="should be LK-en_accuracy"):
        read_metrics_csv(renamed)
