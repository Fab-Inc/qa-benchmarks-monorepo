"""Tests for the bootstrap confidence intervals on per-model metrics."""

import math

import numpy as np
import pandas as pd
import pytest
from phonics_benchmark.bootstrap import bootstrap_metric_cis
from phonics_benchmark.scoring import summarize_by_model


def scored_frame(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal scored frame, filling the columns the metrics ignore.

    `summarize_by_model` needs more columns than the bootstrap does, so its
    require_columns list is satisfied here to let the two be compared on one
    frame.
    """
    frame = pd.DataFrame(rows)
    frame["has_actual_nondecodable"] = frame["fn_count"] + frame["tp_count"] > 0
    frame["has_predicted_nondecodable"] = frame["fp_count"] + frame["tp_count"] > 0
    frame["precision"] = math.nan
    frame["recall"] = math.nan
    frame["f1"] = math.nan
    return frame


def synthetic_frame(model_to_correct: dict[str, list[bool]]) -> pd.DataFrame:
    """Build one item per flag per model, correct where the flag is True."""
    rows = []
    for model, correct_flags in model_to_correct.items():
        for position, correct in enumerate(correct_flags):
            rows.append(
                {
                    "model": model,
                    "benchmark_item_id": f"p0_SW-KE-T{position:03d}",
                    "perfect_match": correct,
                    "tp_count": 1 if correct else 0,
                    "fp_count": 0 if correct else 1,
                    "fn_count": 0,
                }
            )
    return scored_frame(rows)


# Two models over four items. m1 gets three of four items exactly right, m2 gets
# two.
TWO_MODEL_ROWS = [
    {
        "model": "m1",
        "benchmark_item_id": "p1_SW-KE-A01",
        "perfect_match": True,
        "tp_count": 2,
        "fp_count": 0,
        "fn_count": 0,
    },
    {
        "model": "m1",
        "benchmark_item_id": "p2_SW-KE-A01",
        "perfect_match": True,
        "tp_count": 1,
        "fp_count": 0,
        "fn_count": 0,
    },
    {
        "model": "m1",
        "benchmark_item_id": "p1_SW-KE-A02",
        "perfect_match": False,
        "tp_count": 1,
        "fp_count": 2,
        "fn_count": 1,
    },
    {
        "model": "m1",
        "benchmark_item_id": "p2_SW-KE-A02",
        "perfect_match": True,
        "tp_count": 3,
        "fp_count": 0,
        "fn_count": 0,
    },
    {
        "model": "m2",
        "benchmark_item_id": "p1_SW-KE-A01",
        "perfect_match": False,
        "tp_count": 0,
        "fp_count": 1,
        "fn_count": 2,
    },
    {
        "model": "m2",
        "benchmark_item_id": "p2_SW-KE-A01",
        "perfect_match": True,
        "tp_count": 1,
        "fp_count": 0,
        "fn_count": 0,
    },
    {
        "model": "m2",
        "benchmark_item_id": "p1_SW-KE-A02",
        "perfect_match": False,
        "tp_count": 2,
        "fp_count": 3,
        "fn_count": 0,
    },
    {
        "model": "m2",
        "benchmark_item_id": "p2_SW-KE-A02",
        "perfect_match": True,
        "tp_count": 3,
        "fp_count": 0,
        "fn_count": 0,
    },
]


def test_point_estimates_match_summarize_by_model() -> None:
    """The CI table's point estimates are the same numbers the summary reports.

    This is the test that catches the two code paths drifting apart.
    """
    frame = scored_frame(TWO_MODEL_ROWS)

    summary = summarize_by_model(frame).set_index("model_name")
    report = bootstrap_metric_cis(frame, n_bootstrap=50).set_index("model")

    for model in ("m1", "m2"):
        for metric in (
            "accuracy",
            "pooled_precision",
            "pooled_recall",
            "pooled_f1",
            "pooled_f2",
        ):
            assert report.loc[model, metric] == pytest.approx(
                summary.loc[model, metric]
            ), f"{model} {metric}"

    # Spelled out for m1: 3 of 4 items exact; pooled tp=7, fp=2, fn=1.
    assert report.loc["m1", "accuracy"] == 0.75
    assert report.loc["m1", "pooled_precision"] == pytest.approx(7 / 9)  # 7/(7+2)
    assert report.loc["m1", "pooled_recall"] == pytest.approx(7 / 8)  # 7/(7+1)
    assert report.loc["m1", "pooled_f1"] == pytest.approx(14 / 17)  # 2*7/(2*7+2+1)
    assert report.loc["m1", "pooled_f2"] == pytest.approx(35 / 41)  # 5*7/(5*7+4*1+2)


def test_precision_and_recall_bracket_the_f_scores() -> None:
    """A model that over-flags shows low precision beside high recall.

    The reason these two columns are worth reporting: flagging every word gives
    recall 1 and precision at the base rate, and F2 alone hides that.
    """
    rows = [
        {
            "model": "flag-all",
            "benchmark_item_id": f"p0_SW-KE-T{i:02d}",
            "perfect_match": False,
            "tp_count": 1,
            "fp_count": 9,
            "fn_count": 0,
        }
        for i in range(8)
    ]

    report = bootstrap_metric_cis(scored_frame(rows), n_bootstrap=200).set_index(
        "model"
    )

    assert report.loc["flag-all", "pooled_recall"] == 1.0  # nothing missed
    assert report.loc["flag-all", "pooled_precision"] == pytest.approx(0.1)  # 8/(8+72)
    # Recall is 1 in every replicate, so its interval is degenerate.
    assert report.loc["flag-all", "pooled_recall_ci_low"] == 1.0
    assert report.loc["flag-all", "pooled_recall_ci_high"] == 1.0
    # Both F-scores land between precision and recall, and F2 sits far above F1
    # because it weights recall four times as heavily.
    flag_all = report.loc["flag-all"]
    assert flag_all["pooled_precision"] < flag_all["pooled_f1"]
    assert flag_all["pooled_f1"] < flag_all["pooled_f2"]
    assert flag_all["pooled_f2"] < flag_all["pooled_recall"]


def test_same_seed_reproduces() -> None:
    """A fixed seed makes reruns of the report give the same interval."""
    frame = synthetic_frame({"m1": [True] * 25 + [False] * 15})

    first = bootstrap_metric_cis(frame, n_bootstrap=200, seed=0)
    again = bootstrap_metric_cis(frame, n_bootstrap=200, seed=0)

    pd.testing.assert_frame_equal(first, again)


def test_models_share_one_draw_per_replicate() -> None:
    """Every model is resampled with the same item indices in a replicate.

    Two models with identical per-item results get identical intervals only if
    they see the same draw; independent draws would give different bounds.
    """
    flags = [True] * 25 + [False] * 15
    report = bootstrap_metric_cis(
        synthetic_frame({"a": flags, "b": flags}), n_bootstrap=200
    ).set_index("model")

    pd.testing.assert_series_equal(report.loc["a"], report.loc["b"], check_names=False)


def test_model_correct_on_every_item_gets_a_degenerate_interval() -> None:
    """With no variation to resample, both bounds sit on the point estimate."""
    rows = [
        {
            "model": "m1",
            "benchmark_item_id": f"p1_SW-KE-A0{i}",
            "perfect_match": True,
            "tp_count": 1,
            "fp_count": 0,
            "fn_count": 0,
        }
        for i in range(1, 5)
    ]

    report = bootstrap_metric_cis(scored_frame(rows), n_bootstrap=100)

    assert report.loc[0, "accuracy"] == 1.0
    assert report.loc[0, "accuracy_ci_low"] == 1.0
    assert report.loc[0, "accuracy_ci_high"] == 1.0


def test_lower_confidence_level_gives_a_narrower_interval() -> None:
    """A 50% interval sits inside the 95% one."""
    frame = scored_frame(TWO_MODEL_ROWS)

    wide = bootstrap_metric_cis(frame, n_bootstrap=500, ci_level=0.95)
    narrow = bootstrap_metric_cis(frame, n_bootstrap=500, ci_level=0.5)

    wide_width = wide["accuracy_ci_high"] - wide["accuracy_ci_low"]
    narrow_width = narrow["accuracy_ci_high"] - narrow["accuracy_ci_low"]
    assert (narrow_width < wide_width).all()


def test_unbalanced_panel_scores_each_model_over_the_items_it_has() -> None:
    """A model missing an item is scored on its own items, leaving others alone.

    Failed provider calls and unparseable responses are dropped before scoring,
    so a run can reach here with one model short of an item.
    """
    dropped = ("m2", "p1_SW-KE-A02")
    rows = [
        row
        for row in TWO_MODEL_ROWS
        if (row["model"], row["benchmark_item_id"]) != dropped
    ]

    report = bootstrap_metric_cis(scored_frame(rows), n_bootstrap=200)
    report = report.set_index("model")

    assert report.loc["m1", "accuracy"] == 0.75  # all four items, unchanged
    assert report.loc["m2", "accuracy"] == pytest.approx(2 / 3)  # its three items
    metrics = [
        "accuracy",
        "pooled_precision",
        "pooled_recall",
        "pooled_f1",
        "pooled_f2",
    ]
    # Replicates that draw only m2's missing item are NaN for m2 and skipped, so
    # the bounds still come out defined.
    columns = [
        column
        for metric in metrics
        for column in (metric, f"{metric}_ci_low", f"{metric}_ci_high")
    ]
    assert not report[columns].isna().to_numpy().any()
    # The draw is over the union of items, so m2 reports 4 though it has 3.
    assert (report["n_items_resampled"] == 4).all()


@pytest.mark.parametrize("bad_level", [0.0, 1.0, 1.5, -0.1])
def test_confidence_level_outside_the_open_unit_interval_is_rejected(
    bad_level: float,
) -> None:
    """A level of 0 or 1 has no percentile interpretation here."""
    with pytest.raises(ValueError, match="ci_level"):
        bootstrap_metric_cis(scored_frame(TWO_MODEL_ROWS), ci_level=bad_level)


@pytest.mark.parametrize("bad_count", [0, -1])
def test_non_positive_bootstrap_count_is_rejected(bad_count: int) -> None:
    """At least one replicate is needed to take a percentile."""
    with pytest.raises(ValueError, match="n_bootstrap"):
        bootstrap_metric_cis(scored_frame(TWO_MODEL_ROWS), n_bootstrap=bad_count)


# Turns any RuntimeWarning from a 0/0 into a failure, so undefined metrics stay
# silent NaNs rather than noise in the report run.
@pytest.mark.filterwarnings("error")
def test_one_item_per_model_does_not_raise() -> None:
    """The smoke run is one item per model, which must still produce a table."""
    rows = [
        {
            "model": "m1",
            "benchmark_item_id": "p1_SW-KE-A01",
            "perfect_match": True,
            "tp_count": 0,
            "fp_count": 0,
            "fn_count": 0,
        },
        {
            "model": "m2",
            "benchmark_item_id": "p2_SW-KE-A02",
            "perfect_match": False,
            "tp_count": 0,
            "fp_count": 1,
            "fn_count": 0,
        },
    ]

    report = bootstrap_metric_cis(scored_frame(rows), n_bootstrap=20)

    assert len(report) == 2
    # m1 predicted nothing on an item with nothing to find, so tp+fp+fn is 0 and
    # its F-scores are undefined. m2's false positive makes its F-scores 0.
    by_model = report.set_index("model")
    assert np.isnan(by_model.loc["m1", "pooled_f1"])
    assert np.isnan(by_model.loc["m1", "pooled_f2"])
    assert by_model.loc["m2", "pooled_f1"] == 0.0
    assert by_model.loc["m2", "pooled_f2"] == 0.0

    # m1 flagged nothing, so precision has no predictions to judge; m2 flagged one
    # word and got it wrong, so its precision is 0. Neither item held a word to
    # find, so recall is undefined for both.
    assert np.isnan(by_model.loc["m1", "pooled_precision"])
    assert by_model.loc["m2", "pooled_precision"] == 0.0
    assert np.isnan(by_model.loc["m1", "pooled_recall"])
    assert np.isnan(by_model.loc["m2", "pooled_recall"])

    # An undefined point estimate is undefined in every replicate too, so its
    # interval is NaN.
    undefined = [
        ("m1", "pooled_f1"),
        ("m1", "pooled_f2"),
        ("m1", "pooled_precision"),
        ("m1", "pooled_recall"),
        ("m2", "pooled_recall"),
    ]
    for model, metric in undefined:
        assert np.isnan(by_model.loc[model, f"{metric}_ci_low"]), f"{model} {metric}"
        assert np.isnan(by_model.loc[model, f"{metric}_ci_high"]), f"{model} {metric}"
