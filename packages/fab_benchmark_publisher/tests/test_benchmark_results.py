"""Tests for the frontiers file, the metadata files and the per-benchmark set."""

import json
from dataclasses import replace

import pandas as pd
import pytest
from fab_benchmark_publisher.benchmark_frontiers import (
    build_frontiers,
    frontiers_file_name,
    validate_frontiers,
)
from fab_benchmark_publisher.benchmark_results import (
    cluster_metadata_file_name,
    gcs_object_name,
    metrics_to_leaderboard_records,
    read_benchmark_results,
    write_benchmark_results,
)
from fab_benchmark_publisher.metrics import aggregate_item_scores
from fab_benchmark_publisher.result_metadata import (
    BenchmarkMeta,
    ClusterMeta,
    MetricMeta,
    read_benchmark_meta,
    read_cluster_meta,
    write_benchmark_meta,
    write_cluster_meta,
)

ACCURACY = MetricMeta(
    metric_slug="accuracy",
    display_name="Accuracy",
    display_description="Share of items answered correctly.",
    unit="%",
    is_higher_better=True,
    is_bad_format_scored=False,
)
META = BenchmarkMeta("LK-en", "LK English", (ACCURACY,))

# model -> (open, size, input_cost, output_cost); blended = (3*in + out) / 4
MODELS = {
    "cheap": (False, "", 1.0, 1.0),
    "mid": (False, "", 2.0, 2.0),
    "pricey": (False, "", 4.0, 4.0),
    "small-open": (True, 7, 0.5, 0.5),
    "big-open": (True, 70, 3.0, 3.0),
}
# model -> per-item scores, items 1-4 in D1, 5-8 in D2
SCORES = {
    "cheap": [1, 0, 0, 0, 1, 1, 0, 0],  # 37.5
    "mid": [1, 1, 0, 0, 1, 0, 0, 0],  # 37.5, no better than cheap
    "pricey": [1, 1, 1, 1, 1, 1, 0, 0],  # 75
    "small-open": [1, 1, 0, 0, 0, 0, 0, 0],  # 25
    "big-open": [1, 1, 1, 0, 1, 1, 1, 0],  # 75
    "unlisted": [1, 1, 1, 1, 1, 1, 1, 1],  # not in models.csv
}


@pytest.fixture
def models_csv(tmp_path):
    path = tmp_path / "models.csv"
    lines = ["model_id,provider,open,size,input_cost,output_cost,release_date"]
    lines += [
        f"{m},P,{op},{size},{i},{o},01/2026" for m, (op, size, i, o) in MODELS.items()
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def make_metrics(metric=ACCURACY, groupings=(("domain",),)):
    rows = [
        (model, item, score, "D1" if item <= 4 else "D2")
        for model, scores in SCORES.items()
        for item, score in enumerate(scores, start=1)
    ]
    return aggregate_item_scores(
        pd.DataFrame(rows, columns=["model_slug", "item_id", "item_score", "domain"]),
        benchmark_slug="LK-en",
        metric=metric,
        experiment_slug="final-137-items",
        subset_groupings=groupings,
        timestamp_metric="2026-11-18T12:00:00+00:00",
        n_bootstrap=100,
    )


@pytest.fixture
def metrics():
    return make_metrics()


def frontier(frontiers, axis, subset=""):
    (match,) = [
        f
        for f in frontiers["frontiers"]
        if f["axis"] == axis and f["benchmark_subset"] == subset
    ]
    return [p["model_slug"] for p in match["points"]]


# --------------------------------------------------------------------------- #
# frontiers
# --------------------------------------------------------------------------- #
def test_frontiers_crosstab_axes_and_subsets(metrics, models_csv) -> None:
    frontiers = build_frontiers(metrics, metric=ACCURACY, models_csv=models_csv)

    keys = {(f["axis"], f["benchmark_subset"]) for f in frontiers["frontiers"]}
    subsets = {"", '{"domain":"D1"}', '{"domain":"D2"}'}
    assert keys == {(a, s) for a in ("price", "size") for s in subsets}
    assert frontiers["frontiers"][0]["benchmark_subset"] == ""
    assert (
        frontiers["benchmark_slug"],
        frontiers["metric_slug"],
        frontiers["experiment_slug"],
    ) == ("LK-en", "accuracy", "final-137-items")


def test_overall_only_metrics_give_one_frontier_per_axis(models_csv) -> None:
    frontiers = build_frontiers(
        make_metrics(groupings=()), metric=ACCURACY, models_csv=models_csv
    )

    assert [(f["axis"], f["benchmark_subset"]) for f in frontiers["frontiers"]] == [
        ("price", ""),
        ("size", ""),
    ]


def test_price_frontier_keeps_only_improving_models(metrics, models_csv) -> None:
    frontiers = build_frontiers(metrics, metric=ACCURACY, models_csv=models_csv)

    # small-open (0.5, 25) -> cheap (1, 37.5) -> big-open (3, 75); mid ties cheap at
    # a higher price and pricey ties big-open at a higher price.
    assert frontier(frontiers, "price") == ["small-open", "cheap", "big-open"]
    assert frontier(frontiers, "price", '{"domain":"D1"}') == [
        "small-open",
        "big-open",
        "pricey",
    ]


def test_size_frontier_is_open_weight_only(metrics, models_csv) -> None:
    frontiers = build_frontiers(metrics, metric=ACCURACY, models_csv=models_csv)

    assert frontier(frontiers, "size") == ["small-open", "big-open"]


def test_models_missing_from_models_csv_are_reported(metrics, models_csv) -> None:
    frontiers = build_frontiers(metrics, metric=ACCURACY, models_csv=models_csv)

    assert frontiers["models_without_metadata"] == ["unlisted"]
    assert all(
        p["model_slug"] != "unlisted"
        for f in frontiers["frontiers"]
        for p in f["points"]
    )


def test_lower_is_better_flips_the_frontier(models_csv) -> None:
    lower = replace(ACCURACY, is_higher_better=False)

    frontiers = build_frontiers(
        make_metrics(lower), metric=lower, models_csv=models_csv
    )

    # cheapest first, then only models with a LOWER score
    assert frontier(frontiers, "price") == ["small-open"]


def test_frontier_disagreeing_with_metrics_is_rejected(metrics, models_csv) -> None:
    frontiers = build_frontiers(metrics, metric=ACCURACY, models_csv=models_csv)
    frontiers["frontiers"][0]["points"][0]["y"] += 1

    with pytest.raises(ValueError, match="differs from metrics"):
        validate_frontiers(frontiers, metrics)


def test_frontiers_file_name_matches_its_metrics_file() -> None:
    assert (
        frontiers_file_name("LK-en", "accuracy", "final-137-items")
        == "LK-en_accuracy_final-137-items_frontiers.json"
    )


# --------------------------------------------------------------------------- #
# metadata
# --------------------------------------------------------------------------- #
def test_benchmark_metadata_holds_metric_metadata_array(tmp_path) -> None:
    path = write_benchmark_meta(META, tmp_path / "benchmark_metadata.json")

    raw = json.loads(path.read_text())
    assert raw == {
        "benchmark_slug": "LK-en",
        "display_name": "LK English",
        "metrics": [
            {
                "metric_slug": "accuracy",
                "display_name": "Accuracy",
                "display_description": "Share of items answered correctly.",
                "unit": "%",
                "is_higher_better": True,
                "is_bad_format_scored": False,
            }
        ],
    }
    assert read_benchmark_meta(path) == META


def test_metric_metadata_with_wrong_types_is_rejected(tmp_path) -> None:
    path = write_benchmark_meta(META, tmp_path / "benchmark_metadata.json")
    raw = json.loads(path.read_text())
    raw["metrics"][0]["is_higher_better"] = "yes"
    path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="is_higher_better must be true or false"):
        read_benchmark_meta(path)


def test_benchmark_metadata_rejects_duplicate_metrics() -> None:
    with pytest.raises(ValueError, match="duplicate metric_slug"):
        BenchmarkMeta("LK-en", "LK English", (ACCURACY, ACCURACY))


def test_cluster_metadata_round_trips_in_order(tmp_path) -> None:
    meta = ClusterMeta(
        "foundational-literacy", "Foundational Literacy", ("LK-sw", "LK-en")
    )

    path = write_cluster_meta(
        meta, tmp_path / cluster_metadata_file_name(meta.cluster_slug)
    )

    assert path.name == "foundational-literacy_cluster_metadata.json"
    assert json.loads(path.read_text())["benchmark_ids"] == ["LK-sw", "LK-en"]
    assert read_cluster_meta(path) == meta


def test_gcs_object_names() -> None:
    base = "website/benchmark-clusters/foundational-literacy"
    assert (
        gcs_object_name(
            "LK-en_accuracy_final-137-items.csv",
            cluster_slug="foundational-literacy",
            benchmark_group="linguistic-knowledge",
            benchmark_slug="LK-en",
        )
        == f"{base}/linguistic-knowledge/LK-en/LK-en_accuracy_final-137-items.csv"
    )
    assert (
        gcs_object_name(
            "foundational-literacy_cluster_metadata.json",
            cluster_slug="foundational-literacy",
        )
        == f"{base}/foundational-literacy_cluster_metadata.json"
    )


# --------------------------------------------------------------------------- #
# the per-benchmark set
# --------------------------------------------------------------------------- #
def test_write_and_read_back_result_set(tmp_path, metrics, models_csv) -> None:
    frontiers = build_frontiers(metrics, metric=ACCURACY, models_csv=models_csv)
    outdir = tmp_path / "LK-en"

    paths = write_benchmark_results(
        outdir, metrics=metrics, frontiers=frontiers, meta=META
    )
    back_metrics, back_frontiers, back_meta = read_benchmark_results(
        outdir, metric_slug="accuracy", experiment_slug="final-137-items"
    )

    assert sorted(paths) == [
        "LK-en_accuracy_final-137-items.csv",
        "LK-en_accuracy_final-137-items_frontiers.json",
        "benchmark_metadata.json",
    ]
    assert back_meta == META
    assert back_frontiers == frontiers
    assert len(back_metrics) == len(metrics)


def test_mismatched_metadata_is_rejected(tmp_path, metrics, models_csv) -> None:
    frontiers = build_frontiers(metrics, metric=ACCURACY, models_csv=models_csv)
    outdir = tmp_path / "out"

    with pytest.raises(ValueError, match="metadata for 'LK-sw'"):
        write_benchmark_results(
            outdir,
            metrics=metrics,
            frontiers=frontiers,
            meta=replace(META, benchmark_slug="LK-sw"),
        )
    assert not outdir.exists()


def test_metric_missing_from_metadata_is_rejected(
    tmp_path, metrics, models_csv
) -> None:
    frontiers = build_frontiers(metrics, metric=ACCURACY, models_csv=models_csv)
    f1 = replace(ACCURACY, metric_slug="f1")

    with pytest.raises(ValueError, match="declares no metric 'accuracy'"):
        write_benchmark_results(
            tmp_path / "out",
            metrics=metrics,
            frontiers=frontiers,
            meta=replace(META, metrics=(f1,)),
        )


def test_overall_rows_convert_to_leaderboard_records(metrics) -> None:
    records = {r.model: r for r in metrics_to_leaderboard_records(metrics, META)}

    assert set(records) == set(SCORES)
    assert records["pricey"].accuracy == 75.0
    assert (records["pricey"].n, records["pricey"].bad_format) == (8, 0.0)
