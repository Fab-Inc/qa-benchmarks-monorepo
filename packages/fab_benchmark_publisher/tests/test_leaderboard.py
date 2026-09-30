"""Tests for the shared leaderboard bundle builder and the records-file seam."""

import json

import pytest
from fab_benchmark_publisher.contracts import (
    BenchmarkPair,
    LeaderboardBundle,
    LeaderboardRecord,
)
from fab_benchmark_publisher.leaderboard import (
    build_leaderboard_bundle,
    load_models,
    load_providers,
    read_records_file,
    read_records_files,
    records_file_name,
    step_segments,
    write_leaderboard_bundle,
    write_records_file,
)

LK_EN = BenchmarkPair(benchmark_domain="LK-en", display_name="LK English")
LK_SW = BenchmarkPair(benchmark_domain="LK-sw", display_name="LK Kiswahili")

MODELS_HEADER = (
    "model_id,provider,open,url,release_date,size,input_cost,output_cost,display_name"
)


def write_models_csv(tmp_path, rows: list[str]):
    path = tmp_path / "models.csv"
    path.write_text("\n".join([MODELS_HEADER, *rows]) + "\n", encoding="utf-8")
    return path


def write_providers_csv(tmp_path, rows: list[str]):
    path = tmp_path / "providers.csv"
    path.write_text("\n".join(["provider,color,svg", *rows]) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def csvs(tmp_path):
    """Two providers and four models, enough to exercise every branch of the build."""
    models = write_models_csv(
        tmp_path,
        [
            "cheap-open,OpenCo,true,https://a,01/2025,1000000000,0.1,0.2,Cheap Open",
            "dear-closed,ClosedCo,false,https://b,01/2026,,10.0,30.0,Dear Closed",
            "mid-open,OpenCo,true,,06/2025,7000000000,1.0,2.0,",
            "free-open,OpenCo,true,,01/2024,500000000,,,Free Open",
        ],
    )
    providers = write_providers_csv(
        tmp_path, ["OpenCo,#112233,<svg>o</svg>", "ClosedCo,#445566,<svg>c</svg>"]
    )
    return models, providers


def record(model: str, accuracy: float, domain: str = "LK-en", **kwargs):
    return LeaderboardRecord(
        benchmark_domain=domain, model=model, accuracy=accuracy, **kwargs
    )


# --------------------------------------------------------------------------- #
# CSV loading
# --------------------------------------------------------------------------- #
def test_load_models_blends_price_three_to_one(csvs) -> None:
    models_csv, _ = csvs

    models_db, _ = load_models(models_csv)

    # (0.1*3 + 0.2) / 4
    assert models_db["cheap-open"]["blended_price"] == 0.125


def test_load_models_leaves_blended_price_unset_without_both_costs(csvs) -> None:
    models_csv, _ = csvs

    models_db, _ = load_models(models_csv)

    assert models_db["free-open"]["blended_price"] is None
    assert models_db["free-open"]["input_cost"] is None


def test_load_models_parses_open_as_a_bool_and_falls_back_to_the_id(csvs) -> None:
    models_csv, _ = csvs

    models_db, _ = load_models(models_csv)

    assert models_db["cheap-open"]["open"] is True
    assert models_db["dear-closed"]["open"] is False
    # blank display_name falls back to the model id
    assert models_db["mid-open"]["display_name"] == "mid-open"
    assert models_db["mid-open"]["url"] is None


def test_load_models_row_index_is_file_order(csvs) -> None:
    models_csv, _ = csvs

    _, row_index = load_models(models_csv)

    assert row_index == {
        "cheap-open": 0,
        "dear-closed": 1,
        "mid-open": 2,
        "free-open": 3,
    }


def test_load_providers_keeps_file_order(csvs) -> None:
    _, providers_csv = csvs

    providers_db = load_providers(providers_csv)

    assert list(providers_db) == ["OpenCo", "ClosedCo"]
    assert providers_db["OpenCo"]["color"] == "#112233"


# --------------------------------------------------------------------------- #
# step_segments
# --------------------------------------------------------------------------- #
def test_step_segments_inserts_a_horizontal_leg_before_each_rise() -> None:
    """The renderer draws a staircase, so each new point needs the previous
    y-value carried across first."""
    segs = step_segments(
        [{"x": 1, "y": 10.0, "providerId": "0"}, {"x": 5, "y": 20.0, "providerId": "1"}]
    )

    assert segs == [
        {"x": 1, "y": 10.0, "providerId": "0"},
        {"x": 5, "y": 10.0, "providerId": "1"},
        {"x": 5, "y": 20.0, "providerId": "1"},
    ]


def test_step_segments_of_a_single_point_is_that_point() -> None:
    assert step_segments([{"x": 1, "y": 10.0, "providerId": "0"}]) == [
        {"x": 1, "y": 10.0, "providerId": "0"}
    ]


# --------------------------------------------------------------------------- #
# The records file
# --------------------------------------------------------------------------- #
def test_records_file_round_trips(tmp_path) -> None:
    records = [
        record("cheap-open", 50.0, n=100, bad_format=1.25),
        record("dear-closed", 90.0, n=100),
    ]

    path = write_records_file(
        records, benchmark=LK_EN, outdir=tmp_path, source="summary.csv"
    )
    loaded = read_records_file(path)

    assert path.name == records_file_name("LK-en")
    assert loaded.benchmark == LK_EN
    assert loaded.source == "summary.csv"
    assert loaded.records == records


def test_records_file_preserves_extra_fields(tmp_path) -> None:
    records = [record("m", 50.0, extra={"custom": 1})]

    loaded = read_records_file(
        write_records_file(records, benchmark=LK_EN, outdir=tmp_path)
    )

    assert loaded.records[0].extra == {"custom": 1}


def test_records_file_key_order_is_stable(tmp_path) -> None:
    """Two runs of the same scores must produce diffable files, so the record keys
    follow the dataclass declaration order rather than a set's iteration order."""
    path = write_records_file(
        [record("m", 50.0, n=10)],
        benchmark=LK_EN,
        outdir=tmp_path,
        generated_at="2026-01-01T00:00:00Z",
    )

    assert list(json.loads(path.read_text(encoding="utf-8"))["records"][0]) == [
        "benchmark_domain",
        "model",
        "accuracy",
        "n",
    ]


def test_records_file_rejects_records_from_another_pair(tmp_path) -> None:
    with pytest.raises(ValueError, match="one records file holds exactly one pair"):
        write_records_file(
            [record("m", 50.0, domain="LK-sw")], benchmark=LK_EN, outdir=tmp_path
        )


def test_records_file_stamps_generated_at_when_not_given(tmp_path) -> None:
    path = write_records_file([record("m", 50.0)], benchmark=LK_EN, outdir=tmp_path)

    assert read_records_file(path).generated_at is not None


def test_read_records_file_rejects_a_bare_list(tmp_path) -> None:
    path = tmp_path / "records.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="bare list"):
        read_records_file(path)


def test_read_records_file_reports_missing_keys(tmp_path) -> None:
    path = tmp_path / "records.json"
    path.write_text(json.dumps({"benchmark_domain": "LK-en"}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required key"):
        read_records_file(path)


def test_read_records_file_reports_invalid_json(tmp_path) -> None:
    path = tmp_path / "records.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        read_records_file(path)


def test_read_records_file_reports_an_unusable_record(tmp_path) -> None:
    path = tmp_path / "records.json"
    path.write_text(
        json.dumps(
            {
                "benchmark_domain": "LK-en",
                "display_name": "LK English",
                "records": [{"model": "m"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unusable record"):
        read_records_file(path)


def test_read_records_file_rejects_a_non_object_record(tmp_path) -> None:
    path = tmp_path / "records.json"
    path.write_text(
        json.dumps(
            {
                "benchmark_domain": "LK-en",
                "display_name": "LK English",
                "records": ["nope"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be an object"):
        read_records_file(path)


def test_read_records_files_rejects_two_files_claiming_one_pair(tmp_path) -> None:
    """Glob-based discovery must fail loudly here, not silently publish whichever
    file the filesystem listed last."""
    first = write_records_file([record("a", 50.0)], benchmark=LK_EN, outdir=tmp_path)
    second = tmp_path / "other.json"
    second.write_text(first.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(ValueError, match="two records files claim"):
        read_records_files([first, second])


def test_read_records_files_preserves_the_order_given(tmp_path) -> None:
    sw = write_records_file(
        [record("a", 50.0, domain="LK-sw")], benchmark=LK_SW, outdir=tmp_path
    )
    en = write_records_file([record("a", 60.0)], benchmark=LK_EN, outdir=tmp_path)

    loaded = read_records_files([en, sw])

    assert [c.benchmark.benchmark_domain for c in loaded] == ["LK-en", "LK-sw"]


# --------------------------------------------------------------------------- #
# build_leaderboard_bundle
# --------------------------------------------------------------------------- #
def test_build_assigns_positional_ids_from_the_csv_row_order(csvs) -> None:
    """IDs are row indices so they stay stable across benchmarks sharing the CSVs."""
    models_csv, providers_csv = csvs

    bundle = build_leaderboard_bundle(
        [record("cheap-open", 50.0), record("dear-closed", 90.0)],
        benchmarks=[LK_EN],
        models_csv=models_csv,
        providers_csv=providers_csv,
        curate_defaults=False,
    )

    by_model = {row["modelId"]: row for row in bundle.payloads["data"]}
    assert set(by_model) == {"0", "1"}
    assert by_model["0"]["providerId"] == "0"
    assert by_model["1"]["providerId"] == "1"


def test_build_drops_models_absent_from_models_csv_with_a_warning(csvs, capsys) -> None:
    models_csv, providers_csv = csvs

    bundle = build_leaderboard_bundle(
        [record("cheap-open", 50.0), record("ghost-model", 90.0)],
        benchmarks=[LK_EN],
        models_csv=models_csv,
        providers_csv=providers_csv,
        curate_defaults=False,
    )

    assert [row["modelId"] for row in bundle.payloads["data"]] == ["0"]
    assert "ghost-model" in capsys.readouterr().out


def test_build_marks_models_scored_elsewhere_as_disabled(csvs) -> None:
    """A model with no result for this pair is greyed out rather than shown as 0."""
    models_csv, providers_csv = csvs

    bundle = build_leaderboard_bundle(
        [record("cheap-open", 50.0), record("dear-closed", 90.0, domain="LK-sw")],
        benchmarks=[LK_EN, LK_SW],
        models_csv=models_csv,
        providers_csv=providers_csv,
        curate_defaults=False,
    )

    meta = bundle.payloads["configmeta"]["benchmarks"]
    assert meta["LK-en"]["disabled_models"] == ["1"]
    assert meta["LK-sw"]["disabled_models"] == ["0"]


def test_build_orders_default_models_by_accuracy(csvs) -> None:
    models_csv, providers_csv = csvs

    bundle = build_leaderboard_bundle(
        [record("cheap-open", 50.0), record("dear-closed", 90.0)],
        benchmarks=[LK_EN],
        models_csv=models_csv,
        providers_csv=providers_csv,
        curate_defaults=False,
    )

    assert bundle.payloads["configmeta"]["benchmarks"]["LK-en"]["default_models"] == [
        "1",
        "0",
    ]


def test_build_keeps_the_benchmark_order_it_was_given(csvs) -> None:
    """Column order on the site comes from the caller, never from the filesystem."""
    models_csv, providers_csv = csvs

    bundle = build_leaderboard_bundle(
        [record("cheap-open", 50.0), record("cheap-open", 60.0, domain="LK-sw")],
        benchmarks=[LK_SW, LK_EN],
        models_csv=models_csv,
        providers_csv=providers_csv,
        curate_defaults=False,
    )

    listed = bundle.payloads["configmeta"]["base_options"]["benchmarks"]
    assert [b["id"] for b in listed] == ["LK-sw", "LK-en"]


def test_build_gives_a_provider_its_providers_csv_row_position(csvs) -> None:
    """providerId has to be the row index, because Azure's providermeta -- which the
    site resolves these ids through -- is generated from the same providers.csv."""
    models_csv, providers_csv = csvs

    bundle = build_leaderboard_bundle(
        [record("dear-closed", 50.0)],
        benchmarks=[LK_EN],
        models_csv=models_csv,
        providers_csv=providers_csv,
        curate_defaults=False,
    )

    # ClosedCo is the second row of providers.csv.
    assert {r["providerId"] for r in bundle.payloads["data"]} == {"1"}


def test_build_gives_an_unknown_provider_an_id_past_the_providers_file(
    tmp_path,
) -> None:
    """A provider in models.csv but not providers.csv still needs an id, and it must
    not collide with a real row's."""
    models_csv = write_models_csv(
        tmp_path, ["m,NewCo,true,,01/2026,1000000000,1.0,2.0,M"]
    )
    providers_csv = write_providers_csv(tmp_path, ["OpenCo,#112233,<svg>o</svg>"])

    bundle = build_leaderboard_bundle(
        [record("m", 50.0)],
        benchmarks=[LK_EN],
        models_csv=models_csv,
        providers_csv=providers_csv,
        curate_defaults=False,
    )

    assert {r["providerId"] for r in bundle.payloads["data"]} == {"1"}


# --------------------------------------------------------------------------- #
# the providermeta gate
# --------------------------------------------------------------------------- #
def provider_meta_listing(*model_ids: str) -> dict:
    """A minimal Azure providermeta holding exactly these model ids."""
    return {"0": {"metadata": {"id": "0"}, "models": {m: {} for m in model_ids}}}


def test_build_publishes_a_model_the_providermeta_lists(csvs) -> None:
    models_csv, providers_csv = csvs

    bundle = build_leaderboard_bundle(
        [record("cheap-open", 50.0)],
        benchmarks=[LK_EN],
        models_csv=models_csv,
        providers_csv=providers_csv,
        provider_meta=provider_meta_listing("0", "1", "2", "3"),
        curate_defaults=False,
    )

    assert bundle.payloads["data"]
    assert bundle.dropped == {}


def test_build_drops_a_model_the_providermeta_does_not_list(csvs, capsys) -> None:
    """models.csv gains models before Azure's providermeta does. Publishing one of
    those would put a row on the site with no name and no price, so it is dropped --
    loudly, and recorded, because a dropped model is otherwise invisible."""
    models_csv, providers_csv = csvs

    bundle = build_leaderboard_bundle(
        [record("cheap-open", 50.0), record("dear-closed", 60.0)],
        benchmarks=[LK_EN],
        models_csv=models_csv,
        providers_csv=providers_csv,
        provider_meta=provider_meta_listing("0"),
        curate_defaults=False,
    )

    published = {r["modelId"] for r in bundle.payloads["data"]}
    assert published == {"0"}
    assert bundle.dropped == {"dear-closed": "not in providermeta"}
    out = capsys.readouterr().out
    assert "not in the leaderboard's providermeta - dropped: dear-closed" in out
    assert "providermeta refresh" in out


def test_build_records_a_model_missing_from_models_csv(csvs, capsys) -> None:
    """The older of the two gates, now recorded the same way: a score for a model the
    catalogue has never heard of is not published, and the manifest has to say so."""
    models_csv, providers_csv = csvs

    bundle = build_leaderboard_bundle(
        [record("cheap-open", 50.0), record("ghost-model", 60.0)],
        benchmarks=[LK_EN],
        models_csv=models_csv,
        providers_csv=providers_csv,
        curate_defaults=False,
    )

    assert {r["modelId"] for r in bundle.payloads["data"]} == {"0"}
    assert bundle.dropped == {"ghost-model": "not in models.csv"}
    assert "no models.csv match - dropped: ghost-model" in capsys.readouterr().out


def test_build_skips_the_providermeta_gate_when_not_given(csvs) -> None:
    """None means 'no providermeta to check against', not 'nothing is listed'."""
    models_csv, providers_csv = csvs

    bundle = build_leaderboard_bundle(
        [record("cheap-open", 50.0)],
        benchmarks=[LK_EN],
        models_csv=models_csv,
        providers_csv=providers_csv,
        provider_meta=None,
        curate_defaults=False,
    )

    assert bundle.payloads["data"]
    assert bundle.dropped == {}


def test_build_rejects_a_record_for_an_unpublished_pair(csvs) -> None:
    models_csv, providers_csv = csvs

    with pytest.raises(ValueError, match="not in the published pairs"):
        build_leaderboard_bundle(
            [record("cheap-open", 50.0, domain="DA-sw")],
            benchmarks=[LK_EN],
            models_csv=models_csv,
            providers_csv=providers_csv,
        )


def test_build_rejects_two_records_for_one_model_and_pair(csvs) -> None:
    models_csv, providers_csv = csvs

    with pytest.raises(ValueError, match="duplicate record"):
        build_leaderboard_bundle(
            [record("cheap-open", 50.0), record("cheap-open", 60.0)],
            benchmarks=[LK_EN],
            models_csv=models_csv,
            providers_csv=providers_csv,
        )


def test_build_defaults_missing_n_and_bad_format_to_zero(csvs) -> None:
    models_csv, providers_csv = csvs

    bundle = build_leaderboard_bundle(
        [record("cheap-open", 50.0)],
        benchmarks=[LK_EN],
        models_csv=models_csv,
        providers_csv=providers_csv,
        curate_defaults=False,
    )

    row = bundle.payloads["data"][0]
    assert row["n"] == 0
    assert row["bad_format"] == 0.0


def test_build_cost_frontier_is_a_staircase_through_real_models(csvs) -> None:
    models_csv, providers_csv = csvs

    bundle = build_leaderboard_bundle(
        [record("cheap-open", 50.0), record("dear-closed", 90.0)],
        benchmarks=[LK_EN],
        models_csv=models_csv,
        providers_csv=providers_csv,
        curate_defaults=False,
    )

    segments = bundle.payloads["data-costfrontier"][0]["lineSegments"]
    assert [s["y"] for s in segments] == [50.0, 50.0, 90.0]


def test_build_size_frontier_covers_open_models_only(csvs) -> None:
    models_csv, providers_csv = csvs

    bundle = build_leaderboard_bundle(
        [record("cheap-open", 50.0), record("dear-closed", 90.0)],
        benchmarks=[LK_EN],
        models_csv=models_csv,
        providers_csv=providers_csv,
        curate_defaults=False,
    )

    segments = bundle.payloads["data-sizefrontier"][0]["lineSegments"]
    assert [s["y"] for s in segments] == [50.0]


def test_build_omits_a_size_frontier_when_no_open_model_has_a_size(tmp_path) -> None:
    models_csv = write_models_csv(tmp_path, ["m,ClosedCo,false,,01/2026,,1.0,2.0,M"])
    providers_csv = write_providers_csv(tmp_path, ["ClosedCo,#445566,<svg>c</svg>"])

    bundle = build_leaderboard_bundle(
        [record("m", 50.0)],
        benchmarks=[LK_EN],
        models_csv=models_csv,
        providers_csv=providers_csv,
        curate_defaults=False,
    )

    assert bundle.payloads["data-sizefrontier"] == []


def test_build_warns_when_a_drawn_frontier_vertex_is_not_pre_selected(
    tmp_path, capsys
) -> None:
    """A vertex nobody pre-selected renders as a line bending at an invisible dot --
    the one curation regression a visitor would actually see.

    `bump` sits below the line joining its neighbours, so the concave peeling smooths
    it away while the drawn (raw Pareto) line still bends at it. All three are open
    so they land in one subset: curation splits open from closed, and two points
    cannot form a concave hull.
    """
    models_csv = write_models_csv(
        tmp_path,
        [
            "cheap,OpenCo,true,,01/2025,1000000000,1.0,1.0,Cheap",
            "bump,OpenCo,true,,01/2025,2000000000,10.0,10.0,Bump",
            "dear,OpenCo,true,,01/2025,3000000000,100.0,100.0,Dear",
        ],
    )
    providers_csv = write_providers_csv(tmp_path, ["OpenCo,#112233,<svg>o</svg>"])

    build_leaderboard_bundle(
        [record("cheap", 10.0), record("bump", 20.0), record("dear", 90.0)],
        benchmarks=[LK_EN],
        models_csv=models_csv,
        providers_csv=providers_csv,
        curate_defaults=True,
        add_missing_models=False,
        filter_by_family=False,
    )

    out = capsys.readouterr().out
    assert "frontier vertices missing from" in out
    assert "bump" in out


def test_build_curated_defaults_are_a_subset_of_scored_models(csvs) -> None:
    models_csv, providers_csv = csvs

    bundle = build_leaderboard_bundle(
        [
            record("cheap-open", 30.0),
            record("mid-open", 60.0),
            record("dear-closed", 90.0),
        ],
        benchmarks=[LK_EN],
        models_csv=models_csv,
        providers_csv=providers_csv,
        curate_defaults=True,
    )

    assert set(bundle.default_models["LK-en"]) <= {
        "cheap-open",
        "mid-open",
        "dear-closed",
    }


def test_build_default_models_record_is_sorted_for_diffing(csvs) -> None:
    models_csv, providers_csv = csvs

    bundle = build_leaderboard_bundle(
        [record("mid-open", 60.0), record("cheap-open", 30.0)],
        benchmarks=[LK_EN],
        models_csv=models_csv,
        providers_csv=providers_csv,
        curate_defaults=False,
    )

    assert bundle.default_models["LK-en"] == sorted(bundle.default_models["LK-en"])


def test_build_keeps_a_pair_with_no_records_as_an_all_disabled_column(csvs) -> None:
    models_csv, providers_csv = csvs

    bundle = build_leaderboard_bundle(
        [record("cheap-open", 50.0)],
        benchmarks=[LK_EN, LK_SW],
        models_csv=models_csv,
        providers_csv=providers_csv,
        curate_defaults=False,
    )

    meta = bundle.payloads["configmeta"]["benchmarks"]["LK-sw"]
    assert meta["default_models"] == []
    assert meta["disabled_models"] == ["0"]


def test_build_ignores_a_manual_default_model_absent_from_models_csv(csvs) -> None:
    """Manual additions bypass every curation filter, but a model with no models.csv
    row has no id the site could resolve, so it must not reach configmeta."""
    models_csv, providers_csv = csvs

    bundle = build_leaderboard_bundle(
        [record("cheap-open", 50.0), record("ghost", 60.0)],
        benchmarks=[LK_EN],
        models_csv=models_csv,
        providers_csv=providers_csv,
        manual_default_models=["ghost"],
    )

    assert bundle.default_models["LK-en"] == ["cheap-open"]


def test_build_rejects_empty_records(csvs) -> None:
    models_csv, providers_csv = csvs

    with pytest.raises(ValueError, match="No leaderboard records"):
        build_leaderboard_bundle(
            [],
            benchmarks=[LK_EN],
            models_csv=models_csv,
            providers_csv=providers_csv,
        )


# --------------------------------------------------------------------------- #
# write_leaderboard_bundle
# --------------------------------------------------------------------------- #
def test_write_bundle_names_files_for_the_run_and_cluster(tmp_path) -> None:
    bundle = LeaderboardBundle(
        payloads={"configmeta": {"a": 1}, "data": [{"b": 2}]},
        default_models={"LK-en": ["m"]},
    )

    run_dir = write_leaderboard_bundle(
        bundle,
        outdir=tmp_path,
        cluster_name="foundational-literacy",
        timestamp="2026-01-02T030405Z",
    )

    assert run_dir.name == "2026-01-02T030405Z"
    assert sorted(p.name for p in run_dir.iterdir()) == [
        "2026-01-02T030405Z_foundational-literacy_configmeta.json",
        "2026-01-02T030405Z_foundational-literacy_data.json",
        "benchmark_default_models.json",
    ]


def test_write_bundle_keeps_the_audit_record_out_of_the_upload_glob(tmp_path) -> None:
    """A glob-based uploader takes <timestamp>_<cluster>_*.json; the default-models
    record must not be swept up with them."""
    bundle = LeaderboardBundle(payloads={"data": []}, default_models={"LK-en": ["m"]})

    run_dir = write_leaderboard_bundle(
        bundle, outdir=tmp_path, cluster_name="cluster", timestamp="2026-01-02T030405Z"
    )

    uploads = list(run_dir.glob("2026-01-02T030405Z_cluster_*.json"))
    assert [p.name for p in uploads] == ["2026-01-02T030405Z_cluster_data.json"]
    assert (run_dir / "benchmark_default_models.json").is_file()


def test_write_bundle_generates_a_timestamp_when_not_given(tmp_path) -> None:
    bundle = LeaderboardBundle(payloads={"data": []}, default_models={})

    run_dir = write_leaderboard_bundle(bundle, outdir=tmp_path, cluster_name="c")

    assert run_dir.name.endswith("Z")
    assert run_dir.parent == tmp_path


def test_write_bundle_round_trips_its_payloads(tmp_path) -> None:
    payload = {"benchmarks": {"LK-en": {"displayName": "LK English"}}}
    bundle = LeaderboardBundle(payloads={"configmeta": payload}, default_models={})

    run_dir = write_leaderboard_bundle(
        bundle, outdir=tmp_path, cluster_name="c", timestamp="2026-01-02T030405Z"
    )

    written = run_dir / "2026-01-02T030405Z_c_configmeta.json"
    assert json.loads(written.read_text(encoding="utf-8")) == payload
