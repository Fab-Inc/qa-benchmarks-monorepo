"""Tests for `fab-leaderboard publish`, the cluster-wide bundle build."""

import json

import pytest
from fab_benchmark_publisher import leaderboard_cli
from fab_benchmark_publisher.contracts import BenchmarkPair, LeaderboardRecord
from fab_benchmark_publisher.leaderboard import find_records_files, write_records_file
from fab_benchmark_publisher.leaderboard_cli import (
    load_cluster_config,
    main,
    order_contributions,
)

LK_EN = BenchmarkPair(benchmark_domain="LK-en", display_name="LK English")
LK_SW = BenchmarkPair(benchmark_domain="LK-sw", display_name="LK Kiswahili")
DA_EN = BenchmarkPair(benchmark_domain="DA-en", display_name="DA English")

PAIRS = {
    "cluster_name": "foundational-literacy",
    "pairs": [
        {"benchmark_domain": "LK-sw", "display_name": "LK Kiswahili"},
        {"benchmark_domain": "LK-en", "display_name": "LK English"},
        {"benchmark_domain": "DA-en", "display_name": "DA English"},
    ],
}


def write_pairs(tmp_path, payload=None):
    path = tmp_path / "leaderboard_pairs.json"
    path.write_text(json.dumps(payload or PAIRS), encoding="utf-8")
    return path


def record(model, accuracy, domain):
    return LeaderboardRecord(
        benchmark_domain=domain, model=model, accuracy=accuracy, n=100
    )


def use_csvs(tmp_path, monkeypatch, *, provider_meta=None):
    """Write fixture models/providers CSVs and make the CLI resolve to them.

    The command has no --models-csv flag on purpose: model ids are row positions in
    the bucket's own files, so it always fetches those. Patching the fetch is how a
    test supplies its own without reaching the network. The same goes for Azure's
    providermeta, which by default is stubbed to list both fixture models.
    """
    models = tmp_path / "models.csv"
    models.write_text(
        "model_id,provider,open,url,release_date,size,input_cost,output_cost,"
        "display_name\n"
        "alpha,OpenCo,true,,01/2025,1000000000,0.1,0.2,Alpha\n"
        "beta,ClosedCo,false,,01/2026,,10.0,30.0,Beta\n",
        encoding="utf-8",
    )
    providers = tmp_path / "providers.csv"
    providers.write_text(
        "provider,color,svg\nOpenCo,#112233,<svg/>\nClosedCo,#445566,<svg/>\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        leaderboard_cli, "ensure_shared_config", lambda name: tmp_path / name
    )
    if provider_meta is None:
        provider_meta = {"0": {"metadata": {"id": "0"}, "models": {"0": {}, "1": {}}}}
    monkeypatch.setattr(leaderboard_cli, "fetch_providermeta", lambda: provider_meta)
    return models, providers


# --------------------------------------------------------------------------- #
# find_records_files
# --------------------------------------------------------------------------- #
def test_find_records_files_collects_across_directories(tmp_path) -> None:
    """Each benchmark keeps its records under its own tree, so discovery spans dirs."""
    lk, phonics = tmp_path / "lk", tmp_path / "phonics"
    write_records_file([record("alpha", 50.0, "LK-en")], benchmark=LK_EN, outdir=lk)
    write_records_file(
        [record("alpha", 60.0, "DA-en")], benchmark=DA_EN, outdir=phonics
    )

    found = find_records_files([lk, phonics])

    assert [p.name for p in found] == [
        "leaderboard_records_LK-en.json",
        "leaderboard_records_DA-en.json",
    ]


def test_find_records_files_ignores_a_missing_directory(tmp_path) -> None:
    """A benchmark that has produced nothing yet must not break the whole build."""
    assert find_records_files([tmp_path / "nope"]) == []


def test_find_records_files_ignores_other_json(tmp_path) -> None:
    (tmp_path / "benchmark_default_models.json").write_text("{}", encoding="utf-8")
    write_records_file(
        [record("alpha", 50.0, "LK-en")], benchmark=LK_EN, outdir=tmp_path
    )

    assert [p.name for p in find_records_files([tmp_path])] == [
        "leaderboard_records_LK-en.json"
    ]


# --------------------------------------------------------------------------- #
# the pairs config
# --------------------------------------------------------------------------- #
def test_load_cluster_config_reads_pairs_in_order(tmp_path) -> None:
    config = load_cluster_config(write_pairs(tmp_path))

    assert config.cluster_name == "foundational-literacy"
    assert [p.benchmark_domain for p in config.pairs] == ["LK-sw", "LK-en", "DA-en"]


def test_load_cluster_config_reports_a_missing_file(tmp_path) -> None:
    with pytest.raises(SystemExit, match="pairs config not found"):
        load_cluster_config(tmp_path / "nope.json")


def test_load_cluster_config_rejects_a_malformed_pair(tmp_path) -> None:
    path = write_pairs(tmp_path, {"pairs": [{"benchmark_domain": "LK-en"}]})

    with pytest.raises(SystemExit, match="malformed"):
        load_cluster_config(path)


def test_load_cluster_config_rejects_an_empty_pair_list(tmp_path) -> None:
    with pytest.raises(SystemExit, match="lists no pairs"):
        load_cluster_config(write_pairs(tmp_path, {"pairs": []}))


def test_load_cluster_config_rejects_a_repeated_pair(tmp_path) -> None:
    payload = {"pairs": [PAIRS["pairs"][0], PAIRS["pairs"][0]]}

    with pytest.raises(SystemExit, match="repeats pair"):
        load_cluster_config(write_pairs(tmp_path, payload))


# --------------------------------------------------------------------------- #
# ordering and missing pairs
# --------------------------------------------------------------------------- #
def test_order_contributions_follows_the_config_not_the_filesystem(tmp_path) -> None:
    """Column order is a published decision; a directory listing is not."""
    config = load_cluster_config(write_pairs(tmp_path))
    da = write_records_file(
        [record("a", 1.0, "DA-en")], benchmark=DA_EN, outdir=tmp_path
    )
    sw = write_records_file(
        [record("a", 2.0, "LK-sw")], benchmark=LK_SW, outdir=tmp_path
    )

    from fab_benchmark_publisher.leaderboard import read_records_files

    ordered, missing = order_contributions(config, read_records_files([da, sw]))

    assert [c.benchmark.benchmark_domain for c in ordered] == ["LK-sw", "DA-en"]
    assert [p.benchmark_domain for p in missing] == ["LK-en"]


def test_order_contributions_takes_display_names_from_the_config(tmp_path) -> None:
    """One place decides what a column is called, or benchmarks rename each other."""
    config = load_cluster_config(write_pairs(tmp_path))
    renamed = BenchmarkPair(benchmark_domain="DA-en", display_name="Something Else")
    path = write_records_file(
        [record("a", 1.0, "DA-en")], benchmark=renamed, outdir=tmp_path
    )

    from fab_benchmark_publisher.leaderboard import read_records_files

    ordered, _ = order_contributions(config, read_records_files([path]))

    assert ordered[0].benchmark.display_name == "DA English"


def test_order_contributions_rejects_a_pair_not_in_the_config(tmp_path) -> None:
    config = load_cluster_config(write_pairs(tmp_path))
    ghost = BenchmarkPair(benchmark_domain="XX-zz", display_name="Mystery")
    path = write_records_file(
        [record("a", 1.0, "XX-zz")], benchmark=ghost, outdir=tmp_path
    )

    from fab_benchmark_publisher.leaderboard import read_records_files

    with pytest.raises(SystemExit, match="not in the config"):
        order_contributions(config, read_records_files([path]))


# --------------------------------------------------------------------------- #
# the command
# --------------------------------------------------------------------------- #
def test_publish_builds_a_bundle_from_several_benchmarks(
    tmp_path, capsys, monkeypatch
) -> None:
    pairs = write_pairs(tmp_path)
    use_csvs(tmp_path, monkeypatch)
    lk, phonics = tmp_path / "lk", tmp_path / "phonics"
    write_records_file([record("alpha", 50.0, "LK-en")], benchmark=LK_EN, outdir=lk)
    write_records_file([record("beta", 60.0, "DA-en")], benchmark=DA_EN, outdir=phonics)
    outdir = tmp_path / "out"

    main(
        [
            "publish",
            "--records-dir",
            str(lk),
            "--records-dir",
            str(phonics),
            "--pairs",
            str(pairs),
            "--outdir",
            str(outdir),
        ]
    )

    run_dir = next(outdir.iterdir())
    names = sorted(p.name for p in run_dir.iterdir())
    assert len(names) == 6  # 4 payloads + default-models record + manifest
    data = json.loads(next(run_dir.glob("*_data.json")).read_text(encoding="utf-8"))
    assert {r["benchmark_domain"] for r in data} == {"LK-en", "DA-en"}
    assert "MISSING: LK-sw" in capsys.readouterr().out


def test_publish_writes_a_manifest_naming_every_contribution(
    tmp_path, monkeypatch
) -> None:
    """The manifest is how anyone answers 'which run is on the site right now'."""
    pairs = write_pairs(tmp_path)
    use_csvs(tmp_path, monkeypatch)
    write_records_file(
        [record("alpha", 50.0, "LK-en")],
        benchmark=LK_EN,
        outdir=tmp_path / "lk",
        source="summary_en.csv",
        generated_at="2026-01-01T00:00:00Z",
    )
    outdir = tmp_path / "out"

    main(
        [
            "publish",
            "--records-dir",
            str(tmp_path / "lk"),
            "--pairs",
            str(pairs),
            "--outdir",
            str(outdir),
        ]
    )

    manifest = json.loads(
        (next(outdir.iterdir()) / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["contributions"] == [
        {
            "benchmark_domain": "LK-en",
            "display_name": "LK English",
            "source": "summary_en.csv",
            "generated_at": "2026-01-01T00:00:00Z",
            "n_records": 1,
        }
    ]
    assert manifest["missing"] == ["LK-sw", "DA-en"]


def test_publish_reports_when_no_records_exist_at_all(tmp_path) -> None:
    pairs = write_pairs(tmp_path)

    with pytest.raises(SystemExit, match="no leaderboard_records"):
        main(
            [
                "publish",
                "--records-dir",
                str(tmp_path / "empty"),
                "--pairs",
                str(pairs),
            ]
        )


def test_publish_reports_a_models_csv_that_cannot_be_fetched(
    tmp_path, monkeypatch
) -> None:
    """A checkout with no models.csv and no way to reach the bucket must exit with a
    message, not a traceback -- and must not go on to build a bundle in which every
    model has no provider, price or size."""
    pairs = write_pairs(tmp_path)
    write_records_file(
        [record("alpha", 50.0, "LK-en")], benchmark=LK_EN, outdir=tmp_path / "lk"
    )

    def unreachable(_name):
        raise FileNotFoundError("models.csv not found and the bucket is unreachable")

    monkeypatch.setattr(leaderboard_cli, "ensure_shared_config", unreachable)
    outdir = tmp_path / "out"

    with pytest.raises(SystemExit, match="models.csv not found"):
        main(
            [
                "publish",
                "--records-dir",
                str(tmp_path / "lk"),
                "--pairs",
                str(pairs),
                "--outdir",
                str(outdir),
            ]
        )
    assert not outdir.exists()


def test_publish_reports_two_files_claiming_one_pair(tmp_path, monkeypatch) -> None:
    pairs = write_pairs(tmp_path)
    use_csvs(tmp_path, monkeypatch)
    first = write_records_file(
        [record("alpha", 50.0, "LK-en")], benchmark=LK_EN, outdir=tmp_path / "lk"
    )
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / first.name).write_text(
        first.read_text(encoding="utf-8"), encoding="utf-8"
    )

    with pytest.raises(SystemExit, match="two records files claim"):
        main(
            [
                "publish",
                "--records-dir",
                str(tmp_path / "lk"),
                "--records-dir",
                str(tmp_path / "other"),
                "--pairs",
                str(pairs),
                "--outdir",
                str(tmp_path / "out"),
            ]
        )


def test_publish_reports_a_duplicated_model_in_a_records_file(
    tmp_path, monkeypatch
) -> None:
    """A summary CSV with a model listed twice reaches here as two records for one
    cell; the build rejects it and the CLI turns that into a clean exit."""
    pairs = write_pairs(tmp_path)
    use_csvs(tmp_path, monkeypatch)
    write_records_file(
        [record("alpha", 50.0, "LK-en"), record("alpha", 60.0, "LK-en")],
        benchmark=LK_EN,
        outdir=tmp_path / "lk",
    )

    with pytest.raises(SystemExit, match="duplicate record"):
        main(
            [
                "publish",
                "--records-dir",
                str(tmp_path / "lk"),
                "--pairs",
                str(pairs),
                "--outdir",
                str(tmp_path / "out"),
            ]
        )


def test_publish_reports_a_providermeta_that_cannot_be_fetched(
    tmp_path, monkeypatch
) -> None:
    """Without it there is no way to know whether the ids in this bundle resolve, and
    a bundle that skipped the check looks exactly like one that passed."""
    pairs = write_pairs(tmp_path)
    use_csvs(tmp_path, monkeypatch)
    write_records_file(
        [record("alpha", 50.0, "LK-en")], benchmark=LK_EN, outdir=tmp_path / "lk"
    )

    def unreachable():
        raise OSError("no network")

    monkeypatch.setattr(leaderboard_cli, "fetch_providermeta", unreachable)
    outdir = tmp_path / "out"

    with pytest.raises(SystemExit, match="cannot read the leaderboard's providermeta"):
        main(
            [
                "publish",
                "--records-dir",
                str(tmp_path / "lk"),
                "--pairs",
                str(pairs),
                "--outdir",
                str(outdir),
            ]
        )
    assert not outdir.exists()


def test_publish_records_a_dropped_model_in_the_manifest(
    tmp_path, monkeypatch, capsys
) -> None:
    """A model Azure's providermeta cannot resolve is left off the site entirely, so
    the manifest is the only place that ever says it scored."""
    pairs = write_pairs(tmp_path)
    # providermeta lists alpha (row 0) but not beta (row 1).
    use_csvs(
        tmp_path,
        monkeypatch,
        provider_meta={"0": {"metadata": {"id": "0"}, "models": {"0": {}}}},
    )
    write_records_file(
        [record("alpha", 50.0, "LK-en"), record("beta", 60.0, "LK-en")],
        benchmark=LK_EN,
        outdir=tmp_path / "lk",
    )
    outdir = tmp_path / "out"

    main(
        [
            "publish",
            "--records-dir",
            str(tmp_path / "lk"),
            "--pairs",
            str(pairs),
            "--outdir",
            str(outdir),
        ]
    )

    run_dir = next(outdir.iterdir())
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dropped"] == {"beta": "not in providermeta"}
    data = json.loads(next(run_dir.glob("*_data.json")).read_text(encoding="utf-8"))
    assert {r["modelId"] for r in data} == {"0"}
    assert "1 model(s) dropped" in capsys.readouterr().out


def test_publish_all_models_skips_curation(tmp_path, monkeypatch) -> None:
    pairs = write_pairs(tmp_path)
    use_csvs(tmp_path, monkeypatch)
    write_records_file(
        [record("alpha", 50.0, "LK-en"), record("beta", 60.0, "LK-en")],
        benchmark=LK_EN,
        outdir=tmp_path / "lk",
    )
    outdir = tmp_path / "out"

    main(
        [
            "publish",
            "--records-dir",
            str(tmp_path / "lk"),
            "--pairs",
            str(pairs),
            "--outdir",
            str(outdir),
            "--all-models",
            "--cluster",
            "test-cluster",
        ]
    )

    run_dir = next(outdir.iterdir())
    config = json.loads(
        next(run_dir.glob("*_configmeta.json")).read_text(encoding="utf-8")
    )
    assert len(config["benchmarks"]["LK-en"]["default_models"]) == 2
    assert list(run_dir.glob("*_test-cluster_*.json"))
