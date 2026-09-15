"""Regression checks for the additive finished-model analysis script."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def analysis():
    path = (
        Path(__file__).parents[1] / "scripts" / "analyze_finished_kiswahili_models.py"
    )
    spec = importlib.util.spec_from_file_location("finished_kiswahili_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


@pytest.fixture
def bundle(tmp_path, analysis):
    items = [
        {
            "benchmark_item_id": f"i{index}",
            "content": text,
            "student_knowledge": {"graphemes": ["a", "m"]},
            "nondecodable_words": gold,
            "language_iso_code": "sw",
        }
        for index, (text, gold) in enumerate(
            [
                ("Mama kuku soko.", ["kuku", "soko"]),
                ("Mama kuku.", ["kuku"]),
                ("Mama.", []),
                ("Mama soko.", ["soko"]),
            ],
            1,
        )
    ]
    items_path = tmp_path / "source" / "items.jsonl"
    write_jsonl(items_path, items)
    items_hash = analysis.digest(items_path)

    def make_run(run_id, predictions, selected=None):
        run = tmp_path / "source" / run_id
        run.mkdir()
        config = {
            "run_id": run_id,
            "language": "sw",
            "items_sha256": items_hash,
            "items_path": "/unavailable/cloud/container/items.jsonl",
            "prompt_name": "decodability_evaluation_clusters",
            "models": selected or list(predictions),
        }
        (run / "config.json").write_text(json.dumps({"resolved_config": config}))
        for model, outputs in predictions.items():
            rows = [
                {
                    "run_id": run_id,
                    "model": model,
                    "benchmark_item_id": f"i{i}",
                    "result_id": f"{model}-{i}",
                    "output": output,
                    "call_status": "error" if output is None else "success",
                    "success": output is not None,
                }
                for i, output in enumerate(outputs, 1)
            ]
            write_jsonl(run / "responses" / f"responses_{model}.jsonl", rows)
        return run

    return items_path, items_hash, make_run


def test_two_runs_metrics_completion_and_exclusions(tmp_path, analysis, bundle):
    items, checksum, make_run = bundle
    run1 = make_run(
        "run1",
        {
            "model-a": ['["kuku"]', '["Kuku", "mama", "kuku"]', "[]", None],
            "partial": ['["kuku", "soko"]'],
            "unparseable": ["no list", "[]", "[]", "[]"],
            "all-errors": [None] * 4,
        },
        selected=["model-a", "partial", "unparseable", "all-errors", "absent"],
    )
    # The same name in another run must not be combined with run1's predictions.
    run2 = make_run(
        "run2",
        {
            "model-a": ['["soko", "kuku"]', '["kuku"]', "[]", '["soko"]'],
        },
    )
    source_bytes = {
        p: p.read_bytes() for p in (tmp_path / "source").rglob("*") if p.is_file()
    }
    output = tmp_path / "analysis"
    summary = analysis.analyze(
        [str(run1), str(run2)], str(items), checksum, output, n_bootstrap=100
    )
    by_model = summary.set_index(["run_id", "model"])
    row = by_model.loc[("run1", "model-a")]
    assert row["pooled_f1"] == pytest.approx(2 / 3)
    assert row["pooled_f2"] == pytest.approx(2 / 3)
    assert row["item_averaged_f1"] == pytest.approx(2 / 3)
    assert row["n_items_with_f1"] == 2
    assert row["n_items_f1_undefined"] == 1
    assert row["perfect_match_accuracy"] == pytest.approx(1 / 3)
    assert row["n_scored"] == 3
    assert row["n_call_failed"] == 1
    assert by_model.loc[("run2", "model-a"), "perfect_match_accuracy"] == 1
    assert pd.isna(by_model.loc[("run1", "all-errors"), "pooled_f1"])
    assert by_model.loc[("run1", "all-errors"), "n_scored"] == 0
    assert pd.isna(by_model.loc[("run1", "all-errors"), "item_averaged_f1"])
    assert by_model.loc[("run1", "all-errors"), "n_items_with_f1"] == 0
    assert by_model.loc[("run1", "all-errors"), "n_items_f1_undefined"] == 0
    assert by_model.loc[("run1", "all-errors"), "pooled_f1_n_bootstrap_valid"] == 0
    assert pd.isna(by_model.loc[("run1", "all-errors"), "pooled_f1_ci_low"])
    assert pd.isna(by_model.loc[("run1", "all-errors"), "pooled_f2"])
    assert pd.isna(by_model.loc[("run1", "all-errors"), "pooled_f2_ci_low"])
    assert by_model.loc[("run1", "all-errors"), "pooled_f2_n_bootstrap_valid"] == 0
    assert by_model.loc[("run2", "model-a"), "perfect_match_accuracy_ci_low"] == 1
    assert by_model.loc[("run2", "model-a"), "perfect_match_accuracy_ci_high"] == 1
    assert by_model.loc[("run2", "model-a"), "item_averaged_f1_ci_low"] == 1
    assert by_model.loc[("run2", "model-a"), "item_averaged_f1_ci_high"] == 1
    assert by_model.loc[("run2", "model-a"), "pooled_f2_ci_low"] == 1
    assert by_model.loc[("run2", "model-a"), "pooled_f2_ci_high"] == 1
    assert by_model.loc[("run1", "unparseable"), "n_parse_failed"] == 1
    assert "partial" not in summary["model"].tolist()
    coverage = pd.read_csv(output / "coverage_by_model.csv").set_index(
        ["run_id", "model"]
    )
    assert coverage.loc[("run1", "partial"), "n_missing"] == 3
    assert coverage.loc[("run1", "partial"), "n_scored"] == 0
    assert coverage.loc[("run1", "absent"), "n_recorded"] == 0
    excluded = pd.read_csv(output / "excluded_responses.csv")
    assert len(excluded) == 6
    assert set(excluded["exclusion_reason"]) == {"call_failed", "parse_failed"}
    assert source_bytes == {p: p.read_bytes() for p in source_bytes}
    output_bytes = {p: p.read_bytes() for p in output.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        analysis.analyze([str(run1)], str(items), checksum, output)
    assert output_bytes == {p: p.read_bytes() for p in output_bytes}
    manifest = json.loads((output / "snapshot_manifest.json").read_text())
    for artifact in manifest["artifacts"]:
        assert analysis.digest(output / artifact["local_path"]) == artifact["sha256"]
    metadata = json.loads((output / "analysis.json").read_text())["bootstrap"]
    assert metadata["n_bootstrap"] == 100
    assert metadata["ci_level"] == 0.95
    assert metadata["seed"] == 0
    assert metadata["shared_draws_across_models"]
    saved_summary = pd.read_csv(output / "summary_by_model.csv")
    pd.testing.assert_frame_equal(saved_summary, summary, check_dtype=False)


def test_item_f1_is_an_unweighted_mean_and_includes_fully_decodable_false_alarms(
    tmp_path, analysis, bundle
):
    items, checksum, make_run = bundle
    run = make_run(
        "run",
        {
            "empty-correct": ['["kuku", "soko", "mama"]', "[]", "[]", '["mama"]'],
            "false-alarm": ['["kuku", "soko", "mama"]', "[]", '["mama"]', '["mama"]'],
        },
    )
    output = tmp_path / "analysis"
    summary = analysis.analyze(
        [str(run)], str(items), checksum, output, n_bootstrap=100
    ).set_index("model")
    assert summary.loc["empty-correct", "pooled_f1"] == 0.5
    assert summary.loc["empty-correct", "item_averaged_f1"] == pytest.approx(0.8 / 3)
    assert summary.loc["empty-correct", "n_items_with_f1"] == 3
    assert summary.loc["empty-correct", "n_items_f1_undefined"] == 1
    assert summary.loc["false-alarm", "item_averaged_f1"] == pytest.approx(0.8 / 4)
    assert summary.loc["false-alarm", "n_items_with_f1"] == 4
    assert summary.loc["false-alarm", "n_items_f1_undefined"] == 0
    scored = pd.read_csv(output / "results_with_scoring.csv").set_index(
        ["model", "benchmark_item_id"]
    )
    assert pd.isna(scored.loc[("empty-correct", "i3"), "item_f1"])
    assert scored.loc[("false-alarm", "i3"), "item_f1"] == 0
    assert scored.loc[("false-alarm", "i3"), "nondecodable_ratio"] == 0
    assert scored.loc[("empty-correct", "i1"), "item_f1"] == 0.8
    assert scored.loc[("empty-correct", "i1"), "num_total_words"] == 3
    assert scored.loc[("empty-correct", "i1"), "num_nondecodable_words"] == 2
    assert scored.loc[("empty-correct", "i1"), "nondecodable_ratio"] == pytest.approx(
        2 / 3
    )


def test_pooled_f2_weights_misses_and_retains_finished_models_with_errors(
    tmp_path, analysis, bundle
):
    items, checksum, make_run = bundle
    run = make_run(
        "run",
        {
            "misses": ["[]", '["kuku", "mama"]', "[]", None],
            "false-alarms": [
                '["kuku", "soko", "mama"]',
                '["kuku", "mama"]',
                '["mama"]',
                None,
            ],
        },
    )
    output = tmp_path / "analysis"
    summary = analysis.analyze(
        [str(run)], str(items), checksum, output, n_bootstrap=100
    ).set_index("model")
    # TP=1, FP=1, FN=2: F2 penalises the misses more strongly than F1.
    assert summary.loc["misses", "pooled_f1"] == pytest.approx(2 / 5)
    assert summary.loc["misses", "pooled_f2"] == pytest.approx(5 / 14)
    # TP=3, FP=3, FN=0: greater recall emphasis increases the score.
    assert summary.loc["false-alarms", "pooled_f1"] == pytest.approx(2 / 3)
    assert summary.loc["false-alarms", "pooled_f2"] == pytest.approx(5 / 6)
    assert summary.finished.all()
    assert summary.n_call_failed.eq(1).all()
    assert summary.n_scored.eq(3).all()
    assert "item_averaged_f2" not in summary.columns
    metadata = json.loads((output / "analysis.json").read_text())
    assert "4 * sum(FN)" in metadata["pooled_f2"]


def test_pooled_f2_bootstrap_recomputes_from_resampled_counts(analysis, monkeypatch):
    draws = iter([np.array([0, 0]), np.array([0, 1]), np.array([1, 1])])

    class FixedRng:
        def integers(self, low, high, size):
            assert (low, high, size) == (0, 2, 2)
            return next(draws)

    monkeypatch.setattr(analysis.np.random, "default_rng", lambda seed: FixedRng())
    scored = pd.DataFrame(
        {
            "run_id": ["r", "r"],
            "model": ["a", "a"],
            "benchmark_item_id": ["i1", "i2"],
            "tp_count": [2, 0],
            "fp_count": [1, 0],
            "fn_count": [0, 3],
            "perfect_match": [False, False],
        }
    )
    result = analysis.bootstrap_intervals(
        scored,
        pd.Series(["i1", "i2"]),
        scored[["run_id", "model"]].drop_duplicates(),
        n_bootstrap=3,
        ci_level=0.5,
    ).iloc[0]
    # Pooled F2 resamples: 10/11, 10/23, 0; not means of per-item F2.
    assert result.pooled_f2_ci_low == pytest.approx(5 / 23)
    assert result.pooled_f2_ci_high == pytest.approx(170 / 253)
    assert result.pooled_f2_n_bootstrap_valid == 3
    assert next(draws, None) is None


@pytest.mark.parametrize(
    "problem",
    ["duplicate", "unknown_id", "wrong_run", "status", "hash", "duplicate_file"],
)
def test_refuses_ambiguous_or_mismatched_results(tmp_path, analysis, bundle, problem):
    items, checksum, make_run = bundle
    run = make_run("run", {"model": ["[]"] * 4})
    path = run / "responses" / "responses_model.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if problem == "duplicate":
        rows[1] = rows[0]
    elif problem == "unknown_id":
        rows[0]["benchmark_item_id"] = "unknown"
    elif problem == "wrong_run":
        rows[0]["run_id"] = "other"
    elif problem == "status":
        rows[0]["success"] = False
    elif problem == "hash":
        config_path = run / "config.json"
        config = json.loads(config_path.read_text())
        config["resolved_config"]["items_sha256"] = "wrong"
        config_path.write_text(json.dumps(config))
    elif problem == "duplicate_file":
        write_jsonl(run / "responses" / "duplicate.jsonl", rows)
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError):
        analysis.analyze([str(run)], str(items), checksum, tmp_path / "analysis")
    assert not (tmp_path / "analysis" / "summary_by_model.csv").exists()


def test_label_drift_stops_before_model_scoring(tmp_path, analysis, bundle):
    items, _, make_run = bundle
    rows = [json.loads(line) for line in items.read_text().splitlines()]
    rows[0]["nondecodable_words"] = []
    items.write_text("\n".join(json.dumps(row) for row in rows))
    output = tmp_path / "analysis"
    with pytest.raises(ValueError, match="Tusome labels differ"):
        analysis.analyze([], str(items), analysis.digest(items), output)
    differences = json.loads((output / "label_differences.json").read_text())
    assert differences == [
        {"benchmark_item_id": "i1", "saved": [], "recomputed": ["kuku", "soko"]}
    ]
    assert not (output / "summary_by_model.csv").exists()


def test_dataset_hash_is_required(tmp_path, analysis, bundle):
    items, _, _ = bundle
    with pytest.raises(ValueError, match="Dataset SHA-256"):
        analysis.analyze([], str(items), "wrong", tmp_path / "analysis")


def test_gold_override_scores_supplied_labels_without_recomputation(
    tmp_path, analysis, bundle, monkeypatch
):
    items, checksum, make_run = bundle
    run = make_run("run", {"model": ['["kuku", "soko"]', '["kuku"]', "[]", '["soko"]']})
    rows = [json.loads(line) for line in items.read_text().splitlines()]
    rows[0].update(
        nondecodable_words=["KUKU", "kuku"],
        num_nondecodable_words=999,
        num_total_words=999,
        nondecodable_ratio=999,
    )
    gold = tmp_path / "source" / "corrected.jsonl"
    write_jsonl(gold, rows[::-1])
    source_bytes = {
        p: p.read_bytes() for p in (tmp_path / "source").rglob("*") if p.is_file()
    }

    def unexpected_recomputation(*args, **kwargs):
        pytest.fail("Supplied gold labels must not be recomputed")

    monkeypatch.setattr(analysis, "add_labels_and_analysis", unexpected_recomputation)
    output = tmp_path / "analysis"
    summary = analysis.analyze(
        [str(run)],
        str(items),
        checksum,
        output,
        gold_labels_source=str(gold),
        n_bootstrap=100,
    )
    row = summary.iloc[0]
    assert (row.pooled_tp, row.pooled_fp, row.pooled_fn) == (3, 1, 0)
    assert row.pooled_f1 == pytest.approx(6 / 7)
    assert row.pooled_f2 == pytest.approx(15 / 16)
    assert row.perfect_match_accuracy == 0.75
    # One seeded draw contains only the empty/empty item: F1 and F2 are undefined.
    assert row.pooled_f1_n_bootstrap_valid == 99
    assert row.pooled_f2_n_bootstrap_valid == 99
    assert row.perfect_match_accuracy_n_bootstrap_valid == 100
    for metric in ["pooled_f1", "pooled_f2", "perfect_match_accuracy"]:
        assert row[f"{metric}_ci_low"] < row[f"{metric}_ci_high"]
    verified = analysis.read_jsonl(output / "verified_items.jsonl")
    assert verified.benchmark_item_id.tolist() == ["i1", "i2", "i3", "i4"]
    assert verified.nondecodable_words.tolist() == [["kuku"], ["kuku"], [], ["soko"]]
    assert verified.iloc[0].num_total_words == 3
    assert verified.iloc[0].num_nondecodable_words == 1
    assert verified.iloc[0].nondecodable_ratio == pytest.approx(1 / 3)
    metadata = json.loads((output / "analysis.json").read_text())
    assert metadata["items_sha256"] == checksum
    assert metadata["gold_labels_sha256"] == analysis.digest(gold)
    assert metadata["gold_labels_source"] == str(gold)
    assert metadata["decodability_config"] is None
    manifest = json.loads((output / "snapshot_manifest.json").read_text())
    for artifact in manifest["artifacts"]:
        assert analysis.digest(output / artifact["local_path"]) == artifact["sha256"]
    assert (output / "inputs" / "gold_labels.jsonl").read_bytes() == gold.read_bytes()
    assert (output / "inputs" / "run-1" / "config.json").read_bytes() == (
        run / "config.json"
    ).read_bytes()
    assert source_bytes == {p: p.read_bytes() for p in source_bytes}


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("benchmark_item_id", "unknown", "original item IDs"),
        ("benchmark_item_id", "i2", "original item IDs"),
        ("content", "Different text.", "original content"),
        ("student_knowledge", {"graphemes": []}, "original student_knowledge"),
        ("language_iso_code", "en", "original language_iso_code"),
        ("nondecodable_words", "kuku", "word lists"),
        ("nondecodable_words", None, "word lists"),
        ("nondecodable_words", [7], "word lists"),
        ("nondecodable_words", [""], "word lists"),
        ("nondecodable_words", ["absent"], "words outside i1"),
    ],
)
def test_gold_override_rejects_changed_inputs_and_invalid_labels(
    tmp_path, analysis, bundle, field, value, message
):
    items, checksum, _ = bundle
    rows = [json.loads(line) for line in items.read_text().splitlines()]
    rows[0][field] = value
    gold = tmp_path / "corrected.jsonl"
    write_jsonl(gold, rows)
    with pytest.raises(ValueError, match=message):
        analysis.verify_items(items, checksum, tmp_path, gold_labels_path=gold)
    assert not (tmp_path / "verified_items.jsonl").exists()


@pytest.mark.parametrize("mismatch", ["source", "run"])
def test_gold_override_preserves_original_dataset_hash_checks(
    tmp_path, analysis, bundle, mismatch
):
    items, checksum, make_run = bundle
    run = make_run("run", {"model": ["[]"] * 4})
    if mismatch == "source":
        checksum = "wrong"
    else:
        config_path = run / "config.json"
        config = json.loads(config_path.read_text())
        config["resolved_config"]["items_sha256"] = "wrong"
        config_path.write_text(json.dumps(config))
    output = tmp_path / "analysis"
    with pytest.raises(ValueError, match="SHA-256|dataset hash"):
        analysis.analyze(
            [str(run)], str(items), checksum, output, gold_labels_source=str(items)
        )
    assert not (output / "summary_by_model.csv").exists()


def test_cloud_reads_pin_generations_and_never_overwrite(tmp_path, analysis):
    class Blob:
        generation = 123

        def reload(self, **kwargs):
            pass

        def download_to_file(self, handle, *, if_generation_match, timeout):
            assert if_generation_match == 123
            handle.write(b"checkpoint")

    snapshotter = analysis.Snapshotter(tmp_path)
    path = tmp_path / "snapshot.jsonl"
    snapshotter.copy("gs://bucket/key", path, blob=Blob())
    assert snapshotter.artifacts[0]["generation"] == 123
    with pytest.raises(FileExistsError):
        snapshotter.copy("gs://bucket/key", path, blob=Blob())
    assert path.read_bytes() == b"checkpoint"


def test_cluster_and_whole_word_rules_are_applied(tmp_path, analysis):
    # All graphemes are known, but vy must additionally be taught as a cluster;
    # a known whole word overrides that requirement.
    rows = [
        {
            "benchmark_item_id": "unknown",
            "content": "vya",
            "student_knowledge": {"graphemes": ["v", "y", "a"]},
        },
        {
            "benchmark_item_id": "cluster",
            "content": "vya",
            "student_knowledge": {"graphemes": ["v", "y", "a"], "clusters": ["vy"]},
        },
        {
            "benchmark_item_id": "whole",
            "content": "vya",
            "student_knowledge": {"graphemes": [], "whole_words": ["vya"]},
        },
    ]
    path = tmp_path / "items.jsonl"
    write_jsonl(path, rows)
    result = analysis.verify_items(path, analysis.digest(path), tmp_path)
    assert result["nondecodable_words"].tolist() == [["vya"], [], []]


def test_bootstrap_preserves_pairing_multiplicity_and_missing_scores(
    analysis, monkeypatch
):
    # Use three known resamples of two items to check the resulting quantiles.
    # The same model name in two runs checks that run identity is retained.
    draws = iter([np.array([0, 0]), np.array([0, 1]), np.array([1, 1])])

    class FixedDraws:
        def integers(self, low, high, size):
            assert (low, high, size) == (0, 2, 2)
            return next(draws)

    monkeypatch.setattr(analysis.np.random, "default_rng", lambda seed: FixedDraws())
    models = pd.DataFrame({"run_id": ["r1", "r2", "r1"], "model": ["a", "a", "b"]})
    rows = []
    for run in ["r1", "r2"]:
        rows.extend(
            [
                {
                    "run_id": run,
                    "model": "a",
                    "benchmark_item_id": "i1",
                    "tp_count": 2,
                    "fp_count": 0,
                    "fn_count": 0,
                    "perfect_match": True,
                },
                {
                    "run_id": run,
                    "model": "a",
                    "benchmark_item_id": "i2",
                    "tp_count": 0,
                    "fp_count": 1,
                    "fn_count": 1,
                    "perfect_match": False,
                },
            ]
        )
    # b has a valid empty/empty prediction only on i2. Sampling i1 twice has no
    # valid score for b; the other draws have accuracy 1 but undefined F1.
    rows.append(
        {
            "run_id": "r1",
            "model": "b",
            "benchmark_item_id": "i2",
            "tp_count": 0,
            "fp_count": 0,
            "fn_count": 0,
            "perfect_match": True,
        }
    )
    result = analysis.bootstrap_intervals(
        pd.DataFrame(rows),
        pd.Series(["i1", "i2"]),
        models,
        n_bootstrap=3,
        ci_level=0.5,
    )
    # Resampled pooled F1s are 1, 2/3, 0. Averaging item F1s would give 1, 1/2, 0.
    assert result.loc[0, "pooled_f1_ci_low"] == pytest.approx(1 / 3)
    assert result.loc[0, "pooled_f1_ci_high"] == pytest.approx(5 / 6)
    assert result.loc[0, "perfect_match_accuracy_ci_low"] == 0.25
    assert result.loc[0, "perfect_match_accuracy_ci_high"] == 0.75
    assert result.loc[0, "item_averaged_f1_ci_low"] == 0.25
    assert result.loc[0, "item_averaged_f1_ci_high"] == 0.75
    assert result.loc[0, "item_averaged_f1_n_bootstrap_valid"] == 3
    pd.testing.assert_series_equal(
        result.iloc[0, 2:], result.iloc[1, 2:], check_names=False
    )
    assert result.loc[2, "perfect_match_accuracy_ci_low"] == 1
    assert result.loc[2, "perfect_match_accuracy_ci_high"] == 1
    assert result.loc[2, "perfect_match_accuracy_n_bootstrap_valid"] == 2
    assert result.loc[2, "pooled_f1_n_bootstrap_valid"] == 0
    assert pd.isna(result.loc[2, "pooled_f1_ci_low"])
    assert result.loc[2, "item_averaged_f1_n_bootstrap_valid"] == 0
    assert pd.isna(result.loc[2, "item_averaged_f1_ci_low"])
    assert result.loc[2, "pooled_f2_n_bootstrap_valid"] == 0
    assert pd.isna(result.loc[2, "pooled_f2_ci_low"])
    assert next(draws, None) is None  # exactly one shared draw per replicate


def test_bootstrap_is_reproducible_and_handles_no_finished_models(analysis):
    models = pd.DataFrame({"run_id": ["r"], "model": ["a"]})
    scores = pd.DataFrame(
        {
            "run_id": ["r"] * 3,
            "model": ["a"] * 3,
            "benchmark_item_id": ["i1", "i2", "i3"],
            "tp_count": [2, 0, 1],
            "fp_count": [0, 1, 3],
            "fn_count": [0, 1, 2],
            "perfect_match": [True, False, False],
        }
    )
    item_ids = pd.Series(["i1", "i2", "i3"])
    expected = analysis.bootstrap_intervals(
        scores,
        item_ids,
        models,
        n_bootstrap=100,
        seed=42,
    )
    repeated = analysis.bootstrap_intervals(
        scores.iloc[::-1],
        item_ids,
        models,
        n_bootstrap=100,
        seed=42,
    )
    pd.testing.assert_frame_equal(expected, repeated)
    empty = analysis.bootstrap_intervals(
        scores.iloc[:0],
        item_ids,
        models.iloc[:0],
        n_bootstrap=10,
    )
    assert empty.empty
    assert "pooled_f1_ci_low" in empty.columns


@pytest.mark.parametrize(
    "options",
    [
        {"n_bootstrap": 0},
        {"n_bootstrap": -1},
        {"ci_level": 0},
        {"ci_level": 1},
        {"ci_level": float("nan")},
        {"seed": -1},
    ],
)
def test_invalid_bootstrap_options_fail_before_creating_outputs(
    tmp_path, analysis, options
):
    output = tmp_path / "analysis"
    with pytest.raises(ValueError):
        analysis.analyze([], "unused", "unused", output, **options)
    assert not output.exists()
