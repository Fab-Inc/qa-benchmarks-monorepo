"""Guard portable reporting inputs, dynamic denominators and metadata integrity."""

import importlib.util
import json
from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def analysis(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "f2_analysis", scripts / "run_f2_analysis.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def run_dir(tmp_path):
    run = tmp_path / "english"
    run.mkdir()
    items = run / "benchmark_items.jsonl"
    items.write_text(
        "".join(json.dumps({"benchmark_item_id": f"item-{i}"}) + "\n" for i in range(5))
    )
    config = {
        "resolved_config": {
            "items_path": str(items),
            "items_sha256": sha256(items.read_bytes()).hexdigest(),
        }
    }
    (run / "config.json").write_text(json.dumps(config))
    (run / "coverage_by_model.csv").write_text(
        "model,expected_items,scorable_items\nexample,5,4\n"
    )
    (run / "item_provenance.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "benchmark_item_id": f"item-{i}",
                    "gold_status": "corrected" if i < 3 else "provisional",
                }
            )
            + "\n"
            for i in range(5)
        )
    )
    return run


def test_sizes_and_gold_notes_come_from_input(analysis, run_dir):
    size, note = analysis.inspect_dataset(run_dir)
    assert size == 5
    assert note == "English gold: 3 corrected items; 2 provisional items."


def test_old_denominator_rejected(analysis, run_dir):
    (run_dir / "coverage_by_model.csv").write_text(
        "model,expected_items,scorable_items\nexample,991,4\n"
    )
    with pytest.raises(ValueError, match="denominator"):
        analysis.inspect_dataset(run_dir)


def test_gold_provenance_ids_must_match(analysis, run_dir):
    (run_dir / "item_provenance.jsonl").write_text(
        '{"benchmark_item_id":"other","gold_status":"corrected"}\n'
    )
    with pytest.raises(ValueError, match="provenance"):
        analysis.inspect_dataset(run_dir)


def test_missing_old_absolute_item_path_has_actionable_error(analysis, run_dir):
    path = run_dir / "config.json"
    config = json.loads(path.read_text())
    config["resolved_config"]["items_path"] = str(run_dir / "missing.jsonl")
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="rebuild the merge/selection locally"):
        analysis.inspect_dataset(run_dir)


def test_changed_input_rejected_before_scoring(analysis, tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text("original\n")
    manifest = {"files": {"data.jsonl": sha256(path.read_bytes()).hexdigest()}}
    (tmp_path / "artifact_manifest.json").write_text(json.dumps(manifest))
    assert analysis.verify_merge(tmp_path) == manifest
    path.write_text("changed\n")
    with pytest.raises(ValueError, match="Merged input has changed"):
        analysis.verify_merge(tmp_path)


def test_manifest_cannot_read_outside_root(analysis, tmp_path):
    (tmp_path / "artifact_manifest.json").write_text(
        json.dumps({"files": {"../outside": "ignored"}})
    )
    with pytest.raises(ValueError, match="leaves the input root"):
        analysis.verify_merge(tmp_path)


def test_bundled_metadata_supports_new_qwen_alias(analysis):
    specs = analysis.hardware.load_metadata(analysis.DEFAULT_METADATA / "models.csv")
    assert specs["qwen3-8b-alibaba"]["size"] == 8e9
    assert (
        analysis.hardware.assign_hardware(specs["qwen3-8b-alibaba"])["hardware_tier"]
        == "Mobile + Tablet"
    )
    models = pd.read_csv(analysis.DEFAULT_METADATA / "models.csv")
    providers = pd.read_csv(analysis.DEFAULT_METADATA / "providers.csv")
    assert models.model_id.is_unique and providers.provider.is_unique
    assert set(models.provider) <= set(providers.provider)


def test_leaderboard_filter_counts_successes_and_preserves_intervals(analysis):
    table = pd.DataFrame(
        {
            "model": ["below", "exact", "above"],
            "scorable_items": [699, 700, 701],
            "response_rows": [991, 991, 991],
            "score": [0.95, 0.9, 0.8],
            "ci_low": [0.94, 0.89, 0.79],
            "ci_high": [0.96, 0.91, 0.81],
        }
    )
    filtered = analysis.leaderboard.eligible_models(table)
    pd.testing.assert_frame_equal(
        filtered, table.iloc[1:].reset_index(drop=True), check_exact=True
    )
    assert analysis.leaderboard.eligible_models(table, 701).model.tolist() == ["above"]


def test_nonpositive_threshold_rejected(analysis):
    with pytest.raises(ValueError, match="positive"):
        analysis.leaderboard.eligible_models(pd.DataFrame(), 0)


def test_coverage_plot_preserves_default_and_minimum_item_outputs(
    analysis, tmp_path, monkeypatch
):
    run = tmp_path / "english"
    report = run / "report"
    report.mkdir(parents=True)
    for suffix in ("", "_min700"):
        for ext in ("png", "pdf", "csv"):
            (report / f"pooled_f2_leaderboard{suffix}.{ext}").write_bytes(b"existing")
    table = pd.DataFrame(
        {
            "model": ["below", "eligible"],
            "display_name": ["Below", "Eligible"],
            "scorable_items": [792, 793],
            "n_items": [792, 793],
            "expected_items": [991, 991],
            "provider": ["test", "test"],
            "color": ["#2a78d6", "#2a78d6"],
            "score": [0.95, 0.9],
            "ci_low": [0.94, 0.89],
            "ci_high": [0.96, 0.91],
            "ci_level": [0.95, 0.95],
            "n_bootstrap": [1000, 1000],
        }
    )
    monkeypatch.setattr(
        analysis.leaderboard,
        "load_plot_table",
        lambda *_: (table, {"test": "#2a78d6"}),
    )
    analysis.save_leaderboards(
        run,
        table,
        "English gold: 991 corrected items.",
        793,
        output_suffix="_coverage_gt80",
        eligibility_label=">80% coverage",
    )
    saved = analysis.leaderboard.read_csv(
        report / "pooled_f2_leaderboard_coverage_gt80.csv"
    )
    pd.testing.assert_frame_equal(saved, table.iloc[1:].reset_index(drop=True))
    for suffix in ("", "_min700"):
        for ext in ("png", "pdf", "csv"):
            assert (
                report / f"pooled_f2_leaderboard{suffix}.{ext}"
            ).read_bytes() == b"existing"
