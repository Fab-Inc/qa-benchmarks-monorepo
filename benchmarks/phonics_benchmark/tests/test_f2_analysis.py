"""Guard portable reporting inputs, dynamic denominators and metadata integrity."""

import json
from hashlib import sha256

import pytest


@pytest.fixture
def analysis():
    from phonics_benchmark import reporting_inputs

    return reporting_inputs


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


def test_relative_run_item_path_is_resolved_from_run_directory(analysis, run_dir):
    path = run_dir / "config.json"
    config = json.loads(path.read_text())
    config["resolved_config"]["items_path"] = "benchmark_items.jsonl"
    path.write_text(json.dumps(config))
    assert analysis.inspect_dataset(run_dir)[0] == 5


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
