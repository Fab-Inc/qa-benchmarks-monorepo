"""Tests for shared_utils."""

import json

import pytest
from linguistic_knowledge_benchmark import shared_utils


def test_english_language_name_maps_kiswahili_to_swahili():
    assert shared_utils.english_language_name("Kiswahili") == "Swahili"


def test_english_language_name_passes_through_unmapped_languages():
    assert shared_utils.english_language_name("Luganda") == "Luganda"


def test_load_dataset_reads_json(tmp_path):
    path = tmp_path / "ds.json"
    path.write_text(json.dumps([{"question_id": 1}]))
    assert shared_utils.load_dataset(path) == [{"question_id": 1}]


def test_load_dataset_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        shared_utils.load_dataset(tmp_path / "nope.json")


def test_git_hash_returns_none_on_failure(monkeypatch, capsys):
    monkeypatch.setattr(
        "subprocess.check_output",
        lambda *a, **kw: (_ for _ in ()).throw(Exception("no git")),
    )
    assert shared_utils.git_hash() is None
    assert "WARNING: could not read git hash" in capsys.readouterr().err


def test_update_log_merges_with_existing_entries(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    log_path = exp_dir / "log.json"
    log_path.write_text(json.dumps({"model-a": {"keep": "yes"}}))

    shared_utils.update_log(exp_dir, "model-a", {"new": "value"})
    shared_utils.update_log(exp_dir, "model-b", {"other": "thing"})

    data = json.loads(log_path.read_text())
    assert data["model-a"]["keep"] == "yes"
    assert data["model-a"]["new"] == "value"
    assert data["model-b"]["other"] == "thing"


def test_update_log_merges_existing_and_skips_none(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    shared_utils.update_log(exp_dir, "model-a", {"a": "1", "skip": None, "b": "2"})
    shared_utils.update_log(exp_dir, "model-a", {"c": "3"})
    data = json.loads((exp_dir / "log.json").read_text())
    assert data["model-a"] == {"a": "1", "b": "2", "c": "3"}
