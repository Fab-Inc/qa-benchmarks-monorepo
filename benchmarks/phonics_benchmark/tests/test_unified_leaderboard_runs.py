"""Protect unified run identity and the adjusted-BA comparison gate."""

import base64
import importlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import google_crc32c
import pandas as pd
import pytest
from google.api_core.exceptions import PreconditionFailed

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
builder = importlib.import_module("build_unified_leaderboard_runs")
analysis = importlib.import_module("analyze_unified_leaderboard_runs")
report = importlib.import_module("report_on_benchmark_run")
publisher = importlib.import_module("publish_unified_leaderboard_runs")


def test_unified_run_rejects_duplicate_model_item(tmp_path):
    run = tmp_path / "english"
    (run / "responses").mkdir(parents=True)
    item = {"benchmark_item_id": "item-1", "language_iso_code": "en"}
    (run / "benchmark_items.jsonl").write_text(json.dumps(item) + "\n")
    (run / "config.json").write_text(
        json.dumps(
            {
                "resolved_config": {
                    "language": "en",
                    "models": ["model-1"],
                    "items_sha256": builder.digest(run / "benchmark_items.jsonl"),
                }
            }
        )
    )
    responses = [
        {
            "model": "model-1",
            "benchmark_item_id": "item-1",
            "model_call_id": call_id,
            "language": "en",
            "run_id": "source-run",
        }
        for call_id in ("call-1", "call-2")
    ]
    (run / "responses/merged.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in responses)
    )
    with pytest.raises(ValueError, match="Invalid merged response"):
        builder.check_run(run, "en")


def test_adjusted_ba_mismatch_is_rejected(tmp_path):
    row = {column: 0.0 for column in analysis.COMPARE_COLUMNS}
    row.update(model="model-1", n_scorable=1, expected_items=1)
    reference = tmp_path / "reference.csv"
    pd.DataFrame([row]).to_csv(reference, index=False)
    changed = pd.DataFrame([{**row, "adjusted_balanced_accuracy": 1.0}])
    with pytest.raises(AssertionError, match="adjusted_balanced_accuracy"):
        analysis.compare(changed, reference)


def test_combined_leaderboard_model_mismatch_is_rejected(tmp_path):
    path = tmp_path / "combined.csv"
    pd.DataFrame(
        [
            {
                "language": "english",
                "model": "other-model",
                "n_scorable": 1,
                "adjusted_balanced_accuracy": 0,
                "adjusted_balanced_accuracy_ci_low": 0,
                "adjusted_balanced_accuracy_ci_high": 0,
            }
        ]
    ).to_csv(path, index=False)
    fresh = pd.DataFrame(
        [{"model": "model-1", "expected_items": 1000, "n_scorable": 901}]
    )
    with pytest.raises(ValueError, match="Eligible model set differs"):
        analysis.compare_combined(fresh, "english", path)


def test_report_reads_bundle_relative_item_path(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "benchmark_items.jsonl").write_text('{"benchmark_item_id":"item-1"}\n')
    items = report.load_run_items(
        {
            "items_path": "benchmark_items.jsonl",
            "items_sha256": builder.digest(run / "benchmark_items.jsonl"),
        },
        run,
    )
    assert items.benchmark_item_id.tolist() == ["item-1"]


def test_unified_config_describes_only_the_analysis_run(tmp_path):
    run = tmp_path / "kiswahili"
    run.mkdir()
    gold = "gs://bucket/datasets/sw-corrected.jsonl"
    (run / "item_provenance.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "benchmark_item_id": f"item-{i}",
                    "gold_sources": [gold],
                    "gold_status": "corrected",
                    "source_datasets": [],
                }
            )
            + "\n"
            for i in range(2)
        )
    )
    source = {
        "analysis_only": True,
        "resolved_config": {
            "run_id": "kiswahili",
            "language": "sw",
            "items_path": "/Users/someone/selection/kiswahili/benchmark_items.jsonl",
            "items_sha256": "abc",
            "models": ["model-1"],
            "metadata": {
                "snapshot_started_at": "2026-09-22T13:58:41+00:00",
                "expected_items": 2,
                "gold_status_counts": {"corrected": 2},
                "gold_update": {"source_uri": "gs://bucket/datasets/english.jsonl"},
                "parent_full_root": "/Users/someone/reports/full",
            },
        },
    }
    config = builder.unified_config(source, run, "sw-run", {"items_source_uri": "x"})
    metadata = config["resolved_config"]["metadata"]
    assert "gold_update" not in metadata and "parent_full_root" not in metadata
    assert metadata["gold_sources"] == {gold: 2}
    assert metadata["snapshot_started_at"] == "2026-09-22T13:58:41+00:00"
    assert config["resolved_config"]["items_path"] == "benchmark_items.jsonl"
    assert builder.local_paths(config) == []
    source["resolved_config"]["metadata"]["gold_status_counts"] = {"corrected": 1}
    with pytest.raises(ValueError, match="Gold status counts"):
        builder.unified_config(source, run, "sw-run", {})
    with pytest.raises(ValueError, match="Local paths"):
        builder.unified_config(
            {
                **source,
                "resolved_config": {**source["resolved_config"], "metadata": {}},
            },
            run,
            "sw-run",
            {"source": "/Users/someone/source"},
        )


def crc32c(content: bytes) -> str:
    return base64.b64encode(google_crc32c.Checksum(content).digest()).decode()


class FakeBlob:
    def __init__(self, client, name):
        self.client, self.name, self.generation = client, name, None

    def upload_from_filename(self, filename, *, if_generation_match):
        self.client.uploads.append((self.name, if_generation_match))
        if self.client.objects.get(self.name, (0, b""))[0] != if_generation_match:
            raise PreconditionFailed(self.name)
        self.generation = self.client.next_generation
        self.client.next_generation += 1
        self.client.objects[self.name] = (self.generation, Path(filename).read_bytes())


class FakeClient:
    """One in-memory bucket with GCS generation preconditions."""

    def __init__(self):
        self.objects, self.uploads, self.next_generation = {}, [], 1

    def bucket(self, name):
        return SimpleNamespace(blob=lambda object_name: FakeBlob(self, object_name))

    def list_blobs(self, bucket_name, *, prefix):
        return [
            SimpleNamespace(name=name, generation=generation, crc32c=crc32c(content))
            for name, (generation, content) in sorted(self.objects.items())
            if name.startswith(prefix)
        ]


def seal_bundle(root, note):
    """Write a one-run bundle, seal it, and record a passing comparison."""
    run, prefix = root / "english", "gs://bucket/runs/en-run/"
    (run / "responses").mkdir(parents=True, exist_ok=True)
    (run / "responses/merged.jsonl").write_text("{}\n")
    (run / "benchmark_items.jsonl").write_text("{}\n")
    builder.write_json(
        run / "config.json",
        {
            "resolved_config": {
                "run_id": "en-run",
                "metadata": {
                    "items_source_uri": prefix + "benchmark_items.jsonl",
                    "note": note,
                },
            }
        },
    )
    builder.write_json(
        root / "bundle_info.json",
        {
            "runs": {
                "english": {
                    "run_id": "en-run",
                    "gcs_prefix": prefix,
                    "response_rows": 1,
                }
            }
        },
    )
    shutil.rmtree(root / "analysis", ignore_errors=True)
    builder.seal_artifacts(root, {"kind": "unified_leaderboard_runs"})
    passed = {"max_absolute_difference": 0.0, "matching_tolerance": 1e-10}
    builder.write_json(
        root / "analysis/comparison.json",
        {
            "source_bundle_manifest_sha256": builder.digest(
                root / "artifact_manifest.json"
            ),
            "languages": {
                "english": {"comparison": passed, "combined_summary_comparison": passed}
            },
        },
    )


def test_publish_refuses_files_outside_the_seal(tmp_path):
    seal_bundle(tmp_path, "first")
    (tmp_path / "english/report").mkdir()
    (tmp_path / "english/report/summary.csv").write_text("model\n")
    client = FakeClient()
    with pytest.raises(ValueError, match="outside the seal.*report/summary.csv"):
        publisher.publish(tmp_path, client=client)
    assert client.uploads == []


def test_publish_overwrites_only_named_files_at_checked_generation(tmp_path):
    seal_bundle(tmp_path, "first")
    client = FakeClient()
    created = publisher.publish(tmp_path, client=client)["english"]
    assert len(created["create"]) == 3 and created["replace"] == []
    assert {generation for _, generation in client.uploads} == {0}
    config = "runs/en-run/config.json"
    checked = client.objects[config][0]

    seal_bundle(tmp_path, "corrected")
    with pytest.raises(ValueError, match="Existing objects differ.*config.json"):
        publisher.publish(tmp_path, client=client)
    with pytest.raises(ValueError, match="not sealed"):
        publisher.publish(tmp_path, replace=("other.json",), client=client)
    uploads = len(client.uploads)
    planned = publisher.publish(
        tmp_path, replace=("config.json",), dry_run=True, client=client
    )["english"]
    assert planned["create"] == []
    assert [entry["previous_generation"] for entry in planned["replace"]] == [checked]
    assert len(client.uploads) == uploads

    publisher.publish(tmp_path, replace=("config.json",), client=client)
    assert client.uploads[uploads:] == [(config, checked)]
    assert b"corrected" in client.objects[config][1]
    again = publisher.publish(tmp_path, replace=("config.json",), client=client)
    assert again["english"]["create"] == again["english"]["replace"] == []
