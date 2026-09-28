"""Protect unified run identity, runner fidelity, and the publish gates."""

import base64
import hashlib
import importlib
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import google_crc32c
import pandas as pd
import pytest
from google.api_core.exceptions import PreconditionFailed
from phonics_benchmark import runner
from phonics_benchmark.config import build_run_config, config_identity_sha256

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
builder = importlib.import_module("build_unified_leaderboard_runs")
analysis = importlib.import_module("analyze_unified_leaderboard_runs")
report = importlib.import_module("report_on_benchmark_run")
publisher = importlib.import_module("publish_unified_leaderboard_runs")
downloader = importlib.import_module("download_unified_leaderboard_runs")
f2 = importlib.import_module("run_f2_analysis")
FIXTURE = Path(__file__).parent / "fixtures/completed_benchmark_items.jsonl"


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


def test_compare_ignores_only_the_dropped_models(tmp_path):
    row = {column: 0.0 for column in analysis.COMPARE_COLUMNS}
    row.update(n_scorable=1, expected_items=1)
    reference = tmp_path / "reference.csv"
    pd.DataFrame([{**row, "model": "model-1"}, {**row, "model": "dropped"}]).to_csv(
        reference, index=False
    )
    fresh = pd.DataFrame([{**row, "model": "model-1"}])
    assert analysis.compare(fresh, reference, ("dropped",))["dropped_models"] == [
        "dropped"
    ]
    with pytest.raises(ValueError, match="Model set differs"):
        analysis.compare(fresh, reference)


def write_gold_provenance(run, item_ids, gold):
    run.mkdir(parents=True, exist_ok=True)
    (run / "item_provenance.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "benchmark_item_id": item_id,
                    "gold_sources": [gold],
                    "gold_status": "corrected",
                    "source_datasets": [],
                }
            )
            + "\n"
            for item_id in item_ids
        )
    )


def test_run_config_is_a_normal_launch_config(tmp_path):
    run = tmp_path / "kiswahili"
    gold = "gs://bucket/datasets/sw-corrected.jsonl"
    write_gold_provenance(run, ["item-0", "item-1"], gold)
    source = {
        "resolved_config": {
            "metadata": {
                "snapshot_started_at": "2026-09-22T13:58:41+00:00",
                "expected_items": 2,
                "gold_status_counts": {"corrected": 2},
                "gold_update": {"source_uri": "gs://bucket/datasets/english.jsonl"},
                "parent_full_root": "/Users/someone/reports/full",
            }
        }
    }
    metadata = builder.run_metadata(source, run, {"source_responses_sha256": "abc"})
    assert "gold_update" not in metadata and "parent_full_root" not in metadata
    assert metadata["gold_sources"] == {gold: 2}
    config = builder.run_config(
        "sw-run", "sw", ["model-1"], metadata, "gs://bucket/phonics"
    )
    assert config["items_path"] == "gs://bucket/phonics/datasets/sw-run.jsonl"
    assert config["results_dest"] == "gs://bucket/phonics/runs/sw-run/"
    assert config["max_tokens"] == 30_000
    assert config["prompt_name"] == "decodability_evaluation_clusters"
    source["resolved_config"]["metadata"]["gold_status_counts"] = {"corrected": 1}
    with pytest.raises(ValueError, match="Gold status counts"):
        builder.run_metadata(source, run, {})
    with pytest.raises(ValueError, match="Local paths"):
        builder.run_config(
            "sw-run", "sw", ["model-1"], {"x": "/Users/someone"}, "gs://bucket/p"
        )


def kiswahili_config(tmp_path, output):
    items = tmp_path / "items.jsonl"
    items.write_text("".join(FIXTURE.read_text().splitlines(keepends=True)[:2]))
    custom_models = tmp_path / "custom_models.yaml"
    custom_models.write_text("TestGroup:\n  models:\n    model-a: {}\n")
    return build_run_config(
        run_id="sw-run",
        language="sw",
        output_dir=output,
        models=["model-a"],
        base_dir=tmp_path,
        task="decodability",
        items_path=items,
        prompt_name="decodability_evaluation_clusters",
        custom_models_path=custom_models,
        max_tokens=30_000,
        model_workers=1,
    )


def answer(config, model_name, item, prompt):
    return {
        "output": "soko",
        "max_tokens": 30_000,
        "tokens_used_completion": 3,
        "timestamp_utc": "2026-09-22T00:00:00Z",
        "latency_seconds": 0.5,
    }


def refuse(*args):
    raise AssertionError("The model was called")


def test_runner_files_match_what_the_runner_writes(tmp_path):
    config = kiswahili_config(tmp_path, tmp_path / "runner")
    model_configs = {"model-a": {"Api_Model_Name": "model-a"}}
    config_id = config_identity_sha256(config)
    runner.run_benchmark(
        config, config_id=config_id, evaluator=answer, model_configs=model_configs
    )
    written = tmp_path / "runner/sw-run"
    responses = written / "responses/responses_model-a.jsonl"
    rows = [json.loads(line) for line in responses.read_text().splitlines()]

    ours = tmp_path / "ours"
    stats = builder.write_runner_run(config, model_configs, rows[::-1], ours)
    assert (ours / "responses/responses_model-a.jsonl").read_bytes() == (
        responses.read_bytes()
    )
    assert json.loads((ours / "config.json").read_text()) == json.loads(
        (written / "config.json").read_text()
    )
    assert stats["missing_calls"] == 0

    shutil.copytree(ours, tmp_path / "rerun/sw-run")
    rerun = replace(config, output_dir=tmp_path / "rerun")
    output = runner.run_benchmark(
        rerun, config_id=config_id, evaluator=refuse, model_configs=model_configs
    )
    assert output["executed_count"] == 0 and output["skipped_count"] == 2

    with pytest.raises(ValueError, match="not calls this config would make"):
        builder.write_runner_run(
            config, {"model-a": {"Api_Model_Name": "other"}}, rows, tmp_path / "x"
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

    def download_to_filename(self, filename, *, if_generation_match):
        generation, content = self.client.objects[self.name]
        if generation != if_generation_match:
            raise PreconditionFailed(self.name)
        Path(filename).write_bytes(content)

    def delete(self, *, if_generation_match):
        self.client.deletes.append((self.name, if_generation_match))
        if self.client.objects[self.name][0] != if_generation_match:
            raise PreconditionFailed(self.name)
        del self.client.objects[self.name]


class FakeClient:
    """One in-memory bucket with GCS generation preconditions."""

    def __init__(self):
        self.objects, self.uploads, self.deletes = {}, [], []
        self.next_generation, self.listed_crc32c = 1, {}

    def bucket(self, name):
        return SimpleNamespace(blob=lambda object_name: FakeBlob(self, object_name))

    def list_blobs(self, bucket_name, *, prefix):
        return [
            SimpleNamespace(
                name=name,
                generation=generation,
                crc32c=self.listed_crc32c.get(name, crc32c(content)),
            )
            for name, (generation, content) in sorted(self.objects.items())
            if name.startswith(prefix)
        ]


ITEMS = b'{"benchmark_item_id": "item-1"}\n'


def seal_bundle(root, note):
    """Write a one-run runner-shaped bundle, seal it, and pass the comparison."""
    run, prefix = root / "english", "gs://bucket/runs/en-run/"
    items_uri = "gs://bucket/datasets/en-run.jsonl"
    (run / "responses").mkdir(parents=True, exist_ok=True)
    (run / "responses/responses_model-a.jsonl").write_text("{}\n")
    (run / "benchmark_items.jsonl").write_bytes(ITEMS)
    builder.write_json(
        run / "config.json",
        {
            "config_id": "config",
            "raw_config": {"items_path": items_uri},
            "resolved_config": {
                "run_id": "en-run",
                "results_dest": prefix,
                "metadata": {"note": note},
            },
        },
    )
    builder.write_json(root / "run-configs/en-run.json", {"run_id": "en-run"})
    builder.write_json(
        root / "bundle_info.json",
        {
            "runs": {
                "english": {
                    "run_id": "en-run",
                    "gcs_prefix": prefix,
                    "items_uri": items_uri,
                    "run_config_uri": "gs://bucket/run-configs/en-run.json",
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


def client_with_items():
    client = FakeClient()
    client.objects["datasets/en-run.jsonl"] = (50, ITEMS)
    return client


def test_publish_refuses_files_outside_the_seal(tmp_path):
    seal_bundle(tmp_path, "first")
    (tmp_path / "english/report").mkdir()
    (tmp_path / "english/report/summary.csv").write_text("model\n")
    client = client_with_items()
    with pytest.raises(ValueError, match="outside the seal.*report/summary.csv"):
        publisher.publish(tmp_path, client=client)
    assert client.uploads == []


def test_publish_needs_the_run_items_in_place(tmp_path):
    seal_bundle(tmp_path, "first")
    client = client_with_items()
    client.objects["datasets/en-run.jsonl"] = (50, b"other\n")
    with pytest.raises(ValueError, match="Published items differ"):
        publisher.publish(tmp_path, client=client)


def test_publish_overwrites_only_named_files_at_checked_generation(tmp_path):
    seal_bundle(tmp_path, "first")
    client = client_with_items()
    created = publisher.publish(tmp_path, client=client)["english"]
    assert len(created["create"]) == 4 and created["replace"] == []
    assert "run-configs/en-run.json" in client.objects
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


def test_publish_deletes_only_named_leftovers_at_checked_generation(tmp_path):
    seal_bundle(tmp_path, "first")
    client = client_with_items()
    leftover = "runs/en-run/responses/merged.jsonl"
    client.objects[leftover] = (9, b"{}\n")
    with pytest.raises(ValueError, match="Unexpected objects.*merged.jsonl"):
        publisher.publish(tmp_path, client=client)
    with pytest.raises(ValueError, match="sealed files"):
        publisher.publish(tmp_path, delete=("config.json",), client=client)
    with pytest.raises(ValueError, match="do not exist.*absent.jsonl"):
        publisher.publish(
            tmp_path,
            delete=("responses/merged.jsonl", "absent.jsonl"),
            client=client,
        )
    planned = publisher.publish(
        tmp_path, delete=("responses/merged.jsonl",), dry_run=True, client=client
    )
    assert planned["english"]["delete"][0]["previous_generation"] == 9
    assert leftover in client.objects
    publisher.publish(tmp_path, delete=("responses/merged.jsonl",), client=client)
    assert client.deletes == [(leftover, 9)] and leftover not in client.objects


def publish_runner_runs(client):
    """Seed GCS with one runner-shaped run per language, as the builder writes."""
    lines = FIXTURE.read_text().splitlines(keepends=True)[:2]
    items = "".join(lines).encode()
    ids = [json.loads(line)["benchmark_item_id"] for line in lines]
    for name, language in builder.LANGUAGES.items():
        run_id = builder.RUN_IDS[name]
        prefix = f"runs/{run_id}/"
        client.objects[f"datasets/{run_id}.jsonl"] = (40, items)
        config = {
            "config_id": "config",
            "raw_config": {
                "items_path": f"gs://bucket/datasets/{run_id}.jsonl",
                "metadata": {"items_source_generation": 40},
            },
            "resolved_config": {
                "run_id": run_id,
                "language": language,
                "models": ["model-a", "model-b"],
                "items_sha256": hashlib.sha256(items).hexdigest(),
            },
        }
        rows = [
            {
                "model": "model-a",
                "benchmark_item_id": item_id,
                "model_call_id": f"call-{item_id}",
                "call_status": "success",
                "output": "soko",
            }
            for item_id in ids
        ]
        provenance = [
            {
                "benchmark_item_id": item_id,
                "gold_sources": [],
                "gold_status": "corrected",
                "source_datasets": [],
            }
            for item_id in ids
        ]
        files = {
            "config.json": json.dumps(config),
            "item_provenance.jsonl": "".join(json.dumps(r) + "\n" for r in provenance),
            "responses/responses_model-a.jsonl": "".join(
                json.dumps(row) + "\n" for row in rows
            ),
            "source_config.json": "{}",
        }
        for relative, content in files.items():
            client.objects[prefix + relative] = (7, content.encode())


def test_download_rebuilds_the_layout_report_scripts_read(tmp_path):
    client = FakeClient()
    publish_runner_runs(client)
    root = tmp_path / "gold"
    summary = downloader.download(root, "gs://bucket/runs", client=client)
    assert summary["runs"]["english"]["objects"]["config.json"]["generation"] == 7
    assert summary["runs"]["kiswahili"]["models"] == 2
    assert f2.verify_merge(root)["kind"] == "unified_leaderboard_download"
    assert f2.inspect_dataset(root / "kiswahili")[0] == 2
    coverage = pd.read_csv(root / "english/coverage_by_model.csv")
    assert coverage.set_index("model").response_rows.to_dict() == {
        "model-a": 2,
        "model-b": 0,
    }
    with pytest.raises(FileExistsError):
        downloader.download(root, "gs://bucket/runs", client=client)
    # The same generations give the same manifest; only downloaded_at differs.
    manifest = builder.digest(root / "artifact_manifest.json")
    shutil.rmtree(root)
    assert downloader.download(root, "gs://bucket/runs", client=client) == summary
    assert builder.digest(root / "artifact_manifest.json") == manifest


RUN = "runs/en-gold-leaderboard-20260922/"


def remove_item_provenance(client):
    del client.objects[RUN + "item_provenance.jsonl"]


def add_notes(client):
    client.objects[RUN + "notes.txt"] = (7, b"x")


def add_merged_responses(client):
    client.objects[RUN + "responses/merged.jsonl"] = (7, b"{}\n")


def corrupt_config(client):
    client.listed_crc32c[RUN + "config.json"] = "AAAAAA=="


def misfile_rows(client):
    rows = client.objects[RUN + "responses/responses_model-a.jsonl"][1]
    client.objects[RUN + "responses/responses_model-b.jsonl"] = (7, rows)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (remove_item_provenance, "Missing objects"),
        (add_notes, "Unexpected objects.*notes.txt"),
        (add_merged_responses, "Unexpected objects.*merged.jsonl"),
        (corrupt_config, "Checksum differs"),
        (misfile_rows, "Response for model-a"),
    ],
)
def test_download_refuses_missing_unknown_and_corrupt_objects(
    tmp_path, change, message
):
    client = FakeClient()
    publish_runner_runs(client)
    change(client)
    with pytest.raises(ValueError, match=message):
        downloader.download(tmp_path / "gold", "gs://bucket/runs", client=client)
    assert not (tmp_path / "gold").exists()


def test_compare_skips_models_whose_rows_differ_on_purpose(tmp_path):
    row = {column: 0.0 for column in analysis.COMPARE_COLUMNS}
    row.update(n_scorable=1, expected_items=1)
    reference = tmp_path / "reference.csv"
    pd.DataFrame(
        [{**row, "model": "model-1"}, {**row, "model": "later", "n_scorable": 2}]
    ).to_csv(reference, index=False)
    fresh = pd.DataFrame([{**row, "model": "model-1"}, {**row, "model": "later"}])
    result = analysis.compare(fresh, reference, skipped=("later",))
    assert result["matched_models"] == 1 and result["skipped_models"] == ["later"]
    with pytest.raises(AssertionError, match="n_scorable"):
        analysis.compare(fresh, reference)
