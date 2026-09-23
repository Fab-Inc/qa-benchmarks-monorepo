"""Exercise full-pool merging, immutable selection and gold/coverage semantics."""

import importlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

downloader = importlib.import_module("download_benchmark_snapshot")
merge = importlib.import_module("merge_local_benchmark_runs")
select = importlib.import_module("select_local_benchmark_subset")


def item(key, words=None):
    words = ["cat"] if words is None else words
    return {
        "benchmark_item_id": key,
        "language_iso_code": "en",
        "content": "The cat sat.",
        "student_knowledge": {"gpcs": [], "sight_words": []},
        "nondecodable_words": words,
        "num_total_words": 3,
        "num_nondecodable_words": len(words),
        "nondecodable_ratio": len(words) / 3,
    }


def snapshot(tmp_path, changed_input=False, conflicting_calls=False):
    root = tmp_path / "snapshot"
    root.mkdir()
    uris = {key: f"gs://bucket/{key}.jsonl" for key in ("old", "extension", "gold")}
    old = [item("a"), item("b")]
    extra = [item("a"), item("c"), item("d")]
    if changed_input:
        extra[0]["content"] = "The dog sat."
        extra[0]["nondecodable_words"] = ["dog"]
    for name, rows in (
        ("old", old),
        ("extension", extra),
        ("gold", [item("a", []), item("b"), item("unrun")]),
    ):
        merge.write_jsonl(root / f"sources/datasets/{name}.jsonl", rows)
    for name, rows in (("old", old), ("extension", extra)):
        run = root / "sources/runs" / name
        merge.write_json(
            run / "config.json",
            {
                "resolved_config": {
                    "run_id": name,
                    "language": "en",
                    "models": ["m", "no-responses"],
                    "prompt_name": "decodability_evaluation_schwa_rule",
                    "items_sha256": merge.digest(
                        root / f"sources/datasets/{name}.jsonl"
                    ),
                    "metadata": {"items_source_uri": uris[name]},
                }
            },
        )
        calls = []
        for row in rows:
            key = row["benchmark_item_id"]
            if key == "d":
                continue
            call_id = f"call-{key}"
            if conflicting_calls and name == "extension" and key == "a":
                call_id += "-different-settings"
            calls.append(
                {
                    "model": "m",
                    "benchmark_item_id": key,
                    "model_call_id": call_id,
                    "result_id": name + key,
                    "run_id": "historical-origin",
                    "language": "en",
                    "prompt": merge.build_decodability_prompt(
                        "decodability_evaluation_schwa_rule", row, "en"
                    ),
                    "call_status": "error" if name == "old" else "success",
                    "timestamp_utc": "20260918" if name == "old" else "20260921",
                    "output": "not a word list" if key == "c" else "[]",
                }
            )
        merge.write_jsonl(run / "responses/results.jsonl", calls)
    manifest = tmp_path / "manifest.json"
    merge.write_json(
        manifest,
        {
            "groups": {"en": ["old"], "en_extension": ["extension"]},
            "group_languages": {"en_extension": "en"},
            "run_prefix": "gs://runs/",
            "gold_overlays": {"en": [uris["gold"]]},
            "source_gold_status": {uris["extension"]: "provisional"},
        },
    )
    merge.write_json(
        root / "download_manifest.json",
        {
            "started_at": "2026-09-21T00:00:00Z",
            "downloads": [],
            "files": [
                {"path": str(p.relative_to(root)), "sha256": merge.digest(p)}
                for p in sorted((root / "sources").rglob("*"))
                if p.is_file()
            ],
        },
    )
    return root, manifest


def selection_file(tmp_path, rule=None):
    path = tmp_path / "selection.json"
    merge.write_json(
        path, {"name": "chosen", "languages": {"en": rule or {"item_ids": ["a", "d"]}}}
    )
    return path


def test_full_pool_includes_unanswered_items_and_keeps_errors(tmp_path):
    root, manifest = snapshot(tmp_path)
    result = merge.merge_snapshot(root, manifest)
    full = root / "full"
    values = merge.read_jsonl(full / "english/benchmark_items.jsonl")
    assert {v["benchmark_item_id"] for v in values} == {"a", "b", "c", "d"}
    assert values[0]["nondecodable_words"] == []
    assert result["languages"]["en"]["gold_status_counts"] == {
        "corrected": 2,
        "provisional": 2,
    }
    assert result["languages"]["en"]["retained_rows"] == 3
    assert result["languages"]["en"]["scorable_rows"] == 1
    assert result["languages"]["en"]["duplicate_rows_removed"] == 1
    rows = merge.read_jsonl(full / "english/responses/merged.jsonl")
    assert rows[0]["run_id"] == "historical-origin"
    assert rows[0]["result_id"] == "extensiona"
    coverage = pd.read_csv(full / "english/coverage_by_model.csv").set_index("model")
    assert coverage.loc["m", "expected_items"] == 4
    assert coverage.loc["m", "missing_responses"] == 1
    assert coverage.loc["m", "call_failed"] == coverage.loc["m", "parse_failed"] == 1
    assert coverage.loc["no-responses", "missing_responses"] == 4
    assert merge.verify_artifacts(full)["kind"] == "full"
    with pytest.raises(FileExistsError):
        merge.merge_snapshot(root, manifest)


def test_subset_preserves_full_responses_and_uses_subset_denominator(tmp_path):
    root, manifest = snapshot(tmp_path)
    merge.merge_snapshot(root, manifest)
    before = (root / "full/artifact_manifest.json").read_bytes()
    spec = selection_file(tmp_path)
    audit = select.select_subset(root / "full", spec)
    target = root / "selections/chosen"
    assert audit["languages"]["en"]["items"] == 2
    coverage = pd.read_csv(target / "english/coverage_by_model.csv").set_index("model")
    assert coverage.loc["m", "scorable_fraction"] == 0.5
    assert coverage.loc["m", "missing_responses"] == 1
    assert (
        merge.read_jsonl(target / "english/responses/merged.jsonl")[0]
        == merge.read_jsonl(root / "full/english/responses/merged.jsonl")[0]
    )
    assert (root / "full/artifact_manifest.json").read_bytes() == before
    merge.verify_artifacts(root / "full")
    merge.verify_artifacts(target)
    with pytest.raises(FileExistsError):
        select.select_subset(root / "full", spec)


@pytest.mark.parametrize(
    "rule,match",
    [
        ({"item_ids": ["unknown"]}, "absent from full"),
        ({"item_ids": ["a", "a"]}, "Duplicate selected"),
        ({"item_ids": []}, "no items"),
    ],
)
def test_invalid_subset_is_not_published(tmp_path, rule, match):
    root, manifest = snapshot(tmp_path)
    merge.merge_snapshot(root, manifest)
    with pytest.raises(ValueError, match=match):
        select.select_subset(root / "full", selection_file(tmp_path, rule))
    assert not (root / "selections/chosen").exists()


def test_full_hash_tampering_blocks_selection(tmp_path):
    root, manifest = snapshot(tmp_path)
    merge.merge_snapshot(root, manifest)
    p = root / "full/english/responses/merged.jsonl"
    p.write_text(p.read_text() + "\n")
    with pytest.raises(ValueError, match="artifact changed"):
        select.select_subset(root / "full", selection_file(tmp_path))


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"changed_input": True}, "Changed content"),
        ({"conflicting_calls": True}, "Multiple distinct calls"),
    ],
)
def test_conflicts_refused_before_full_publication(tmp_path, kwargs, match):
    root, manifest = snapshot(tmp_path, **kwargs)
    with pytest.raises(ValueError, match=match):
        merge.merge_snapshot(root, manifest)
    assert not (root / "full").exists()


def test_unknown_group_is_not_assumed_kiswahili():
    with pytest.raises(ValueError, match="group_languages"):
        list(merge.manifest_runs({"groups": {"en_extension": ["run"]}}))


def test_mixture_quotas_and_order_independent_selection(tmp_path):
    expected = [100, 419, 204, 97, 49, 29, 22, 20, 20, 20, 20]
    assert select.mixture_quotas({}) == expected
    items = {}
    for band, quota in enumerate(expected):
        for index in range(quota + 2):
            key = f"{band}:{index}"
            items[key] = {
                "benchmark_item_id": key,
                "num_total_words": 10,
                "num_nondecodable_words": band,
                "nondecodable_ratio": band / 10,
            }
    rule = {"mixture": {"seed": 42}}
    chosen, audit = select.choose_items(items, rule, "en", tmp_path)
    reversed_items = dict(reversed(list(items.items())))
    assert select.choose_items(reversed_items, rule, "en", tmp_path)[0] == chosen
    assert len(chosen) == 1000
    assert audit["shortfall_items"] == 0
    assert [row["selected"] for row in audit["allocation"]] == expected


def test_shortfall_is_explicit_and_never_filled_from_other_bins(tmp_path):
    items = {f"item-{i}": item(f"item-{i}", []) for i in range(5)}
    with pytest.raises(ValueError, match="Insufficient items"):
        select.choose_items(items, {"mixture": {"size": 10}}, "en", tmp_path)
    chosen, audit = select.choose_items(
        items, {"mixture": {"size": 10, "shortfall": "keep_available"}}, "en", tmp_path
    )
    assert len(chosen) == 1
    assert audit["shortfall_items"] == 9


def test_id_file_does_not_override_gold(tmp_path):
    source = tmp_path / "ids.jsonl"
    merge.write_jsonl(source, [item("a", ["sat"])])
    chosen, _ = select.choose_items(
        {"a": item("a", [])}, {"items_file": source.name}, "en", tmp_path
    )
    assert chosen["a"]["nondecodable_words"] == []


def test_downloader_discovers_source_and_overlay_files_without_cloud_writes(
    tmp_path, monkeypatch
):
    manifest = tmp_path / "source.json"
    merge.write_json(
        manifest,
        {
            "groups": {"en": ["run"]},
            "run_prefix": "gs://bucket/runs/",
            "gold_overlays": {"en": ["gs://bucket/labels.jsonl"]},
        },
    )
    calls = []

    def fake_download(source, destination):
        calls.append((source, destination))
        if "/runs/" in source:
            merge.write_json(
                destination / "config.json",
                {
                    "resolved_config": {
                        "metadata": {"items_source_uri": "gs://bucket/items.jsonl"}
                    }
                },
            )
        else:
            merge.write_jsonl(destination, [item("a")])
        return {"source": source, "returncode": 0}

    monkeypatch.setattr(downloader, "download", fake_download)
    root = tmp_path / "download"
    downloader.download_snapshot(root, manifest)
    assert {source for source, _ in calls} == {
        "gs://bucket/runs/run",
        "gs://bucket/items.jsonl",
        "gs://bucket/labels.jsonl",
    }
    assert all(destination.is_relative_to(root) for _, destination in calls)
    assert json.loads((root / "source_manifest.json").read_text()) == json.loads(
        manifest.read_text()
    )


def test_current_manifest_applies_all_three_corrected_kiswahili_overlays(tmp_path):
    manifest = json.loads((SCRIPTS / "merged_sources_20260921.json").read_text())
    filenames = [
        "benchmark_items_syllabic_nasals_corrected.jsonl",
        "benchmark_items_additional_syllabic_nasals_corrected.jsonl",
        "benchmark_items_additional_linear_syllabic_nasals_corrected_20260921.jsonl",
    ]
    root = tmp_path / "snapshot"
    source_uri = "gs://bucket/source.jsonl"
    rows = [
        {
            "benchmark_item_id": key,
            "language_iso_code": "sw",
            "content": "Mama ana kuku.",
            "student_knowledge": {"graphemes": [], "whole_words": []},
            "nondecodable_words": [],
        }
        for key in ("original", "additional", "extension")
    ]
    source = root / "sources/datasets/source.jsonl"
    merge.write_jsonl(source, rows)
    for filename, row in zip(filenames, rows, strict=True):
        merge.write_jsonl(
            root / "sources/datasets" / filename,
            [{**row, "nondecodable_words": ["kuku"]}],
        )
    merge.write_json(
        root / "sources/runs/run/config.json",
        {
            "resolved_config": {
                "run_id": "run",
                "language": "sw",
                "models": ["m"],
                "prompt_name": "decodability_evaluation_clusters",
                "items_sha256": merge.digest(source),
                "metadata": {"items_source_uri": source_uri},
            }
        },
    )
    manifest["groups"] = {"sw_original": ["run"]}
    manifest["gold_overlays"].pop("en")
    manifest["source_gold_status"] = {source_uri: "corrected"}
    manifest_path = tmp_path / "manifest.json"
    merge.write_json(manifest_path, manifest)
    merge.write_json(
        root / "download_manifest.json",
        {
            "started_at": "2026-09-22T00:00:00Z",
            "downloads": [],
            "files": [
                {"path": str(path.relative_to(root)), "sha256": merge.digest(path)}
                for path in sorted((root / "sources").rglob("*"))
                if path.is_file()
            ],
        },
    )
    merge.merge_snapshot(root, manifest_path)
    items = merge.read_jsonl(root / "full/kiswahili/benchmark_items.jsonl")
    assert {row["benchmark_item_id"] for row in items} == {
        "original",
        "additional",
        "extension",
    }
    assert all(row["nondecodable_words"] == ["kuku"] for row in items)
    assert all(row["nondecodable_ratio"] == pytest.approx(1 / 3) for row in items)
    assert merge.read_jsonl(source) == rows


def test_merged_coverage_excludes_exactly_90_percent_successes(tmp_path):
    items = {str(i): item(str(i)) for i in range(10)}
    responses = [
        (
            {
                "model": model,
                "benchmark_item_id": key,
                "model_call_id": f"{model}-{key}",
                "call_status": "success",
                "output": "unparseable" if model == "exact90" and key == "9" else "[]",
            },
            {},
            [],
        )
        for model in ("exact90", "complete")
        for key in items
    ]
    stats = merge.write_analysis_run(
        tmp_path / "run",
        tmp_path / "run",
        "en",
        items,
        responses,
        {"exact90", "complete"},
        {key: {"benchmark_item_id": key, "gold_status": "corrected"} for key in items},
        {},
    )
    coverage = pd.read_csv(tmp_path / "run/coverage_by_model.csv").set_index("model")
    assert coverage.response_rows.eq(10).all()
    assert coverage.loc["exact90", "scorable_items"] == 9
    assert not coverage.loc["exact90", "passes_90_percent_gate"]
    assert coverage.loc["complete", "passes_90_percent_gate"]
    assert stats["models_over90"] == 1
