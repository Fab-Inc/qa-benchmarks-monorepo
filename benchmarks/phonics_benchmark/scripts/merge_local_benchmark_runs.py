"""Build analysis-only runs from a downloaded snapshot, without network access.

Run from the worktree root. Original responses and configs are kept under sources/.
The generated configs point at local scoring labels, not the original run labels.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

import pandas as pd
from phonics_benchmark.labeling import get_vocabulary_set
from phonics_benchmark.parsing import parse_word_list
from phonics_benchmark.prompts import build_decodability_prompt

DEFAULT_ROOT = Path("outputs/phonics-merged-20260918")
DEFAULT_MANIFEST = Path(__file__).with_name("merged_sources_20260918.json")
EN_ITEMS = "en-ufli-schwa-stratified-20260915.jsonl"
SW_ORIGINAL = "sw-tusome-clusters-20260905.jsonl"
SW_ADDITIONAL = "sw-tusome-clusters-additional-20260916.jsonl"
SW_UPDATED = "benchmark_items_additional_syllabic_nasals_ungated.jsonl"
SW_BALANCED = "benchmark_items_balanced1000_syllabic_nasals_ungated.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def unique_items(rows: list[dict]) -> tuple[dict, list[dict]]:
    """Collapse exact duplicate items; refuse conflicting ID collisions."""
    items, duplicates = {}, []
    for line, row in enumerate(rows, 1):
        item_id = row["benchmark_item_id"]
        if item_id in items:
            if row != items[item_id]:
                raise ValueError(f"Conflicting item rows for {item_id}")
            duplicates.append({"benchmark_item_id": item_id, "duplicate_line": line})
        else:
            items[item_id] = row
    return items, duplicates


def validate_same_input(left: dict, right: dict) -> None:
    """Gold labels may change; the model's text and knowledge must not."""
    for field in ("benchmark_item_id", "content", "student_knowledge"):
        if left[field] != right[field]:
            raise ValueError(f"Changed {field} for {left['benchmark_item_id']}")


def replace_gold_labels(
    original: dict, updated: dict, language: str
) -> tuple[dict, list]:
    """Replace scoring labels only after validating item identity and vocabulary."""
    if original.keys() != updated.keys():
        raise ValueError("Updated gold must contain exactly the original item IDs")
    gold_fields = {"nondecodable_words", "num_nondecodable_words", "nondecodable_ratio"}
    frame = pd.DataFrame(list(original.values())).set_index(
        "benchmark_item_id", drop=False
    )
    vocabularies = get_vocabulary_set(frame, language)
    target, changes = {}, []
    for item_id, source in original.items():
        replacement = updated[item_id]
        validate_same_input(source, replacement)
        for field in (source.keys() | replacement.keys()) - gold_fields:
            if source.get(field) != replacement.get(field):
                raise ValueError(f"Changed non-gold field {field} for {item_id}")
        words = sorted({word.lower() for word in replacement["nondecodable_words"]})
        vocab = vocabularies.loc[item_id]
        if not set(words) <= vocab:
            raise ValueError(f"Gold words outside vocabulary for {item_id}")
        target[item_id] = {
            **source,
            "nondecodable_words": words,
            "num_nondecodable_words": len(words),
            "nondecodable_ratio": len(words) / len(vocab) if vocab else 0.0,
        }
        old_words = {word.lower() for word in source["nondecodable_words"]}
        if old_words != set(words):
            changes.append(
                {
                    "benchmark_item_id": item_id,
                    "old_nondecodable_words": sorted(old_words),
                    "new_nondecodable_words": words,
                    "removed_words": sorted(old_words - set(words)),
                    "added_words": sorted(set(words) - old_words),
                }
            )
    return target, changes


def deduplicate_calls(entries: list[tuple[dict, dict]]) -> tuple[list, list]:
    """Prefer success, then newest attempt; record every duplicate candidate.

    Parsing and scoring never affect selection. Original result/run IDs survive.
    The source path and line break exact timestamp ties deterministically.
    """
    by_call = defaultdict(list)
    for row, source in entries:
        if not row.get("model_call_id"):
            raise ValueError(f"Missing model_call_id at {source}")
        by_call[row["model_call_id"]].append((row, source))
    selected, audit = [], []
    for call_id, candidates in sorted(by_call.items()):
        identities = {
            (r["language"], r["model"], r["benchmark_item_id"], r["prompt"])
            for r, _ in candidates
        }
        if len(identities) != 1:
            raise ValueError(f"Conflicting call identity: {call_id}")
        ranked = sorted(
            candidates,
            key=lambda pair: (
                pair[0]["call_status"] == "success",
                pair[0].get("timestamp_utc", ""),
                pair[1]["file"],
                pair[1]["line"],
            ),
        )
        winner, winner_source = ranked[-1]
        selected.append((winner, winner_source, [s for _, s in ranked]))
        if len(ranked) > 1:
            success_outputs = {
                r.get("output") for r, _ in ranked if r["call_status"] == "success"
            }
            audit.append(
                {
                    "model_call_id": call_id,
                    "model": winner["model"],
                    "benchmark_item_id": winner["benchmark_item_id"],
                    "conflicting_success_outputs": len(success_outputs) > 1,
                    "selected_source": winner_source,
                    "candidates": [
                        {
                            **source,
                            "call_status": row["call_status"],
                            "timestamp_utc": row.get("timestamp_utc"),
                            "output_sha256": sha256(
                                (row.get("output") or "").encode()
                            ).hexdigest(),
                        }
                        for row, source in ranked
                    ],
                }
            )
    return selected, audit


def source_dataset_uri(snapshot: dict) -> str:
    config = snapshot["resolved_config"]
    uri = config.get("metadata", {}).get("items_source_uri") or snapshot.get(
        "raw_config", {}
    ).get("items_path")
    if not uri or not uri.startswith("gs://"):
        raise ValueError(f"Missing GCS dataset URI for {config['run_id']}")
    return uri


def merge_snapshot(root: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    downloads = json.loads((root / "download_manifest.json").read_text())
    for entry in downloads["files"]:
        path = root / entry["path"]
        if sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Downloaded snapshot changed: {path}")
    failed_downloads = {
        d["source"]: d for d in downloads["downloads"] if d["returncode"]
    }
    for failure in failed_downloads.values():
        if "matched no objects or files" not in failure.get("error", ""):
            raise ValueError(f"Unresolved download failure: {failure}")

    datasets = {
        path.name: unique_items(read_jsonl(path))[0]
        for path in (root / "sources/datasets").glob("*.jsonl")
    }
    original, additional = datasets[SW_ORIGINAL], datasets[SW_ADDITIONAL]
    overlaps = sorted(original.keys() & additional.keys())
    for item_id in overlaps:
        validate_same_input(original[item_id], additional[item_id])
        if (
            original[item_id]["nondecodable_words"]
            != additional[item_id]["nondecodable_words"]
        ):
            raise ValueError(f"Conflicting original gold for overlap {item_id}")
    source_sw = original | additional
    balanced, duplicate_items = unique_items(
        read_jsonl(root / "sources/datasets" / SW_BALANCED)
    )
    missing_ids = sorted(balanced.keys() - source_sw.keys())
    # Explicit user decision: one exact duplicate and 84 absent items are excluded.
    target_sw = {k: v for k, v in balanced.items() if k in source_sw}
    for item_id, item in target_sw.items():
        validate_same_input(item, source_sw[item_id])
        if item_id in datasets[SW_UPDATED]:
            updated = datasets[SW_UPDATED][item_id]
            validate_same_input(item, updated)
            if item["nondecodable_words"] != updated["nondecodable_words"]:
                raise ValueError(f"Updated gold disagrees for {item_id}")

    audit_dir = root / "audit"
    write_jsonl(
        audit_dir / "excluded_missing_sw_items.jsonl",
        [balanced[k] for k in missing_ids],
    )
    write_csv(audit_dir / "duplicate_sw_items.csv", duplicate_items)
    label_changes = [
        {
            "benchmark_item_id": k,
            "old_nondecodable_words": source_sw[k]["nondecodable_words"],
            "new_nondecodable_words": v["nondecodable_words"],
        }
        for k, v in target_sw.items()
        if set(source_sw[k]["nondecodable_words"]) != set(v["nondecodable_words"])
    ]
    write_jsonl(audit_dir / "sw_gold_label_changes.jsonl", label_changes)
    target_en, english_changes = datasets[EN_ITEMS], []
    english_gold_uri = manifest.get("english_gold")
    if english_gold_uri:
        target_en, english_changes = replace_gold_labels(
            target_en, datasets[english_gold_uri.rsplit("/", 1)[-1]], "en"
        )
    write_jsonl(audit_dir / "en_gold_label_changes.jsonl", english_changes)
    write_json(root / "source_manifest.json", manifest)

    entries = {"en": [], "sw": []}
    expected_models = {"en": set(), "sw": set()}
    source_audit = []
    for group, runs in manifest["groups"].items():
        language = "en" if group == "en" else "sw"
        for run in runs:
            run_dir = root / "sources/runs" / run
            uri = manifest["run_prefix"] + run
            if not (run_dir / "config.json").exists():
                if uri not in failed_downloads:
                    raise ValueError(f"Missing config without download record: {run}")
                source_audit.append(
                    {
                        "destination": uri,
                        "group": group,
                        "status": "no_objects",
                        "dataset_uri": "",
                        "response_rows": 0,
                    }
                )
                continue
            snapshot = json.loads((run_dir / "config.json").read_text())
            config = snapshot["resolved_config"]
            if config["language"] != language:
                raise ValueError(f"Wrong language for {run}")
            dataset_uri = source_dataset_uri(snapshot)
            dataset_path = root / "sources/datasets" / dataset_uri.rsplit("/", 1)[-1]
            if sha256(dataset_path.read_bytes()).hexdigest() != config["items_sha256"]:
                raise ValueError(f"Dataset does not match destination config: {run}")
            items = datasets[dataset_path.name]
            expected_models[language].update(config["models"])
            expected_prompt = (
                "decodability_evaluation_schwa_rule"
                if language == "en"
                else "decodability_evaluation_clusters"
            )
            if config["prompt_name"] != expected_prompt:
                raise ValueError(f"Unexpected prompt for {run}")
            prompts = {
                k: build_decodability_prompt(expected_prompt, v, language)
                for k, v in items.items()
            }
            count = 0
            for path in sorted((run_dir / "responses").glob("*.jsonl")):
                for line, row in enumerate(read_jsonl(path), 1):
                    item_id = row["benchmark_item_id"]
                    if row["language"] != language or row["prompt"] != prompts[item_id]:
                        raise ValueError(
                            f"Response inputs differ from config: {path}:{line}"
                        )
                    if row["call_status"] not in {"success", "error"}:
                        raise ValueError(f"Unknown call status: {path}:{line}")
                    entries[language].append(
                        (
                            row,
                            {
                                "destination": uri,
                                "dataset_uri": dataset_uri,
                                "file": str(path.relative_to(root)),
                                "line": line,
                            },
                        )
                    )
                    expected_models[language].add(row["model"])
                    count += 1
            source_audit.append(
                {
                    "destination": uri,
                    "group": group,
                    "status": "downloaded",
                    "dataset_uri": dataset_uri,
                    "response_rows": count,
                }
            )

    # Planned models come from each destination's own config, even at zero coverage.
    write_csv(audit_dir / "sources.csv", source_audit)

    totals = {}
    for language, items, run_id in (
        ("en", target_en, "english"),
        ("sw", target_sw, "kiswahili"),
    ):
        selected, duplicate_calls = deduplicate_calls(entries[language])
        retained = [
            (r, s, origins)
            for r, s, origins in selected
            if r["benchmark_item_id"] in items
        ]
        retained.sort(
            key=lambda value: (value[0]["model"], value[0]["benchmark_item_id"])
        )
        pairs = [(r["model"], r["benchmark_item_id"]) for r, _, _ in retained]
        if len(pairs) != len(set(pairs)):
            raise ValueError(f"Multiple distinct calls per model/item in {language}")
        run_dir = root / run_id
        items_path = run_dir / "benchmark_items.jsonl"
        write_jsonl(items_path, items.values())
        write_jsonl(run_dir / "responses/merged.jsonl", (r for r, _, _ in retained))
        write_jsonl(
            run_dir / "provenance.jsonl",
            (
                {
                    "model_call_id": r["model_call_id"],
                    "selected_source": s,
                    "all_sources": origins,
                }
                for r, s, origins in retained
            ),
        )
        write_jsonl(audit_dir / f"{language}_duplicate_calls.jsonl", duplicate_calls)
        write_json(
            run_dir / "config.json",
            {
                "analysis_only": True,
                "resolved_config": {
                    "run_id": run_id,
                    "language": language,
                    "items_path": str(items_path),
                    "items_sha256": sha256(items_path.read_bytes()).hexdigest(),
                    "models": sorted(expected_models[language]),
                    "metadata": {
                        "source_manifest": str(
                            manifest_path.resolve().relative_to(Path.cwd())
                        ),
                        "gold_relabelled": language == "sw" or bool(english_gold_uri),
                        "gold_source_uri": english_gold_uri
                        if language == "en"
                        else next(
                            uri
                            for uri in manifest["updated_gold"]
                            if uri.endswith(SW_BALANCED)
                        ),
                        "snapshot_started_at": downloads["started_at"],
                    },
                },
            },
        )
        counts = defaultdict(Counter)
        excluded = []
        with contextlib.redirect_stdout(io.StringIO()):
            for row, source, _ in retained:
                count = counts[row["model"]]
                count["response_rows"] += 1
                reason = ""
                if row["call_status"] != "success":
                    reason = "call_failed"
                elif not parse_word_list(row["output"]).parse_ok:
                    reason = "parse_failed"
                count[reason or "scorable_items"] += 1
                if reason:
                    excluded.append(
                        {
                            "model": row["model"],
                            "benchmark_item_id": row["benchmark_item_id"],
                            "model_call_id": row["model_call_id"],
                            "exclusion_reason": reason,
                        }
                    )
        coverage = []
        for model in sorted(expected_models[language]):
            c = counts[model]
            coverage.append(
                {
                    "model": model,
                    "expected_items": len(items),
                    "response_rows": c["response_rows"],
                    "scorable_items": c["scorable_items"],
                    "missing_responses": len(items) - c["response_rows"],
                    "call_failed": c["call_failed"],
                    "parse_failed": c["parse_failed"],
                    "scorable_fraction": c["scorable_items"] / len(items),
                    "complete_scorable": c["scorable_items"] == len(items),
                }
            )
        write_csv(run_dir / "coverage_by_model.csv", coverage)
        write_jsonl(run_dir / "excluded_responses.jsonl", excluded)
        totals[language] = {
            "items": len(items),
            "raw_rows": len(entries[language]),
            "distinct_calls_all_items": len(selected),
            "duplicate_rows_removed": len(entries[language]) - len(selected),
            "responses_outside_target": len(selected) - len(retained),
            "retained_rows": len(retained),
            "scorable_rows": sum(c["scorable_items"] for c in coverage),
            "expected_models": len(coverage),
            "models_with_responses": sum(c["response_rows"] > 0 for c in coverage),
            "models_with_scores": sum(c["scorable_items"] > 0 for c in coverage),
            "models_complete_scorable": sum(c["complete_scorable"] for c in coverage),
            "conflicting_success_calls_all_items": sum(
                d["conflicting_success_outputs"] for d in duplicate_calls
            ),
        }
    audit = {
        "snapshot_started_at": downloads["started_at"],
        "source_runs": len(source_audit),
        "unavailable_runs": [
            s["destination"] for s in source_audit if s["status"] == "no_objects"
        ],
        "overlapping_sw_ids": overlaps,
        "duplicate_sw_target_rows_removed": len(duplicate_items),
        "sw_target_items_missing_from_sources": len(missing_ids),
        "sw_gold_labels_changed": len(label_changes),
        "en_gold_labels_changed": len(english_changes),
        "english_gold_source_uri": english_gold_uri,
        "deduplication_policy": (
            "model_call_id; prefer success, then newest timestamp_utc; "
            "source file and line break ties. "
            "Selection never uses parse or score quality."
        ),
        "languages": totals,
    }
    write_json(audit_dir / "merge_summary.json", audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    print(json.dumps(merge_snapshot(args.root, args.manifest), indent=2))


if __name__ == "__main__":
    main()
