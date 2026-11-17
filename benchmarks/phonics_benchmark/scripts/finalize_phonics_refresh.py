"""Audit a historical gold refresh and compare fixed-ID snapshots.

Reads local files only. Optional saved GCS metadata verifies the downloaded gold
file, not whether that generation is still the latest remote object.
"""

from __future__ import annotations

import argparse
import base64
import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from merge_local_benchmark_runs import validate_same_input
from phonics_benchmark.hardware import minimum_scorable_items
from phonics_benchmark.reporting import read_csv
from phonics_benchmark.reporting_inputs import inspect_dataset, verify_merge

METRIC_COLUMNS = [
    name
    for metric in ("pooled_f1", "pooled_f2")
    for name in (metric, metric + "_ci_low", metric + "_ci_high")
]


def jsonl(path, key):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate {key} in {path}")
    return result


def adjusted_points(frame):
    """Normalize fractional and point summaries without guessing by range."""
    result = frame.copy()
    expected = 2 * result.balanced_accuracy - 1
    if "adjusted_balanced_accuracy_scale" in result:
        scale = result.adjusted_balanced_accuracy_scale
        if not scale.isin([1, 100]).all():
            raise ValueError("Unknown adjusted balanced accuracy scale")
    elif np.allclose(result.adjusted_balanced_accuracy, expected, rtol=0, atol=1e-11):
        scale = 1
    elif np.allclose(
        result.adjusted_balanced_accuracy, 100 * expected, rtol=0, atol=1e-11
    ):
        scale = 100
    else:
        raise ValueError("Adjusted scores disagree with balanced accuracy")
    for suffix in ("", "_ci_low", "_ci_high"):
        column = "adjusted_balanced_accuracy" + suffix
        result[column] = result[column] * 100 / scale
        np.testing.assert_allclose(
            result[column],
            100 * (2 * result["balanced_accuracy" + suffix] - 1),
            rtol=0,
            atol=1e-11,
        )
    result["adjusted_balanced_accuracy_scale"] = 100
    return result


def inspect_gold_metadata(path, gold_path, now=None):
    """Validate saved gcloud storage ls -L metadata against a local gold file."""
    metadata = {
        line.strip().split(":", 1)[0]: line.strip().split(":", 1)[1].strip()
        for line in path.read_text().splitlines()
        if ": " in line
    }
    data = gold_path.read_bytes()
    encoded_md5 = base64.b64encode(hashlib.md5(data).digest()).decode()
    if metadata.get("Hash (MD5)") != encoded_md5:
        raise ValueError("Gold file does not match the saved GCS metadata")
    updated = datetime.datetime.fromisoformat(
        metadata["Update Time"].replace("Z", "+00:00")
    )
    now = now or datetime.datetime.now(datetime.UTC)
    local_updated = updated.astimezone(ZoneInfo("Europe/London"))
    today = now.astimezone(ZoneInfo("Europe/London")).date()
    return dict(
        generation=metadata["Generation"],
        updated_utc=updated.isoformat(),
        updated_london=local_updated.isoformat(),
        updated_today_or_yesterday=local_updated.date()
        in {today, today - datetime.timedelta(days=1)},
        source_sha256=hashlib.sha256(data).hexdigest(),
        metadata_path=str(path),
        remote_latest_checked=False,
    )


def compare_snapshot(root, previous, language, summary):
    before = jsonl(previous / language / "benchmark_items.jsonl", "benchmark_item_id")
    after = jsonl(root / language / "benchmark_items.jsonl", "benchmark_item_id")
    if before.keys() != after.keys():
        raise ValueError("Fixed-ID comparisons require identical target item IDs")
    changed = removed = added = 0
    for key, item in after.items():
        validate_same_input(before[key], item)
        old_gold, new_gold = (
            set(before[key]["nondecodable_words"]),
            set(item["nondecodable_words"]),
        )
        changed += old_gold != new_gold
        removed += len(old_gold - new_gold)
        added += len(new_gold - old_gold)
    prior = jsonl(previous / language / "responses/merged.jsonl", "model_call_id")
    current = jsonl(root / language / "responses/merged.jsonl", "model_call_id")
    additions = [current[key] for key in current.keys() - prior.keys()]
    prior_ba = adjusted_points(
        read_csv(previous / "balanced_accuracy" / f"{language}_summary.csv")
    )
    columns = ["model", "n_scorable", "adjusted_balanced_accuracy"]
    comparison = summary[columns].merge(
        prior_ba[columns],
        on="model",
        how="outer",
        suffixes=("_new", "_old"),
        validate="one_to_one",
    )
    comparison["language"] = language
    comparison["adjusted_ba_change_points"] = (
        comparison.adjusted_balanced_accuracy_new
        - comparison.adjusted_balanced_accuracy_old
    )
    comparison["scored_items_added"] = comparison.n_scorable_new.fillna(
        0
    ) - comparison.n_scorable_old.fillna(0)
    details = dict(
        gold_changed_items=changed,
        gold_words_removed=removed,
        gold_words_added=added,
        new_saved_responses=len(additions),
        new_successful_responses=sum(bool(row.get("success")) for row in additions),
        removed_saved_responses=len(prior.keys() - current.keys()),
        updated_saved_responses=sum(
            prior[key] != current[key] for key in prior.keys() & current.keys()
        ),
        scored_items_added=int(comparison.scored_items_added.sum()),
    )
    return comparison, details


def main(args):
    root = args.root.resolve()
    previous = args.previous_root.resolve() if args.previous_root else None
    merged_before = verify_merge(root)
    prior_before = verify_merge(previous) if previous else None
    destination = args.output_dir.resolve() if args.output_dir else root
    if previous and (
        destination == previous
        or destination in previous.parents
        or destination.is_relative_to(previous)
    ):
        raise ValueError("Output directory must not overwrite the previous snapshot")
    gold = args.gold.resolve() if args.gold else None
    if args.gold_metadata and not gold:
        raise ValueError("--gold-metadata requires --gold")
    record = dict(
        checked_at_utc=datetime.datetime.now(datetime.UTC).isoformat(),
        adjusted_ba_formula="100 * (TP/(TP+FN) - FP/(FP+TN)) = 100 * (2*BA - 1)",
        adjusted_ba_scale=100,
        adjusted_ba_bounds=[-100, 100],
        negative_values_retained=True,
        root=str(root),
        previous_root=str(previous) if previous else None,
        languages={},
        data_access="local files only",
    )
    if gold:
        source_items = jsonl(gold, "benchmark_item_id")
        selected_items = jsonl(
            root / "english/benchmark_items.jsonl", "benchmark_item_id"
        )
        if source_items.keys() != selected_items.keys():
            raise ValueError("English gold IDs do not match the fixed selection")
        for key, item in selected_items.items():
            validate_same_input(source_items[key], item)
            if {w.lower() for w in source_items[key]["nondecodable_words"]} != set(
                item["nondecodable_words"]
            ):
                raise ValueError(f"English gold differs for {key}")
        record["gold"] = dict(
            path=str(gold), source_sha256=hashlib.sha256(gold.read_bytes()).hexdigest()
        )
        if args.gold_metadata:
            record["gold"].update(inspect_gold_metadata(args.gold_metadata, gold))
    if args.download_root:
        manifest = json.loads(
            (args.download_root / "download_manifest.json").read_text()
        )
        downloads = manifest["downloads"]
        if any(row["returncode"] != 0 for row in downloads):
            raise ValueError("Source snapshot contains failed downloads")
        record["download"] = dict(
            root=str(args.download_root.resolve()),
            started_at=manifest["started_at"],
            finished_at=max((row["finished_at"] for row in downloads), default=None),
            run_folders_refreshed=sum("/runs/" in row["source"] for row in downloads),
            failed_downloads=0,
        )
    changes = []
    for language in ("english", "kiswahili"):
        size, gold_note = inspect_dataset(root / language)
        report_dir = args.report_root / language
        config = json.loads((report_dir / "report_config.json").read_text())
        if (
            str((root / language / "config.json").resolve())
            not in config["input_hashes"]
        ):
            raise ValueError("Report was not generated from the current snapshot")
        for path, expected in config["input_hashes"].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Report input has changed: {path}")
        summary = adjusted_points(read_csv(report_dir / "model_summary_all_scored.csv"))
        if not summary.expected_items.eq(size).all():
            raise ValueError("Report coverage denominator differs from snapshot")
        eligible = summary[summary.passes_coverage_filter]
        details = dict(items=size, gold_note=gold_note, eligible_models=len(eligible))
        if previous:
            comparison, delta = compare_snapshot(root, previous, language, summary)
            changes.append(comparison)
            details.update(delta)
        record["languages"][language] = details
    destination.mkdir(parents=True, exist_ok=True)
    if changes:
        pd.concat(changes, ignore_index=True).to_csv(
            destination / "model_metric_changes.csv", index=False
        )
    if verify_merge(root) != merged_before or (
        previous and verify_merge(previous) != prior_before
    ):
        raise ValueError("Merged inputs changed")
    record["merged_inputs_unchanged"] = True
    (destination / "refresh_validation.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, required=True, help="Current selected dataset/report root"
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        required=True,
        help="Consolidated reports with english/ and kiswahili/ subdirectories",
    )
    parser.add_argument(
        "--previous-root", type=Path, help="Previous reports on the same item IDs"
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Historical audit directory (default: --root)"
    )
    parser.add_argument(
        "--gold", type=Path, help="Authoritative English gold JSONL for this selection"
    )
    parser.add_argument(
        "--gold-metadata",
        type=Path,
        help="Saved gcloud storage ls -L output for --gold",
    )
    parser.add_argument(
        "--download-root", type=Path, help="Optional refreshed source snapshot to audit"
    )
    main(parser.parse_args())
