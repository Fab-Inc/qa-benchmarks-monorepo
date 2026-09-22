"""Validate local F1/F2/BA reports and export combined summaries and comparisons.

Reads local files only. Optional saved GCS metadata verifies the downloaded gold
file, not whether that generation is still the latest remote object.
"""

from __future__ import annotations

import argparse
import base64
import datetime
import hashlib
import json
import os
import shutil
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from analyze_phonics_hardware_tiers import minimum_scorable_items
from merge_local_benchmark_runs import validate_same_input
from plot_report_f2_leaderboards import read_csv
from run_f2_analysis import inspect_dataset, verify_merge

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
    frames, changes = [], []
    for language in ("english", "kiswahili"):
        size, gold_note = inspect_dataset(root / language)
        summary = read_csv(root / "balanced_accuracy" / f"{language}_summary.csv")
        if not summary.adjusted_balanced_accuracy_scale.eq(100).all():
            raise ValueError("Current summaries must store adjusted BA in points")
        normalized = adjusted_points(summary)
        count_frame = read_csv(
            root / "balanced_accuracy" / f"{language}_item_counts.csv"
        )
        if count_frame.duplicated(["model", "benchmark_item_id"]).any():
            raise ValueError("Duplicate scored model/item pairs")
        totals = count_frame.groupby("model")[
            ["tp_count", "fn_count", "tn_count", "fp_in_vocabulary"]
        ].sum()
        values = normalized.set_index("model").loc[totals.index]
        expected = 100 * (
            totals.tp_count / (totals.tp_count + totals.fn_count)
            - totals.fp_in_vocabulary / (totals.fp_in_vocabulary + totals.tn_count)
        )
        np.testing.assert_allclose(
            values.adjusted_balanced_accuracy, expected, rtol=0, atol=1e-11
        )
        eligible = read_csv(
            root
            / "adjusted_balanced_accuracy"
            / f"{language}_adjusted_balanced_accuracy_leaderboard_coverage_gt80.csv"
        )
        f1 = read_csv(
            root / language / "report/pooled_f1_leaderboard_coverage_gt80.csv"
        )
        f2 = read_csv(
            root / language / "report/pooled_f2_leaderboard_coverage_gt80.csv"
        )
        if set(eligible.model) != set(f1.model) or set(eligible.model) != set(f2.model):
            raise ValueError("Eligible model sets differ between metrics")
        np.testing.assert_allclose(
            eligible.score, eligible.adjusted_balanced_accuracy, rtol=0, atol=0
        )
        bootstrap = read_csv(root / language / "report/bootstrap_ci.csv")
        combined = summary.merge(
            bootstrap[["model", *METRIC_COLUMNS]], on="model", validate="one_to_one"
        )
        metadata = read_csv(root / "plot_metadata/models.csv")[
            ["model_id", "display_name"]
        ].rename(columns={"model_id": "model"})
        combined = combined.merge(
            metadata, on="model", how="left", validate="one_to_one"
        )
        combined["language"] = language
        required = minimum_scorable_items(size, 700, 0.8)
        combined["passes_coverage_filter"] = combined.n_scorable.ge(required)
        if set(combined.loc[combined.passes_coverage_filter, "model"]) != set(
            eligible.model
        ):
            raise ValueError("Coverage gate disagrees with plotted models")
        columns = [
            "language",
            "model",
            "display_name",
            "n_scorable",
            "expected_items",
            "coverage",
            "passes_coverage_filter",
            *METRIC_COLUMNS,
            "balanced_accuracy",
            "balanced_accuracy_ci_low",
            "balanced_accuracy_ci_high",
            "adjusted_balanced_accuracy",
            "adjusted_balanced_accuracy_ci_low",
            "adjusted_balanced_accuracy_ci_high",
            "adjusted_balanced_accuracy_scale",
            "recall",
            "specificity",
            "false_positive_rate",
            "tp_count",
            "fn_count",
            "tn_count",
            "fp_in_vocabulary",
            "extra_predictions",
            "n_bootstrap",
            "ci_level",
        ]
        frames.append(combined[columns])
        details = dict(
            items=size,
            gold_note=gold_note,
            minimum_scorable_items=required,
            eligible_models=len(eligible),
            models_at_or_below_zero=int(eligible.score.le(0).sum()),
            ci_contains_zero=int(
                (eligible.ci_low.le(0) & eligible.ci_high.ge(0)).sum()
            ),
        )
        if previous:
            comparison, delta = compare_snapshot(root, previous, language, summary)
            changes.append(comparison)
            details.update(delta)
        record["languages"][language] = details
    destination.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(root / "model_summary_all_scored.csv", index=False)
    combined[combined.passes_coverage_filter].to_csv(
        root / "model_summary_coverage_gt80.csv", index=False
    )
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
    archive = destination / "reproduction"
    archive.mkdir(exist_ok=True)
    scripts = Path(__file__).resolve().parent
    for name in (
        "apply_refreshed_gold.py",
        "run_balanced_accuracy_reports.py",
        "plot_adjusted_balanced_accuracy.py",
        "finalize_phonics_refresh.py",
        "analyze_balanced_accuracy.py",
        "download_benchmark_snapshot.py",
        "merge_local_benchmark_runs.py",
        "select_local_benchmark_subset.py",
        "run_f2_analysis.py",
        "plot_report_f1_leaderboards.py",
        "plot_report_f2_leaderboards.py",
        "analyze_phonics_hardware_tiers.py",
    ):
        source, target = scripts / name, archive / name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
    # Link relative to either the selection or the snapshot-level index.
    relative = Path(os.path.relpath(root, destination)).as_posix()
    lines = [
        "# Phonics reports",
        "",
        "Combined summaries and plots use the same strict >80% coverage gate "
        "and a 700-item minimum.",
        "",
    ]
    for language, details in record["languages"].items():
        lines.append(
            f"- {language.title()}: {details['items']:,} target items; "
            f"{details['eligible_models']} eligible models; at least "
            f"{details['minimum_scorable_items']} scored items."
        )
        for metric in ("pooled_f1", "pooled_f2"):
            lines.append(
                f"  [{metric} "
                f"leaderboard]({relative}/{language}/report/{metric}_leaderboard_coverage_gt80.png)"
            )
        lines.append(
            f"  [Adjusted BA "
            f"leaderboard]({relative}/adjusted_balanced_accuracy/{language}_adjusted_balanced_accuracy_leaderboard_coverage_gt80.png)"
        )
    lines.extend(
        [
            "",
            f"[Eligible model "
            f"summary]({relative}/model_summary_coverage_gt80.csv) · [All "
            f"scored models]({relative}/model_summary_all_scored.csv)",
            "",
        ]
    )
    for metric in (
        "pooled_f1",
        "pooled_f2",
        "balanced_accuracy",
        "adjusted_balanced_accuracy",
    ):
        lines.append(
            f"- [{metric} hardware "
            f"comparison]({relative}/hardware_tiers/{metric}/coverage_gt80/hardware_tiers_top10_mean_slide.png)"
        )
    lines.extend(
        [
            "",
            "Adjusted BA = 100 × (2 × BA − 1): chance/constant = 0, perfect = "
            "100, inverted = −100. Negative scores and CI tails are retained. "
            "All adjusted scores and CIs in CSVs are points; F1/F2/BA remain "
            "fractions. Hardware means use up to ten highest-scoring eligible "
            "models per group, excluding Workstation, without group CIs.",
            "",
            f"[BA methods and bootstrap "
            f"settings]({relative}/balanced_accuracy/README.md) · "
            f"[Validation](refresh_validation.json)",
        ]
    )
    if changes:
        lines.extend(
            [
                "",
                "[Changes against the previous fixed-ID "
                "snapshot](model_metric_changes.csv). Changes combine any "
                "refreshed responses and gold edits; they do not isolate the "
                "gold effect.",
            ]
        )
    if "gold" in record and "updated_london" in record["gold"]:
        lines.extend(
            [
                "",
                f"Saved GCS metadata: generation "
                f"`{record['gold']['generation']}`, updated "
                f"`{record['gold']['updated_london']}`. Updated "
                f"today/yesterday: "
                f"{record['gold']['updated_today_or_yesterday']}. The local "
                f"file matches the metadata MD5. This local report does not "
                f"check whether GCS has a newer generation.",
            ]
        )
    (destination / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, required=True, help="Current selected dataset/report root"
    )
    parser.add_argument(
        "--previous-root", type=Path, help="Previous reports on the same item IDs"
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Report index/audit directory (default: --root)"
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
