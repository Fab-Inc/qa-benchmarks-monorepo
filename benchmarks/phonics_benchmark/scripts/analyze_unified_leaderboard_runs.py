"""Score unified raw responses and verify adjusted BA against a saved report.

This analysis starts from the bundle's raw JSONL responses and corrected items.
It parses and scores the responses anew, then uses the production 1,000-draw,
seed-0 item bootstrap. A comparison mismatch prevents analysis publication.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
from pathlib import Path

import analyze_balanced_accuracy as ba
import numpy as np
import pandas as pd
from merge_local_benchmark_runs import digest, read_jsonl, verify_artifacts, write_json
from phonics_benchmark.parsing import parse_word_list
from phonics_benchmark.scoring import add_per_item_confusion_matrix

DEFAULT_ROOT = Path("outputs/phonics-unified-leaderboard-20260922")
DEFAULT_REFERENCE = Path(
    "outputs/phonics-reports-20260922-coverage90/selections/fixed-991-en-1000-sw"
)
LANGUAGES = {"english": "en", "kiswahili": "sw"}
COMPARE_COLUMNS = (
    "n_scorable",
    "expected_items",
    *ba.COUNTS,
    *(
        column
        for metric in ba.METRICS
        for column in (metric, f"{metric}_ci_low", f"{metric}_ci_high")
    ),
)


def score_one(run: Path, language: str) -> tuple[pd.DataFrame, dict]:
    config = json.loads((run / "config.json").read_text())
    if not config.get("analysis_only"):
        raise ValueError(f"Expected an analysis-only run: {run}")
    resolved = config["resolved_config"]
    if resolved["language"] != language:
        raise ValueError(f"Wrong run language: {run}")
    items_path = Path(resolved["items_path"])
    if not items_path.is_absolute():
        items_path = run / items_path
    if digest(items_path) != resolved["items_sha256"]:
        raise ValueError(f"Configured items changed: {run}")
    items = ba.load_items(items_path, language)
    item_metadata = pd.read_json(items_path, lines=True)[
        ["benchmark_item_id", "content", "student_knowledge", "nondecodable_words"]
    ]
    scorable, raw_count, failed, unparseable = [], 0, 0, 0
    with contextlib.redirect_stdout(io.StringIO()):
        for row in read_jsonl(run / "responses/merged.jsonl"):
            raw_count += 1
            if row["call_status"] != "success":
                failed += 1
                continue
            parsed = parse_word_list(row["output"])
            if not parsed.parse_ok:
                unparseable += 1
                continue
            scorable.append(
                {
                    "model": row["model"],
                    "benchmark_item_id": row["benchmark_item_id"],
                    "response_parsed": parsed.words,
                }
            )
    if not scorable:
        raise ValueError(f"No scorable responses: {run}")
    frame = pd.DataFrame(scorable).merge(
        item_metadata, on="benchmark_item_id", validate="many_to_one"
    )
    if len(frame) != len(scorable):
        raise ValueError(f"Unknown item IDs in responses: {run}")
    scored = add_per_item_confusion_matrix(frame, language)
    summary, _ = ba.summarize(
        scored[
            [
                "model",
                "benchmark_item_id",
                "tp_count",
                "fp_count",
                "fn_count",
                "tn_count",
            ]
        ],
        items,
        n_bootstrap=1000,
        seed=0,
    )
    coverage = pd.read_csv(run / "coverage_by_model.csv").set_index("model")
    actual = summary.set_index("model").n_scorable.reindex(coverage.index, fill_value=0)
    if (
        not actual.equals(coverage.scorable_items)
        or int(coverage.response_rows.sum()) != raw_count
        or int(coverage.call_failed.sum()) != failed
        or int(coverage.parse_failed.sum()) != unparseable
    ):
        raise ValueError(f"Fresh scoring differs from bundle coverage: {run}")
    details = {
        "items": len(items),
        "raw_responses": raw_count,
        "call_failed": failed,
        "parse_failed": unparseable,
        "scored_responses": len(scorable),
        "scored_models": len(summary),
    }
    return summary, details


def compare(summary: pd.DataFrame, reference_path: Path) -> dict:
    old = pd.read_csv(reference_path, float_precision="round_trip").set_index("model")
    new = summary.set_index("model")
    if set(old.index) != set(new.index):
        raise ValueError(f"Model set differs from {reference_path}")
    old, new = old.sort_index(), new.sort_index()
    max_difference = 0.0
    for column in COMPARE_COLUMNS:
        values = (new[column] - old[column]).abs()
        difference = float(values.max())
        max_difference = max(max_difference, difference)
        np.testing.assert_allclose(
            new[column],
            old[column],
            rtol=0,
            atol=1e-10,
            err_msg=f"Adjusted BA analysis differs in {column}",
        )
    return {
        "reference_sha256": digest(reference_path),
        "matched_models": len(new),
        "max_absolute_difference": max_difference,
        "matching_tolerance": 1e-10,
    }


def compare_combined(summary: pd.DataFrame, language: str, path: Path) -> dict:
    combined = pd.read_csv(path, float_precision="round_trip")
    old = combined.loc[combined.language.eq(language)].set_index("model")
    size = int(summary.expected_items.iloc[0])
    required = max(700, 9 * size // 10 + 1)
    new = summary.loc[summary.n_scorable.ge(required)].set_index("model")
    if set(old.index) != set(new.index):
        raise ValueError(f"Eligible model set differs from {path}: {language}")
    old, new = old.sort_index(), new.sort_index()
    columns = (
        "n_scorable",
        "adjusted_balanced_accuracy",
        "adjusted_balanced_accuracy_ci_low",
        "adjusted_balanced_accuracy_ci_high",
    )
    differences = [(new[column] - old[column]).abs().max() for column in columns]
    for column in columns:
        np.testing.assert_allclose(new[column], old[column], rtol=0, atol=1e-10)
    return {
        "reference_sha256": digest(path),
        "eligible_models": len(new),
        "max_absolute_difference": float(max(differences)),
    }


def run(root: Path = DEFAULT_ROOT, reference: Path | None = DEFAULT_REFERENCE) -> dict:
    root = root.resolve()
    reference = reference.resolve() if reference else None
    original_manifest = verify_artifacts(root)
    if original_manifest.get("kind") != "unified_leaderboard_runs":
        raise ValueError("Expected unified leaderboard bundles")
    output = root / "analysis"
    if output.exists():
        raise FileExistsError(f"Preserving existing analysis: {output}")
    summaries, details = {}, {}
    for name, language in LANGUAGES.items():
        summary, stats = score_one(root / name, language)
        # The item minimum is 700; the coverage threshold is strictly >90%.
        required = max(700, 9 * stats["items"] // 10 + 1)
        stats["eligible_models"] = int(summary.n_scorable.ge(required).sum())
        if reference:
            stats["comparison"] = compare(
                summary, reference / "balanced_accuracy" / f"{name}_summary.csv"
            )
            stats["combined_summary_comparison"] = compare_combined(
                summary, name, reference / "model_summary_coverage_gt90.csv"
            )
        summaries[name] = summary
        details[name] = stats
    if verify_artifacts(root) != original_manifest:
        raise ValueError("Unified inputs changed during scoring")
    with tempfile.TemporaryDirectory(prefix=".unified-analysis-", dir=root) as tmp:
        staging = Path(tmp) / "analysis"
        staging.mkdir()
        for name, summary in summaries.items():
            summary.to_csv(
                staging / f"{name}_adjusted_balanced_accuracy.csv", index=False
            )
            required = max(700, 9 * details[name]["items"] // 10 + 1)
            summary.loc[summary.n_scorable.ge(required)].to_csv(
                staging / f"{name}_leaderboard_coverage_gt90.csv", index=False
            )
        record = {
            "source_bundle_manifest_sha256": digest(root / "artifact_manifest.json"),
            "reference": str(reference) if reference else None,
            "bootstrap": {"draws": 1000, "seed": 0, "unit": "item"},
            "languages": details,
        }
        write_json(staging / "comparison.json", record)
        staging.rename(output)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.reference), indent=2))
