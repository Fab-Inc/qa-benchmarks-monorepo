"""Analyze a benchmark results run: per-model accuracy, confusion, and a
sample of items for manual inspection across item difficulty.

Run from the repository root with:

    uv run --package phonics-benchmark python \
        benchmarks/phonics_benchmark/scripts/analyze_results.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from phonics_benchmark.analysis import (
    save_accuracy_by_nondecodable_count_plot,
    save_confusion_bar_plot,
)
from phonics_benchmark.results import (
    accuracy_by_nondecodable_count,
    add_word_level_counts,
    attach_item_metadata,
    load_results,
    sample_items_for_inspection,
    summarize_by_model,
)

BENCHMARK_DIR = Path(__file__).parents[1]
LANGUAGE = "sw"

RESULTS_PATH = BENCHMARK_DIR / "outputs" / "20260716T100738Z_results_trial.jsonl"
ITEMS_WITH_ANALYSIS_PATH = (
    BENCHMARK_DIR / "data" / "kiswahili" / "tusome_benchmark_items_with_analysis.csv"
)
OUTPUT_DIR = BENCHMARK_DIR / "outputs" / "analysis"


def main() -> None:
    results = load_results(RESULTS_PATH)
    items = pd.read_csv(ITEMS_WITH_ANALYSIS_PATH)
    results = attach_item_metadata(results, items, LANGUAGE)
    results = results.merge(
        items[["benchmark_item_id", "num_nondecodable_words"]],
        on="benchmark_item_id",
        how="left",
    )
    results = add_word_level_counts(results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Per-model accuracy + pooled word-level confusion (precision/recall/f1).
    summary = summarize_by_model(results)
    print("Per-model summary:")
    print(summary.to_string(index=False))
    summary.to_csv(OUTPUT_DIR / "summary_by_model.csv", index=False)

    # Does accuracy degrade as items get harder (more non-decodable words)?
    accuracy_by_count = accuracy_by_nondecodable_count(results)
    save_accuracy_by_nondecodable_count_plot(
        accuracy_by_count, OUTPUT_DIR / "accuracy_by_nondecodable_count.png"
    )
    save_confusion_bar_plot(summary, OUTPUT_DIR / "confusion_by_model.png")

    # 2 items from each of <0.05, 0.05-0.1, 0.1-0.2 nondecodable-ratio bins,
    # with every model's response, for manual spot-checking.
    sampled_items = sample_items_for_inspection(
        items, bin_edges=(0.0, 0.05, 0.1, 0.2), n_per_bin=2
    )
    inspection = sampled_items.merge(
        results[
            [
                "benchmark_item_id",
                "model_name",
                "response_parsed",
                "correct_answer",
                "perfect_match",
                "tp",
                "fp",
                "fn",
            ]
        ],
        on="benchmark_item_id",
        how="left",
    )
    inspection_path = OUTPUT_DIR / "sampled_items_for_inspection.csv"
    inspection.to_csv(inspection_path, index=False)
    print(f"\nSampled {len(sampled_items)} items x every model -> {inspection_path}")


if __name__ == "__main__":
    main()
