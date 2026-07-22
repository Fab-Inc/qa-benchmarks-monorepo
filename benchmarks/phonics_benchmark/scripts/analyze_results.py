"""Analyze a benchmark results run: per-model accuracy, confusion, and a
sample of items for manual inspection across item difficulty.

Must run ``benchmarks/phonics_benchmark/scripts/llm_output_trial.py``
first to produce benchmark results.

Data paths are hard-coded for now but must be parameterized later.

Run from the repository root with:

    uv run --package phonics-benchmark python \
        benchmarks/phonics_benchmark/scripts/analyze_results.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from phonics_benchmark.plots import (
    save_accuracy_by_nondecodable_count_plot,
    save_confusion_bar_plot,
)
from phonics_benchmark.results import (
    accuracy_by_nondecodable_count,
    add_word_level_confusion_matrix,
    load_results,
    sample_items_for_inspection,
    summarize_by_model,
)

BENCHMARK_DIR = Path(__file__).parents[1]
LANGUAGE = "sw"

RESULTS_PATH = BENCHMARK_DIR / "outputs" / "20260722T105304Z_results_trial.jsonl"
ITEMS_WITH_ANALYSIS_PATH = (
    BENCHMARK_DIR / "data" / "kiswahili" / "tusome_benchmark_items_with_analysis2.csv"
)
OUTPUT_DIR = BENCHMARK_DIR / "outputs" / "analysis"


def main() -> None:
    results = load_results(RESULTS_PATH)
    items = pd.read_csv(
        ITEMS_WITH_ANALYSIS_PATH,
        converters={"vocab_set": eval, "non_decodable_words": eval},
    )
    save_dir = OUTPUT_DIR / RESULTS_PATH.stem
    save_dir.mkdir(parents=True, exist_ok=True)

    metadata_columns = [
        "benchmark_item_id",
        "content",
        "nondecodable_ratio",
        "vocab_set",
        "non_decodable_words",
    ]
    results = results.merge(items[metadata_columns], on="benchmark_item_id", how="left")
    results = results.merge(
        items[["benchmark_item_id", "num_nondecodable_words"]],
        on="benchmark_item_id",
        how="left",
    )
    results = add_word_level_confusion_matrix(results)

    # Per-model accuracy + pooled word-level confusion (precision/recall/f1).
    summary = summarize_by_model(results)
    print("Per-model summary:")
    print(summary.to_string(index=False))
    summary.to_csv(save_dir / "summary_by_model.csv", index=False)

    # Does accuracy degrade as items get harder (more non-decodable words)?
    accuracy_by_count = accuracy_by_nondecodable_count(results)
    save_accuracy_by_nondecodable_count_plot(
        accuracy_by_count, save_dir / "accuracy_by_nondecodable_count.png"
    )
    save_confusion_bar_plot(summary, save_dir / "confusion_by_model.png")

    # 2 items from each of <0.05, 0.05-0.1, 0.1-0.2 nondecodable-ratio bins,
    # with every model's response, for manual spot-checking.
    sampled_items = sample_items_for_inspection(
        items, bin_edges=(0.0, 0.05, 0.1, 0.2), n_per_bin=4
    )
    inspection = sampled_items.merge(
        results[
            [
                "benchmark_item_id",
                "model_name",
                "prompt",
                "response_raw",
                "response_parsed",
                "tp_count",
                "fp_count",
                "fn_count",
                "tn",
            ]
        ],
        on="benchmark_item_id",
        how="left",
    )
    inspection_path = save_dir / "sampled_items_for_inspection.csv"
    inspection.to_csv(inspection_path, index=False)
    print(f"\nSampled {len(sampled_items)} items x every model -> {inspection_path}")


if __name__ == "__main__":
    main()
