"""Analyze a benchmark results run: per-model accuracy, confusion, and a
sample of items for manual inspection across item difficulty.

Run from the repository root with:

    uv run --package phonics-benchmark python \
        benchmarks/phonics_benchmark/scripts/analyze_results.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from phonics_benchmark.data_io import convert_records_to_dataframe, read_benchmark_items
from phonics_benchmark.plots import (
    save_accuracy_by_nondecodable_count_plot,
)
from phonics_benchmark.results import (
    accuracy_by_nondecodable_count,
    add_per_item_confusion_matrix,
    load_results,
    sample_items_for_inspection,
    summarize_by_model,
)

BENCHMARK_DIR = Path(__file__).parents[1]
LANGUAGE = "kiswahili"

OUTPUT_DIR = BENCHMARK_DIR / "outputs" / "sw-tusome-260810"


def main() -> None:

    result_dfs = []
    for result_file in os.listdir(OUTPUT_DIR / "responses"):
        result_path = OUTPUT_DIR / "responses" / result_file
        model_results_df = load_results(result_path)
        result_dfs.append(model_results_df)

    result_df = pd.concat(result_dfs, ignore_index=True)

    items = read_benchmark_items(LANGUAGE, BENCHMARK_DIR / "data")
    items_df = convert_records_to_dataframe(items)
    metadata_columns = [
        "benchmark_item_id",
        "student_knowledge",
        "content",
        "nondecodable_words",
        "nondecodable_ratio",
        "num_nondecodable_words",
        "num_total_words",
    ]

    results_df = result_df.merge(
        items_df[metadata_columns], on="benchmark_item_id", how="left"
    )

    results_df = add_per_item_confusion_matrix(results_df, "sw")
    results_df.to_csv(OUTPUT_DIR / "results_with_scoring.csv", index=False)

    # Per-model accuracy + pooled word-level confusion (precision/recall/f1).
    summary = summarize_by_model(results_df)
    print("Per-model summary:")
    print(summary.to_string(index=False))
    summary.to_csv(OUTPUT_DIR / "summary_by_model.csv", index=False)

    # Does accuracy degrade as items get harder (more non-decodable words)?
    accuracy_by_count = accuracy_by_nondecodable_count(results_df)
    save_accuracy_by_nondecodable_count_plot(
        accuracy_by_count, OUTPUT_DIR / "accuracy_by_nondecodable_count.png"
    )

    # 2 items from each of <0.05, 0.05-0.1, 0.1-0.2 nondecodable-ratio bins,
    # with every model's response, for manual spot-checking.
    # sampled_items = sample_items_for_inspection(
    #     items_df, bin_edges=(0.0, 0.05, 0.1, 0.2), n_per_bin=4
    # )
    # sampled_items = items_df
    # inspection = sampled_items.merge(
    #     results[
    #         [
    #             "benchmark_item_id",
    #             "model_name",
    #             "prompt",
    #             "response_raw",
    #             "response_parsed",
    #             "tp_count",
    #             "fp_count",
    #             "fn_count",
    #             "tn",
    #         ]
    #     ],
    #     on="benchmark_item_id",
    #     how="left",
    # )
    # inspection_path = OUTPUT_DIR / "sampled_items_for_inspection.csv"
    # inspection.to_csv(inspection_path, index=False)
    # print(f"\nSampled {len(sampled_items)} items x every model -> {inspection_path}")


if __name__ == "__main__":
    main()
