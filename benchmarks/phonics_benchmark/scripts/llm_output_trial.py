"""
Trial runner for the Kiswahili decodability benchmark.

Wires configuration and file I/O to the reusable pipeline in
``phonics_benchmark`` (dataset construction, decodability labelling, filtering,
and running models). Only runner concerns live here — the pipeline logic itself
is in src/ so it can be re-run with different data, filters, decodability config,
languages, or models.

Run this script with:
uv run --frozen --package phonics-benchmark python \
    benchmarks/phonics_benchmark/scripts/llm_output_trial.py
"""

from __future__ import annotations

import ast
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from fdllm import register_models
from phonics_benchmark.llm import call_llm
from phonics_benchmark.parsing import parse_word_list
from phonics_benchmark.prompts import build_decodability_prompt as build_prompt
from phonics_benchmark.dataset import (
    add_analysis_columns,
    build_benchmark_items,
    filter_benchmark_items,
)
from phonics_benchmark.labeling import label_nondecodable_words
from phonics_benchmark.plots import save_nondecodable_ratio_histogram
from phonics_benchmark.run import load_completed_keys, run_benchmark

REPO_ROOT = Path(__file__).parents[3]
DATA_DIR = REPO_ROOT / "benchmarks" / "phonics_benchmark" / "data"
OUTPUTS_DIR = Path(__file__).parents[1] / "outputs"
BENCHMARK_ITEMS_PATH = (
    Path(__file__).parents[1]
    / "data"
    / "kiswahili"
    / "tusome_kenya_llm_trial_cases.jsonl"
)
BENCHMARK_CANDIDATES_PATH = (
    DATA_DIR / "kiswahili" / "tusome_benchmark_item_candidates_with_analysis.csv"
)
BENCHMARK_ITEMS_PATH = (
    DATA_DIR / "kiswahili" / "tusome_benchmark_items_with_analysis.csv"
)
STUDENT_KNOWLEDGE_PROFILES_PATH = (
    DATA_DIR / "kiswahili" / "student_knowledge_profiles.jsonl"
)
TEXTS_PATH = DATA_DIR / "kiswahili" / "decodable_texts.jsonl"

# ISO 639 language code; keys the decodability bundle in phonics_benchmark.languages.
# In the full benchmark this will be a CLI-supplied value.
LANGUAGE = "sw"

# Decodability scoring config: which per-word scorers and which aggregation(s) to
# apply. Names are resolved against the language's decodability registries.
DECODABILITY_CONFIG = {
    "scoring_methods": [
        "score_known_graphemes_kiswahili",
        "score_whole_words_kiswahili",
    ],
    "aggregations": {
        "kenya_tusome_score": {"name": "kiswahili_kenya_tusome_aggregator"},
    },
}

# Temporary subset of models to experiment with.
# Use the same model names as in custom_models.yaml.
models = [
    "gpt-5.6-terra",
    # "claude-opus-4-8",
    "claude-sonnet-4-6",
    "gemini-3.5-flash",
    "deepseek-v4-pro",
    "deepseek-v4-flash",
    "gemma-4-26b-a4b-it",
]

# Option to end each model's loop early for quicker tests; 0 disables the cap.
max_rows_per_model = 0

language = "sw"  # hardcoded registry key; in reality this is CLI-supplied


def build_decodability_prompt(benchmark_item: Any, language: str) -> str:
    """Render the decodability prompt for a prepared benchmark item."""
    return build_prompt(
        {
            "benchmark_item_id": benchmark_item["benchmark_item_id"],
            "content": benchmark_item["content"],
            "student_knowledge": benchmark_item["student_knowledge"],
        },
        language,
    )
    

def submit_benchmark_item_to_model(
    model: str, benchmark_item: Any, language: str
) -> dict[str, Any]:
    """Submit a benchmark item to a model via fdllm."""
    benchmark_item_id = benchmark_item["benchmark_item_id"]
    print(f"\nBenchmark Item ID: {benchmark_item_id}")

    prompt = build_decodability_prompt(benchmark_item, language)

    # TODO: handle failure/retries, maybe not here though
    return call_llm(
        prompt,
        model=model,
        max_tokens=30000,
        tag=f"{benchmark_item_id}",
    )

# Max concurrent in-flight LLM calls per model (fdllm retries 429s; it has no
# concurrency cap, so we set one here).
MAX_CONCURRENCY = 10

# To resume an interrupted run, set this to that run's results JSONL — completed
# (model, benchmark_item_id) pairs are skipped and new results are appended to it.
# Set to None for a fresh run.
RESUME_FROM: Path | None = OUTPUTS_DIR / "20260716T100738Z_results_trial.jsonl"


def save_benchmark_item_candidates() -> None:
    """Build, label, analyse and filter benchmark items; write CSVs + histograms.

    Produces the candidates CSV (all cross-product items with analysis columns)
    and the filtered items CSV that ``main`` consumes.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    benchmark_items = build_benchmark_items(STUDENT_KNOWLEDGE_PROFILES_PATH, TEXTS_PATH)
    benchmark_items["non_decodable_words"] = label_nondecodable_words(
        benchmark_items, DECODABILITY_CONFIG, LANGUAGE
    )
    benchmark_items = add_analysis_columns(benchmark_items, LANGUAGE)
    benchmark_items.to_csv(BENCHMARK_CANDIDATES_PATH, index=False)
    save_nondecodable_ratio_histogram(
        benchmark_items, OUTPUTS_DIR / "nondecodable_ratio_histogram.png"
    )

    filtered_items = filter_benchmark_items(benchmark_items)
    save_nondecodable_ratio_histogram(
        filtered_items,
        OUTPUTS_DIR / "nondecodable_ratio_histogram_filtered.png",
        title="Distribution of Non-Decodable Word Ratios (Filtered)",
    )
    filtered_items.to_csv(BENCHMARK_ITEMS_PATH, index=False)


def process_model_response_into_result(
    call_record: dict[str, Any], benchmark_item: Any
) -> dict[str, Any]:
    response_raw: str = call_record["response_text"]
    response_parsed, response_parse_ok = parse_word_list(response_raw)
    response_parsed = [word.lower() for word in response_parsed]
    print(f"Parsed response word list: {response_parsed}")
    if not response_parse_ok:
        # TODO: maybe retry the LLM call instead of just recording the failure
        print("WARNING: model response did not match expected pattern")

    correct_answer_parsed = [
        word.lower() for word in benchmark_item["non_decodable_words"]
    ]

    # Assess perfect one-to-one correspondence ignoring order (dupes must match too)
    perfect_match = Counter(response_parsed) == Counter(correct_answer_parsed)

    # TODO: log all LLM API config params (even defaults) if possible, e.g. "effort"
    # Longer term this metadata compilation should live somewhere else
    return {
        "model_name": call_record["model"],
        "benchmark_item_id": benchmark_item["benchmark_item_id"],
        "curriculum_stage": int(benchmark_item["curriculum_stage"]),
        "student_knowledge_id": benchmark_item["student_knowledge_id"],
        "response_raw": response_raw,
        "response_parsed": response_parsed,
        "response_parse_ok": response_parse_ok,
        "correct_answer": correct_answer_parsed,
        "perfect_match": perfect_match,
        "max_tokens": call_record["max_tokens"],
        "tokens_used_completion": call_record["tokens_used_completion"],
        "latency_seconds": call_record["latency_seconds"],
        "timestamp_utc": call_record["timestamp_utc"],
    }


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with BENCHMARK_ITEMS_PATH.open(encoding="utf-8") as file:
        benchmark_items = [json.loads(line) for line in file]
    benchmark_items = pd.read_csv(
        BENCHMARK_ITEMS_PATH,
        converters={"non_decodable_words": ast.literal_eval},
    )
    # CSVs produced before the student_knowledge rename use `taught_knowledge`;
    # accept either, parse the dict string, and normalize to `student_knowledge`.
    knowledge_col = (
        "student_knowledge"
        if "student_knowledge" in benchmark_items.columns
        else "taught_knowledge"
    )
    benchmark_items[knowledge_col] = benchmark_items[knowledge_col].apply(
        ast.literal_eval
    )
    benchmark_items = benchmark_items.rename(
        columns={knowledge_col: "student_knowledge"}
    )

    # This env setup will be handled by a CLI callback in the full benchmark.
    load_dotenv(REPO_ROOT / ".env")
    register_models(REPO_ROOT / "custom_models.yaml")

    timestamp: str = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    results_path = OUTPUTS_DIR / f"{timestamp}_results_trial.jsonl"
    results = []

    # TODO: add parallelisation using concurrent.futures, though this
    # higher-level orchestration will normally be handled elsewhere by the runner.
    for model in models:
        print(f"Running decodability benchmark for model: {model}")
        rows = (
            benchmark_items
            if max_rows_per_model == 0
            else benchmark_items[:max_rows_per_model]
        )
        for benchmark_item in rows:
            call_record = submit_benchmark_item_to_model(
                model, benchmark_item, language
            )
            result = process_model_response_into_result(call_record, benchmark_item)
            results.append(result)
    if RESUME_FROM is not None:
        results_path = RESUME_FROM
        completed_keys = load_completed_keys(results_path)
        print(
            f"Resuming from {results_path.name}: "
            f"{len(completed_keys)} (model, item) pairs already complete"
        )
    else:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        results_path = OUTPUTS_DIR / f"{timestamp}_results_trial.jsonl"
        completed_keys = set()

    def append_result_to_jsonl(result: dict) -> None:
        # Stream each result to JSONL as a fallback in case of a crash.
        with results_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")

    run_benchmark(
        benchmark_items,
        models,
        language,
        max_rows_per_model=max_rows_per_model,
        max_concurrency=MAX_CONCURRENCY,
        completed_keys=completed_keys,
        on_result=append_result_to_jsonl,
    )

    # Rebuild the CSV from the full JSONL (prior + new) so a resumed run's CSV is
    # complete; the JSONL is the source of truth.
    all_results = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    final_path = results_path.with_suffix(".csv")
    tmp_path = results_path.with_suffix(".tmp.csv")
    pd.DataFrame(all_results).to_csv(tmp_path, index=False)
    os.replace(tmp_path, final_path)
    print(f"\nCSV saved to {final_path} ({len(all_results)} rows)")


if __name__ == "__main__":
    main()
