"""
Script focused on generating responses to benchmark items from LLMs, and
parsing each response into a word list that can later be scored (outside of scope
but implemented crudely here as demonstration of concept).

The idea is that the functions defined here will be integrated into src/ once
the outer benchmark scaffolding is ready, and most of the rest of the boilerplate
here won't be since being handled elsewhere.

Run this script with:
uv run --frozen --package phonics-benchmark python \
    benchmarks/phonics_benchmark/scripts/llm_output_trial.py
"""

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

REPO_ROOT = Path(__file__).parents[3]
OUTPUTS_DIR = Path(__file__).parents[1] / "outputs"
BENCHMARK_ITEMS_PATH = (
    Path(__file__).parents[1]
    / "data"
    / "kiswahili"
    / "tusome_kenya_llm_trial_cases.csv"
)


# temporary subset of models to experiment with
# use same model names as in custom_models.yaml
models = [
    "gemini-3.1-pro-preview",
    "claude-sonnet-4-6",
    # "claude-haiku-4-5-20251001",
    # "gemini-2.5-flash-preview-09-2025",
    # "deepseek-v3-0324-fp8"
]

# option to end loop early for quicker tests; set to 0 to disable
max_rows_per_model = 10

language = "sw"  # hardcoded registry key; in reality this is CLI-supplied


def build_decodability_prompt(benchmark_item: Any, language: str) -> str:
    """Render the decodability prompt for a prepared benchmark item."""
    return build_prompt(
        {
            "benchmark_item_id": benchmark_item["benchmark_item_id"],
            "content": benchmark_item["content"],
            "student_knowledge": {
                "graphemes": json.loads(benchmark_item["graphemes"]),
                "whole_words": json.loads(benchmark_item["whole_words"]),
            },
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
        word.lower() for word in json.loads(benchmark_item["non_decodable_words"])
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
    benchmark_items = pd.read_csv(BENCHMARK_ITEMS_PATH)

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
            else benchmark_items.head(max_rows_per_model)
        )
        for _, benchmark_item in rows.iterrows():
            call_record = submit_benchmark_item_to_model(
                model, benchmark_item, language
            )
            result = process_model_response_into_result(call_record, benchmark_item)
            results.append(result)

            # Append result to JSONL fallback in case of crash.
            with results_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(result, ensure_ascii=False) + "\n")

    # Save a CSV copy using a temp file and rename.
    final_path = results_path.with_suffix(".csv")
    tmp_path = results_path.with_suffix(".tmp.csv")
    pd.DataFrame(results).to_csv(tmp_path, index=False)
    os.replace(tmp_path, final_path)
    print(f"\nCSV saved to {final_path}")


if __name__ == "__main__":
    main()
