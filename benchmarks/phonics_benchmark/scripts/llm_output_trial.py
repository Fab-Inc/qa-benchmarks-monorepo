"""
Script focused on generating responses to benchmark items from LLMs, and
parsing each response into a word list that can later be scored (outside of scope
but implemented crudely here as demonstration of concept).

The idea is that the functions defined here will be integrated into src/ once
the outer benchmark scaffolding is ready, and most of the rest of the boilerplate
here won't be since being handled elsewhere.

Run this script with:
uv run path/to/this/script.py
"""

import json
import os
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from fdllm import register_models
from phonics_benchmark.llm import call_llm
from phonics_benchmark.parsing import parse_word_list_from_response
from phonics_benchmark.prompts import get_prompt

# --- resurrect once data/ dir is mature and holds enough usable benchmark items ---
# from phonics_benchmark.models import (
#     BenchmarkItem,
#     DecodableText,
#     StudentKnowledgeProfile,
# )
# from phonics_benchmark.data_io import (
#     read_decodable_texts,
#     read_student_knowledge_profiles,
# )

# decodable_texts = read_decodable_texts(language)
# student_knowledge_profiles = read_student_knowledge_profiles(language)

# # calculate outer product of texts vs student knowledge profiles
# benchmark_items = []
# for text in decodable_texts:
#     for profile in student_knowledge_profiles:
#         benchmark_items.append(
#             {
#                 "language": language,
#                 "student_knowledge_profile": profile,
#                 "decodable_text": text,
#             }
#         )

OUTPUTS_DIR = Path(__file__).parents[1] / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
REPO_ROOT = Path(__file__).parents[3]

# temporary Excel data source during prototyping
BENCH_ITEMS_PATH = (
    Path(__file__).parents[1]
    / "data"
    / "kiswahili"
    / "kiswahili_kenya_authentic_tusome_decodable_text_examples_extended.xlsx"
)
# warning: this Excel currently has the headers on row 4; could change
benchmark_items = pd.read_excel(BENCH_ITEMS_PATH, header=3)

# this env setup will be handled by a CLI callback when running the benchmark properly
load_dotenv(REPO_ROOT / ".env")
register_models(REPO_ROOT / "custom_models.yaml")

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
max_rows_per_model = 5

language = "kiswahili"  # hardcoded for now, in reality will be CLI-supplied value

timestamp: str = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
results_path = OUTPUTS_DIR / f"{timestamp}_results_trial.jsonl"


def submit_benchmark_item_to_model(
    model: str, benchmark_item: Any, language: str
) -> dict[str, Any]:
    """Submit a benchmark item to a model via fdllm."""
    # Temporary field names based on sample Excel data column headers
    benchmark_item_id = benchmark_item["ID"]
    text = benchmark_item["Text"]
    student_knowledge = {
        "taught_sounds": benchmark_item["Taught sounds (summary)"],
        "sight_words_known": benchmark_item["Whole words known"],
    }
    print(f"\nBenchmark Item ID: {benchmark_item_id}")

    # alternative handling when using JSON data from data/ dir
    # text: DecodableText = benchmark_item["decodable_text"]
    # student_knowledge: StudentKnowledgeProfile = benchmark_item[
    #     "student_knowledge_profile"
    # ]

    prompt = get_prompt(
        "decodability_evaluation",
        language=language,
        text=text,
        student_knowledge=student_knowledge,
    )

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
    response_parsed = parse_word_list_from_response(response_raw)
    print(f"Parsed response word list: {response_parsed}")

    # basic scoring approach while testing; scoring will be properly handled elsewhere
    correct_answer_raw = benchmark_item["Non-decodable words (cannot read)"]
    correct_answer_parsed = parse_word_list_from_response(correct_answer_raw)

    # Assess perfect one-to-one correspondence ignoring order (dupes must match too)
    perfect_match = Counter(response_parsed) == Counter(correct_answer_parsed)
    # TODO: consider converting everything to lowercase before comparison

    # TODO: log all LLM API config params (even defaults) if possible, e.g. "effort"
    # Longer term this metadata compilation should live somewhere else
    return {
        "model_name": call_record["model"],
        "benchmark_item_id": benchmark_item["ID"],
        "response_raw": response_raw,
        "response_parsed": response_parsed,
        "correct_answer": correct_answer_parsed,
        "perfect_match": perfect_match,
        "max_tokens": call_record["max_tokens"],
        "tokens_used": call_record["tokens_used"],
        "tokens_used_completion": call_record["tokens_used_completion"],
        "tokens_used_reasoning": call_record["tokens_used_reasoning"],
        "may_be_truncated": call_record["may_be_truncated"],
        "latency_seconds": call_record["latency_seconds"],
        "timestamp_utc": call_record["timestamp_utc"],
    }


results = []
# TODO: add parallelisation using concurrent.futures , though this
# higher-level orchestration will normally be handled elsewhere by the runner
for model in models:
    print(f"Running decodability benchmark for model: {model}")
    for _, benchmark_item in benchmark_items.iterrows():
        if max_rows_per_model and len(results) >= max_rows_per_model:
            break

        call_record = submit_benchmark_item_to_model(model, benchmark_item, language)
        result = process_model_response_into_result(call_record, benchmark_item)

        results.append(result)

        # Append result to JSONL fallback in case of crash
        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

# Save the generated DataFrame to an Excel file for easier human readibility
# using a temp file and rename for safety
final_path = results_path.with_suffix(".xlsx")
tmp_path = results_path.with_suffix(".tmp.xlsx")
pd.DataFrame(results).to_excel(tmp_path, index=False, freeze_panes=(1, 0))
os.replace(tmp_path, final_path)
print(f"\nExcel saved to {final_path}")
