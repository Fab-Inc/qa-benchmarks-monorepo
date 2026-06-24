"""
Script focused on generating realistic responses (to benchmark items) from LLMs and
parsing each response into a word list that can later be scored (outside scope though
implemented crudely here as demonstration of concept).

Some of the boilerplate here will be replaced/rewritten once itegrated into src/,
especially the IO parts, and anything that overlaps with other code.
"""

import json
import os
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fdllm import register_models
from phonics_benchmark.llm import call_llm
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

# # calculate outer product of texts vs knowledge profiles
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

load_dotenv(REPO_ROOT / ".env")
register_models(REPO_ROOT / "custom_models.yaml")

# temporary subset of models to experiment with
models = [
    "gemini-3.1-pro-preview",
    "claude-sonnet-4-6",
    # "claude-haiku-4-5-20251001",
    # "gemini-2.5-flash-preview-09-2025",
    # "deepseek-v3-0324-fp8"
]

# option to end loop early for quicker tests; set to 0 to disable
max_rows = 5

language = "kiswahili"  # hardcoded for now, will be replaced with CLI-supplied value

timestamp: str = datetime.now(UTC).strftime("%Y%m%d_%H%M%SZ")
results_path = OUTPUTS_DIR / f"{timestamp}_results_trial.jsonl"


def strip_non_alpha(word: str) -> str:
    """Strip non-alphabetic characters from the start and end of a word.

    Alphabetic characters within the word are preserved, and ``str.isalpha``
    is Unicode-aware so this works for any alphabet/language.
    """
    start, end = 0, len(word)
    while start < end and not word[start].isalpha():
        start += 1
    while end > start and not word[end - 1].isalpha():
        end -= 1
    return word[start:end]


def parse_word_list_from_response(response: str) -> list[str]:
    """Split response word list by commas, then remove all remaining whitespace
    and special characters to leave pure alphabetic words.
    """
    # TODO: we probably want to handle this more scientifically in longer term
    # by evaluating and logging whether the response met the expected schema,
    # so that invalid responses get flagged and not just ignored / marked wrong.
    # For now just brute forcing everything into a list of words.

    if response == "[]" or response == "":
        return []

    if re.search(r"\ball\b", response, re.IGNORECASE):
        print(
            "WARNING: model response contains the word 'all' so may be "
            "erroneously trying to say 'all words decodable' or similar"
        )

    words_raw = [word.strip() for word in response.split(",")]

    # WARNING: this will fail if the word is legitimately supposed to begin
    # or end with punctuation, e.g. apostrophe could be problematic
    words_stripped = [strip_non_alpha(word) for word in words_raw]

    # remove any empty strings that may have occurred (would be unusual)
    words_filtered = [w for w in words_stripped if w]

    return words_filtered


results = []
# TODO: add parallelisation using concurrent.futures
for model in models:
    print(f"Running decodability benchmark for model: {model}")
    for _, benchmark_item in benchmark_items.iterrows():
        if max_rows and len(results) >= max_rows:
            break

        # alternative format when using JSON data from data/ dir
        # text: DecodableText = benchmark_item["decodable_text"]
        # knowledge_profile: StudentKnowledgeProfile =
        #    benchmark_item["student_knowledge_profile"]

        # Temporary field names based on sample Excel data column headers
        id = benchmark_item["ID"]
        text = benchmark_item["Text"]
        knowledge_profile = {
            "taught_sounds": benchmark_item["Taught sounds (summary)"],
            "sight_words_known": benchmark_item["Whole words known"],
        }
        print(f"\nBenchmark Item ID: {id}")

        # temporary approach while testing; scoring will be properly handled elsewhere
        correct_answer_raw = benchmark_item["Non-decodable words (cannot read)"]
        correct_answer_parsed = parse_word_list_from_response(correct_answer_raw)

        prompt = get_prompt(
            "decodability_evaluation",
            language=language,
            text=text,
            taught_knowledge=knowledge_profile,
        )

        # TODO: handle failure/retries, maybe not here though
        result = call_llm(
            prompt,
            model=model,
            max_tokens=30000,
            tag=f"{id}",
            should_log=True,
            log_base_dir=OUTPUTS_DIR,
        )

        response_raw: str = result["response_text"]
        response_parsed = parse_word_list_from_response(response_raw)
        print(f"Parsed response word list: {response_parsed}")

        # Perfect one-to-one correspondence ignoring order (duplicates must match too)
        perfect_match = Counter(response_parsed) == Counter(correct_answer_parsed)

        # TODO: log all LLM API config params (even defaults) if possible, e.g. "effort"
        result = {
            "model_name": model,
            "language": language,
            "benchmark_item_id": id,
            "response_raw": response_raw,
            "response_parsed": response_parsed,
            "correct_answer": correct_answer_parsed,
            "perfect_match": perfect_match,
            "max_tokens": result["max_tokens"],
            "tokens_used": result["tokens_used"],
            "tokens_used_completion": result["tokens_used_completion"],
            "tokens_used_reasoning": result["tokens_used_reasoning"],
            "may_be_truncated": result["may_be_truncated"],
            "duration_seconds": result["duration_seconds"],
            "timestamp_utc": result["timestamp_utc"],
        }

        # Append result to JSONL fallback in case of crash
        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        results.append(result)

# Save the generated DataFrame to an Excel file for easier human readibility
# using a temp file and rename for safety
final_path = results_path.with_suffix(".xlsx")
tmp_path = results_path.with_suffix(".tmp.xlsx")
pd.DataFrame(results).to_excel(tmp_path, index=False, freeze_panes=(1, 0))
os.replace(tmp_path, final_path)
print(f"\nExcel saved to {final_path}")
