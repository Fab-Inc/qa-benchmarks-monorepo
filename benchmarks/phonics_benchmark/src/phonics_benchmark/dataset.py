"""Build and filter the benchmark item set from the raw data files.

Benchmark items are the cross product of student knowledge profiles and
decodable texts: every text is evaluated against every profile. Loading is done
with pandas (rather than the validated readers in ``data_io``) so the pipeline
runs against the current ``taught_knowledge`` data shape without depending on the
models module; student knowledge is validated downstream in ``labeling`` via the
language-specific decodability model.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from decodability import get_decodability_objects


def build_benchmark_items(
    student_knowledge_profiles_path: Path,
    texts_path: Path,
) -> pd.DataFrame:
    """Cross-join student profiles × texts into a benchmark item DataFrame.

    Adds a ``benchmark_item_id`` of the form ``<student_knowledge_id>_<text_id>``.
    """
    profiles = pd.read_json(student_knowledge_profiles_path, lines=True)
    texts = pd.read_json(texts_path, lines=True)

    # TODO: align on benchmark item ID format. Potentially drop student and text IDs
    benchmark_items = profiles.merge(texts, how="cross")
    # Align the in-memory column name with models.StudentKnowledge; the raw JSONL
    # keeps the `taught_knowledge` key (owned by the ingestion / QFB-187 work).
    benchmark_items = benchmark_items.rename(
        columns={"taught_knowledge": "student_knowledge"}
    )
    benchmark_items["benchmark_item_id"] = (
        benchmark_items["student_knowledge_id"].astype(str)
        + "_"
        + benchmark_items["text_id"].astype(str)
    )
    return benchmark_items


def add_analysis_columns(benchmark_items: pd.DataFrame, language: str) -> pd.DataFrame:
    """Add num_nondecodable_words, num_total_words and nondecodable_ratio.

    Expects a ``non_decodable_words`` column (see ``labeling``) and a ``content``
    column. Total word count uses the language's tokenizer. Returns a copy.
    """
    tokenizer = get_decodability_objects(language).tokenizer

    benchmark_items = benchmark_items.copy()
    benchmark_items["num_nondecodable_words"] = benchmark_items[
        "non_decodable_words"
    ].apply(len)
    benchmark_items["num_total_words"] = benchmark_items["content"].apply(
        lambda content: len(tokenizer(content))
    )
    benchmark_items["nondecodable_ratio"] = (
        benchmark_items["num_nondecodable_words"] / benchmark_items["num_total_words"]
    )
    return benchmark_items


def filter_benchmark_items(
    benchmark_items: pd.DataFrame,
    random_state: int = 42,
) -> pd.DataFrame:
    """Down-sample to a manageable mix of easy and moderately hard items.

    - Keep at most ``max_zero_nondecodable`` items with a 0.0 ratio (random sample).
    - Keep at most ``max_one_nondecodable`` items with exactly one non-decodable
      word (random sample).
    - Drop items harder than ``max_nondecodable_ratio``.

    Sampling is seeded by ``random_state`` for reproducibility. This heuristic is
    expected to be refined as the item distribution becomes clearer.
    """
    mask_zero = benchmark_items["nondecodable_ratio"] == 0.0
    max_zero_nondecodable = 20
    zero_nondecodable = benchmark_items[mask_zero]
    if len(zero_nondecodable) > max_zero_nondecodable:
        zero_nondecodable = zero_nondecodable.sample(
            max_zero_nondecodable, random_state=random_state
        )
    benchmark_items = pd.concat([zero_nondecodable, benchmark_items[~mask_zero]])

    mask_one = benchmark_items["num_nondecodable_words"] == 1
    one_nondecodable = benchmark_items[mask_one]
    max_one_nondecodable = 40
    if len(one_nondecodable) > max_one_nondecodable:
        one_nondecodable = one_nondecodable.sample(
            max_one_nondecodable, random_state=random_state
        )
    benchmark_items = pd.concat([one_nondecodable, benchmark_items[~mask_one]])

    max_nondecodable_ratio = 0.2
    benchmark_items = benchmark_items[
        benchmark_items["nondecodable_ratio"] <= max_nondecodable_ratio
    ]
    return benchmark_items
