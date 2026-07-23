"""Script to generate benchmark items for Kiswahili decodability.

Must run
``benchmarks/phonics_benchmark/scripts/ingest_benchmark_data_template.py``
first to produce the student knowledge profiles and decodable texts JSONL files.

Data paths are hard-coded for now but must be parameterized for other
languages and datasets.

Run this script with:
    uv run --package phonics-benchmark python \
        benchmarks/phonics_benchmark/scripts/build_benchmark_items.py
"""

from pathlib import Path

from phonics_benchmark.dataset import (
    add_analysis_columns,
    build_benchmark_items,
    filter_benchmark_items,
)
from phonics_benchmark.labeling import get_vocabulary_set, label_nondecodable_words
from phonics_benchmark.plots import save_nondecodable_ratio_histogram

REPO_ROOT = Path(__file__).parents[3]
DATA_DIR = REPO_ROOT / "benchmarks" / "phonics_benchmark" / "data"

OUTPUTS_DIR = Path(__file__).parents[1] / "outputs"
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


def save_benchmark_item_candidates() -> None:
    """Build, label, analyse and filter benchmark items; write CSVs + histograms.

    Produces the candidates CSV (all cross-product items with analysis columns)
    and the filtered items CSV that ``main`` consumes.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    benchmark_items = build_benchmark_items(STUDENT_KNOWLEDGE_PROFILES_PATH, TEXTS_PATH)
    benchmark_items["vocab_set"] = get_vocabulary_set(benchmark_items, LANGUAGE)
    benchmark_items["non_decodable_words"] = label_nondecodable_words(
        benchmark_items, DECODABILITY_CONFIG, LANGUAGE
    )
    benchmark_items = add_analysis_columns(benchmark_items, LANGUAGE)
    benchmark_items.to_csv(BENCHMARK_CANDIDATES_PATH, index=False)
    save_nondecodable_ratio_histogram(
        benchmark_items, OUTPUTS_DIR / "nondecodable_ratio_histogram.png"
    )

    filtered_items = filter_benchmark_items(
        benchmark_items,
        max_zero_nondecodable=3,
        max_one_nondecodable=5,
        max_nondecodable_ratio=0.2,
        sample_size_other=8,
        random_state=42,
    )
    save_nondecodable_ratio_histogram(
        filtered_items,
        OUTPUTS_DIR / "nondecodable_ratio_histogram_filtered.png",
        title="Distribution of Non-Decodable Word Ratios (Filtered)",
    )
    filtered_items.to_csv(BENCHMARK_ITEMS_PATH, index=False)


if __name__ == "__main__":
    save_benchmark_item_candidates()
