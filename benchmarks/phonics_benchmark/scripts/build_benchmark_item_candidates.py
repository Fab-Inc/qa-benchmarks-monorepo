"""Pair student profiles with decodable texts, then analyse the pairs.

Stage 2 of the pipeline. Run ``ingest_benchmark_data.py`` first to produce the
student knowledge profiles and decodable texts JSONL files.

Run from the repository root with (as an example):

    uv run --package phonics-benchmark python \\
        benchmarks/phonics_benchmark/scripts/build_benchmark_item_candidates.py \\
        --language kiswahili

Produces, under ``data/<language>/``:

- ``benchmark_item_candidates.jsonl`` — every (profile, text) pairing, ID-ed by
  ``benchmark_item_id``, with its non-decodable words and analysis columns
- ``nondecodable_ratio_histogram_for_benchmark_candidates.png`` — the
  non-decodable ratio distribution across the full cross-product
- ``nondecodable_summary_for_benchmark_candidates.json`` — summary statistics of
  the non-decodable counts and ratios across the full cross-product

These are intermediate artifacts and are gitignored; re-run this script to
reproduce them.

What's next
1. Inspect the histogram and summary to choose filtering thresholds.
2. Run ``filter_benchmark_items.py`` to cut the cross-product down to the set
   the benchmark actually runs.
"""

import json
from pathlib import Path

import typer
from phonics_benchmark.curation import (
    add_labels_and_analysis,
    get_nondecodable_word_analysis,
)
from phonics_benchmark.data_io import (
    build_benchmark_item_candidates_dataframe,
    convert_dataframe_to_records,
    read_decodable_texts,
    read_student_knowledge_profiles,
    resolve_language_iso_code,
    validate_student_knowledge,
    write_dataset,
    write_json,
)
from phonics_benchmark.models import LabelledBenchmarkItem
from phonics_benchmark.plots import save_nondecodable_ratio_histogram

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"

# Config for the decodability package. ``score_column`` picks which aggregation
# output decides decodability, and so differs by language.
DECODABILITY_CONFIGS: dict[str, dict] = {
    "sw": {
        "scoring_methods": [
            "score_known_graphemes_kiswahili",
            "score_known_clusters_and_patterns_kiswahili",
            "score_whole_words_kiswahili",
        ],
        "aggregations": {
            "kenya_tusome_score": {"name": "kiswahili_kenya_tusome_aggregator"},
        },
        "score_column": "kenya_tusome_score",
    },
    "en": {
        "scoring_methods": [
            "score_known_gpcs_english",
            "score_sight_words_english",
        ],
        "aggregations": {
            "ufli_score": {"name": "ufli_curriculum_aggregator"},
        },
        "score_column": "ufli_score",
    },
}


def build_candidate_benchmark_items(
    language: str = typer.Option(
        ...,
        "--language",
        "-l",
        help="Language folder name under the data root, for example 'kiswahili'.",
    ),
    data_root: Path = typer.Option(
        DEFAULT_DATA_ROOT,
        "--data-root",
        exists=True,
        file_okay=False,
        readable=True,
        help="Root data directory. Defaults to the package's data/ folder.",
    ),
) -> None:
    """Pair profiles with texts, analyse every pairing, write the candidates."""
    texts = read_decodable_texts(language, data_root)
    profiles = read_student_knowledge_profiles(language, data_root)
    validate_student_knowledge(profiles, strict=True)

    language_iso_code = resolve_language_iso_code(profiles, texts)
    config = DECODABILITY_CONFIGS.get(language_iso_code)
    if config is None:
        raise ValueError(
            f"No decodability config for language '{language_iso_code}'. "
            "Add an entry to DECODABILITY_CONFIGS."
        )

    candidates_df = build_benchmark_item_candidates_dataframe(profiles, texts)
    candidates_with_labels_df = add_labels_and_analysis(
        candidates_df, config, language_iso_code, score_column=config["score_column"]
    )
    candidates_with_labels = convert_dataframe_to_records(
        candidates_with_labels_df, LabelledBenchmarkItem
    )

    data_dir = write_dataset(
        language, data_root, benchmark_candidates=candidates_with_labels
    )

    analysis = get_nondecodable_word_analysis(candidates_with_labels_df)
    write_json(
        analysis, data_dir / "nondecodable_summary_for_benchmark_candidates.json"
    )

    histogram_path = (
        data_dir / "nondecodable_ratio_histogram_for_benchmark_candidates.png"
    )
    save_nondecodable_ratio_histogram(
        candidates_with_labels_df,
        histogram_path,
        title=(
            "Distribution of Non-Decodable Word Ratios\n"
            "(for all benchmark item candidates)"
        ),
    )

    typer.echo(
        f"Paired {len(profiles)} student knowledge profiles × {len(texts)} texts "
        f"into {len(candidates_with_labels)} benchmark item candidates in {data_dir}\n"
    )
    typer.echo(f"Analysis summary:\n{json.dumps(analysis, indent=4)}\n")
    typer.echo(
        f"Next: inspect {histogram_path}, then run filter_benchmark_items.py "
        "with your chosen thresholds."
    )


def main() -> None:
    """Entry point."""
    typer.run(build_candidate_benchmark_items)


if __name__ == "__main__":
    main()
