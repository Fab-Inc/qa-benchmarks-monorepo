"""Build the decodable-text-generation benchmark item set.

Crosses every canonical student knowledge profile with every generation scenario
and writes two item files: the full set and a smaller stratified dev set.

Profiles are read from ``student_knowledge_profiles.jsonl`` in the language's
data folder — the single source of truth maintained by the data team. Each
profile's ``student_knowledge`` is validated against the language's model in
`decodability` before any items are built, so a malformed grapheme or an
inconsistent cluster/grapheme pairing fails here rather than at run time.

Usage:

    uv run --package phonics-benchmark python \\
        benchmarks/phonics_benchmark/scripts/build_benchmark_items.py \\
        --language kiswahili
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from phonics_benchmark.curation import (
    build_generation_items,
    profiles_to_dataframe,
    sample_dev_profiles,
    summarize_items,
)
from phonics_benchmark.data_io import (
    convert_dataframe_to_records,
    get_data_dir,
    read_student_knowledge_profiles,
    resolve_language_iso_code,
    validate_student_knowledge,
    write_dataset,
    write_json,
)
from phonics_benchmark.models import GenerationBenchmarkItem
from phonics_benchmark.plots import save_item_difficulty_histogram
from phonics_benchmark.prompts import get_scenarios

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"

app = typer.Typer(add_completion=False)


@app.command()
def main(
    language: str = typer.Option(
        "kiswahili",
        "--language",
        "-l",
        help="Language folder name under the data root, for example 'kiswahili'.",
    ),
    data_root: Path = typer.Option(
        DEFAULT_DATA_ROOT,
        "--data-root",
        file_okay=False,
        help="Root data directory. Defaults to the package's data/ folder.",
    ),
) -> None:
    """Build and write the full and dev generation item sets."""
    profiles = read_student_knowledge_profiles(language, data_root)
    validate_student_knowledge(profiles, strict=True)
    language_iso_code = resolve_language_iso_code(profiles)

    scenarios = get_scenarios()
    typer.echo(
        f"Read {len(profiles)} student knowledge profiles ({language_iso_code}); "
        f"{len(scenarios)} scenarios ({', '.join(sorted(scenarios))})"
    )

    profiles_frame = profiles_to_dataframe(profiles)
    items = build_generation_items(profiles_frame, scenarios)
    dev_items = build_generation_items(sample_dev_profiles(profiles_frame), scenarios)

    write_dataset(
        language,
        data_root,
        benchmark_items=convert_dataframe_to_records(items, GenerationBenchmarkItem),
        benchmark_items_dev=convert_dataframe_to_records(
            dev_items, GenerationBenchmarkItem
        ),
    )

    data_dir = get_data_dir(language, data_root)
    summary = {"full": summarize_items(items), "dev": summarize_items(dev_items)}
    write_json(summary, data_dir / "benchmark_items_summary.json")
    save_item_difficulty_histogram(
        items, data_dir / "benchmark_items_by_difficulty.png"
    )

    typer.echo(json.dumps(summary, indent=2))
    typer.echo(f"Wrote item sets to {data_dir}")


if __name__ == "__main__":
    app()
