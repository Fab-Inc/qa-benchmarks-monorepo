"""Build the decodable-text-generation benchmark item set.

Crosses every distinct student knowledge profile with every generation scenario
and writes two item files: the full set and a smaller stratified dev set.

Profiles are read from an existing JSONL of records carrying
``student_knowledge`` (either ``student_knowledge_profiles.jsonl`` or a legacy
labelled-items file — only the profile fields are used; any text/gold-label
columns are ignored).

Usage:

    uv run --package phonics-benchmark python \\
        benchmarks/phonics_benchmark/scripts/build_generation_benchmark_items.py \\
        --language kiswahili \\
        --profiles-file phonics_datasets_kiswahili-with-clusters_benchmark_items.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from phonics_benchmark.curation import (
    build_generation_items,
    extract_unique_profiles,
    sample_dev_profiles,
    summarize_items,
)
from phonics_benchmark.data_io import (
    convert_dataframe_to_records,
    get_data_dir,
    write_dataset,
    write_json,
)
from phonics_benchmark.models import GenerationBenchmarkItem
from phonics_benchmark.plots import save_item_difficulty_histogram
from phonics_benchmark.prompts import get_scenarios

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"

app = typer.Typer(add_completion=False)


def _read_profile_records(path: Path) -> list[dict]:
    records = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON — {exc}") from exc
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


@app.command()
def main(
    language: str = typer.Option(
        "kiswahili", help="Data folder name, e.g. 'kiswahili'."
    ),
    profiles_file: str = typer.Option(
        ...,
        help="JSONL filename inside the language data folder to read profiles from.",
    ),
    data_root: Path = typer.Option(DEFAULT_DATA_ROOT, help="Root of the data folder."),
) -> None:
    """Build and write the full and dev generation item sets."""
    data_dir = get_data_dir(language, data_root)
    records = _read_profile_records(data_dir / profiles_file)

    profiles = extract_unique_profiles(records)
    language_iso_code = profiles["language_iso_code"].iloc[0]
    if profiles["language_iso_code"].nunique() > 1:
        raise ValueError(
            "Expected one language_iso_code across profiles, found: "
            f"{sorted(profiles['language_iso_code'].unique())}"
        )

    scenarios = get_scenarios()
    typer.echo(
        f"Read {len(records)} records -> {len(profiles)} distinct profiles; "
        f"{len(scenarios)} scenarios ({', '.join(sorted(scenarios))})"
    )

    items = build_generation_items(profiles, scenarios, language_iso_code)
    dev_items = build_generation_items(
        sample_dev_profiles(profiles), scenarios, language_iso_code
    )

    full_records = convert_dataframe_to_records(items, GenerationBenchmarkItem)
    dev_records = convert_dataframe_to_records(dev_items, GenerationBenchmarkItem)

    write_dataset(
        language,
        data_root,
        benchmark_items=full_records,
        benchmark_items_dev=dev_records,
    )

    summary = {"full": summarize_items(items), "dev": summarize_items(dev_items)}
    write_json(summary, data_dir / "generation_items_summary.json")
    save_item_difficulty_histogram(
        items, data_dir / "generation_items_by_difficulty.png"
    )

    typer.echo(json.dumps(summary, indent=2))
    typer.echo(f"Wrote item sets to {data_dir}")


if __name__ == "__main__":
    app()
