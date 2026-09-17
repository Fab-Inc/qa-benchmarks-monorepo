"""Cut the benchmark item candidates down to the set the benchmark runs.

Stage 3 of the pipeline. Run ``build_benchmark_item_candidates.py`` first, then inspect
``data/<language>/nondecodable_ratio_histogram_for_benchmark_candidates.png`` to choose
thresholds.

Reads ``data/<language>/benchmark_item_candidates.jsonl`` and writes
``data/<language>/benchmark_items.jsonl`` — the file a run config's ``items_path``
should point at.

Run from the repository root, e.g.:

    uv run --package phonics-benchmark python \\
        benchmarks/phonics_benchmark/scripts/filter_benchmark_items.py \\
        --language kiswahili \\
        --max-zero-nondecodable 20 \\
        --max-one-nondecodable 40 \\
        --max-nondecodable-ratio 0.2 \\
        --sample-size-other 128
"""

import json
import logging
from pathlib import Path

import typer
from phonics_benchmark.curation import (
    filter_benchmark_items,
    get_nondecodable_word_analysis,
)
from phonics_benchmark.data_io import (
    convert_dataframe_to_records,
    convert_records_to_dataframe,
    read_benchmark_item_candidates,
    write_dataset,
    write_json,
)
from phonics_benchmark.models import LabelledBenchmarkItem
from phonics_benchmark.plots import save_nondecodable_ratio_histogram

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


def format_params_for_title(params: dict) -> str:
    """Format filter parameters for display in a plot title, with line breaks."""
    params.pop("random_state", None)

    param_formatted = ""
    for i, (k, v) in enumerate(params.items()):
        param_formatted += f"{k}={v}"
        if i == len(params) - 1:
            break
        if i % 2 == 1:
            param_formatted += "\n"
        else:
            param_formatted += ", "
    return param_formatted


def filter_benchmark_item_candidates(
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
    max_zero_nondecodable: int | None = typer.Option(
        None,
        "--max-zero-nondecodable",
        min=0,
        help="Keep at most this many items whose non-decodable ratio is 0.0.",
    ),
    max_one_nondecodable: int | None = typer.Option(
        None,
        "--max-one-nondecodable",
        min=0,
        help="Keep at most this many items with exactly one non-decodable word.",
    ),
    max_nondecodable_ratio: float | None = typer.Option(
        None,
        "--max-nondecodable-ratio",
        min=0.0,
        max=1.0,
        help="Drop items whose non-decodable ratio exceeds this.",
    ),
    sample_size_other: int | None = typer.Option(
        None,
        "--sample-size-other",
        min=0,
        help="Keep at most this many of the remaining items. Omit the value for all.",
    ),
    random_state: int = typer.Option(
        42,
        "--random-state",
        help="Sampling seed.",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level. Defaults to 'INFO'.",
    ),
) -> None:
    """Cut the candidates down to the set the benchmark runs, using thresholds."""
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))
    candidates = read_benchmark_item_candidates(language, data_root)
    candidates_df = convert_records_to_dataframe(candidates)

    params = {
        "max_zero_nondecodable": max_zero_nondecodable,
        "max_one_nondecodable": max_one_nondecodable,
        "max_nondecodable_ratio": max_nondecodable_ratio,
        "sample_size_other": sample_size_other,
        "random_state": random_state,
    }
    filtered_df = filter_benchmark_items(
        candidates_df,
        **params,
    )
    filtered = convert_dataframe_to_records(filtered_df, LabelledBenchmarkItem)
    data_dir = write_dataset(language, data_root, benchmark_items=filtered)

    analysis = get_nondecodable_word_analysis(filtered_df)
    analysis["filter_params"] = params
    write_json(analysis, data_dir / "nondecodable_summary_for_benchmark_items.json")

    histogram_path = data_dir / "nondecodable_ratio_histogram_for_benchmark_items.png"
    save_nondecodable_ratio_histogram(
        filtered_df,
        histogram_path,
        title=(
            "Distribution of Non-Decodable Word Ratios (Filtered)\n"
            f"{format_params_for_title(params)}"
        ),
    )

    typer.echo(
        f"Filtered {len(candidates)} candidates down to {len(filtered)} benchmark "
        f"items in {data_dir}\n"
    )
    typer.echo(f"Analysis summary:\n{json.dumps(analysis, indent=4)}\n")
    typer.echo(
        f"Saved a histogram of the filtered non-decodable ratios to {histogram_path}\n"
    )
    typer.echo(
        "Next: point a run config's items_path at benchmark_items.jsonl to use them."
    )


def main() -> None:
    """Entry point. Kept a plain function so cmd.py can register it as a command."""
    typer.run(filter_benchmark_item_candidates)


if __name__ == "__main__":
    main()
