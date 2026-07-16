"""Estimate Rasch item difficulty and model capability from trial results.

Each ``model_name`` is treated as an IRT respondent and ``perfect_match`` is
treated as its binary response to a ``benchmark_item_id``. The script fits a
Rasch/1PL model with GIRTH, using marginal maximum likelihood (MML) for item
difficulty and expected a posteriori (EAP) scoring for model capability.

Run against the supplied trial output with:

uv run --frozen --package phonics-benchmark python \
    benchmarks/phonics_benchmark/scripts/estimate_irt.py \
    benchmarks/phonics_benchmark/outputs/20260715T131507Z_results_trial.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from girth import ability_eap, rasch_mml

REQUIRED_COLUMNS = {"model_name", "benchmark_item_id", "perfect_match"}
ITEM_METADATA_COLUMNS = ["curriculum_stage", "student_knowledge_id"]
DISCRIMINATION = 1.0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Fit a Rasch/1PL model to phonics benchmark trial results, treating "
            "each model_name as a respondent."
        )
    )
    parser.add_argument("results_csv", type=Path, help="Trial results CSV to fit")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (defaults to the input CSV's directory)",
    )
    return parser.parse_args()


def _binary_scores(values: pd.Series) -> pd.Series:
    """Convert boolean-like perfect-match values to integer IRT responses."""
    normalized = values.astype("string").str.strip().str.lower()
    mapping = {"true": 1, "false": 0, "1": 1, "0": 0}
    scores = normalized.map(mapping)
    if scores.isna().any():
        invalid = sorted(normalized[scores.isna()].dropna().unique().tolist())
        raise ValueError(
            "perfect_match must contain only True/False or 1/0 values; "
            f"found: {invalid}"
        )
    return scores.astype(int)


def load_responses(results_csv: Path) -> pd.DataFrame:
    """Load and validate one scored response per model and benchmark item."""
    results = pd.read_csv(results_csv)
    missing_columns = REQUIRED_COLUMNS.difference(results.columns)
    if missing_columns:
        raise ValueError(
            "Results CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    if results.empty:
        raise ValueError("Results CSV contains no responses")

    results = results.copy()
    results["perfect_match"] = _binary_scores(results["perfect_match"])

    response_keys = ["model_name", "benchmark_item_id"]
    duplicates = results.duplicated(response_keys, keep=False)
    if duplicates.any():
        duplicate_count = results.loc[duplicates, response_keys].drop_duplicates()
        raise ValueError(
            "IRT fitting requires one response per model and item; found "
            f"{len(duplicate_count)} duplicated model/item pairs"
        )

    for column in ITEM_METADATA_COLUMNS:
        if column not in results:
            continue
        values_per_item = results.groupby("benchmark_item_id", sort=False)[
            column
        ].nunique(dropna=False)
        if (values_per_item > 1).any():
            raise ValueError(
                f"Column {column!r} has conflicting values for a benchmark item"
            )

    return results


def estimate_rasch(
    results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate item difficulties and model capabilities from scored results."""
    item_order = results["benchmark_item_id"].drop_duplicates().tolist()
    model_order = results["model_name"].drop_duplicates().tolist()
    if len(model_order) < 2:
        raise ValueError("IRT fitting requires responses from at least two models")

    response_frame = results.pivot(
        index="benchmark_item_id",
        columns="model_name",
        values="perfect_match",
    ).reindex(index=item_order, columns=model_order)
    if response_frame.isna().any().any():
        missing_count = int(response_frame.isna().sum().sum())
        raise ValueError(
            "Every model must have a response for every item; "
            f"the response matrix has {missing_count} missing values"
        )

    # GIRTH expects an items-by-respondents matrix. Integer 0/1 values avoid
    # NumPy interpreting all-False/all-True item rows as boolean index masks.
    response_matrix = response_frame.to_numpy(dtype=int)
    estimates = rasch_mml(response_matrix, discrimination=DISCRIMINATION)
    difficulties = np.asarray(estimates["Difficulty"], dtype=float)
    capabilities = np.asarray(
        ability_eap(response_matrix, difficulties, DISCRIMINATION), dtype=float
    )

    correct_by_item = response_matrix.sum(axis=1)
    item_results = pd.DataFrame(
        {
            "benchmark_item_id": item_order,
            "difficulty_logit_mml": difficulties,
            "models_correct": correct_by_item,
            "models_total": len(model_order),
            "response_rate": correct_by_item / len(model_order),
        }
    )
    item_results["difficulty_estimate_status"] = np.select(
        [
            item_results["models_correct"] == len(model_order),
            item_results["models_correct"] == 0,
        ],
        ["lower_bound_all_correct", "upper_bound_all_incorrect"],
        default="interior",
    )

    metadata_columns = [
        column for column in ITEM_METADATA_COLUMNS if column in results.columns
    ]
    if metadata_columns:
        item_metadata = (
            results[["benchmark_item_id", *metadata_columns]]
            .drop_duplicates("benchmark_item_id")
            .set_index("benchmark_item_id")
            .reindex(item_order)
            .reset_index()
        )
        item_results = item_metadata.merge(
            item_results, on="benchmark_item_id", validate="one_to_one"
        )

    correct_by_model = response_matrix.sum(axis=0)
    model_results = pd.DataFrame(
        {
            "model_name": model_order,
            "capability_logit_eap": capabilities,
            "items_correct": correct_by_model,
            "items_total": len(item_order),
            "raw_accuracy": correct_by_model / len(item_order),
        }
    )
    model_results["capability_rank"] = (
        model_results["capability_logit_eap"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    return item_results, model_results


def output_paths(results_csv: Path, output_dir: Path) -> tuple[Path, Path]:
    """Build descriptive output paths from the input filename."""
    return (
        output_dir / f"{results_csv.stem}_irt_items.csv",
        output_dir / f"{results_csv.stem}_irt_models.csv",
    )


def main() -> None:
    """Fit the model and write item and model parameter CSVs."""
    args = parse_args()
    results = load_responses(args.results_csv)
    item_results, model_results = estimate_rasch(results)

    output_dir = args.output_dir or args.results_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    items_path, models_path = output_paths(args.results_csv, output_dir)
    item_results.to_csv(items_path, index=False)
    model_results.to_csv(models_path, index=False)

    n_models = len(model_results)
    boundary_items = item_results["difficulty_estimate_status"] != "interior"
    print(f"Fitted Rasch/1PL model to {len(item_results)} items and {n_models} models.")
    print(f"Item difficulties: {items_path}")
    print(f"Model capabilities: {models_path}")
    if n_models == 2:
        print(
            "WARNING: With only two model respondents, item response rates and "
            "difficulty estimates can take only three distinct levels."
        )
    if boundary_items.any():
        print(
            "WARNING: "
            f"{int(boundary_items.sum())} items were answered identically by all "
            "models; their difficulty estimates are optimizer bounds, not finite "
            "point estimates."
        )


if __name__ == "__main__":
    main()
