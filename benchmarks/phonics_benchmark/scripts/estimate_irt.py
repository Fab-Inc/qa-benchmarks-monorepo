"""Estimate Rasch item difficulty and model capability from trial results.

Each ``model_name`` is treated as an IRT respondent and ``perfect_match`` is
treated as its binary response to a ``benchmark_item_id``. The script fits a
Rasch/1PL model with GIRTH, using marginal maximum likelihood (MML) for item
difficulty and expected a posteriori (EAP) scoring for model capability.

Adapted from the ``explore-irt-item-difficulty`` branch to run on the JSONL
trial outputs produced by ``llm_output_trial.py`` (the original read a CSV).
The two changes that matter:

  * Input is read from JSONL (``.jsonl``/``.json``) as well as CSV, matching the
    ``outputs/*_results_trial.jsonl`` files this benchmark writes.
  * Models need not have answered every item. Missing model x item responses are
    tagged for GIRTH rather than rejected, so trials with a few dropped items
    (e.g. parse failures or API errors) still fit. Per-item and per-model rates
    are computed over answered items only.

Run against the supplied trial output with:

uv run --frozen --package phonics-benchmark python \
    benchmarks/phonics_benchmark/scripts/estimate_irt.py \
    benchmarks/phonics_benchmark/outputs/20260716T100738Z_results_trial.jsonl

Pass ``--models`` to fit on a subset of respondents. Degenerate models that
answer near-identically regardless of item difficulty break IRT's monotonicity
assumption and both distort difficulty estimates and manufacture spurious
discrimination, so it is often worth dropping them:

uv run --frozen --package phonics-benchmark python \
    benchmarks/phonics_benchmark/scripts/estimate_irt.py \
    benchmarks/phonics_benchmark/outputs/20260716T100738Z_results_trial.jsonl \
    --models deepseek-v4-flash gemini-3.5-flash gpt-5.6-terra deepseek-v4-pro
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from girth import ability_eap, rasch_mml, tag_missing_data

REQUIRED_COLUMNS = {"model_name", "benchmark_item_id", "perfect_match"}
ITEM_METADATA_COLUMNS = ["curriculum_stage", "student_knowledge_id"]
DISCRIMINATION = 1.0
# Values GIRTH accepts as real dichotomous responses; anything else in the
# model x item matrix (i.e. a model that never answered the item) is tagged as
# missing and skipped by the estimators.
VALID_RESPONSES = [0, 1]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Fit a Rasch/1PL model to phonics benchmark trial results, treating "
            "each model_name as a respondent."
        )
    )
    parser.add_argument(
        "results_path",
        type=Path,
        help="Trial results file to fit (.jsonl, .json, or .csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (defaults to the input file's directory)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL_NAME",
        help=(
            "Restrict the fit to these model_name respondents (default: all "
            "models in the file). Useful for dropping degenerate respondents "
            "that answer near-identically regardless of item difficulty, which "
            "violate IRT's monotonicity assumption and distort difficulty "
            "estimates."
        ),
    )
    return parser.parse_args()


def _binary_scores(values: pd.Series) -> pd.Series:
    """Convert boolean-like perfect-match values to integer IRT responses.

    Handles native booleans (as loaded from JSONL) and the True/False or 1/0
    strings that a CSV round-trip produces.
    """
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


def _read_results(results_path: Path) -> pd.DataFrame:
    """Read a results file, dispatching on its suffix (JSONL/JSON or CSV)."""
    suffix = results_path.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        # This benchmark writes one JSON object per line (one model x item row).
        return pd.read_json(results_path, lines=True)
    if suffix == ".csv":
        return pd.read_csv(results_path)
    raise ValueError(
        f"Unsupported results format {suffix!r}; expected .jsonl, .json, or .csv"
    )


def load_responses(results_path: Path) -> pd.DataFrame:
    """Load and validate one scored response per model and benchmark item."""
    results = _read_results(results_path)
    missing_columns = REQUIRED_COLUMNS.difference(results.columns)
    if missing_columns:
        raise ValueError(
            "Results file is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    if results.empty:
        raise ValueError("Results file contains no responses")

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


def select_models(results: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Restrict responses to the requested subset of model respondents."""
    available = set(results["model_name"].unique())
    requested = set(models)
    unknown = requested.difference(available)
    if unknown:
        raise ValueError(
            "Requested models are not present in the results file: "
            + ", ".join(sorted(unknown))
            + f"; available models are: {', '.join(sorted(available))}"
        )
    if len(requested) < 2:
        raise ValueError(
            "IRT fitting requires at least two model respondents; "
            f"--models selected {len(requested)}"
        )
    return results[results["model_name"].isin(requested)].copy()


def estimate_rasch(
    results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate item difficulties and model capabilities from scored results."""
    item_order = results["benchmark_item_id"].drop_duplicates().tolist()
    model_order = results["model_name"].drop_duplicates().tolist()
    if len(model_order) < 2:
        raise ValueError("IRT fitting requires responses from at least two models")

    # items-by-respondents, with NaN for any model that never answered an item.
    response_frame = results.pivot(
        index="benchmark_item_id",
        columns="model_name",
        values="perfect_match",
    ).reindex(index=item_order, columns=model_order)

    # GIRTH expects an items-by-respondents integer matrix. Tag the unanswered
    # cells (NaN) so the estimators skip them, then cast to int so all-0/all-1
    # rows aren't mistaken for boolean index masks.
    response_matrix = tag_missing_data(
        response_frame.to_numpy(dtype=float), VALID_RESPONSES
    ).astype(int)
    estimates = rasch_mml(response_matrix, discrimination=DISCRIMINATION)
    difficulties = np.asarray(estimates["Difficulty"], dtype=float)
    capabilities = np.asarray(
        ability_eap(response_matrix, difficulties, DISCRIMINATION), dtype=float
    )

    # Rates are over answered items only, so a model that skipped an item is not
    # penalised as if it had answered incorrectly.
    answered_by_item = response_frame.notna().sum(axis=1).to_numpy(dtype=int)
    correct_by_item = response_frame.sum(axis=1).to_numpy(dtype=int)
    item_results = pd.DataFrame(
        {
            "benchmark_item_id": item_order,
            "difficulty_logit_mml": difficulties,
            "models_correct": correct_by_item,
            "models_answered": answered_by_item,
            "models_total": len(model_order),
            "response_rate": correct_by_item / answered_by_item,
        }
    )
    item_results["difficulty_estimate_status"] = np.select(
        [
            item_results["models_correct"] == item_results["models_answered"],
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

    answered_by_model = response_frame.notna().sum(axis=0).to_numpy(dtype=int)
    correct_by_model = response_frame.sum(axis=0).to_numpy(dtype=int)
    model_results = pd.DataFrame(
        {
            "model_name": model_order,
            "capability_logit_eap": capabilities,
            "items_correct": correct_by_model,
            "items_answered": answered_by_model,
            "items_total": len(item_order),
            "raw_accuracy": correct_by_model / answered_by_model,
        }
    )
    model_results["capability_rank"] = (
        model_results["capability_logit_eap"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    return item_results, model_results


def output_paths(
    results_path: Path, output_dir: Path, suffix: str = ""
) -> tuple[Path, Path]:
    """Build descriptive output paths from the input filename."""
    stem = f"{results_path.stem}{suffix}"
    return (
        output_dir / f"{stem}_irt_items.csv",
        output_dir / f"{stem}_irt_models.csv",
    )


def main() -> None:
    """Fit the model and write item and model parameter CSVs."""
    args = parse_args()
    results = load_responses(args.results_path)

    # A subset run gets its own filenames (e.g. ``_4models``) so it never
    # overwrites the full-panel outputs.
    suffix = ""
    if args.models:
        results = select_models(results, args.models)
        suffix = f"_{results['model_name'].nunique()}models"

    item_results, model_results = estimate_rasch(results)

    output_dir = args.output_dir or args.results_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    items_path, models_path = output_paths(args.results_path, output_dir, suffix)
    item_results.to_csv(items_path, index=False)
    model_results.to_csv(models_path, index=False)

    n_models = len(model_results)
    missing_responses = int(
        (model_results["items_total"] - model_results["items_answered"]).sum()
    )
    boundary_items = item_results["difficulty_estimate_status"] != "interior"
    print(f"Fitted Rasch/1PL model to {len(item_results)} items and {n_models} models.")
    print(f"Item difficulties: {items_path}")
    print(f"Model capabilities: {models_path}")
    if missing_responses:
        print(
            f"NOTE: {missing_responses} model x item responses were missing and "
            "tagged as such; rates are computed over answered items only."
        )
    if n_models == 2:
        print(
            "WARNING: With only two model respondents, item response rates and "
            "difficulty estimates can take only three distinct levels."
        )
    if boundary_items.any():
        print(
            "WARNING: "
            f"{int(boundary_items.sum())} items were answered identically by all "
            "answering models; their difficulty estimates are optimizer bounds, "
            "not finite point estimates."
        )


if __name__ == "__main__":
    main()
