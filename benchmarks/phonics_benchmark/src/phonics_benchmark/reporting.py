"""Score local runs and export reusable report tables, without merge prerequisites."""

from __future__ import annotations

import json
import os
from collections import Counter
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from phonics_benchmark import balanced_accuracy as ba
from phonics_benchmark.bootstrap import (
    bootstrap_metric_cis,
    summarize_balanced_accuracy,
)
from phonics_benchmark.parsing import parse_word_list
from phonics_benchmark.results import CALL_STATUS_SUCCESS, load_results
from phonics_benchmark.scoring import add_per_item_confusion_matrix, summarize_by_model
from phonics_benchmark.utils import require_columns

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "outputs"

INSPECTION_RANDOM_STATE = 42

# Item metadata joined onto each response row. `nondecodable_words` is the gold
# answer; the rest are the difficulty columns the breakdowns group by.
ITEM_METADATA_COLUMNS = [
    "benchmark_item_id",
    "student_knowledge",
    "content",
    "nondecodable_words",
    "nondecodable_ratio",
    "num_nondecodable_words",
    "num_total_words",
]

# Enough to judge a response by eye without the noise of the full row.
INSPECTION_COLUMNS = [
    "run_id",
    "benchmark_item_id",
    "model",
    "prompt",
    "latency_seconds",
    "tokens_used_completion",
    "content",
    "student_knowledge",
    "nondecodable_words",
    "response_parsed",
    "tp_count",
    "fp_count",
    "fn_count",
    "tn",
    "perfect_match",
]


def resolve_run_dir(run_id: str, output_root: Path) -> Path:
    """Return the run's output directory, checking it looks like one."""
    run_dir = output_root / run_id
    if not run_dir.is_dir():
        raise typer.BadParameter(f"Run directory does not exist: {run_dir}")
    if not (run_dir / "responses").is_dir():
        raise typer.BadParameter(
            f"{run_dir} has no responses/ subdirectory — is it a run directory?"
        )
    return run_dir


def read_run_snapshot(run_dir: Path) -> dict:
    """Return the ``resolved_config`` block of a run's config snapshot."""
    snapshot_path = run_dir / "config.json"
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise typer.BadParameter(
            f"Missing {snapshot_path}; the run's language and item set are read "
            "from it. Was this run written by an older version of the runner?"
        ) from None
    return snapshot["resolved_config"]


def load_run_responses(run_dir: Path) -> pd.DataFrame:
    """Concatenate every model's responses JSONL and parse each raw response."""
    response_file_paths = sorted((run_dir / "responses").glob("*.jsonl"))
    if not response_file_paths:
        raise typer.BadParameter(f"No responses/*.jsonl found in {run_dir}")

    frames = [load_results(path) for path in response_file_paths]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise typer.BadParameter(f"Every responses file in {run_dir} was empty")

    results = pd.concat(frames, ignore_index=True)

    parsed = results["output"].apply(parse_word_list)
    results["response_parsed"] = [result.words for result in parsed]
    results["parse_ok"] = [result.parse_ok for result in parsed]

    return results


@require_columns(["call_status", "parse_ok"])
def split_scorable_responses(
    responses: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split loaded responses into included vs. excluded rows for scoring.

    Two types of rows are excluded:

    * ``call_status`` is not ``success``
    * ``parse_ok`` is False

    Returns ``(scorable, excluded)`` with ``excluded`` having an ``exclusion_reason``
    column.
    """
    call_failed = responses["call_status"].ne(CALL_STATUS_SUCCESS)
    parse_failed = ~responses["parse_ok"] & ~call_failed

    reason = pd.Series(index=responses.index, dtype="object")
    reason[parse_failed] = "parse_failed"
    reason[call_failed] = "call_failed"

    excluded = responses[reason.notna()].copy()
    excluded["exclusion_reason"] = reason[reason.notna()]
    scorable = responses[reason.isna()].copy()
    return scorable, excluded


def warn_about_exclusions(excluded: pd.DataFrame) -> None:
    """Print a per-model, per-reason count of the rows left out of scoring.

    Counts only: the excluded rows themselves stay in the run's own
    ``responses/*.jsonl``, so the report has no reason to copy them.
    """
    breakdown = (
        excluded.groupby(["model", "exclusion_reason"]).size().unstack(fill_value=0)
    )
    typer.secho(
        f"WARNING: excluded {len(excluded)} response row(s) from scoring.\n",
        fg=typer.colors.YELLOW,
        err=True,
    )
    typer.secho(breakdown.to_string(), fg=typer.colors.YELLOW, err=True)


def load_run_items(resolved_config: dict, run_dir: Path | None = None) -> pd.DataFrame:
    """Load the exact item set the run was executed against.

    Warns if the file's bytes don't hash to the ``items_sha256`` recorded at
    run time, which would mean the scores are being computed against different
    items than the models saw.
    """
    items_path = Path(resolved_config["items_path"])
    if not items_path.is_absolute() and run_dir is not None:
        items_path = run_dir / items_path
    if not items_path.is_file():
        raise typer.BadParameter(f"The run's items file no longer exists: {items_path}")

    recorded_sha256 = resolved_config.get("items_sha256")
    # actual sha256 from items.py:24
    actual_sha256 = sha256(items_path.read_bytes()).hexdigest()
    if recorded_sha256 and recorded_sha256 != actual_sha256:
        typer.secho(
            f"WARNING: {items_path} has changed since this run.\n"
            f"  recorded items_sha256: {recorded_sha256}\n"
            f"  current  items_sha256: {actual_sha256}",
            fg=typer.colors.YELLOW,
            err=True,
        )

    items = pd.read_json(items_path, lines=True)
    return items


def sample_items_for_inspection(
    scored_results: pd.DataFrame,
    sample_size: int,
) -> pd.DataFrame:
    """Return every model's row for a random sample of benchmark items."""
    item_ids = pd.Series(scored_results["benchmark_item_id"].unique())
    sampled_ids = item_ids.sample(
        min(sample_size, len(item_ids)), random_state=INSPECTION_RANDOM_STATE
    )
    sampled = scored_results[scored_results["benchmark_item_id"].isin(sampled_ids)]
    columns = [column for column in INSPECTION_COLUMNS if column in sampled.columns]
    return sampled[columns].sort_values(["benchmark_item_id", "model"])


def read_csv(path: Path) -> pd.DataFrame:
    """Retain saved float values when replotting or comparing reports."""
    return pd.read_csv(path, float_precision="round_trip")


def coverage_by_model(raw, excluded, items, models):
    """Count all configured models against the full target set, including absences."""
    index = pd.Index(sorted(set(models) | set(raw.model)), name="model")
    result = pd.DataFrame(index=index)
    result["expected_items"] = len(items)
    result["response_rows"] = raw.groupby("model").size().reindex(index, fill_value=0)
    for reason in ("call_failed", "parse_failed"):
        result[reason] = (
            excluded[excluded.exclusion_reason.eq(reason)]
            .groupby("model")
            .size()
            .reindex(index, fill_value=0)
        )
    result["scorable_items"] = (
        result.response_rows - result.call_failed - result.parse_failed
    )
    present = (
        raw.groupby("model").benchmark_item_id.nunique().reindex(index, fill_value=0)
    )
    result["missing_responses"] = len(items) - present
    result["coverage"] = result.scorable_items / len(items)
    return result.reset_index()


def display_metadata(models, directory: Path | None = None):
    """Use supplied display metadata, with readable defaults for unknown models."""
    frame = pd.DataFrame({"model": list(models)})
    frame["display_name"] = frame.model
    frame["provider"] = "Unspecified"
    colors = {"Unspecified": "#376b94"}
    if directory is not None:
        metadata = read_csv(directory / "models.csv").set_index("model_id")
        providers = read_csv(directory / "providers.csv").set_index("provider")
        if not metadata.index.is_unique or not providers.index.is_unique:
            raise ValueError("Metadata identifiers must be unique")
        overrides_path = directory / "display_overrides.json"
        if overrides_path.exists():
            for model, values in json.loads(overrides_path.read_text()).items():
                for key in ("display_name", "provider"):
                    metadata.loc[model, key] = values[key]
        for key in ("display_name", "provider"):
            frame[key] = frame.model.map(metadata[key]).fillna(frame[key])
        colors.update(providers.color.to_dict())
    frame["color"] = frame.provider.map(colors)
    if frame.color.isna().any():
        raise ValueError("Some providers lack colours")
    return frame, colors


def report_on_run(
    run_id: str,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    inspect_sample_size: int | None = None,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 0,
    *,
    output_dir: Path | None = None,
    metadata: Path | None = None,
    min_items: int = 1,
    min_coverage: float | None = None,
    make_plots: bool = True,
) -> Path:
    """Score a normal runner directory and save all metrics in a single report."""
    from phonics_benchmark.hardware import minimum_scorable_items

    if n_bootstrap < 0 or not 0 < ci_level < 1:
        raise ValueError("Invalid bootstrap count or confidence level")
    if inspect_sample_size is not None and inspect_sample_size < 1:
        raise ValueError("Inspection sample size must be positive")
    run = resolve_run_dir(run_id, Path(output_root))
    destination = Path(output_dir) if output_dir is not None else run / "report"
    config = read_run_snapshot(run)
    items = load_run_items(config, run)
    if (
        items.empty
        or items.benchmark_item_id.isna().any()
        or not items.benchmark_item_id.is_unique
    ):
        raise ValueError("Expected non-empty unique target item IDs")
    item_path = Path(config["items_path"])
    if not item_path.is_absolute():
        item_path = run / item_path
    vocabulary = ba.load_items(item_path, config["language"])
    required = minimum_scorable_items(len(items), min_items, min_coverage)
    sources = [
        run / "config.json",
        item_path,
        *sorted((run / "responses").glob("*.jsonl")),
    ]
    if metadata is not None:
        sources.extend(path for path in metadata.iterdir() if path.is_file())
    if (run / "item_provenance.jsonl").exists():
        sources.append(run / "item_provenance.jsonl")
    before = {
        str(path.resolve()): sha256(path.read_bytes()).hexdigest() for path in sources
    }
    raw = load_run_responses(run)
    if not raw.benchmark_item_id.isin(items.benchmark_item_id).all():
        raise ValueError("Unknown item IDs in responses")
    responses, excluded = split_scorable_responses(raw)
    if not excluded.empty:
        warn_about_exclusions(excluded)
    if responses.empty:
        raise ValueError(f"Nothing to score in {run}")
    if responses.duplicated(["model", "benchmark_item_id"]).any():
        raise ValueError("Duplicate scored model/item pairs")
    # Older runner datasets can omit derived difficulty metadata.
    for column, values in {
        "num_nondecodable_words": vocabulary.n_gold,
        "num_total_words": vocabulary.n_vocab,
        "nondecodable_ratio": vocabulary.n_gold / vocabulary.n_vocab,
    }.items():
        if column not in items:
            items[column] = items.benchmark_item_id.map(values)
    results = responses.merge(
        items[ITEM_METADATA_COLUMNS], on="benchmark_item_id", validate="many_to_one"
    )
    scored = add_per_item_confusion_matrix(results, config["language"])
    summary = summarize_by_model(scored)
    balanced, counts = summarize_balanced_accuracy(
        scored[
            [
                "model",
                "benchmark_item_id",
                "tp_count",
                "fp_count",
                "fn_count",
                "tn_count",
            ]
        ],
        vocabulary,
        n_bootstrap=n_bootstrap,
        seed=seed,
        ci_level=ci_level,
    )
    coverage = coverage_by_model(raw, excluded, items, config.get("models", []))
    combined = balanced.copy()
    if n_bootstrap:
        intervals = bootstrap_metric_cis(
            scored,
            n_bootstrap=n_bootstrap,
            ci_level=ci_level,
            seed=seed,
            item_ids=items.benchmark_item_id,
        )
        columns = [
            "model",
            *[c for c in intervals if c.startswith(("pooled_", "accuracy"))],
        ]
        combined = combined.merge(intervals[columns], on="model", validate="one_to_one")
    else:
        intervals = None
        combined = combined.merge(
            summary[["model_name", "accuracy", "pooled_f1", "pooled_f2"]].rename(
                columns={"model_name": "model"}
            ),
            on="model",
            validate="one_to_one",
        )
        for metric in ("pooled_f1", "pooled_f2"):
            combined[f"{metric}_ci_low"] = np.nan
            combined[f"{metric}_ci_high"] = np.nan
    display, _ = display_metadata(combined.model, metadata)
    combined = combined.merge(display, on="model", validate="one_to_one")
    combined = combined.merge(
        coverage.drop(columns=["expected_items", "coverage"]),
        on="model",
        validate="one_to_one",
    )
    if not combined.n_scorable.eq(combined.scorable_items).all():
        raise ValueError("Scoring counts disagree with response coverage")
    language = {"en": "english", "sw": "kiswahili"}.get(
        config["language"], config["language"]
    )
    gold_note = ""
    provenance_path = run / "item_provenance.jsonl"
    if provenance_path.exists():
        provenance = pd.read_json(provenance_path, lines=True)
        if not provenance.benchmark_item_id.is_unique or set(
            provenance.benchmark_item_id
        ) != set(items.benchmark_item_id):
            raise ValueError("Gold provenance disagrees with target item IDs")
        statuses = Counter(provenance.gold_status)
        gold_note = (
            f"{language.title()} gold: "
            + "; ".join(
                f"{n:,} {status} items" for status, n in sorted(statuses.items())
            )
            + "."
        )
    combined["language"] = language
    combined["minimum_scorable_items"] = required
    combined["passes_coverage_filter"] = combined.n_scorable.ge(required)
    destination.mkdir(parents=True, exist_ok=True)
    for name, frame in {
        "results_with_scoring.csv": scored,
        "summary_by_model.csv": summary,
        "balanced_accuracy_summary.csv": balanced,
        "balanced_accuracy_item_counts.csv": counts[
            ["model", "benchmark_item_id", *ba.COUNTS]
        ],
        "coverage_by_model.csv": coverage,
        "model_summary_all_scored.csv": combined,
        "model_summary_eligible.csv": combined[combined.passes_coverage_filter],
    }.items():
        frame.to_csv(destination / name, index=False)
    baseline_models, baseline_counts = compare_with_chance(
        combined[combined.passes_coverage_filter]
    )
    baseline_models.to_csv(destination / "baseline_comparison.csv", index=False)
    baseline_counts.to_csv(destination / "baseline_counts.csv", index=False)
    if intervals is not None:
        intervals.to_csv(destination / "bootstrap_ci.csv", index=False)
    elif (destination / "bootstrap_ci.csv").exists():
        (destination / "bootstrap_ci.csv").unlink()
    if inspect_sample_size:
        sample_items_for_inspection(scored, inspect_sample_size).to_csv(
            destination / "sampled_items_for_inspection.csv",
            index=False,
        )
    record = {
        "run_id": config["run_id"],
        "language": language,
        "expected_items": len(items),
        "n_bootstrap": n_bootstrap,
        "ci_level": ci_level,
        "seed": seed,
        "bootstrap_unit": "item",
        "bootstrap_population": "all target items",
        "bootstrap_shared_across_models_and_metrics": True,
        "min_items": min_items,
        "min_coverage": min_coverage,
        "minimum_scorable_items": required,
        "input_hashes": before,
        "gold_note": gold_note,
        "all_words_pooled_f1": float(
            2
            * vocabulary.n_gold.sum()
            / (vocabulary.n_gold.sum() + vocabulary.n_vocab.sum())
        ),
    }
    if any(
        sha256(Path(path).read_bytes()).hexdigest() != digest
        for path, digest in before.items()
    ):
        raise ValueError("Run inputs changed while generating reports")
    (destination / "report_config.json").write_text(json.dumps(record, indent=2) + "\n")
    write_report_index(destination, destination, record, [])
    if make_plots:
        plot_report(destination)
    return destination


def plot_report(report_dir: Path, *, output_dir: Path | None = None) -> list[Path]:
    """Render saved tables only; never load responses or call a bootstrap routine."""
    import matplotlib.pyplot as plt

    from phonics_benchmark.hardware import coverage_suffix
    from phonics_benchmark.plots import (
        METRIC_SPECS,
        plot_leaderboard,
        save_accuracy_by_nondecodable_count_plot,
    )
    from phonics_benchmark.scoring import accuracy_by_nondecodable_count

    report_dir = Path(report_dir)
    destination = Path(output_dir) if output_dir is not None else report_dir
    destination.mkdir(parents=True, exist_ok=True)
    config = json.loads((report_dir / "report_config.json").read_text())
    all_models = read_csv(report_dir / "model_summary_all_scored.csv")
    eligible = all_models[all_models.passes_coverage_filter].copy()
    colors = dict(zip(all_models.provider, all_models.color, strict=True))
    label = f"≥{config['minimum_scorable_items']:,} evaluated items"
    if config["min_coverage"] is not None:
        label = f">{100 * config['min_coverage']:g}% coverage · " + label
    outputs = []
    suffix = coverage_suffix(config["min_coverage"])
    for metric in METRIC_SPECS:
        table = eligible.drop(
            columns=["score", "ci_low", "ci_high"], errors="ignore"
        ).copy()
        table["score"] = table[metric]
        table["ci_low"] = table[f"{metric}_ci_low"]
        table["ci_high"] = table[f"{metric}_ci_high"]
        table["n_items"] = table.n_scorable
        table = (
            table[table.score.notna()]
            .sort_values(["score", "model"], ascending=[False, True])
            .reset_index(drop=True)
        )
        stem = destination / f"{metric}_leaderboard{suffix}"
        table.to_csv(stem.with_suffix(".csv"), index=False)
        if table.empty:
            continue
        fig = plot_leaderboard(
            table,
            colors,
            config["language"],
            metric=metric,
            eligibility_label=label,
            gold_note=config.get("gold_note", ""),
            baseline=config.get("all_words_pooled_f1")
            if metric == "pooled_f1"
            else None,
        )
        try:
            for extension in ("png", "pdf"):
                path = stem.with_suffix(f".{extension}")
                fig.savefig(path, dpi=150, bbox_inches="tight")
                outputs.append(path)
        finally:
            plt.close(fig)
    scored_path = report_dir / "results_with_scoring.csv"
    if scored_path.exists():
        scored = read_csv(scored_path)
        save_accuracy_by_nondecodable_count_plot(
            accuracy_by_nondecodable_count(scored),
            destination / "accuracy_by_nondecodable_count.png",
        )
    write_report_index(destination, report_dir, config, outputs)
    return outputs


def export_collection(reports, destination):
    """Combine already scored runs, retaining run identity and their coverage gates."""
    frames, links = [], []
    for report in reports:
        frame = read_csv(report / "model_summary_all_scored.csv")
        config = json.loads((report / "report_config.json").read_text())
        frame.insert(0, "run_id", config["run_id"])
        frames.append(frame)
        relative = Path(os.path.relpath(report, destination))
        links.append(f"- [{config['run_id']}]({relative}/README.md)")
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(destination / "model_summary_all_scored.csv", index=False)
    combined[combined.passes_coverage_filter].to_csv(
        destination / "model_summary_eligible.csv", index=False
    )
    (destination / "README.md").write_text(
        "# Benchmark reports\n\n[All scored models](model_summary_all_scored.csv) · "
        "[Eligible models](model_summary_eligible.csv)\n\n" + "\n".join(links) + "\n"
    )


def write_report_index(destination, source, config, figures):
    """Link plots to their source tables, including when they live elsewhere."""
    relative = Path(os.path.relpath(source, destination))
    links = [
        f"- [{path.stem}]({path.name})" for path in figures if path.suffix == ".png"
    ]
    (destination / "README.md").write_text(
        f"# {config['run_id']} report\n\n"
        f"[All scored models]({relative}/model_summary_all_scored.csv) · "
        f"[Eligible models]({relative}/model_summary_eligible.csv)\n\n"
        + "\n".join(links)
        + "\n"
    )


def compare_with_chance(eligible):
    """Retain the BA baseline audit using exact integer confusion products."""
    result = eligible[eligible.balanced_accuracy.notna()].copy()
    determinant = result.tp_count.astype("int64") * result.tn_count.astype(
        "int64"
    ) - result.fp_in_vocabulary.astype("int64") * result.fn_count.astype("int64")
    result["point_below_50"] = determinant.lt(0)
    result["point_at_50"] = determinant.eq(0)
    low, high = result.balanced_accuracy_ci_low, result.balanced_accuracy_ci_high
    result["entire_ci_below_or_at_50"] = high.le(0.5)
    summary = pd.DataFrame(
        [
            {
                "models": len(result),
                "below_50": int(determinant.lt(0).sum()),
                "at_50": int(determinant.eq(0).sum()),
                "below_or_at_50": int(determinant.le(0).sum()),
                "entire_ci_below_or_at_50": int(high.le(0.5).sum()),
                "ci_contains_50": int((low.le(0.5) & high.ge(0.5)).sum()),
                "entire_ci_above_50": int(low.gt(0.5).sum()),
            }
        ]
    )
    return result, summary
