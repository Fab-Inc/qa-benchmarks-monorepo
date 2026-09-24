"""Score a full local merge or selection and produce F2 leaderboards/tier plots.

Run from the repository root with uv run --frozen --package phonics-benchmark
python benchmarks/phonics_benchmark/scripts/run_f2_analysis.py --root PATH.
Reads local inputs only; writes derived reports, metadata and audits under PATH.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from hashlib import sha256
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import analyze_phonics_hardware_tiers as hardware
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plot_report_f2_leaderboards as leaderboard
from report_on_benchmark_run import report_on_run

DEFAULT_METADATA = Path(__file__).resolve().parent / "f2_plot_metadata"
LANGUAGES = ("english", "kiswahili")


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def verify_merge(root: Path) -> dict:
    """Verify the merger/selector manifest before scoring or writing reports."""
    manifest = json.loads((root / "artifact_manifest.json").read_text())
    for relative, expected in manifest["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"Manifest path leaves the input root: {relative}")
        if digest(path) != expected:
            raise ValueError(f"Merged input has changed: {relative}")
    return manifest


def inspect_dataset(run: Path) -> tuple[int, str]:
    items_path = run / "benchmark_items.jsonl"
    items = pd.read_json(items_path, lines=True)
    ids = items.benchmark_item_id
    if items.empty or not ids.is_unique:
        raise ValueError(f"Expected non-empty unique item IDs in {run}")
    config = json.loads((run / "config.json").read_text())["resolved_config"]
    configured_path = Path(config["items_path"])
    if not configured_path.is_absolute():
        configured_path = run / configured_path
    if not configured_path.is_file() or digest(configured_path) != digest(items_path):
        raise ValueError(
            f"Config must resolve to this merged dataset in {run}; "
            "rebuild the merge/selection locally if its old absolute "
            "path is unavailable."
        )
    if config.get("items_sha256") != digest(items_path):
        raise ValueError(f"Dataset hash disagrees with config in {run}")
    coverage = pd.read_csv(run / "coverage_by_model.csv")
    if not coverage.expected_items.eq(len(items)).all():
        raise ValueError(f"Coverage denominator disagrees with dataset size in {run}")
    provenance = pd.read_json(run / "item_provenance.jsonl", lines=True)
    if not provenance.benchmark_item_id.is_unique or set(
        provenance.benchmark_item_id
    ) != set(ids):
        raise ValueError(f"Gold provenance disagrees with dataset IDs in {run}")
    statuses = Counter(provenance.gold_status)
    details = "; ".join(
        f"{n:,} {status} items" for status, n in sorted(statuses.items())
    )
    return len(items), f"{run.name.title()} gold: {details}."


def validate_scores(run: Path, min_items: int = 700) -> pd.DataFrame:
    table, _ = leaderboard.load_plot_table(run, run.parent / "plot_metadata")
    counts = pd.read_csv(
        run / "report/results_with_scoring.csv",
        float_precision="round_trip",
        usecols=["model", "benchmark_item_id", "tp_count", "fp_count", "fn_count"],
    )
    if counts.duplicated(["model", "benchmark_item_id"]).any():
        raise ValueError("Duplicate scored model/item pairs")
    coverage = leaderboard.read_csv(run / "coverage_by_model.csv").set_index("model")
    actual = counts.groupby("model").size().reindex(coverage.index, fill_value=0)
    if not actual.equals(coverage.scorable_items):
        raise ValueError("Scoring counts disagree with merge coverage")
    grouped = counts.groupby("model")[["tp_count", "fp_count", "fn_count"]].sum()
    score = (
        5
        * grouped.tp_count
        / (5 * grouped.tp_count + 4 * grouped.fn_count + grouped.fp_count)
    )
    np.testing.assert_allclose(
        table.set_index("model").score.sort_index(),
        score.sort_index(),
        rtol=0,
        atol=1e-14,
    )
    table["minimum_scorable_items"] = min_items
    table["passes_minimum_items"] = table.scorable_items.ge(min_items)
    return table


def save_leaderboards(
    run: Path,
    table: pd.DataFrame,
    gold_note: str,
    min_items: int = 700,
    *,
    output_suffix: str | None = None,
    eligibility_label: str | None = None,
) -> None:
    _, colors = leaderboard.load_plot_table(run, run.parent / "plot_metadata")
    selected = leaderboard.eligible_models(table, min_items)
    label = eligibility_label or f"≥{min_items:,} evaluated items"
    suffixes = ("", f"_min{min_items}") if output_suffix is None else (output_suffix,)
    for suffix in suffixes:
        if selected.empty:
            raise ValueError(f"No models qualify for {label} in {run.name}")
        fig = leaderboard.plot_leaderboard(selected, colors, run.name)
        ax = fig.axes[0]
        ax.set_title(
            f"{run.name.title()} decodability · Pooled F2\nLeaderboard · {label}",
            loc="left",
            fontsize=12,
            fontweight="semibold",
            color=leaderboard.INK_PRIMARY,
            pad=35,
        )
        ax.annotate(
            gold_note,
            (0, 0),
            xytext=(0, -57),
            xycoords="axes fraction",
            textcoords="offset points",
            fontsize=8,
            color=leaderboard.INK_SECONDARY,
            va="top",
        )
        stem = run / "report" / f"pooled_f2_leaderboard{suffix}"
        try:
            for extension in ("png", "pdf"):
                fig.savefig(
                    stem.with_suffix(f".{extension}"),
                    dpi=300,
                    bbox_inches="tight",
                    facecolor=leaderboard.SURFACE,
                )
        finally:
            plt.close(fig)
        selected.to_csv(stem.with_suffix(".csv"), index=False)
        written = leaderboard.read_csv(stem.with_suffix(".csv"))
        pd.testing.assert_frame_equal(
            written[["score", "ci_low", "ci_high"]],
            selected[["score", "ci_low", "ci_high"]].reset_index(drop=True),
            check_exact=True,
        )


def run_analysis(
    root: Path,
    metadata: Path,
    n_bootstrap: int,
    seed: int,
    min_items: int = 700,
    min_coverage: float | None = None,
) -> dict:
    if min_items < 1:
        raise ValueError("min_items must be positive")
    suffix = hardware.coverage_suffix(min_coverage)
    root, metadata = root.resolve(), metadata.resolve()
    before = verify_merge(root)
    datasets = {lang: inspect_dataset(root / lang) for lang in LANGUAGES}
    local_metadata = root / "plot_metadata"
    local_metadata.mkdir(exist_ok=True)
    for name in (
        "models.csv",
        "providers.csv",
        "display_overrides.json",
        "provenance.json",
    ):
        source, destination = metadata / name, local_metadata / name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
    summaries, validation = [], {}
    for language, (size, gold_note) in datasets.items():
        report_on_run(
            run_id=language,
            output_root=root,
            inspect_sample_size=None,
            n_bootstrap=n_bootstrap,
            ci_level=0.95,
            seed=seed,
        )
        required_items = hardware.minimum_scorable_items(size, min_items, min_coverage)
        table = validate_scores(root / language, required_items)
        if min_coverage is not None:
            table["minimum_coverage_exclusive"] = min_coverage
        save_leaderboards(
            root / language,
            table,
            gold_note,
            required_items,
            output_suffix=suffix or None,
            eligibility_label=(
                f">{100 * min_coverage:g}% coverage · "
                f"≥{required_items:,} evaluated items"
                if min_coverage is not None
                else None
            ),
        )
        summaries.append(table.assign(language=language))
        validation[language] = {
            "target_items": size,
            "models_scored": len(table),
            "models_meeting_minimum_items": int(table.passes_minimum_items.sum()),
            "minimum_scorable_items": required_items,
            "gold_note": gold_note,
        }
    hardware.main(
        root,
        top_n=10,
        denominators={lang: data[0] for lang, data in datasets.items()},
        metric="pooled_f2",
        gold_note=datasets["english"][1],
        min_items=min_items,
        min_coverage=min_coverage,
    )
    hardware_dir = root / "hardware_tiers/pooled_f2"
    if suffix:
        hardware_dir = hardware_dir / suffix.lstrip("_")
    for language in LANGUAGES:
        audit = leaderboard.read_csv(
            hardware_dir / f"hardware_tiers_{language}_audit.csv"
        )
        included = audit[audit.included]
        if not set(included.hardware_tier) <= {"Mobile + Tablet", "Cloud"}:
            raise ValueError("Excluded hardware tier entered comparison")
        if not included.passes_coverage.all():
            raise ValueError("Ineligible model entered hardware comparison")
        validation[language]["hardware_models"] = (
            included.groupby("hardware_tier").size().to_dict()
        )
    all_models = pd.concat(summaries, ignore_index=True)
    all_models.to_csv(root / f"audit/f2_model_eligibility{suffix}.csv", index=False)
    all_models[all_models.passes_minimum_items].to_csv(
        root / f"pooled_f2_summary_by_language_and_model{suffix}.csv", index=False
    )
    if verify_merge(root) != before:
        raise ValueError("Input manifest changed during analysis")
    scripts = [
        Path(__file__),
        Path(hardware.__file__),
        Path(leaderboard.__file__),
        Path(__file__).with_name("report_on_benchmark_run.py"),
    ]
    record = {
        "metric": "pooled_f2",
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "ci_level": 0.95,
        "minimum_scorable_items": min_items,
        "minimum_coverage_exclusive": min_coverage,
        "eligibility_comparison": (
            "scorable_items >= min_items and "
            "scorable_items / expected_items > min_coverage"
            if min_coverage is not None
            else "scorable_items >= min_items"
        ),
        "input_manifest_sha256": digest(root / "artifact_manifest.json"),
        "scripts_sha256": {p.name: digest(p) for p in scripts},
        "metadata_sha256": {
            p.name: digest(p) for p in local_metadata.iterdir() if p.is_file()
        },
        "validation": validation,
        "merged_inputs_unchanged": True,
    }
    (root / f"audit/f2_analysis_run{suffix}.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print(json.dumps(validation, indent=2))
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Full merge or selected subset parent containing english/ and kiswahili/.",
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--min-items",
        type=int,
        default=700,
        help="Inclusive minimum of successfully scored items per model.",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        help="Require coverage > this fraction (e.g. 0.9); save separate outputs.",
    )
    args = parser.parse_args()
    if args.n_bootstrap < 1:
        parser.error(
            "--n-bootstrap must be positive: plots require confidence intervals"
        )
    if args.min_items < 1:
        parser.error("--min-items must be positive")
    if args.min_coverage is not None and not 0 <= args.min_coverage < 1:
        parser.error("--min-coverage must be in [0, 1)")
    run_analysis(
        args.root,
        args.metadata,
        args.n_bootstrap,
        args.seed,
        args.min_items,
        args.min_coverage,
    )


if __name__ == "__main__":
    main()
