"""LKB-style pooled F1 leaderboards using the report's saved estimates and CIs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import analyze_phonics_hardware_tiers as hardware
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from phonics_benchmark.labeling import get_vocabulary_set
from plot_report_f2_leaderboards import (
    SURFACE,
    digest,
    eligible_models,
    load_plot_table,
    plot_leaderboard,
    read_csv,
)
from run_f2_analysis import inspect_dataset, verify_merge


def main(
    root: Path,
    min_items: int = 700,
    min_coverage: float | None = None,
    hardware_tiers: bool = False,
):
    if min_items < 1:
        raise ValueError("min_items must be positive")
    suffix = hardware.coverage_suffix(min_coverage)
    root = root.resolve()
    merge_before = verify_merge(root)
    datasets = {lang: inspect_dataset(root / lang) for lang in ("english", "kiswahili")}
    inputs = [
        root / "plot_metadata" / name
        for name in ("models.csv", "providers.csv", "display_overrides.json")
    ]
    for language in ("english", "kiswahili"):
        inputs.extend(
            root / language / name
            for name in (
                "report/summary_by_model.csv",
                "report/bootstrap_ci.csv",
                "coverage_by_model.csv",
            )
        )
        inputs.append(root / language / "benchmark_items.jsonl")
    before = {str(path.relative_to(root)): digest(path) for path in inputs}
    outputs, details, summaries = [], {}, []
    for language in ("english", "kiswahili"):
        run = root / language
        table, colors = load_plot_table(run, root / "plot_metadata")
        table = table.drop(columns=["score", "ci_low", "ci_high"])
        summary = read_csv(run / "report/summary_by_model.csv").rename(
            columns={"model_name": "model"}
        )
        ci = read_csv(run / "report/bootstrap_ci.csv")
        values = summary[["model", "pooled_f1"]].merge(
            ci[["model", "pooled_f1", "pooled_f1_ci_low", "pooled_f1_ci_high"]],
            on="model",
            how="outer",
            validate="one_to_one",
            suffixes=("", "_bootstrap"),
        )
        np.testing.assert_allclose(
            values.pooled_f1, values.pooled_f1_bootstrap, rtol=0, atol=1e-14
        )
        table = table.merge(
            values.drop(columns="pooled_f1_bootstrap"),
            on="model",
            validate="one_to_one",
        )
        table = table.rename(
            columns={
                "pooled_f1": "score",
                "pooled_f1_ci_low": "ci_low",
                "pooled_f1_ci_high": "ci_high",
            }
        )
        if (
            not table[["score", "ci_low", "ci_high"]]
            .apply(lambda c: c.between(0, 1))
            .all()
            .all()
        ):
            raise ValueError("Undefined or out-of-range pooled F1 estimates")
        if not table.ci_low.le(table.ci_high).all():
            raise ValueError("Reversed interval")
        table = table.sort_values(
            ["score", "model"], ascending=[False, True]
        ).reset_index(drop=True)
        size, gold_note = datasets[language]
        required_items = hardware.minimum_scorable_items(size, min_items, min_coverage)
        table["minimum_scorable_items"] = required_items
        table["passes_minimum_items"] = table.scorable_items.ge(required_items)
        table["minimum_coverage_exclusive"] = min_coverage
        summaries.append(table.assign(language=language))
        table = eligible_models(table, required_items)
        if table.empty:
            raise ValueError(f"No eligible {language} models")
        items = pd.read_json(run / "benchmark_items.jsonl", lines=True)
        vocab = get_vocabulary_set(items, "en" if language == "english" else "sw")
        gold = items.nondecodable_words.map(lambda words: {w.lower() for w in words})
        if any(g - v for g, v in zip(gold, vocab, strict=True)):
            raise ValueError("Gold words absent from item vocabulary")
        n_gold, n_vocab = gold.map(len).sum(), vocab.map(len).sum()
        baseline = 2 * n_gold / (n_gold + n_vocab)
        table["all_words_full_dataset_pooled_f1"] = baseline
        fig = plot_leaderboard(table, colors, language)
        ax = fig.axes[0]
        ax.set_xlabel("Pooled F1")
        label = f"≥{required_items:,} evaluated items"
        if min_coverage is not None:
            label = f">{100 * min_coverage:g}% coverage · " + label
        ax.set_title(
            f"{language.title()} decodability · Pooled F1\nLeaderboard · {label}",
            loc="left",
            fontsize=12,
            fontweight="semibold",
            pad=35,
        )
        for text in ax.texts:
            if text.get_text() == "F2":
                text.set_text("F1")
            elif text.get_text().startswith("Items = scorable"):
                text.set_text(
                    "Items = scorable responses; coverage varies. "
                    f"Dashed line = full-set all-words baseline ({baseline:.1%})."
                )
        ax.axvline(baseline, color="#454545", ls="--", lw=1, zorder=5)
        ax.annotate(
            gold_note,
            (0, 0),
            xytext=(0, -57),
            xycoords="axes fraction",
            textcoords="offset points",
            fontsize=8,
            color="#52514e",
            va="top",
        )
        suffixes = (suffix,) if suffix else ("", f"_min{min_items}")
        try:
            for output_suffix in suffixes:
                stem = run / "report" / f"pooled_f1_leaderboard{output_suffix}"
                for extension in ("png", "pdf"):
                    path = stem.with_suffix(f".{extension}")
                    fig.savefig(path, dpi=250, bbox_inches="tight", facecolor=SURFACE)
                    outputs.append(path)
                csv_path = stem.with_suffix(".csv")
                table.to_csv(csv_path, index=False)
                pd.testing.assert_frame_equal(
                    read_csv(csv_path)[["score", "ci_low", "ci_high"]],
                    table[["score", "ci_low", "ci_high"]],
                    check_exact=True,
                )
                outputs.append(csv_path)
        finally:
            plt.close(fig)
        details[language] = {
            "n_models": len(table),
            "target_items": size,
            "minimum_scorable_items": required_items,
            "all_words_baseline": baseline,
            "gold_note": gold_note,
        }
        print(
            f"{language}: {len(table)} pooled F1 scores and CIs preserved exactly; "
            f"all-words baseline {baseline:.2%}"
        )
    all_models = pd.concat(summaries, ignore_index=True)
    all_models.to_csv(root / f"audit/f1_model_eligibility{suffix}.csv", index=False)
    all_models[all_models.passes_minimum_items].to_csv(
        root / f"pooled_f1_summary_by_language_and_model{suffix}.csv",
        index=False,
    )
    if hardware_tiers:
        hardware.main(
            root,
            top_n=10,
            denominators={lang: size for lang, (size, _) in datasets.items()},
            metric="pooled_f1",
            gold_note=datasets["english"][1],
            min_items=min_items,
            min_coverage=min_coverage,
        )
    if verify_merge(root) != merge_before:
        raise ValueError("Merged input manifest changed")
    after = {str(path.relative_to(root)): digest(path) for path in inputs}
    if before != after:
        raise ValueError("Source reports or metadata changed")
    manifest = {
        "script_sha256": digest(Path(__file__)),
        "metric": "pooled_f1",
        "minimum_scorable_items": min_items,
        "minimum_coverage_exclusive": min_coverage,
        "eligibility_comparison": (
            "scorable_items >= min_items and "
            "scorable_items / expected_items > min_coverage"
            if min_coverage is not None
            else "scorable_items >= min_items"
        ),
        "merged_inputs_verified": True,
        "hardware_tiers_generated": hardware_tiers,
        "input_sha256": before,
        "output_sha256": {
            str(path.relative_to(root)): digest(path) for path in outputs
        },
        "scores_recomputed": False,
        "bootstrap_recomputed": False,
        "plotted_bounds_equal_saved_bounds": True,
        "datasets": details,
    }
    (root / f"audit/f1_plot_manifest{suffix}.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--min-items", type=int, default=700)
    parser.add_argument(
        "--min-coverage",
        type=float,
        help="Require coverage > this fraction (e.g. 0.8); save separate outputs.",
    )
    parser.add_argument("--hardware-tiers", action="store_true")
    args = parser.parse_args()
    if args.min_items < 1:
        parser.error("--min-items must be positive")
    if args.min_coverage is not None and not 0 <= args.min_coverage < 1:
        parser.error("--min-coverage must be in [0, 1)")
    main(args.root, args.min_items, args.min_coverage, args.hardware_tiers)
