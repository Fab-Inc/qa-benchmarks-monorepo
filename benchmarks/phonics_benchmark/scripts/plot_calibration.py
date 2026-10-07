"""Calibration of the predicted vs actual non-decodable ratio, per hardware tier.

For every scored (model, item) the actual ratio is (TP + FN) / vocabulary size and
the predicted ratio is (TP + FP) / vocabulary size. Rows are grouped into ten
equal-width bins of actual ratio (0-10%, ..., 90-100%) and each bin plots the mean
actual against the mean predicted ratio over the selected models in the tier.
Points below the diagonal mean the models under-flag; above, they over-flag.
Writes error_analysis/calibration.{csv,png} next to the hardware-tier outputs.

How to run (from the repo root). Run analyze_phonics_hardware_tiers.py on the same
--root first, since this script reads its hardware_tiers_top{N}_models.csv:

    uv run --frozen --package phonics-benchmark python \\
      benchmarks/phonics_benchmark/scripts/plot_calibration.py \\
      --root outputs/phonics-mixture-20260922/selections/beta1-8-provisional

Options: --language (english or kiswahili), --top-n (default 10), --metric
(ranking metric, default pooled_f2) and --min-coverage (e.g. 0.9, to use the
coverage-filtered hardware-tier outputs).
"""

from __future__ import annotations

import argparse

import hardware_tier_errors as common
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

N_BINS = 10


def item_ratios(scored: pd.DataFrame) -> pd.DataFrame:
    """Actual and predicted non-decodable ratio for each (model, item)."""
    sizes = {c: scored[c].map(len) for c in common.WORD_COLUMNS}
    total = sum(sizes.values())
    ratios = scored[["model", "hardware_tier", "benchmark_item_id"]].copy()
    ratios["actual"] = (sizes["tp"] + sizes["fn"]) / total
    ratios["predicted"] = (sizes["tp"] + sizes["fp"]) / total
    return ratios[total > 0]


def calibration_table(ratios: pd.DataFrame) -> pd.DataFrame:
    # Upper edge is inclusive for the last bin only, so a ratio of 1.0 stays in 90-100%.
    edges = np.linspace(0, 1, N_BINS + 1)
    index = np.minimum(np.digitize(ratios.actual, edges[1:-1]), N_BINS - 1)
    ratios = ratios.assign(actual_bin=index)
    table = (
        ratios.groupby(["hardware_tier", "actual_bin"])
        .agg(
            mean_actual=("actual", "mean"),
            mean_predicted=("predicted", "mean"),
            n_rows=("actual", "size"),
            n_items=("benchmark_item_id", "nunique"),
        )
        .reset_index()
    )
    table["bin_low"] = edges[table.actual_bin]
    table["bin_high"] = edges[table.actual_bin + 1]
    return table


def plot(table: pd.DataFrame, title_suffix: str, path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.plot([0, 100], [0, 100], "--", color="#898781", linewidth=1.5, zorder=1)
    for tier in common.tiers_present(table):
        series = table[table.hardware_tier.eq(tier)].sort_values("actual_bin")
        ax.plot(
            100 * series.mean_actual,
            100 * series.mean_predicted,
            "o-",
            color=common.TIER_COLORS[tier],
            linewidth=2.5,
            markersize=7,
            label=tier,
        )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.grid(color="#e1e0d9", linewidth=0.8)
    common.style_axes(ax)
    ax.set_xlabel("Actual % non-decodable words")
    ax.set_ylabel("Predicted % non-decodable words")
    # Region labels are neutral so they are not read as belonging to one tier.
    ax.text(6, 91, "over-flagging\n(predicted > actual)", color="#52514e", fontsize=9)
    ax.text(58, 19, "under-flagging\n(predicted < actual)", color="#52514e", fontsize=9)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title(
        f"Calibration: predicted vs actual non-decodable ratio\n{title_suffix}",
        fontweight="bold",
    )
    common.finish(
        fig,
        path,
        "Restricted to the highest-ranked models per hardware tier, per "
        "hardware_tiers_top_n_models.csv.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(parser)
    args = parser.parse_args()
    models = common.load_top_models(
        args.root, args.language, args.top_n, args.metric, args.min_coverage
    )
    scored = common.load_scored_items(args.root, args.language, models)
    table = calibration_table(item_ratios(scored))
    table.insert(0, "language", args.language)
    out = common.tier_dir(args.root, args.metric, args.min_coverage) / "error_analysis"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "calibration.csv", index=False)
    plot(table, common.tier_title(models), out / "calibration.png")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
