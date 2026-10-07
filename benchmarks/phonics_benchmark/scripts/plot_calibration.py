"""Calibration of the predicted vs actual non-decodable ratio, per hardware tier.

For every scored (model, item) the actual ratio is (TP + FN) / vocabulary size and
the predicted ratio is (TP + FP) / vocabulary size. Rows are grouped into ten
equal-width bins of actual ratio (0-10%, ..., 90-100%) and each bin plots the mean
actual against the mean predicted ratio over the selected models in the tier.
Points below the diagonal mean the models under-flag; above, they over-flag.
Each line gets a shaded confidence band for the predicted ratio from an item-level
bootstrap (1,000 replicates, 95% by default; --n-bootstrap 0 turns it off). The band
reflects which benchmark items were drawn, not label error or the choice of top
models. Writes error_analysis/calibration_<language>.{csv,png,pdf} next to the
hardware-tier outputs.

How to run (from the repo root). Run report_on_benchmark_run.py --hardware-tiers on
the same --root first, since this script reads its hardware_tiers_top{N}_models.csv.
--root is the run folder that holds reports/ (reports/<language>/ and
reports/hardware_tiers/<metric>/). Only pass --min-coverage when the hardware-tier
outputs sit in a coverage_gt<N> subfolder of reports/hardware_tiers/<metric>/; the
gold leaderboard runs keep them directly in <metric>/, so omit it there:

    uv run --frozen --package phonics-benchmark python \\
      benchmarks/phonics_benchmark/scripts/plot_calibration.py \\
      --root $ROOT

Both languages are plotted unless --language english or --language kiswahili is
given. Other options: --top-n (default 10), --metric (ranking metric, default
pooled_f2), --min-coverage (e.g. 0.9, to use the coverage-filtered hardware-tier
outputs), and --n-bootstrap, --ci-level and --seed for the bands.
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
    total = sizes["tp"] + sizes["fp"] + sizes["fn"] + sizes["tn"]
    ratios = scored[["model", "hardware_tier", "benchmark_item_id"]].copy()
    ratios["actual"] = (sizes["tp"] + sizes["fn"]) / total
    ratios["predicted"] = (sizes["tp"] + sizes["fp"]) / total
    return ratios.loc[total > 0]


def calibration_table(
    ratios: pd.DataFrame,
    n_bootstrap: int = 0,
    ci_level: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """Mean actual and predicted ratio per (tier, actual-ratio bin).

    With ``n_bootstrap`` > 0 the table also has ``*_ci_low`` and ``*_ci_high`` columns
    for both means from resampling items. The point estimates stay the plain means.
    """
    # Upper edge is inclusive for the last bin only, so a ratio of 1.0 stays in 90-100%.
    edges = np.arange(N_BINS + 1) / N_BINS
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
    if n_bootstrap > 0:
        interval = mean_interval(ratios, n_bootstrap, ci_level, seed)
        table = table.merge(interval, on=["hardware_tier", "actual_bin"], how="left")
        table["n_bootstrap"] = n_bootstrap
        table["ci_level"] = ci_level
    return table


def mean_interval(
    ratios: pd.DataFrame, n_bootstrap: int, ci_level: float, seed: int
) -> pd.DataFrame:
    """Bootstrap intervals of both means for every (tier, bin).

    ``ratios`` must already have ``actual_bin``. An item's actual ratio comes from the
    gold labels, so it stays in one bin whichever model is scored, and a bin's mean is
    a ratio of sums over the drawn items. A bin no drawn item falls in is skipped.
    """
    items = sorted(ratios.benchmark_item_id.unique())
    tiers = common.tiers_present(ratios)
    bins = list(range(N_BINS))
    per_item = (
        ratios.assign(n_rows=1)
        .groupby(["benchmark_item_id", "hardware_tier", "actual_bin"])
        .agg(
            sum_actual=("actual", "sum"),
            sum_predicted=("predicted", "sum"),
            n_rows=("n_rows", "sum"),
        )
    )
    counts = common.item_count_array(per_item, items, tiers, bins)
    totals = common.bootstrap_totals(counts, n_bootstrap, seed)
    index = pd.MultiIndex.from_product(
        [tiers, bins], names=["hardware_tier", "actual_bin"]
    )
    columns = {}
    for position, name in enumerate(["mean_actual", "mean_predicted"]):
        means = common.safe_ratio(totals[..., position], totals[..., 2])
        low, high = common.percentile_interval(means, ci_level)
        columns[f"{name}_ci_low"] = low.ravel()
        columns[f"{name}_ci_high"] = high.ravel()
    return pd.DataFrame(columns, index=index).reset_index()


def plot(table: pd.DataFrame, title_suffix: str, ranking: str, path) -> None:
    has_ci = "mean_predicted_ci_low" in table
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.plot([0, 100], [0, 100], "--", color="#898781", linewidth=1.5, zorder=1)
    for tier in common.tiers_present(table):
        series = table[table.hardware_tier.eq(tier)].sort_values("actual_bin")
        color = common.TIER_COLORS[tier]
        if has_ci:
            # Vertical band at each bin's mean actual ratio.
            ax.fill_between(
                100 * series.mean_actual,
                100 * series.mean_predicted_ci_low,
                100 * series.mean_predicted_ci_high,
                color=color,
                alpha=0.25,
                linewidth=0,
                zorder=1,
            )
        ax.plot(
            100 * series.mean_actual,
            100 * series.mean_predicted,
            "o-",
            color=color,
            linewidth=2.5,
            markersize=7,
            label=tier,
            zorder=2,
        )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.grid(color="#e1e0d9", linewidth=0.8)
    common.style_axes(ax)
    ax.set_xlabel("Actual % non-decodable words")
    ax.set_ylabel("Predicted % non-decodable words")
    # Region labels are neutral so they are not read as belonging to one tier. They sit
    # in the corners, clear of the lines: under-flagging tiers can run along the lower
    # half, so the label is pinned to the bottom-right corner and the legend moves up.
    ax.text(
        4,
        80,
        "over-flagging\n(predicted > actual)",
        color="#52514e",
        fontsize=9,
        va="top",
    )
    ax.text(
        97,
        4,
        "under-flagging\n(predicted < actual)",
        color="#52514e",
        fontsize=9,
        ha="right",
        va="bottom",
    )
    ax.legend(frameon=False, loc="upper left")
    ax.set_title(
        f"Calibration: predicted vs actual non-decodable ratio\n{title_suffix}",
        fontweight="bold",
    )
    note = f"Restricted to the {ranking} per hardware tier."
    if has_ci:
        n_bootstrap, ci_level = table.n_bootstrap.iloc[0], table.ci_level.iloc[0]
        note += "\n" + common.ci_note(int(n_bootstrap), float(ci_level))
    common.finish(fig, path, note)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(parser)
    args = parser.parse_args()
    common.validate_bootstrap_args(parser, args)
    out = common.tier_dir(args.root, args.metric, args.min_coverage) / "error_analysis"
    out.mkdir(parents=True, exist_ok=True)
    for language in args.language:
        models = common.load_top_models(
            args.root, language, args.top_n, args.metric, args.min_coverage
        )
        scored = common.load_scored_items(args.root, language, models)
        table = calibration_table(
            item_ratios(scored), args.n_bootstrap, args.ci_level, args.seed
        )
        table.insert(0, "language", language)
        stem = f"calibration_{language}"
        table.to_csv(out / f"{stem}.csv", index=False)
        ranking = common.ranking_phrase(args.metric, language)
        plot(table, common.tier_title(models), ranking, out / f"{stem}.png")
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
