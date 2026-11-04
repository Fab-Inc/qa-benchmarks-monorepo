"""Pooled false positive or false negative rate by word length, per hardware tier.

--error fp: share of decodable words wrongly flagged as non-decodable, FP / (FP + TN).
--error fn: share of non-decodable words the model missed, FN / (FN + TP).

Counts are pooled over every selected model in a tier: each rate is the summed errors
divided by the summed words that could have been errors, not a mean of per-model
rates. A model that saw more words in a bin counts for more.
Each line gets a shaded confidence band from an item-level bootstrap (1,000
replicates, 95% by default; --n-bootstrap 0 turns it off).

Plots are saved to <root>/reports/hardware_tiers/<metric>/error_analysis/
{fp,fn}_rate_by_word_length_<language>.{csv,png,pdf}.

How to run (from the repo root). Run report_on_benchmark_run.py --hardware-tiers on
the same --root first, since this script reads its hardware_tiers_top{N}_models.csv.
--root is the run folder that holds reports/ (reports/<language>/ and
reports/hardware_tiers/<metric>/). Only pass --min-coverage when the hardware-tier
outputs sit in a coverage_gt<N> subfolder of reports/hardware_tiers/<metric>/; the
gold leaderboard runs keep them directly in <metric>/, so omit it there:

    uv run --frozen --package phonics-benchmark python \\
      benchmarks/phonics_benchmark/scripts/plot_error_rate_by_word_length.py \\
      --root $ROOT \\
      --error fp

Omit --error to make both plots. Both languages are plotted unless --language
english or --language kiswahili is given. Other options: --top-n (default 10),
--metric (ranking metric, default pooled_f2), --min-coverage (see above), and
--n-bootstrap, --ci-level and --seed for the bands.
"""

from __future__ import annotations

import argparse

import hardware_tier_errors as common
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# error -> (numerator outcome, denominator outcomes, axis label)
ERRORS = {
    "fp": ("fp", ("fp", "tn"), "false positive rate"),
    "fn": ("fn", ("fn", "tp"), "false negative rate"),
}
BIN_LABELS = [common.length_label(lo, hi) for lo, hi in common.LENGTH_BINS]


def rate_by_length(
    words: pd.DataFrame,
    error: str,
    n_bootstrap: int = 0,
    ci_level: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """Pooled error rate per (tier, length bin), with the counts behind it.

    With ``n_bootstrap`` > 0 the table also has ``rate_ci_low`` and ``rate_ci_high``
    from resampling items. The point estimate stays the plain pooled rate.
    """
    numerator, denominator, _ = ERRORS[error]
    pool = words[words.outcome.isin(denominator)]
    grouped = pool.groupby(["hardware_tier", "length_bin"])
    table = grouped.size().rename("opportunities").to_frame()
    table["errors"] = (
        pool[pool.outcome.eq(numerator)].groupby(["hardware_tier", "length_bin"]).size()
    )
    table["errors"] = table.errors.fillna(0).astype(int)
    table["rate"] = table.errors / table.opportunities
    table["n_models"] = grouped.model.nunique()
    table = table.reset_index()
    if n_bootstrap > 0:
        interval = rate_interval(words, error, n_bootstrap, ci_level, seed)
        table = table.merge(interval, on=["hardware_tier", "length_bin"], how="left")
        table["n_bootstrap"] = n_bootstrap
        table["ci_level"] = ci_level
    return table


def rate_interval(
    words: pd.DataFrame, error: str, n_bootstrap: int, ci_level: float, seed: int
) -> pd.DataFrame:
    """Bootstrap interval of the pooled rate for every (tier, length bin).

    The resampling population is every item with a scored word for these models,
    so items with no word in this rate's denominator still count as draws.
    """
    numerator, denominator, _ = ERRORS[error]
    items = sorted(words.benchmark_item_id.unique())
    tiers = common.tiers_present(words)
    pool = words[words.outcome.isin(denominator)].assign(
        errors=lambda d: d.outcome.eq(numerator).astype(int), opportunities=1
    )
    per_item = pool.groupby(["benchmark_item_id", "hardware_tier", "length_bin"])[
        ["errors", "opportunities"]
    ].sum()
    counts = common.item_count_array(per_item, items, tiers, BIN_LABELS)
    totals = common.bootstrap_totals(counts, n_bootstrap, seed)
    rates = common.safe_ratio(totals[..., 0], totals[..., 1])
    low, high = common.percentile_interval(rates, ci_level)
    index = pd.MultiIndex.from_product(
        [tiers, BIN_LABELS], names=["hardware_tier", "length_bin"]
    )
    interval = pd.DataFrame(
        {"rate_ci_low": low.ravel(), "rate_ci_high": high.ravel()}, index=index
    )
    return interval.reset_index()


def plot(
    table: pd.DataFrame,
    error: str,
    language: str,
    title_suffix: str,
    ranking: str,
    path,
) -> None:
    _, _, label = ERRORS[error]
    x = np.arange(len(BIN_LABELS))
    has_ci = "rate_ci_low" in table
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for tier in common.tiers_present(table):
        series = (
            table[table.hardware_tier.eq(tier)]
            .set_index("length_bin")
            .reindex(BIN_LABELS)
        )
        color = common.TIER_COLORS[tier]
        if has_ci:
            ax.fill_between(
                x,
                series.rate_ci_low,
                series.rate_ci_high,
                color=color,
                alpha=0.25,
                linewidth=0,
                zorder=1,
            )
        ax.plot(x, series.rate, "o-", color=color, linewidth=2.5, label=tier, zorder=2)
        last = series.rate.dropna()
        if not last.empty:
            ax.annotate(
                f"{100 * last.iloc[-1]:.1f}%",
                (x[BIN_LABELS.index(last.index[-1])], last.iloc[-1]),
                xytext=(8, 0),
                textcoords="offset points",
                color=color,
                fontsize=9,
                va="center",
            )
    ax.set_xticks(x, BIN_LABELS)
    common.percent_axis(ax)
    common.style_axes(ax)
    ax.set_xlabel("word length (letters)")
    ax.set_ylabel(f"{label} (%)")
    # Fixed position, shared with the calibration plot, so it never moves between plots.
    ax.legend(frameon=False, loc="upper left")
    ax.set_title(
        f"Pooled {label} by word length ({language.title()})\n{title_suffix}",
        fontweight="bold",
    )
    note = f"{error.upper()} rate pooled across the {ranking} in each hardware tier"
    if has_ci:
        n_bootstrap, ci_level = table.n_bootstrap.iloc[0], table.ci_level.iloc[0]
        note += "\n" + common.ci_note(int(n_bootstrap), float(ci_level))
    common.finish(fig, path, note)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(parser)
    parser.add_argument(
        "--error", nargs="+", choices=list(ERRORS), default=list(ERRORS)
    )
    args = parser.parse_args()
    common.validate_bootstrap_args(parser, args)
    out = common.tier_dir(args.root, args.metric, args.min_coverage) / "error_analysis"
    out.mkdir(parents=True, exist_ok=True)
    for language in args.language:
        models = common.load_top_models(
            args.root, language, args.top_n, args.metric, args.min_coverage
        )
        scored = common.load_scored_items(args.root, language, models)
        words = common.explode_words(scored)
        ranking = common.ranking_phrase(args.metric, language)
        for error in args.error:
            table = rate_by_length(
                words, error, args.n_bootstrap, args.ci_level, args.seed
            )
            table.insert(0, "language", language)
            stem = f"{error}_rate_by_word_length_{language}"
            table.to_csv(out / f"{stem}.csv", index=False)
            title = common.tier_title(models)
            plot(table, error, language, title, ranking, out / f"{stem}.png")
            print(table.to_string(index=False))


if __name__ == "__main__":
    main()
