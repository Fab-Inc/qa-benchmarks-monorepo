"""Average false positive or false negative rate by word length, per hardware tier.

--error fp: share of decodable words wrongly flagged as non-decodable, FP / (FP + TN).
--error fn: share of non-decodable words the model missed, FN / (FN + TP).

Counts are pooled over every selected model in a tier, so each rate is
opportunity-weighted: a model that saw more words in a bin counts for more.
Writes error_analysis/{fp,fn}_rate_by_word_length.{csv,png,pdf} next to the
hardware-tier outputs.

How to run (from the repo root). Run analyze_phonics_hardware_tiers.py on the same
--root first, since this script reads its hardware_tiers_top{N}_models.csv:

    uv run --frozen --package phonics-benchmark python \\
      benchmarks/phonics_benchmark/scripts/plot_error_rate_by_word_length.py \\
      --root outputs/phonics-mixture-20260922/selections/beta1-8-provisional \\
      --error fp

Omit --error to make both plots. Other options: --language (english or kiswahili),
--top-n (default 10), --metric (ranking metric, default pooled_f2) and
--min-coverage (e.g. 0.9, to use the coverage-filtered hardware-tier outputs).
"""

from __future__ import annotations

import argparse

import hardware_tier_errors as common
import matplotlib.pyplot as plt
import pandas as pd

# error -> (numerator outcome, denominator outcomes, axis label)
ERRORS = {
    "fp": ("fp", ("fp", "tn"), "false positive rate"),
    "fn": ("fn", ("fn", "tp"), "false negative rate"),
}


def rate_by_length(words: pd.DataFrame, error: str) -> pd.DataFrame:
    """Pooled error rate per (tier, length bin), with the counts behind it."""
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
    return table.reset_index()


def plot(table: pd.DataFrame, error: str, title_suffix: str, path) -> None:
    _, _, label = ERRORS[error]
    bins = [common.length_label(lo, hi) for lo, hi in common.LENGTH_BINS]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for tier in common.tiers_present(table):
        series = (
            table[table.hardware_tier.eq(tier)].set_index("length_bin").reindex(bins)
        )
        color = common.TIER_COLORS[tier]
        ax.plot(bins, series.rate, "o-", color=color, linewidth=2.5, label=tier)
        last = series.rate.dropna()
        if not last.empty:
            ax.annotate(
                f"{100 * last.iloc[-1]:.1f}%",
                (last.index[-1], last.iloc[-1]),
                xytext=(8, 0),
                textcoords="offset points",
                color=color,
                fontsize=9,
                va="center",
            )
    common.percent_axis(ax)
    common.style_axes(ax)
    ax.set_xlabel("word length (letters)")
    ax.set_ylabel(f"{label} (%)")
    ax.legend(frameon=False, loc="upper left" if error == "fp" else "upper right")
    ax.set_title(f"Average {label} by word length\n{title_suffix}", fontweight="bold")
    abbreviation = error.upper()
    common.finish(
        fig,
        path,
        f"{abbreviation} rate averaged (opportunity-weighted) across the top-ranked "
        "models in each hardware tier.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(parser)
    parser.add_argument(
        "--error", nargs="+", choices=list(ERRORS), default=list(ERRORS)
    )
    args = parser.parse_args()
    models = common.load_top_models(
        args.root, args.language, args.top_n, args.metric, args.min_coverage
    )
    scored = common.load_scored_items(args.root, args.language, models)
    words = common.explode_words(scored)
    out = common.tier_dir(args.root, args.metric, args.min_coverage) / "error_analysis"
    out.mkdir(parents=True, exist_ok=True)
    for error in args.error:
        table = rate_by_length(words, error)
        table.insert(0, "language", args.language)
        table.to_csv(out / f"{error}_rate_by_word_length.csv", index=False)
        plot(
            table,
            error,
            common.tier_title(models),
            out / f"{error}_rate_by_word_length.png",
        )
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
