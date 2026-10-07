"""Shared helpers for the hardware-tier error analyses.

Used by plot_error_rate_by_word_length.py and plot_calibration.py. Both read the
top-N models per hardware tier chosen by report_on_benchmark_run.py --hardware-tiers
(hardware_tiers_top{N}_models.csv) and the word-level scoring in each language's
reports/<language>/results_with_scoring.csv. No model calls and no rescoring.

Confidence intervals come from an item-level bootstrap, as in the run report: the
benchmark items are resampled with replacement and one draw is shared across tiers,
so a tier's rate is recomputed as a ratio of sums over the drawn items. They show
item-sampling uncertainty only.
"""

from __future__ import annotations

import argparse
import ast
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
from phonics_benchmark import hardware

# Copied from HARDWARE_TIER_COLORS in
# linguistic_knowledge_benchmark/scripts/hardware_tiers.py, so tiers look the same
# across benchmarks.
TIER_COLORS = {"Mobile + Tablet": "#59a14f", "Cloud": "#bab0ac"}
# Upper-inclusive bin edges for word length in letters; the last bin is open-ended.
LENGTH_BINS = [(1, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, None)]
WORD_COLUMNS = ["tp", "fp", "fn", "tn"]


def length_label(lo: int, hi: int | None) -> str:
    if hi is None:
        return f"{lo}+"
    return str(lo) if lo == hi else f"{lo}-{hi}"


def length_bin(n_letters: int) -> str:
    """Map a word length to its bin label (lengths below 1 are not expected)."""
    for lo, hi in LENGTH_BINS:
        if n_letters >= lo and (hi is None or n_letters <= hi):
            return length_label(lo, hi)
    raise ValueError(f"word length {n_letters} is outside the bins")


def parse_word_set(value: str) -> set[str]:
    """Parse a set written by pandas, e.g. "{'a', 'b'}" or "set()"."""
    value = value.strip()
    return set() if value == "set()" else set(ast.literal_eval(value))


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Run folder holding reports/<language>/ and reports/hardware_tiers/.",
    )
    parser.add_argument(
        "--language",
        nargs="+",
        choices=list(hardware.SERIES),
        default=list(hardware.SERIES),
        help="Languages to plot, one set of files each. Defaults to all.",
    )
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--metric", default="pooled_f2", choices=list(hardware.METRICS))
    parser.add_argument(
        "--min-coverage",
        type=float,
        help="Use the hardware_tiers outputs made with this --min-coverage (e.g. 0.9).",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Item-level bootstrap replicates for the confidence bands. 0 skips them.",
    )
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Bootstrap seed. Fixed by default so reruns give the same interval.",
    )


def validate_bootstrap_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.n_bootstrap < 0:
        parser.error("--n-bootstrap must be 0 or more")
    if not 0 < args.ci_level < 1:
        parser.error("--ci-level must be between 0 and 1")


def item_count_array(
    per_item: pd.DataFrame, items: list, tiers: list, bins: list
) -> np.ndarray:
    """Spread per-(item, tier, bin) counts into an (items, tiers, bins, columns) array.

    ``per_item`` is indexed by (item, tier, bin) with one column per count. Cells with
    no rows stay 0, and ``items`` is the full resampling population, including items
    that contribute nothing to a given rate.
    """
    counts = np.zeros((len(items), len(tiers), len(bins), per_item.shape[1]))
    levels = per_item.index
    item_at = pd.Index(items).get_indexer(levels.get_level_values(0))
    tier_at = pd.Index(tiers).get_indexer(levels.get_level_values(1))
    bin_at = pd.Index(bins).get_indexer(levels.get_level_values(2))
    counts[item_at, tier_at, bin_at] = per_item.to_numpy(dtype=float)
    return counts


def bootstrap_totals(counts: np.ndarray, n_bootstrap: int, seed: int) -> np.ndarray:
    """Item-level bootstrap totals of per-item counts.

    ``counts`` has items on axis 0. Each replicate draws as many items as there are,
    with replacement, and sums their counts, giving (n_bootstrap, ...) totals. A
    multinomial draw of item multiplicities is the same as drawing items with
    replacement, and it turns every replicate into one matrix product. The draw is
    shared by everything in ``counts``, so tiers and bins are resampled together.
    """
    n_items = counts.shape[0]
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n_items, np.full(n_items, 1 / n_items), size=n_bootstrap)
    totals = draws @ counts.reshape(n_items, -1)
    return totals.reshape(n_bootstrap, *counts.shape[1:])


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Elementwise numerator / denominator, NaN where the denominator is 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator > 0, numerator / denominator, np.nan)


def percentile_interval(
    replicates: np.ndarray, ci_level: float
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile interval over axis 0, skipping the NaNs of undefined replicates."""
    alpha = 1.0 - ci_level
    with warnings.catch_warnings():  # all-NaN cells, e.g. an empty bin
        warnings.simplefilter("ignore", RuntimeWarning)
        low, high = np.nanpercentile(
            replicates, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0
        )
    return low, high


def ci_note(n_bootstrap: int, ci_level: float) -> str:
    """Footnote sentence describing the bands, empty when bootstrapping is off."""
    if n_bootstrap < 1:
        return ""
    return (
        f"Shaded band: {100 * ci_level:g}% CI from {n_bootstrap:,} "
        "item-level bootstrap resamples."
    )


def ranking_phrase(metric: str, language: str) -> str:
    """Footnote wording, e.g. "highest-pooled_f2 English models"."""
    return f"highest-{metric} {language.title()} models"


def tier_dir(root: Path, metric: str, min_coverage: float | None) -> Path:
    path = root / "reports" / "hardware_tiers" / metric
    suffix = hardware.coverage_suffix(min_coverage)
    return path / suffix.lstrip("_") if suffix else path


def load_top_models(
    root: Path, language: str, top_n: int, metric: str, min_coverage: float | None
) -> pd.DataFrame:
    """Return model and hardware_tier for the selected models of one language."""
    path = (
        tier_dir(root, metric, min_coverage) / f"hardware_tiers_top{top_n}_models.csv"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run report_on_benchmark_run.py --hardware-tiers first"
        )
    selected = pd.read_csv(path)
    selected = selected[selected.language.eq(language)]
    if selected.empty:
        raise ValueError(f"No selected models for {language} in {path}")
    return selected[["model", "hardware_tier"]].reset_index(drop=True)


def load_scored_items(root: Path, language: str, models: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, item) with parsed word sets, for the selected models."""
    path = root / "reports" / language / "results_with_scoring.csv"
    usecols = ["model", "benchmark_item_id", "success", "parse_ok", *WORD_COLUMNS]
    scored = pd.read_csv(path, usecols=usecols)
    # Failed calls and unparseable outputs carry no word labels, so they are skipped.
    scored = scored[scored.success & scored.parse_ok]
    scored = scored.merge(models, on="model", how="inner")
    for column in WORD_COLUMNS:
        scored[column] = scored[column].map(parse_word_set)
    return scored


def explode_words(scored: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, item, word) with its outcome and length bin."""
    frames = []
    for outcome in WORD_COLUMNS:
        part = scored[["model", "hardware_tier", "benchmark_item_id", outcome]].rename(
            columns={outcome: "word"}
        )
        part = part.explode("word").dropna(subset=["word"])
        part["outcome"] = outcome
        frames.append(part)
    words = pd.concat(frames, ignore_index=True)
    # Letters only, so an apostrophe or hyphen does not lengthen a word.
    words["letters"] = words.word.map(lambda w: sum(c.isalpha() for c in w))
    words = words[words.letters > 0]
    words["length_bin"] = words.letters.map(length_bin)
    return words


def style_axes(ax: Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)


def finish(fig: Figure, path: Path, note: str) -> None:
    """Save the figure as PNG at ``path`` and as a PDF beside it."""
    fig.text(0.5, 0.01, note, fontsize=8, color="#898781", ha="center", va="bottom")
    # Leave room under the axes for each footnote line.
    fig.tight_layout(rect=(0, 0.02 + 0.025 * (1 + note.count("\n")), 1, 1))
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def percent_axis(ax: Axes) -> None:
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)


def tier_title(models: pd.DataFrame) -> str:
    counts = models.groupby("hardware_tier").size()
    return (
        f"(top {counts.get('Cloud', 0)} Cloud, "
        f"top {counts.get('Mobile + Tablet', 0)} Mobile+Tablet)"
    )


def tiers_present(frame: pd.DataFrame) -> list[str]:
    return [t for t in hardware.TIER_ORDER if t in set(frame.hardware_tier)]


__all__ = [
    "TIER_COLORS",
    "LENGTH_BINS",
    "add_common_args",
    "bootstrap_totals",
    "ci_note",
    "explode_words",
    "finish",
    "item_count_array",
    "length_bin",
    "length_label",
    "load_scored_items",
    "load_top_models",
    "parse_word_set",
    "percent_axis",
    "percentile_interval",
    "ranking_phrase",
    "safe_ratio",
    "style_axes",
    "tier_dir",
    "tier_title",
    "tiers_present",
    "validate_bootstrap_args",
]
