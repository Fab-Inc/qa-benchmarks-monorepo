"""Shared helpers for the hardware-tier error analyses.

Used by plot_error_rate_by_word_length.py and plot_calibration.py. Both read the
top-N models per hardware tier chosen by analyze_phonics_hardware_tiers.py
(hardware_tiers_top{N}_models.csv) and the word-level scoring in each run's
report/results_with_scoring.csv. No model calls and no rescoring.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import analyze_phonics_hardware_tiers as hardware
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

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
        help="Analysis root holding <language>/ runs and hardware_tiers/.",
    )
    parser.add_argument("--language", default="english", choices=list(hardware.SERIES))
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--metric", default="pooled_f2", choices=list(hardware.METRICS))
    parser.add_argument(
        "--min-coverage",
        type=float,
        help="Use the hardware_tiers outputs made with this --min-coverage (e.g. 0.9).",
    )


def tier_dir(root: Path, metric: str, min_coverage: float | None) -> Path:
    path = root / "hardware_tiers" / metric
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
            f"{path} not found; run analyze_phonics_hardware_tiers.py first"
        )
    selected = pd.read_csv(path)
    selected = selected[selected.language.eq(language)]
    if selected.empty:
        raise ValueError(f"No selected models for {language} in {path}")
    return selected[["model", "hardware_tier"]].reset_index(drop=True)


def load_scored_items(root: Path, language: str, models: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, item) with parsed word sets, for the selected models."""
    path = root / language / "report/results_with_scoring.csv"
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


def style_axes(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)


def finish(fig: plt.Figure, path: Path, note: str) -> None:
    fig.text(0.01, 0.01, note, fontsize=8, color="#898781", ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def percent_axis(ax: plt.Axes) -> None:
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
    "explode_words",
    "finish",
    "length_bin",
    "length_label",
    "load_scored_items",
    "load_top_models",
    "np",
    "parse_word_set",
    "percent_axis",
    "style_axes",
    "tier_dir",
    "tier_title",
    "tiers_present",
]
