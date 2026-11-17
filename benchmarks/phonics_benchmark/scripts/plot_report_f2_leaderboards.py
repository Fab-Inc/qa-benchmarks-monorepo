"""Compatibility imports for historical analyses; use plot_benchmark_report.py."""

from phonics_benchmark.historical_reports import load_plot_table
from phonics_benchmark.plots import (
    BASELINE,
    GRIDLINE,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SURFACE,
    eligible_models,
    plot_leaderboard,
)
from phonics_benchmark.reporting import read_csv
from phonics_benchmark.reporting_inputs import digest
