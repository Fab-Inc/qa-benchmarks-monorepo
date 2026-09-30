"""Inspection plots for the benchmark item distribution and model results."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import typer
from matplotlib import patheffects
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

from phonics_benchmark.utils import require_columns

# Model display names and provider colours, shared across the benchmarks in this
# repo rather than vendored per benchmark. They are gitignored and synced from
# the fab-open-configs bucket by whichever CLI runs first; the paths are fixed
# here because that sync writes them to the repository root by those names.
# Nothing in this module fetches them — a report is offline work, and the chart
# falls back to model IDs and one bar colour when they are not there.
REPO_ROOT = Path(__file__).resolve().parents[4]
MODELS_CSV = REPO_ROOT / "models.csv"
PROVIDERS_CSV = REPO_ROOT / "providers.csv"

# Design tokens from LKB's paper_visualizations.py, so a phonics leaderboard
# sits beside an LKB one without looking like a different document. Copied
# rather than imported: LKB is a separate benchmark package here.
SURFACE, INK_PRIMARY, INK_SECONDARY = "#fcfcfb", "#0b0b0b", "#52514e"
INK_MUTED, GRIDLINE, BASELINE_RULE = "#898781", "#e1e0d9", "#c3c2b7"
DEFAULT_BAR_COLOR = "#4e79a7"
BASELINE_FLAG_ALL = "baseline:flag-all"

# Where the two right-hand value columns sit, in axis-width fractions past the
# plot area. The metric is read against the bar it belongs to, so it sits
# close; the item count is a separate reading and gets its own column.
METRIC_COLUMN_X = 1.02
ITEM_COUNT_COLUMN_X = 1.13

# How each decodability language is named in a title. Keyed by the ISO code the
# runs record; "sw" is written Kiswahili throughout this repo, not Swahili.
LANGUAGE_DISPLAY_NAMES = {"en": "English", "sw": "Kiswahili"}


def language_display_name(language: str | None) -> str:
    """Name a language for a title, falling back to the code itself."""
    if not language:
        return "Decodability"
    return LANGUAGE_DISPLAY_NAMES.get(language, language.upper())


@require_columns(["nondecodable_ratio"])
def save_nondecodable_ratio_histogram(
    benchmark_items: pd.DataFrame,
    output_path: Path,
    *,
    column: str = "nondecodable_ratio",
    title: str = "Distribution of Non-Decodable Word Ratios",
    bins: int = 20,
) -> None:
    """Save a histogram of ``column`` to ``output_path``.

    The caller passes the full output path, so the same function serves the
    pre-filter, post-filter, or any other view by varying path and title.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    benchmark_items[column].hist(bins=bins)
    plt.title(title)
    plt.xlabel("Non-Decodable Word Ratio")
    plt.ylabel("Frequency")
    plt.savefig(output_path)
    plt.close()


@require_columns(["model", "num_nondecodable_words", "accuracy"])
def save_accuracy_by_nondecodable_count_plot(
    accuracy_by_count: pd.DataFrame,
    output_path: Path,
    *,
    title: str = "Accuracy vs. Number of Non-Decodable Words in Item",
) -> None:
    """Save a per-model line plot of accuracy across item difficulty.

    ``accuracy_by_count`` is the long-format frame from
    ``results.accuracy_by_nondecodable_count`` (model,
    num_nondecodable_words, accuracy). A plain histogram would only show how
    many items fall at each difficulty level; plotting accuracy against it
    directly answers whether a model's exact-match rate drops off as items
    get harder.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for model_name, group in accuracy_by_count.groupby("model"):
        group = group.sort_values("num_nondecodable_words")
        plt.plot(
            group["num_nondecodable_words"],
            group["accuracy"],
            marker="o",
            label=model_name,
        )
    plt.title(title)
    plt.xlabel("Number of Non-Decodable Words in Item")
    plt.ylabel("Accuracy (Exact Match Rate)")
    plt.legend()
    plt.savefig(output_path)
    plt.close()


def _style_lkb_axes(axis: Axes) -> None:
    """LKB paper axes: quiet rules, a solid value-axis grid, percentage scores."""
    axis.figure.set_facecolor(SURFACE)
    axis.set_facecolor(SURFACE)
    axis.set_axisbelow(True)
    axis.grid(axis="x", color=GRIDLINE, linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(BASELINE_RULE)
        axis.spines[side].set_linewidth(0.8)
    axis.tick_params(colors=INK_MUTED, length=3, width=0.8, labelsize=9)
    axis.xaxis.set_major_formatter(PercentFormatter(1))


def _draw_intervals(axis: Axes, table: pd.DataFrame) -> None:
    """Draw each whisker between its own endpoints, not as a bar offset.

    A percentile interval is not centred on the point estimate and can even
    exclude it, so drawing it as ``estimate ± half-width`` would misplace it.
    Centring on the interval's own midpoint keeps both ends where the bootstrap
    actually put them.
    """
    valid = table["ci_low"].notna() & table["ci_high"].notna()
    low, high = table.loc[valid, "ci_low"], table.loc[valid, "ci_high"]
    midpoint = (low + high) / 2
    bars = axis.errorbar(
        midpoint,
        table.loc[valid, "position"],
        xerr=[midpoint - low, high - midpoint],
        fmt="none",
        ecolor=INK_PRIMARY,
        elinewidth=1,
        capsize=3,
        zorder=4,
    )
    # Provider colours include black; keep both ends of its CI visible.
    for artist in (*bars.lines[1], *bars.lines[2]):
        artist.set_path_effects(
            [
                patheffects.Stroke(linewidth=2.5, foreground=SURFACE),
                patheffects.Normal(),
            ]
        )


def _default_subtitle(ci_report: pd.DataFrame) -> str:
    """State what the bars were measured on, from the CI frame's own columns."""
    parts = []
    if "n_items_resampled" in ci_report.columns:
        parts.append(f"{int(ci_report['n_items_resampled'].max()):,} items")
    if {"ci_level", "n_bootstrap"} <= set(ci_report.columns):
        level = float(ci_report["ci_level"].iloc[0])
        replicates = int(ci_report["n_bootstrap"].iloc[0])
        parts.append(f"{level:.0%} bootstrap CIs ({replicates:,} resamples)")
    return " · ".join(parts)


def _default_footnote(baseline_f2: float | None) -> str:
    """Explain the two things a reader cannot infer from the bars themselves."""
    lines = [
        "Items = benchmark items scored for that model; failed and unparseable "
        "responses excluded.",
    ]
    if baseline_f2 is not None:
        lines.append(
            "Dashed line = flag-all baseline "
            f"({baseline_f2:.1%}), which marks every word non-decodable. A bar "
            "at or below it bought its F2 by over-flagging."
        )
    return "\n".join(lines)


def load_model_metadata(
    models_csv: Path = MODELS_CSV,
    providers_csv: Path = PROVIDERS_CSV,
) -> tuple[pd.DataFrame | None, dict[str, str]]:
    """Read display names, providers and provider colours, if they are present.

    Returns ``(None, {})`` when either file is missing or unreadable. The
    metadata only decides how the leaderboard is labelled and coloured, so a
    missing file is worth a word but must never stop a report being written —
    which is also why this catches broadly rather than naming failure modes.
    """
    missing = [path for path in (models_csv, providers_csv) if not path.is_file()]
    if missing:
        typer.secho(
            f"Model metadata not found ({', '.join(str(p) for p in missing)}); "
            "the leaderboard will use model IDs and a single bar colour.",
            fg=typer.colors.YELLOW,
            err=True,
        )
        return None, {}

    try:
        models = pd.read_csv(
            models_csv, usecols=["model_id", "provider", "display_name"]
        ).set_index("model_id")
        if not models.index.is_unique or models.index.hasnans:
            raise ValueError(f"{models_csv}: model_id must be unique and non-null")
        providers = pd.read_csv(providers_csv, usecols=["provider", "color"])
        colors = dict(zip(providers["provider"], providers["color"], strict=True))
    except Exception as error:
        typer.secho(
            f"WARNING: could not read model metadata ({error}); the leaderboard "
            "will use model IDs and a single bar colour.",
            fg=typer.colors.YELLOW,
            err=True,
        )
        return None, {}

    return models, colors


@require_columns(["model", "pooled_f2", "pooled_f2_ci_low", "pooled_f2_ci_high"])
def save_pooled_f2_plot(
    ci_report: pd.DataFrame,
    output_path: Path,
    *,
    item_counts: pd.Series | None = None,
    models_csv: Path = MODELS_CSV,
    providers_csv: Path = PROVIDERS_CSV,
    language: str | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    footnote: str | None = None,
) -> None:
    """Save an LKB-style pooled-F2 leaderboard, one horizontal bar per model.

    ``ci_report`` is the frame from ``bootstrap.bootstrap_metric_cis``, so the
    intervals come from one resampling of one item set and are comparable
    between the models on the axis. Pooling several runs before this point is
    what makes that true; whiskers taken from separately bootstrapped runs are
    not on the same footing even when the item set matches.

    ``baseline:flag-all`` becomes the dashed floor rather than a bar. Its recall
    is 1 by construction, so it is the score a recall-weighted metric has to
    beat — a bar at or below the line bought its F2 by over-flagging. Other
    baseline rows are left out entirely.

    Display names and provider colours are read from the shared CSVs at the
    repository root, and the chart falls back to model IDs in one colour when
    they are absent. ``models_csv`` and ``providers_csv`` override where they
    are looked for.

    ``item_counts`` is items per model, from ``summary_by_model``'s ``n_items``,
    shown as its own column: a bar drawn from fewer items is not comparable
    with the rest, and that is worth stating for every model rather than only
    the short ones.

    The title names the language, since that is what the leaderboard ranks
    within, and the subtitle says what the bars were measured on. Either can be
    replaced outright with ``title`` and ``subtitle``. Which runs the rows came
    from is not on the chart: that is what the report's
    ``scoring_provenance.json`` records, beside the hashes it depends on.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_metadata, provider_colors = load_model_metadata(models_csv, providers_csv)

    table = ci_report.rename(
        columns={
            "pooled_f2": "score",
            "pooled_f2_ci_low": "ci_low",
            "pooled_f2_ci_high": "ci_high",
        }
    ).copy()

    floor = table.loc[table["model"] == BASELINE_FLAG_ALL, "score"]
    baseline_f2 = float(floor.iloc[0]) if not floor.empty else None
    if "is_baseline" in table.columns:
        table = table[~table["is_baseline"].astype(bool)]
    else:
        table = table[table["model"] != BASELINE_FLAG_ALL]
    if table.empty:
        raise ValueError("No non-baseline models to plot.")

    table["display_name"] = table["model"]
    table["color"] = DEFAULT_BAR_COLOR
    if model_metadata is not None:
        joined = table.join(model_metadata, on="model", rsuffix="_meta")
        table["display_name"] = joined["display_name_meta"].fillna(table["model"])
        table["provider"] = joined["provider"].fillna("Unknown provider")
        if provider_colors:
            table["color"] = (
                table["provider"].map(provider_colors).fillna(DEFAULT_BAR_COLOR)
            )

    table["n_items_scored"] = (
        table["model"].map(item_counts) if item_counts is not None else pd.NA
    )

    plotted = table.sort_values("score", ascending=False, kind="stable").copy()
    plotted["position"] = list(range(len(plotted)))

    figure, axis = plt.subplots(figsize=(11.5, max(4, 1.5 + 0.34 * len(plotted))))
    axis.barh(
        plotted["position"],
        plotted["score"],
        height=0.65,
        color=plotted["color"],
        zorder=3,
    )
    _draw_intervals(axis, plotted)

    if baseline_f2 is not None:
        # Unlabelled at the line itself: the footnote says what it is, which
        # keeps the plot area clear when a bar ends near the baseline.
        axis.axvline(
            baseline_f2, color=INK_PRIMARY, linestyle=(0, (4, 4)), linewidth=1, zorder=2
        )

    axis.set_yticks(plotted["position"], list(plotted["display_name"]))
    axis.set_ylim(len(plotted) - 0.3, -1.4)
    axis.set_xlim(0, 1)
    axis.set_xlabel("Pooled F2 (word level)", color=INK_SECONDARY)

    # Score and coverage read as two right-hand columns rather than as
    # annotations: the item count belongs beside every model, not only the ones
    # that fall short, and a column of counts is scannable where a suffix on
    # some labels is not.
    axis.text(
        METRIC_COLUMN_X,
        -1.15,
        "F2",
        transform=axis.get_yaxis_transform(),
        va="center",
        fontsize=9,
        color=INK_PRIMARY,
    )
    if item_counts is not None:
        axis.text(
            ITEM_COUNT_COLUMN_X,
            -1.15,
            "Items",
            transform=axis.get_yaxis_transform(),
            va="center",
            fontsize=9,
            color=INK_PRIMARY,
        )
    for row in plotted.itertuples():
        axis.text(
            METRIC_COLUMN_X,
            float(row.position),
            f"{row.score:.1%}",
            transform=axis.get_yaxis_transform(),
            va="center",
            fontsize=9,
            color=INK_SECONDARY,
        )
        if not pd.isna(row.n_items_scored):
            axis.text(
                ITEM_COUNT_COLUMN_X,
                float(row.position),
                f"{int(row.n_items_scored)}",
                transform=axis.get_yaxis_transform(),
                va="center",
                fontsize=9,
                color=INK_MUTED,
            )

    _style_lkb_axes(axis)
    if model_metadata is not None and provider_colors:
        present = set(plotted["provider"])
        palette = {p: c for p, c in provider_colors.items() if p in present}
        palette.update(
            {p: DEFAULT_BAR_COLOR for p in sorted(present - provider_colors.keys())}
        )
        axis.legend(
            handles=[Patch(facecolor=c, label=p) for p, c in palette.items()],
            title="Provider",
            loc="center left",
            # Clear of both value columns, not just the bars.
            bbox_to_anchor=(1.22, 0.5),
            frameon=False,
            fontsize=8,
            title_fontsize=9,
        )

    # Title and subtitle are set after styling so the axes furniture is in
    # place and the title pad can hold the subtitle beneath it.
    if title is None:
        title = (
            f"{language_display_name(language)} Decodability - Pooled F2 Leaderboard"
        )
    subtitle_text = textwrap.fill(
        subtitle if subtitle is not None else _default_subtitle(ci_report), 110
    )
    # The subtitle sits in the title's padding, so the padding has to grow with
    # it: naming several pooled runs wraps to more than one line, and a fixed
    # pad would put the second line through the title.
    subtitle_lines = subtitle_text.count("\n") + 1
    axis.set_title(
        textwrap.fill(title, 80),
        color=INK_PRIMARY,
        fontsize=12,
        fontweight="bold",
        loc="left",
        pad=13 + 12 * subtitle_lines,
    )
    # Offsets in points, not axis fractions: the figure grows with the number
    # of bars, so a fraction that clears the x-label on a short chart leaves a
    # gap on a tall one, and vice versa.
    axis.annotate(
        subtitle_text,
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(0, 7),
        textcoords="offset points",
        va="bottom",
        ha="left",
        fontsize=9,
        color=INK_SECONDARY,
    )
    axis.annotate(
        footnote if footnote is not None else _default_footnote(baseline_f2),
        xy=(0.0, 0.0),
        xycoords="axes fraction",
        # Below the tick labels and the x-axis label, both of which sit in this
        # gap and are sized in points themselves.
        xytext=(0, -56),
        textcoords="offset points",
        va="top",
        ha="left",
        fontsize=8,
        color=INK_SECONDARY,
    )

    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
