"""LKB-style comparisons of perfect-match accuracy, pooled F1 and pooled F2.

Reads a finished-model analysis; never rescores responses or changes its bootstrap.
Metadata CSVs may be local paths or gs:// URIs. Each run snapshots its inputs into
a NEW output directory and writes PNG/PDF figures, their plotted-value CSVs and
README.md. Pass --formats png svg to change the export formats.

Example (from the repository root)::

    uv run --frozen --package phonics-benchmark python \
        benchmarks/phonics_benchmark/scripts/plot_lkb_style_metrics.py \
        --models-csv path/to/models.csv --providers-csv path/to/providers.csv

Family assignments default to this benchmark's data/model_families.csv; override
with --families-csv. Groups with one scored model remain visible.

Outputs for each metric: leaderboard, vs_cost, vs_size, hardware, hardware_swarm,
hardware_mean_ci, and families (when scored models have family assignments).

Adapted from LKB's analyze_results.py, paper_visualizations.py, validate_results.py
and the raw compute_frontier in website/frontier_analysis.py on branch
qfb-194-implement-lkb-workflow (5e67f9f). Kept here so neither benchmark needs a
branch-specific runtime import. The paper palette, canonical provider legend,
3:1 price blend, total-parameter thresholds and staircase frontier follow LKB.
Horizontal bars keep model names readable; unpublished sizes have a separate tier.

Per-model whiskers show the saved item-bootstrap bounds. The separate tier-mean
figure follows LKB's 1,000 model resamples, seed 42, 95% percentile interval. Swarm
boxes show quartiles and ordinary 1.5-IQR whiskers, not confidence intervals.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from analyze_finished_kiswahili_models import Snapshotter, write_json  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402
from plot_model_metrics import (  # noqa: E402
    OUTPUTS_DIR,
    interval_note,
    latest_summary_csv,
    load_summary,
)

METRICS = {
    "perfect_match_accuracy": "Perfect-match accuracy",
    "pooled_f1": "Pooled F1",
    "pooled_f2": "Pooled F2",
}
DEFAULT_FAMILIES = Path(__file__).resolve().parents[1] / "data" / "model_families.csv"
PLOT_LABELS = {
    "leaderboard": "Leaderboard",
    "vs_cost": "Performance vs cost",
    "vs_size": "Performance vs model size",
    "hardware": "Models by hardware tier",
    "hardware_swarm": "Model scores within hardware tiers",
    "hardware_mean_ci": "Mean model score by hardware tier",
    "families": "Models by family",
}
# LKB paper_visualizations.py design tokens.
SURFACE, INK_PRIMARY, INK_SECONDARY = "#fcfcfb", "#0b0b0b", "#52514e"
INK_MUTED, GRIDLINE, BASELINE = "#898781", "#e1e0d9", "#c3c2b7"
DEFAULT_BAR_COLOR = "#4e79a7"
HARDWARE_TIERS = [
    "Mobile (<4B)",
    "Tablet (4–12B)",
    "Workstation (12–35B)",
    "Cloud (≥35B)",
    "Unknown size",
]


def read_metadata_csv(path, key, columns):
    table = pd.read_csv(path, usecols=[key, *columns]).set_index(key)
    if not table.index.is_unique or table.index.hasnans:
        raise ValueError(f"{path}: {key} must be unique and non-null")
    return table


def load_metadata(models_path: Path, providers_path: Path, families_path=None):
    """Keep model IDs as join keys; provider CSV order determines legend order."""
    models = read_metadata_csv(
        models_path,
        "model_id",
        [
            "display_name",
            "provider",
            "size",
            "input_cost",
            "output_cost",
        ],
    )
    providers = read_metadata_csv(providers_path, "provider", ["color"])
    colors = providers["color"].to_dict()
    for column in ("size", "input_cost", "output_cost"):
        models[column] = pd.to_numeric(models[column], errors="raise")
    models["blended_price"] = 0.75 * models["input_cost"] + 0.25 * models["output_cost"]
    models["family"] = pd.NA
    if families_path:
        families = read_metadata_csv(families_path, "model_id", ["family"])
        models["family"] = families["family"]
    return models, colors


def metric_table(summary_path, metric, metadata, colors):
    """Preserve the existing estimate and interval; attach plotting metadata only."""
    summary = load_summary(summary_path, metric)
    columns = ["model", metric, f"{metric}_ci_low", f"{metric}_ci_high"]
    columns += [
        c
        for c in ("run_id", "n_scored", "n_call_failed", "n_parse_failed")
        if c in summary
    ]
    table = (
        summary[columns]
        .rename(
            columns={
                metric: "score",
                f"{metric}_ci_low": "ci_low",
                f"{metric}_ci_high": "ci_high",
            }
        )
        .merge(
            metadata,
            left_on="model",
            right_index=True,
            how="left",
            validate="one_to_one",
        )
    )
    table["metric"] = metric
    table["display_name"] = table["display_name"].fillna(table["model"])
    table["provider"] = table["provider"].fillna("Unknown provider")
    table["color"] = table["provider"].map(colors).fillna(DEFAULT_BAR_COLOR)
    sizes = table["size"].where(np.isfinite(table["size"]) & table["size"].gt(0))
    tiers = pd.cut(
        sizes, [0, 4e9, 12e9, 35e9, np.inf], labels=HARDWARE_TIERS[:-1], right=False
    )
    table["hardware_tier"] = tiers.cat.add_categories("Unknown size").fillna(
        "Unknown size"
    )
    return table.reset_index(drop=True)


def style_axes(ax, value_axis="y"):
    """LKB paper axes: quiet rules, a solid value-axis grid, percentage scores."""
    ax.figure.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    ax.grid(axis=value_axis, color=GRIDLINE, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, length=3, width=0.8, labelsize=9)
    axis = ax.xaxis if value_axis == "x" else ax.yaxis
    axis.set_major_formatter(PercentFormatter(1))


def provider_legend(ax, table, colors):
    present = set(table["provider"])
    palette = {p: c for p, c in colors.items() if p in present}
    palette.update({p: DEFAULT_BAR_COLOR for p in sorted(present - colors.keys())})
    ax.legend(
        handles=[Patch(facecolor=c, label=p) for p, c in palette.items()],
        title="Provider",
        loc="center left",
        bbox_to_anchor=(1.18, 0.5),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )


def draw_intervals(ax, table, positions, *, horizontal=False):
    """Use endpoints directly, even when a percentile interval excludes its estimate."""
    valid = table["ci_low"].notna() & table["ci_high"].notna()
    low, high = table.loc[valid, "ci_low"], table.loc[valid, "ci_high"]
    midpoint = (low + high) / 2
    positions = np.asarray(positions)[valid]
    error = [midpoint - low, high - midpoint]
    if horizontal:
        bars = ax.errorbar(
            midpoint,
            positions,
            xerr=error,
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
    else:
        ax.errorbar(
            positions,
            midpoint,
            yerr=error,
            fmt="none",
            ecolor=INK_PRIMARY,
            elinewidth=1,
            capsize=3,
            zorder=2,
        )


def plot_bars(table, metric, colors, group=None):
    """One common renderer for the leaderboard, hardware and family comparisons."""
    groups = (
        [(None, table)]
        if group is None
        else table.groupby(group, observed=True, sort=True)
    )
    frames, headings, cursor = [], [], 0
    for label, rows in groups:
        rows = rows.sort_values("score", ascending=False, kind="stable").copy()
        if group:
            headings.append((cursor, label))
            cursor += 1.1
        rows["position"] = np.arange(len(rows)) + cursor
        frames.append(rows)
        cursor += len(rows) + (0.8 if group else 0)
    plotted = pd.concat(frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(11.5, max(4, 1.5 + 0.34 * cursor)))
    ax.barh(
        plotted["position"],
        plotted["score"],
        height=0.65,
        color=plotted["color"],
        zorder=3,
    )
    draw_intervals(ax, plotted, plotted["position"], horizontal=True)
    ax.set_yticks(plotted["position"], plotted["display_name"])
    ax.set_ylim(cursor - 0.3, -0.8)
    ax.set_xlim(0, 1)
    ax.set_xlabel(METRICS[metric], color=INK_SECONDARY)
    for row in plotted.itertuples():
        ax.text(
            1.02,
            row.position,
            f"{row.score:.1%}",
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=9,
            color=INK_SECONDARY,
        )
    for position, label in headings:
        ax.text(
            0,
            position,
            str(label),
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=10,
            fontweight="semibold",
            color=INK_PRIMARY,
        )
    style_axes(ax, "x")
    provider_legend(ax, plotted, colors)
    return fig, plotted


def compute_frontier(table, column):
    """LKB's raw Pareto rule: a strict score improvement as cost/size increases."""
    ordered = table.sort_values(
        [column, "score", "model"], ascending=[True, False, True]
    )
    frontier, best = [], -np.inf
    for index, row in ordered.iterrows():
        if row["score"] > best:
            frontier.append(index)
            best = row["score"]
    return table.loc[frontier]


def si_params(value, _):
    for scale, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if value >= scale:
            return f"{value / scale:g}{suffix}"
    return f"{value:g}"


def plot_tradeoff(table, metric, colors, column):
    """LKB log-axis scatter and provider-coloured staircase, with saved score CIs."""
    plotted = table.loc[np.isfinite(table[column]) & table[column].gt(0)].copy()
    if plotted.empty:
        return None, plotted
    frontier = compute_frontier(plotted, column)
    plotted["on_frontier"] = plotted["model"].isin(frontier["model"])
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    xs, ys = frontier[column].to_numpy(), frontier["score"].to_numpy()
    frontier_colors = frontier["color"].tolist()
    for i in range(len(frontier)):
        end = xs[i + 1] if i + 1 < len(xs) else plotted[column].max()
        ax.plot(
            [xs[i], end],
            [ys[i], ys[i]],
            color=frontier_colors[i],
            linestyle=(0, (5, 3)),
            linewidth=1.6,
        )
        if i + 1 < len(xs):
            ax.plot(
                [end, end],
                [ys[i], ys[i + 1]],
                color=frontier_colors[i + 1],
                linestyle=(0, (5, 3)),
                linewidth=1.6,
            )
    draw_intervals(ax, plotted, plotted[column])
    ax.scatter(
        plotted[column],
        plotted["score"],
        c=plotted["color"],
        s=42,
        edgecolors=SURFACE,
        linewidths=0.9,
        zorder=3,
    )
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(
        FuncFormatter(si_params if column == "size" else lambda v, _: f"{v:g}")
    )
    ax.set_xlabel(
        "Model size (total parameters)" if column == "size" else "Cost ($/1M tokens)",
        color=INK_SECONDARY,
    )
    ax.set_ylabel(METRICS[metric], color=INK_SECONDARY)
    ax.set_ylim(0, 1.02)
    style_axes(ax)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8, linestyle=(0, (4, 4)))
    provider_legend(ax, plotted, colors)
    return fig, plotted


def hardware_means(table):
    """LKB's bootstrap of model scores within each tier, separate from item CIs."""
    rng = np.random.default_rng(42)
    rows = []
    for tier, group in table.groupby("hardware_tier", observed=True, sort=True):
        values = group["score"].to_numpy()
        samples = rng.choice(values, size=(1000, len(values)), replace=True)
        low, high = np.percentile(samples.mean(axis=1), [2.5, 97.5])
        rows.append(
            {
                "hardware_tier": tier,
                "metric": group["metric"].iloc[0],
                "score": values.mean(),
                "ci_low": low,
                "ci_high": high,
                "n_models": len(values),
                "n_bootstrap": 1000,
                "seed": 42,
                "ci_level": 0.95,
            }
        )
    return pd.DataFrame(rows)


def plot_hardware_distribution(table, metric, colors, *, means=False):
    fig, ax = plt.subplots(figsize=(10, 5.6))
    if means:
        plotted = hardware_means(table)
        positions = np.arange(len(plotted))
        ax.scatter(positions, plotted["score"], color=INK_PRIMARY, marker="D", zorder=3)
        draw_intervals(ax, plotted, positions)
        labels = [f"{r.hardware_tier}\nn={r.n_models}" for r in plotted.itertuples()]
    else:
        rng = np.random.default_rng(42)
        groups = list(table.groupby("hardware_tier", observed=True, sort=True))
        positions, labels, frames = np.arange(len(groups)), [], []
        for position, (tier, rows) in zip(positions, groups):
            rows = rows.copy()
            rows["position"] = position + rng.uniform(-0.22, 0.22, len(rows))
            frames.append(rows)
            labels.append(f"{tier}\nn={len(rows)}")
            ax.scatter(
                rows["position"],
                rows["score"],
                c=rows["color"],
                s=28,
                alpha=0.75,
                edgecolors=SURFACE,
                linewidths=0.5,
                zorder=3,
            )
        ax.boxplot(
            [rows["score"] for _, rows in groups],
            positions=positions,
            widths=0.65,
            patch_artist=True,
            manage_ticks=False,
            showfliers=False,
            boxprops={"facecolor": BASELINE, "alpha": 0.2},
            medianprops={"color": INK_PRIMARY},
            whiskerprops={"color": INK_MUTED},
            capprops={"color": INK_MUTED},
        )
        plotted = pd.concat(frames, ignore_index=True)
        provider_legend(ax, plotted, colors)
    ax.set_xticks(positions, labels, fontsize=8)
    ax.set_ylabel(
        f"Mean model {METRICS[metric]}" if means else METRICS[metric],
        color=INK_SECONDARY,
    )
    ax.set_ylim(-0.02, 1.02)
    style_axes(ax)
    return fig, plotted


def save_figure(fig, table, output, stem, title, caption, formats, dpi):
    fig.axes[0].set_title(
        title, loc="left", fontsize=12, fontweight="semibold", color=INK_PRIMARY, pad=14
    )
    fig.tight_layout()
    try:
        for fmt in formats:
            with (output / f"{stem}.{fmt}").open("xb") as handle:
                fig.savefig(
                    handle, format=fmt, dpi=dpi, bbox_inches="tight", facecolor=SURFACE
                )
    finally:
        plt.close(fig)
    table.to_csv(output / f"{stem}.csv", index=False, mode="x")
    links = " · ".join(f"[{fmt.upper()}]({stem}.{fmt})" for fmt in formats)
    title = title.replace("\n", " · ")
    return f"- **{title}** — {links} · [CSV]({stem}.csv)\n\n  {caption}\n"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--summary-csv", type=Path, help="Default: newest local analysis"
    )
    parser.add_argument("--models-csv", required=True, help="Local CSV or gs:// URI")
    parser.add_argument("--providers-csv", required=True, help="Local CSV or gs:// URI")
    parser.add_argument(
        "--families-csv",
        default=str(DEFAULT_FAMILIES),
        help="Local CSV or gs:// URI (default: data/model_families.csv)",
    )
    parser.add_argument(
        "--metrics", nargs="+", choices=list(METRICS), default=list(METRICS)
    )
    parser.add_argument(
        "--formats", nargs="+", choices=["png", "pdf", "svg"], default=["png", "pdf"]
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--label", default="Kiswahili decodability")
    parser.add_argument("--output-dir", type=Path, help="Must not already exist")
    args = parser.parse_args()
    if args.dpi < 1:
        parser.error("--dpi must be positive")
    summary_source = args.summary_csv or latest_summary_csv()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = (
        args.output_dir
        or OUTPUTS_DIR / f"kiswahili-lkb-plots-{stamp}-{uuid4().hex[:8]}"
    )
    output.mkdir(parents=True, exist_ok=False)
    snapshots = Snapshotter(output)
    inputs = output / "inputs"
    summary_path = snapshots.copy(str(summary_source), inputs / "summary_by_model.csv")
    if summary_source.with_name("analysis.json").is_file():
        snapshots.copy(
            str(summary_source.with_name("analysis.json")), inputs / "analysis.json"
        )
    models_path = snapshots.copy(args.models_csv, inputs / "models.csv")
    providers_path = snapshots.copy(args.providers_csv, inputs / "providers.csv")
    families_path = (
        snapshots.copy(args.families_csv, inputs / "model_families.csv")
        if args.families_csv
        else None
    )
    metadata, colors = load_metadata(models_path, providers_path, families_path)
    notes = [
        f"# {args.label}\n",
        f"Source analysis: `{summary_source}`.\n",
        "Scores use scorable responses from finished models, "
        "including models with recorded errors. "
        "Perfect-match accuracy is the mean of the per-item exact-set-match indicator. "
        "Pooled F1/F2 and all per-model bootstrap bounds are unchanged.\n",
        "Provider colours and display names come from the saved metadata. "
        "Cost = 0.75 × input + 0.25 × output price per million tokens, "
        "not measured run cost. Hardware tiers use LKB's total-parameter thresholds "
        "as rough hardware proxies, "
        "not measured RAM requirements; unpublished sizes are separate.\n",
    ]
    for metric in dict.fromkeys(args.metrics):
        table = metric_table(summary_path, metric, metadata, colors)
        ci_note = interval_note(
            load_summary(summary_path, metric), metric, summary_path
        )
        unknown = table.loc[~table["model"].isin(metadata.index), "model"].tolist()
        if unknown:
            notes.append(
                f"Metadata absent for: {', '.join(unknown)}. "
                "Kept with ID labels and fallback colour.\n"
            )
        jobs = [
            ("leaderboard", *plot_bars(table, metric, colors), ci_note),
            (
                "vs_cost",
                *plot_tradeoff(table, metric, colors, "blended_price"),
                ci_note,
            ),
            ("vs_size", *plot_tradeoff(table, metric, colors, "size"), ci_note),
            ("hardware", *plot_bars(table, metric, colors, "hardware_tier"), ci_note),
            (
                "hardware_swarm",
                *plot_hardware_distribution(table, metric, colors),
                "Each point is a model. Boxes: median and quartiles; "
                "whiskers: 1.5 IQR, not CIs.",
            ),
            (
                "hardware_mean_ci",
                *plot_hardware_distribution(table, metric, colors, means=True),
                "Equal-weight mean of model scores per tier. Whiskers: 95% percentile "
                "CI from 1,000 model resamples within each tier, seed 42. "
                "These are not item-bootstrap CIs.",
            ),
        ]
        families = table.dropna(subset=["family"])
        if not families.empty:
            jobs.append(
                ("families", *plot_bars(families, metric, colors, "family"), ci_note)
            )
        else:
            notes.append(
                f"{METRICS[metric]}: family comparison skipped; "
                "no scored model has a family assignment.\n"
            )
        for name, fig, plotted, caption in jobs:
            if fig is None:
                notes.append(
                    f"{METRICS[metric]} / {name}: skipped; "
                    "no positive, finite metadata values.\n"
                )
                continue
            if name in ("vs_cost", "vs_size", "families"):
                omitted = table.loc[
                    ~table["model"].isin(plotted["model"]), "model"
                ].tolist()
                caption += f" Plotted {len(plotted)}/{len(table)} models."
                if omitted:
                    caption += f" Omitted: {', '.join(omitted)}."
            if name.startswith("vs_"):
                caption += (
                    " Dashed staircase: Pareto frontier of point estimates; "
                    "it is not a significance test."
                )
            title = f"{args.label} · {METRICS[metric]}\n{PLOT_LABELS[name]}"
            footnote = {
                "hardware_swarm": "Boxes: quartiles; whiskers: 1.5 IQR (not CIs).",
                "hardware_mean_ci": "95% CI: 1,000 model resamples within each tier.",
            }.get(
                name, ci_note.replace("bootstrap intervals", "item-bootstrap intervals")
            )
            fig.text(0.01, -0.015, footnote, fontsize=8, color=INK_SECONDARY)
            notes.append(
                save_figure(
                    fig,
                    plotted,
                    output,
                    f"{metric}_{name}",
                    title,
                    caption,
                    dict.fromkeys(args.formats),
                    args.dpi,
                )
            )
    write_json(
        output / "manifest.json",
        {
            "inputs": snapshots.artifacts,
            "options": vars(args)
            | {
                "summary_csv": str(summary_source),
                "output_dir": str(output),
            },
        },
    )
    with (output / "README.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(notes))
    print(f"Wrote figures, tables and captions to {output / 'README.md'}")


if __name__ == "__main__":
    main()
