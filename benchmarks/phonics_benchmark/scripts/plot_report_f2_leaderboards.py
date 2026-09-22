"""Plot the reporting branch's saved pooled F2 estimates and bootstrap bounds.

Leaderboard styling is adapted from plot_lkb_style_metrics.py at f53c09c,
which generated the September 15 LKB plots. No responses are rescored and no
bootstrap is run here. Model/provider metadata is a local snapshot of that plot
folder; the Astra display metadata comes from its downloaded run config.
"""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

SURFACE, INK_PRIMARY, INK_SECONDARY = "#fcfcfb", "#0b0b0b", "#52514e"
INK_MUTED, GRIDLINE, BASELINE = "#898781", "#e1e0d9", "#c3c2b7"
METADATA_REFERENCE = "kiswahili-lkb-plots-20260915T103825Z-3fdb146a"
STYLE_COMMIT = "f53c09cccda01ee8b9946fdfd862920c8db74178"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def load_plot_table(run: Path, metadata: Path) -> tuple[pd.DataFrame, dict]:
    summary = read_csv(run / "report/summary_by_model.csv").rename(
        columns={"model_name": "model"}
    )
    ci = read_csv(run / "report/bootstrap_ci.csv")
    coverage = read_csv(run / "coverage_by_model.csv")
    models = read_csv(metadata / "models.csv").set_index("model_id")
    providers = read_csv(metadata / "providers.csv").set_index("provider")
    for table in (models, providers):
        if not table.index.is_unique:
            raise ValueError("Metadata identifiers must be unique")
    overrides = json.loads((metadata / "display_overrides.json").read_text())
    for model, override in overrides.items():
        models.loc[model, "display_name"] = override["display_name"]
        models.loc[model, "provider"] = override["provider"]

    values = summary[["model", "pooled_f2", "n_items"]].merge(
        ci[
            [
                "model",
                "pooled_f2",
                "pooled_f2_ci_low",
                "pooled_f2_ci_high",
                "n_bootstrap",
                "ci_level",
            ]
        ],
        on="model",
        validate="one_to_one",
        how="outer",
        suffixes=("", "_bootstrap"),
    )
    np.testing.assert_allclose(
        values.pooled_f2, values.pooled_f2_bootstrap, rtol=0, atol=1e-14
    )
    values = values.drop(columns="pooled_f2_bootstrap").merge(
        coverage[
            [
                "model",
                "expected_items",
                "scorable_items",
                "call_failed",
                "parse_failed",
                "missing_responses",
            ]
        ],
        on="model",
        how="left",
        validate="one_to_one",
    )
    if not values.n_items.eq(values.scorable_items).all():
        raise ValueError("Scored item counts disagree with coverage")
    values = values.merge(
        models[["display_name", "provider"]],
        left_on="model",
        right_index=True,
        how="left",
        validate="one_to_one",
    )
    if values[["display_name", "provider"]].isna().any().any():
        raise ValueError("Some scored models lack display metadata")
    colors = providers.color.to_dict()
    values["color"] = values.provider.map(colors)
    if values.color.isna().any():
        raise ValueError("Some providers lack saved colours")
    values = values.rename(
        columns={
            "pooled_f2": "score",
            "pooled_f2_ci_low": "ci_low",
            "pooled_f2_ci_high": "ci_high",
        }
    )
    if (
        not values[["score", "ci_low", "ci_high"]]
        .apply(lambda c: c.between(0, 1))
        .all()
        .all()
    ):
        raise ValueError("F2 estimates and intervals must be defined within [0, 1]")
    if not values.ci_low.le(values.ci_high).all():
        raise ValueError("Reversed confidence interval")
    for column in ("ci_level", "n_bootstrap", "expected_items"):
        if values[column].nunique() != 1:
            raise ValueError(f"Inconsistent {column}")
    return values.sort_values(["score", "model"], ascending=[False, True]).reset_index(
        drop=True
    ), colors


def plot_leaderboard(table: pd.DataFrame, colors: dict, language: str):
    """Original LKB bars, palette, rules and CI treatment, plus an item-count column."""
    n = len(table)
    fig, ax = plt.subplots(figsize=(13, max(5, 2.0 + 0.34 * n)))
    fig.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    positions = np.arange(n)
    ax.barh(positions, table.score, height=0.65, color=table.color, zorder=3)
    # Endpoints are drawn directly: percentile bounds need not enclose the estimate.
    midpoint = (table.ci_low + table.ci_high) / 2
    bars = ax.errorbar(
        midpoint,
        positions,
        xerr=[midpoint - table.ci_low, table.ci_high - midpoint],
        fmt="none",
        ecolor=INK_PRIMARY,
        elinewidth=1,
        capsize=3,
        zorder=4,
    )
    for artist in (*bars.lines[1], *bars.lines[2]):
        artist.set_path_effects(
            [
                patheffects.Stroke(linewidth=2.5, foreground=SURFACE),
                patheffects.Normal(),
            ]
        )
    ax.set_yticks(positions, table.display_name)
    ax.set_ylim(n - 0.3, -0.8)
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xlabel("Pooled F2", color=INK_SECONDARY)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, length=3, width=0.8, labelsize=9)
    for position, row in enumerate(table.itertuples()):
        ax.text(
            1.02,
            position,
            f"{row.score:.1%}",
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=9,
            color=INK_SECONDARY,
        )
        ax.text(
            1.145,
            position,
            f"{row.n_items:,}",
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=8.5,
            color=INK_MUTED,
        )
    ax.text(1.02, 1.018, "F2", transform=ax.transAxes, color=INK_SECONDARY, fontsize=9)
    ax.text(
        1.145, 1.018, "Items", transform=ax.transAxes, color=INK_SECONDARY, fontsize=9
    )
    present = set(table.provider)
    ax.legend(
        handles=[
            Patch(facecolor=color, label=provider)
            for provider, color in colors.items()
            if provider in present
        ],
        title="Provider",
        loc="center left",
        bbox_to_anchor=(1.30, 0.5),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )
    level, reps, target = table.iloc[0][["ci_level", "n_bootstrap", "expected_items"]]
    ax.set_title(
        f"{language.title()} decodability · Pooled F2\nLeaderboard",
        loc="left",
        fontsize=12,
        fontweight="semibold",
        color=INK_PRIMARY,
        pad=35,
    )
    ax.annotate(
        f"{int(target):,} target items · {level:.0%} bootstrap CIs "
        f"({int(reps):,} resamples)",
        (0, 1),
        xytext=(0, 8),
        xycoords="axes fraction",
        textcoords="offset points",
        fontsize=8.5,
        color=INK_SECONDARY,
        va="bottom",
    )
    ax.annotate(
        "Items = scorable responses per model; coverage varies.",
        (0, 0),
        xytext=(0, -43),
        xycoords="axes fraction",
        textcoords="offset points",
        fontsize=8,
        color=INK_SECONDARY,
        va="top",
    )
    fig.tight_layout()
    return fig


def eligible_models(table: pd.DataFrame, min_items: int = 700) -> pd.DataFrame:
    """Select models by successfully scored items, not saved response rows."""
    if min_items < 1:
        raise ValueError("min_items must be positive")
    return table[table.scorable_items.ge(min_items)].reset_index(drop=True)


def main(root: Path, min_items: int = 700) -> None:
    metadata = root / "plot_metadata"
    source_paths = [
        metadata / filename
        for filename in ("models.csv", "providers.csv", "display_overrides.json")
    ]
    for language in ("english", "kiswahili"):
        run = root / language
        source_paths.extend(
            [
                run / "report/summary_by_model.csv",
                run / "report/bootstrap_ci.csv",
                run / "coverage_by_model.csv",
            ]
        )
    before = {str(p.relative_to(root)): digest(p) for p in source_paths}
    outputs = []
    for language in ("english", "kiswahili"):
        run = root / language
        table, colors = load_plot_table(run, metadata)
        table = eligible_models(table, min_items)
        if table.empty:
            raise ValueError(
                f"No {language} models have at least {min_items} scored items"
            )
        fig = plot_leaderboard(table, colors, language)
        fig.axes[0].set_title(
            f"{language.title()} decodability · Pooled F2\n"
            f"Leaderboard · ≥{min_items:,} evaluated items",
            loc="left",
            fontsize=12,
            fontweight="semibold",
            color=INK_PRIMARY,
            pad=35,
        )
        try:
            for fmt in ("png", "pdf"):
                path = run / "report" / f"pooled_f2_leaderboard.{fmt}"
                fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=SURFACE)
                outputs.append(path)
        finally:
            plt.close(fig)
        path = run / "report/pooled_f2_leaderboard.csv"
        table.to_csv(path, index=False)
        written = read_csv(path)
        pd.testing.assert_frame_equal(
            written[["score", "ci_low", "ci_high"]],
            table[["score", "ci_low", "ci_high"]],
            check_exact=True,
        )
        outputs.append(path)
        print(
            f"{language}: {len(table)} F2 estimates and confidence intervals "
            "preserved exactly"
        )
    after = {str(p.relative_to(root)): digest(p) for p in source_paths}
    if before != after:
        raise ValueError("Input summaries or metadata changed during plotting")
    manifest = {
        "style_source_commit": STYLE_COMMIT,
        "metadata_reference": METADATA_REFERENCE,
        "script_sha256": digest(Path(__file__)),
        "input_sha256": before,
        "output_sha256": {str(p.relative_to(root)): digest(p) for p in outputs},
        "scores_recomputed": False,
        "bootstrap_recomputed": False,
        "minimum_scorable_items": min_items,
        "plotted_bounds_equal_saved_bounds": True,
    }
    (root / "audit/f2_plot_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/phonics-merged-20260917")
    )
    parser.add_argument("--min-items", type=int, default=700)
    args = parser.parse_args()
    if args.min_items < 1:
        parser.error("--min-items must be positive")
    main(args.root, args.min_items)
