"""LKB hardware-tier comparisons using saved F1, F2 and balanced accuracy.

Adapted from hardware_tiers.py on qfb-194-implement-lkb-workflow at acfbac1.
Uses exactly the same access gate, total-parameter RAM budgets, precision band,
tier merges, medians and top-N model means. Only Mobile + Tablet and Cloud
enter this comparison; the intermediate tier remains in the classification audit.
Models must have at least 700 scorable items by default. Dataset sizes describe
coverage. An optional strict minimum coverage also uses each dataset's size.
Neither response scoring nor the per-model bootstrap is rerun.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REFERENCE_COMMIT = "acfbac1745fc5e710b59b77d7dec947ac6c1eae6"
REFERENCE_SOURCE_SHA256 = (
    "184eb4c18ab297c8138688bd2927b5111db56e39b60eda217ed76c6736eb1cac"
)
REFERENCE_URL = (
    "https://github.com/Fab-Inc/qa-benchmarks-monorepo/blob/"
    f"{REFERENCE_COMMIT}/benchmarks/linguistic_knowledge_benchmark/"
    "scripts/hardware_tiers.py"
)
OVERHEAD = 1.25
HARDWARE_TIERS = [
    ("Mobile", 4, 0.50, 0.5),
    ("Tablet", 12, 0.50, 0.5),
    ("Workstation", 32, 0.85, 1.0),
]
TIER_ORDER = ["Mobile + Tablet", "Cloud"]
TIER_MERGES = {
    "Mobile": "Mobile + Tablet",
    "Tablet": "Mobile + Tablet",
    "Cloud (API-only)": "Cloud",
    "Cloud (capacity)": "Cloud",
}
SURFACE, INK, SECONDARY = "#fcfcfb", "#0b0b0b", "#52514e"
MUTED, GRID, CONNECTOR = "#898781", "#e1e0d9", "#c3c2b7"
SERIES = {"english": "#2a78d6", "kiswahili": "#eb6834"}
METRICS = {
    "pooled_f1": "Pooled F1",
    "pooled_f2": "Pooled F2",
    "balanced_accuracy": "Balanced accuracy",
    "adjusted_balanced_accuracy": "Adjusted balanced accuracy",
}


def tier_max_params(ram: float, fraction: float, bytes_per_param: float) -> float:
    return ram * fraction / (bytes_per_param * OVERHEAD) * 1e9


TIER_MAX_PARAMS = [
    (label, tier_max_params(ram, fraction, precision))
    for label, ram, fraction, precision in HARDWARE_TIERS
]
PRECISION_BAND = (tier_max_params(32, 0.85, 1.0), tier_max_params(32, 0.85, 0.5))


def assign_hardware(spec: dict | None) -> dict:
    """Same gates as the reference; retain raw tiers and the deciding gate."""
    raw, reason, band = None, "unplaced (not in models.csv)", False
    if spec is not None:
        if not spec["open"]:
            raw, reason = "Cloud (API-only)", "access gate (closed weights)"
        elif spec["size"] is None or pd.isna(spec["size"]):
            reason = "unplaced (open, size unpublished)"
        else:
            size = spec["size"]
            raw = next(
                (tier for tier, ceiling in TIER_MAX_PARAMS if size <= ceiling),
                "Cloud (capacity)",
            )
            reason = "capacity gate (size)"
            band = PRECISION_BAND[0] < size <= PRECISION_BAND[1]
    return {
        "raw_hardware_tier": raw,
        "hardware_tier": TIER_MERGES.get(raw, raw) if raw is not None else None,
        "placed_by": reason,
        "precision_dependent": band,
    }


def minimum_scorable_items(
    expected_items: int, min_items: int = 700, min_coverage: float | None = None
) -> int:
    """Combine an inclusive count floor with a strict coverage threshold."""
    if expected_items < 1 or min_items < 1:
        raise ValueError("Dataset size and min_items must be positive")
    if min_coverage is None:
        return min_items
    if not 0 <= min_coverage < 1:
        raise ValueError("min_coverage must be in [0, 1)")
    # Decimal input as a fraction preserves exact boundaries such as 800/1000.
    strict_minimum = int(Fraction(str(min_coverage)) * expected_items // 1) + 1
    return max(min_items, strict_minimum)


def coverage_suffix(min_coverage: float | None) -> str:
    if min_coverage is None:
        return ""
    minimum_scorable_items(1, 1, min_coverage)
    return f"_coverage_gt{100 * min_coverage:g}"


def passes_coverage(scorable_items: int, min_items: int = 700) -> bool:
    """The minimum counts successfully scored items and includes the boundary."""
    if min_items < 1:
        raise ValueError("min_items must be positive")
    return scorable_items >= min_items


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def load_metadata(path: Path) -> dict:
    models = read_csv(path)
    if models.model_id.isna().any() or not models.model_id.is_unique:
        raise ValueError("Model metadata must have unique non-null IDs")
    models["open"] = (
        models["open"].astype(str).str.lower().map({"true": True, "false": False})
    )
    if models["open"].isna().any():
        raise ValueError("Unrecognized open-weights flag")
    specs = {}
    for row in models.itertuples():
        size = None if pd.isna(row.size) else float(row.size)  # pyright: ignore[reportArgumentType]
        if size is not None and (not np.isfinite(size) or size <= 0):
            raise ValueError(f"Invalid total parameter count for {row.model_id}")
        specs[row.model_id] = {
            "size": size,
            "open": bool(row.open),
            "display_name": row.display_name,
            "provider": row.provider,
        }
    return specs


def build_audit(
    run: Path,
    specs: dict,
    denominator: int,
    metric: str = "pooled_f2",
    *,
    min_items: int = 700,
    min_coverage: float | None = None,
) -> pd.DataFrame:
    summary = read_csv(run / "report/summary_by_model.csv").rename(
        columns={"model_name": "model"}
    )
    coverage = read_csv(run / "coverage_by_model.csv")
    if not coverage.expected_items.eq(denominator).all():
        raise ValueError("Coverage denominator disagrees with dataset size")
    if metric == "balanced_accuracy":
        summary = read_csv(run.parent / "balanced_accuracy" / f"{run.name}_summary.csv")
        summary = summary.rename(columns={"n_scorable": "n_items"})
        ci = summary
    else:
        ci = read_csv(run / "report/bootstrap_ci.csv")
    table = coverage.merge(
        summary[["model", metric, "n_items"]],
        on="model",
        how="left",
        validate="one_to_one",
    )
    table = table.merge(
        ci[
            [
                "model",
                metric,
                f"{metric}_ci_low",
                f"{metric}_ci_high",
                "n_bootstrap",
                "ci_level",
            ]
        ],
        on="model",
        how="left",
        validate="one_to_one",
        suffixes=("", "_bootstrap"),
    )
    if not table.n_items.fillna(0).eq(table.scorable_items).all():
        raise ValueError("Summary scored counts disagree with coverage")
    np.testing.assert_allclose(
        table[metric], table[f"{metric}_bootstrap"], rtol=0, atol=1e-14, equal_nan=True
    )
    table = table.drop(columns=f"{metric}_bootstrap")
    return audit_models(
        table,
        specs,
        denominator,
        run.name,
        metric,
        min_items=min_items,
        min_coverage=min_coverage,
    )


def audit_models(
    table,
    specs,
    denominator,
    language,
    metric="pooled_f2",
    *,
    min_items=700,
    min_coverage=None,
):
    """Classify already scored models; unknown specifications stay in the audit."""
    required_items = minimum_scorable_items(denominator, min_items, min_coverage)
    rows = []
    for row in table.to_dict(orient="records"):
        spec = specs.get(row["model"])
        row.update(assign_hardware(spec))
        row.update(
            {
                "language": language,
                "metric": metric,
                "coverage_denominator": denominator,
                "minimum_scorable_items": required_items,
                "minimum_coverage_exclusive": min_coverage,
                "actual_dataset_coverage": row["scorable_items"]
                / row["expected_items"],
                "passes_coverage": passes_coverage(
                    int(row["scorable_items"]), required_items
                ),
                "display_name": spec["display_name"] if spec else row["model"],
                "open_weights": spec["open"] if spec else None,
                "total_parameters": spec["size"] if spec else None,
            }
        )
        reasons = []
        if not row["passes_coverage"]:
            reasons.append("below_minimum_scorable_items")
        if not pd.notna(row[metric]):
            reasons.append(f"no_defined_{metric}")
        if row["hardware_tier"] is None:
            reasons.append("no_hardware_tier")
        elif row["hardware_tier"] not in TIER_ORDER:
            reasons.append("tier_excluded_from_comparison")
        row["included"] = not reasons
        row["exclusion_reason"] = ";".join(reasons)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_tiers(
    audit: pd.DataFrame, top_n: int, metric: str = "pooled_f2"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if top_n < 1:
        raise ValueError("top_n must be positive")
    medians, means, selected = [], [], []
    eligible = audit[audit.included]
    for tier in TIER_ORDER:
        for language in SERIES:
            group = eligible[
                eligible.hardware_tier.eq(tier) & eligible.language.eq(language)
            ]
            if group.empty:
                continue
            group = group.sort_values([metric, "model"], ascending=[False, True])
            medians.append(
                {
                    "hardware_tier": tier,
                    "language": language,
                    "n_models": len(group),
                    f"median_{metric}": group[metric].median(),
                    "n_placed_by_capacity": int(
                        group.placed_by.str.startswith("capacity").sum()
                    ),
                    "n_precision_dependent": int(group.precision_dependent.sum()),
                }
            )
            top = group.head(top_n).copy()
            top["rank_within_tier"] = np.arange(1, len(top) + 1)
            selected.append(top)
            means.append(
                {
                    "hardware_tier": tier,
                    "language": language,
                    "n_available": len(group),
                    "n_used": len(top),
                    "requested_top_n": top_n,
                    f"top_n_mean_{metric}": top[metric].mean(),
                    f"top_n_min_{metric}": top[metric].min(),
                    f"top_n_max_{metric}": top[metric].max(),
                    "models_used": json.dumps(top.model.tolist()),
                }
            )
    return (
        pd.DataFrame(medians),
        pd.DataFrame(means),
        pd.concat(selected, ignore_index=True) if selected else eligible.copy(),
    )


def plot_dumbbell(
    stats: pd.DataFrame,
    value: str,
    title: str,
    subtitle: str,
    denominators: dict,
    *,
    slide: bool = False,
    metric_label: str = "Pooled F2",
    gold_note: str = "",
    min_items: int = 700,
    min_coverage: float | None = None,
    point_units: bool = False,
):
    metric_label = metric_label[0].lower() + metric_label[1:]
    tiers = [t for t in TIER_ORDER if t in set(stats.hardware_tier)]
    fig, ax = plt.subplots(figsize=(11, 1.2 * len(tiers) + 4.0))
    fig.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for y, tier in enumerate(tiers):
        group = stats[stats.hardware_tier.eq(tier)]
        points = dict(zip(group.language, group[value] * 100, strict=True))
        close = len(points) == 2 and max(points.values()) - min(points.values()) < 3
        # Y is categorical: a small vertical offset keeps near-equal X values visible.
        point_y = {
            lang: y + ({"english": -0.12, "kiswahili": 0.12}[lang] if close else 0)
            for lang in points
        }
        if len(points) == 2:
            low, high = min(points.values()), max(points.values())
            if close:
                ax.plot(
                    list(points.values()),
                    list(point_y.values()),
                    color=CONNECTOR,
                    lw=2,
                    zorder=1,
                )
            elif high - low > 1.2:
                ax.plot(
                    [low + 0.6, high - 0.6], [y, y], color=CONNECTOR, lw=2, zorder=1
                )
        for language, score in points.items():
            ax.scatter(
                score,
                point_y[language],
                s=170,
                color=SERIES[language],
                zorder=3,
                edgecolors=SURFACE,
                linewidths=2,
            )
            others = [v for lang, v in points.items() if lang != language]
            left = bool(others) and score <= min(others)
            ax.annotate(
                f"{score:.1f}" + ("" if point_units else "%"),
                (score, point_y[language]),
                xytext=(-14 if left else 14, 0),
                textcoords="offset points",
                ha="right" if left else "left",
                va="center",
                fontsize=15,
                color=INK,
            )
        counts = []
        for language in SERIES:
            rows = group[group.language.eq(language)]
            if not rows.empty:
                row = rows.iloc[0]
                used = int(row["n_used"] if "n_used" in row else row["n_models"])
                available = int(row.get("n_available", used))
                counts.append(
                    f"{language.title()}: {used}"
                    + (f" of {available}" if available != used else "")
                    + (" model" if used == 1 else " models")
                )
        ax.annotate(
            " · ".join(counts),
            (0.01, y),
            xycoords=("axes fraction", "data"),
            xytext=(0, -30),
            textcoords="offset points",
            fontsize=9,
            color=SECONDARY,
            ha="left",
            va="top",
        )
        labels = {
            "Cloud": (
                "Frontier models" if slide else "Cloud",
                "Cloud" if slide else "Datacenter or API-only",
            ),
            "Mobile + Tablet": (
                "Small models" if slide else "Mobile + tablet",
                "On-device, offline" if slide else "Up to 12GB RAM, INT4",
            ),
        }
        name, detail = labels[tier]
        for text, dy, size, color in [(name, 7, 16, INK), (detail, -9, 11, MUTED)]:
            ax.annotate(
                text,
                (-0.025, y),
                xycoords=("axes fraction", "data"),
                xytext=(0, dy),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=size,
                color=color,
            )
    ax.set_xlim(
        min(-2.5, float(stats[value].min()) * 100 - 10) if point_units else 0, 112
    )
    if point_units:
        ax.axvline(0, color=SECONDARY, ls="--", lw=1)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_ylim(-0.85, len(tiers) - 0.45)
    ax.set_yticks([])
    ax.set_xlabel(
        f"{'Median' if value.startswith('median') else 'Mean'} "
        f"{metric_label} ({'points' if point_units else '%'})",
        fontsize=12,
        color=MUTED,
    )
    ax.tick_params(axis="x", labelsize=12, colors=MUTED, length=0)
    ax.grid(axis="x", color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    fig.subplots_adjust(left=0.25, right=0.96, top=0.62, bottom=0.27)
    fig.text(
        0.025,
        0.96,
        title,
        va="top",
        fontsize=20,
        fontweight="bold",
        color=INK,
        linespacing=1.25,
    )
    fig.text(0.025, 0.82, subtitle, va="top", fontsize=11.5, color=SECONDARY)
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=11,
            markerfacecolor=SERIES[lang],
            markeredgecolor=SURFACE,
            label=lang.title(),
        )
        for lang in ("kiswahili", "english")
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.017, 0.75),
        ncol=2,
        frameon=False,
        fontsize=12,
        labelcolor=SECONDARY,
    )
    targets = "; ".join(
        f"{lang.title()}: {size:,} target items" for lang, size in denominators.items()
    )
    eligibility = f"≥{min_items:,} scorable items per model ({targets})"
    if min_coverage is not None:
        thresholds = {
            lang: minimum_scorable_items(size, min_items, min_coverage)
            for lang, size in denominators.items()
        }
        eligibility = f">{100 * min_coverage:g}% coverage"
        eligibility += "; minimum scored items: " + ", ".join(
            f"{lang.title()} {thresholds[lang]:,}/{size:,}"
            for lang, size in denominators.items()
        )
    note = (
        f"Eligibility: {eligibility}.\n"
        "Device tiers assume quantized weights; benchmark scores do not "
        "establish quality after quantization."
    )
    if gold_note:
        note += "\n" + gold_note
    fig.text(
        0.025,
        0.025,
        note,
        va="bottom",
        fontsize=8 if gold_note else 8.5,
        color=SECONDARY,
        linespacing=1.55,
    )
    return fig
