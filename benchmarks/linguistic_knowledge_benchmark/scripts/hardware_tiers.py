#!/usr/bin/env python3
"""
Hardware-tier assignment for the Linguistic Knowledge Benchmark, and the tier figure.

This module owns ONE question: what is the smallest class of device that could run
each model? Everything the assignment depends on -- the memory model, the tier caps,
the special cases -- lives here rather than being spread through the plotting code, so
the rules can be read (and changed) in one place. analyze_results.py imports from here
for its own tier plots; nothing here imports it back at module level.

A tier answers "what is the smallest device class that runs this model at a defensible
quality standard", not "where do the weights physically fit", and it assumes the model
is the FOREGROUND workload with background apps evicted.

HOW A MODEL IS PLACED -- three gates, in order (see hardware_tier() for the code)

  1. ACCESS. Closed weights -> Cloud (API-only), whatever the parameter count. Whether
     a user can obtain the weights is a different question from whether their hardware
     could hold them, so it is settled before any arithmetic.
  2. DATA SUFFICIENCY. Open weights with no published size -> excluded from the
     figures. Size unknown is not size small.
  3. CAPACITY. A per-tier RAM budget divided by a per-tier precision gives a parameter
     ceiling; the model takes the first tier it fits under, else Cloud (capacity).

      budget(tier)     = device_RAM(tier) x usable_fraction(tier)
      max_params(tier) = budget(tier) / (bytes_per_param(tier) x OVERHEAD)

      Mobile       4GB x 0.50 = 2.0GB,  INT4 -> 3.2B params
      Tablet      12GB x 0.50 = 6.0GB,  INT4 -> 9.6B params
      Workstation 32GB x 0.85 = 27.2GB, INT8 -> 21.8B params

  The precision differs BY TIER on purpose: INT4 on Mobile/Tablet is descriptive of
  what ships, while INT8 on Workstation is a quality floor we impose. Because of that
  the tiers are NOT a common capacity scale -- the parameter gap between adjacent tiers
  is not a capability gap.

  Models between 21.8B and 43.5B are flagged `precision_dependent`: Workstation if you
  accept INT4 quality, Cloud if you want INT8. They default to Cloud so the dataset is
  unambiguous, and a figure should render them as a transition band.

  Sparse (MoE) models get NO active-parameter discount -- see the memory model below.

MANDATORY CAVEAT FOR ANY FIGURE BUILT ON THIS
  Capability is measured at FULL precision (BF16 in almost every case) while
  deployability is computed at REDUCED precision. A model plotted at Mobile carries a
  score earned by BF16 weights and a footprint earned by an INT4 build whose quality we
  have not measured, so the chart is optimistic in that specific direction -- most so at
  the Mobile/Tablet end, where small models tolerate quantization worst. State this
  wherever the figure appears.

INPUTS
  models.csv (repo root, synced from the fab-open-configs bucket) for each model's
  `size` and `open`; one judge-bench summary CSV per language for the accuracies.
  The summaries default to DEFAULT_EXPERIMENT_ID below and are overridable per
  language on the CLI.

OUTPUTS (under --outdir, default results/analysis/)
  hardware_tiers_<language>_<experiment-id>.csv   per-model tier + accuracy, the audit
                                                  trail behind every tier statistic
  hardware_tiers_medians.csv                      the plotted values (table-view twin)
  hardware_tiers_medians.png                      tier medians, both languages

Run:
    uv run python benchmarks/linguistic_knowledge_benchmark/scripts/hardware_tiers.py
    uv run python .../hardware_tiers.py --format pdf --outdir /tmp/figs
    uv run python .../hardware_tiers.py \
        --experiment-id-english final-137-items_fallback \
        --experiment-id-kiswahili final-135-items_fallback
    uv run python .../hardware_tiers.py --summary-english /path/to/summary.csv
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from linguistic_knowledge_benchmark.shared_utils import (
    MODELS_CSV,
    require_synced_config,
)

# --------------------------------------------------------------------------- #
# The memory model
# --------------------------------------------------------------------------- #
# A tier answers "what is the smallest device class that runs this model at a
# defensible quality standard", not "where do the weights physically fit", and it
# assumes the model is the FOREGROUND workload with background apps evicted.
#
# Three assumptions turn a parameter count into a tier, and two of them are set PER
# TIER rather than globally:
#
#   1. Quantization. Mobile and Tablet use INT4 (0.5 bytes/param) because that is what
#      ships -- both are ARM SoCs running the same runtimes and the same INT4 artefacts,
#      and a 12GB phone runs a BIGGER INT4 model, never a higher-precision one.
#      Workstation uses INT8 (1.0) as a QUALITY FLOOR: there the constraint lifts and
#      precision becomes a real choice, so we decline to assume the cheapest option that
#      fits. That floor is NORMATIVE, not descriptive -- Q4 is common on workstations
#      too -- and it is the single most consequential assumption here, which is why the
#      Workstation/Cloud boundary ships as a band (see PRECISION_BAND below).
#   2. Overhead: a flat 25% for runtime, activations and a short-to-medium KV cache.
#      This is a MODELING ALLOWANCE, not a physical relationship -- KV cache scales with
#      context and layer count, not with parameter count, so the form is tight for small
#      models and generous for very large ones. Adopted because it is reproducible from
#      one dataset column. Long context (>8K tokens) falls outside it.
#   3. RAM fraction. Tier RAM is TOTAL DEVICE RAM (the spec-sheet number) and the model
#      gets a fraction of it. The fraction tracks whether memory is shared with an OS or
#      dedicated to an accelerator: 50% on Mobile/Tablet (Android holds ~1.5GB resident;
#      iOS terminates a foreground app well below installed RAM), 85% on Workstation
#      (dedicated VRAM competes only with a compositor). The 50% is a CEILING -- shipped
#      on-device models sit nearer 20% of device RAM, which would give 1.3B and 3.8B
#      thresholds instead of 3.2B and 9.6B.
#
# MoE models get NO discount: total parameters drive RAM exactly as for dense models,
# and models.csv's `size_active` is deliberately not read. The router can pick any
# expert for any token, so the whole model has to be resident; paging experts from disk
# works but collapses throughput. gpt-oss-120b (117B total, 5.1B active) settled it --
# on active params it would fit a tablet, while OpenAI quote "within 80GB" and real
# local runs need ~64GB of unified memory, both tracking total parameters.
OVERHEAD = 1.25

# (label, device_ram_gb, usable_fraction, bytes_per_param), smallest tier first.
# `device_ram_gb` is the spec-sheet figure: 4GB the entry-level Android floor, 12GB
# premium-phone/iPad-Pro-class unified memory, 32GB the VRAM of a top consumer card
# (RTX 5090). Above that needs datacenter accelerators or multi-GPU.
HARDWARE_TIERS = [
    ("Mobile", 4, 0.50, 0.5),
    ("Tablet", 12, 0.50, 0.5),
    ("Workstation", 32, 0.85, 1.0),
]

# A model reaches the cloud for one of two distinct reasons: the weights cannot be
# obtained (access), or the model is too large for consumer hardware (capacity). They
# are reported as ONE tier -- HARDWARE_TIER_MERGES folds them together below -- but the
# distinction is not lost: build_tier_table's `placed_by` records which gate fired for
# every model, so a reader can always split the tier back apart. Worth doing before
# quoting a cloud statistic, since most of the tier is the access gate rather than size.
CLOUD_API_ONLY = "Cloud (API-only)"
CLOUD_CAPACITY = "Cloud (capacity)"
CLOUD_TIERS = (CLOUD_CAPACITY, CLOUD_API_ONLY)


def tier_max_params(device_ram_gb: float, fraction: float, bytes_per_param: float):
    """Largest parameter count a tier admits: its RAM budget / (bytes per param x
    overhead). Derived rather than hardcoded so changing a fraction or a precision moves
    the threshold with it."""
    return device_ram_gb * fraction / (bytes_per_param * OVERHEAD) * 1e9


# label -> max params, in tier order. Mobile 3.2B, Tablet 9.6B, Workstation 21.8B.
TIER_MAX_PARAMS = [
    (label, tier_max_params(ram, fraction, bytes_per_param))
    for label, ram, fraction, bytes_per_param in HARDWARE_TIERS
]

# The one boundary where our quantization choice, not the hardware, decides the answer:
# the Workstation budget admits 21.8B at the INT8 quality floor but 43.5B at INT4.
# Models in between (Gemma 3 27B, Qwen3-32B and similar) are genuinely run on single
# consumer GPUs today at Q4, so they are flagged `precision_dependent` -- Workstation if
# you accept INT4 quality, Cloud if you want INT8. Their default tier stays Cloud so the
# dataset is unambiguous; the flag is what a figure should render as a transition band.
# There is no band at the lower boundaries: INT4 is the assumption on BOTH sides there,
# so those cuts are set purely by device memory and have no precision ambiguity.
_WORKSTATION = HARDWARE_TIERS[-1]
PRECISION_BAND = (
    tier_max_params(_WORKSTATION[1], _WORKSTATION[2], _WORKSTATION[3]),  # 21.8B, INT8
    tier_max_params(_WORKSTATION[1], _WORKSTATION[2], 0.5),  # 43.5B, INT4
)

# Mobile and Tablet hold only a handful of the ~120 scored models -- too few to read as
# separate distributions -- so they are reported as one group. Empty this dict to
# un-merge them; nothing else has to change.
CLOUD = "Cloud"
HARDWARE_TIER_MERGES = {
    "Mobile": "Mobile + Tablet",
    "Tablet": "Mobile + Tablet",
    CLOUD_CAPACITY: CLOUD,
    CLOUD_API_ONLY: CLOUD,
}
# Tier labels as plotted: tier order, de-duplicated by the merges above.
PLOTTED_HARDWARE_TIERS = list(
    dict.fromkeys(
        [HARDWARE_TIER_MERGES.get(label, label) for label, *_ in HARDWARE_TIERS]
        + [HARDWARE_TIER_MERGES.get(label, label) for label in CLOUD_TIERS]
    )
)
HARDWARE_TIER_COLORS = {  # keyed by raw and merged labels alike, so merges stay optional
    "Mobile": "#59a14f",
    "Tablet": "#4e79a7",
    "Mobile + Tablet": "#59a14f",
    "Workstation": "#f28e2b",
    CLOUD_CAPACITY: "#bab0ac",
    CLOUD_API_ONLY: "#bab0ac",
    CLOUD: "#bab0ac",
}

# Figure labels: the tier name on its own line, what bounds it underneath. Split out of
# the tier labels so the plot can typeset the two at different weights.
TIER_DISPLAY = {
    "Mobile + Tablet": ("Mobile + tablet", "Under 12GB RAM, INT4"),
    "Workstation": ("Workstation", "32GB VRAM, INT8"),
    CLOUD: ("Cloud", "Datacenter or API-only"),
}

# The slide cut: the same statistic, reduced to the contrast a non-technical audience is
# actually being asked to hold -- the best models anywhere versus the best that run on a
# device in the room. Workstation is dropped rather than relabelled: it is neither of
# those two things, it is the thinnest tier (4 models), and on a slide a third row
# invites the question "which of these is my laptop?" that the figure cannot answer in
# the time available. The full three-tier figure remains the one to cite.
SLIDE_TIERS = ["Mobile + Tablet", CLOUD]
SLIDE_TIER_DISPLAY = {
    "Mobile + Tablet": ("Small models", "On-device, offline"),
    CLOUD: ("Frontier models", "Cloud"),
}

LANGUAGES = ["english", "kiswahili"]
LANGUAGE_LABEL = {"english": "English", "kiswahili": "Kiswahili"}

# The released experiments, scored with the MCQ fallback pass. Override per language
# with --experiment-id-<language>, or point at an arbitrary file with --summary-<language>.
DEFAULT_EXPERIMENT_ID = {
    "english": "final-137-items_fallback",
    "kiswahili": "final-135-items_fallback",
}

# --------------------------------------------------------------------------- #
# Design tokens -- shared with paper_visualizations.py so the paper's figures read as
# one set. The two series colours were validated together as a categorical pair on
# this surface (adjacent CVD dE 24.7 protan / 32.7 tritan, normal 33.6, both >= 3:1
# contrast); do not substitute without re-running the palette validator.
# --------------------------------------------------------------------------- #
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
CONNECTOR = "#c3c2b7"
SERIES = {
    "kiswahili": {"color": "#eb6834", "label": "Kiswahili"},
    "english": {"color": "#2a78d6", "label": "English"},
}

_RULE = "=" * 70


def _section(title: str) -> None:
    print(f"\n{_RULE}\n{title}\n{_RULE}")


# --------------------------------------------------------------------------- #
# Assignment
# --------------------------------------------------------------------------- #
def load_model_hardware(models_csv: str) -> dict:
    """model_id -> {"size", "open"} from models.csv, for hardware_tier().

    `size` stays a raw total parameter count (the CSV stores 4e9, not 4); a blank cell is
    None. models.csv also carries `size_active` for MoE models, deliberately not loaded:
    tiers charge every model for its total parameters (see HARDWARE_TIERS), and reading it
    here is what would let it drift back into the tier maths. Aborts if models.csv is
    missing -- an empty dict would silently untier every model rather than fail.
    """
    require_synced_config(models_csv)
    m = pd.read_csv(models_csv)
    if "size" not in m.columns:
        return {}

    def _number(value) -> float | None:
        return None if value is None or pd.isna(value) else float(value)

    open_col = m["open"] if "open" in m.columns else pd.Series(False, index=m.index)
    return {
        row.model_id: {
            "size": _number(row.size),
            "open": bool(open_col[i]) if not pd.isna(open_col[i]) else False,
        }
        for i, row in zip(m.index, m.itertuples())
    }


def hardware_tier(spec: dict | None) -> str | None:
    """Smallest tier this model can be deployed on, from a load_model_hardware() spec.

    Three gates, in order. DEPLOYMENT ACCESS IS RESOLVED BEFORE ANY MEMORY ARITHMETIC,
    because whether a user can obtain the weights is a different question from whether
    their hardware could hold them:

      1. access -- closed weights go to Cloud (API-only) whatever their parameter count.
         No consumer tier is reachable when the weights cannot be downloaded. (An
         earlier version applied this only when `size` was also missing, so a
         closed-weights model with a published size was assigned to a tier nobody can
         deploy it on.)
      2. data sufficiency -- open weights with no published size return None
         (excluded). Size unknown is not size small, and a default would bias whichever
         tier received it.
      3. capacity -- the first tier in TIER_MAX_PARAMS whose parameter ceiling the model
         fits under, else Cloud (capacity). Total parameters, MoE included.

    Storage does not enter the assignment: at each tier's own quantization the file on
    disk is far inside the free space that class of device has (1.6GB at the top of
    Mobile, 4.8GB at the top of Tablet, 21.8GB at the top of Workstation), so the test
    never fired. It would only bind under a disk-offload pattern, which is exactly what
    the MoE rule rules out.
    """
    if spec is None:
        return None
    if not spec.get("open"):
        return HARDWARE_TIER_MERGES.get(CLOUD_API_ONLY, CLOUD_API_ONLY)
    size = spec.get("size")
    if size is None or pd.isna(size):
        return None
    for label, max_params in TIER_MAX_PARAMS:
        if size <= max_params:
            return HARDWARE_TIER_MERGES.get(label, label)
    return HARDWARE_TIER_MERGES.get(CLOUD_CAPACITY, CLOUD_CAPACITY)


def is_precision_dependent(spec: dict | None) -> bool:
    """True where the tier depends on the quantization we assume rather than on the
    hardware: above the Workstation INT8 ceiling but inside what INT4 would admit (see
    PRECISION_BAND). These models sit in Cloud by default and should be rendered as a
    transition band -- Workstation at INT4, Cloud at INT8."""
    if spec is None or not spec.get("open"):
        return False
    size = spec.get("size")
    if size is None or pd.isna(size):
        return False
    return PRECISION_BAND[0] < size <= PRECISION_BAND[1]


# main() calls the reporter from every tier-plot branch, but --model-size --hardware
# --boxplot --swarplot in one invocation would then print the same breakdown up to six
# times; remember what has already been reported so each (note, model set) prints once.
_REPORTED_TIER_NOTES: set = set()


def report_hardware_tiers(models: list[str], specs: dict, note: str = "") -> None:
    """Print how `models` split across the hardware tiers, naming any model that could
    not be placed (see hardware_tier) so none is dropped from the tier plots silently."""
    key = (note, tuple(sorted(models)))
    if key in _REPORTED_TIER_NOTES:
        return
    _REPORTED_TIER_NOTES.add(key)

    tiers = {model: hardware_tier(specs.get(model)) for model in models}
    label = f" ({note})" if note else ""
    print(f"\nhardware tiers{label}:")
    for tier_label in PLOTTED_HARDWARE_TIERS:
        count = sum(1 for tier in tiers.values() if tier == tier_label)
        print(f"  {tier_label:<30} {count:3d}")
    unplaceable = sorted(model for model, tier in tiers.items() if tier is None)
    if unplaceable:
        print(
            f"  excluded from the tier plots ({len(unplaceable)}): no published "
            "parameter count in models.csv, and open-weights so not assumable as Cloud:"
        )
        for model in unplaceable:
            print(f"    {model}")


def build_tier_table(summary: pd.DataFrame, specs: dict) -> pd.DataFrame:
    """model / hardware_tier / accuracy / placed_by, ordered by tier then accuracy.

    `placed_by` records WHICH GATE decided the tier -- the access gate (closed weights,
    so Cloud whatever the size) or the capacity gate (the memory model). Most of the
    cloud tier is the access gate, so a cloud statistic quoted without that split is
    easy to over-read as "these models are enormous". `precision_dependent` marks the
    Workstation/Cloud band, where the INT8 quality floor rather than the hardware
    decides. Models that could not be placed are kept with an empty tier rather than
    dropped, so the table accounts for every scored model.
    """

    def _placed_by(model: str) -> str:
        """Which gate decided this model's tier (see hardware_tier)."""
        spec = specs.get(model)
        if spec is None:
            return "unplaced (not in models.csv)"
        if not spec.get("open"):
            return "access gate (closed weights)"
        size = spec.get("size")
        if size is None or pd.isna(size):
            return "unplaced (open, size unpublished)"
        return "capacity gate (size)"

    table = pd.DataFrame(
        {
            "model": summary["model"],
            "hardware_tier": summary["model"].map(
                lambda model: hardware_tier(specs.get(model))
            ),
            "accuracy": summary["accuracy"],
            "placed_by": summary["model"].map(_placed_by),
            "precision_dependent": summary["model"].map(
                lambda model: is_precision_dependent(specs.get(model))
            ),
            "size": summary["model"].map(
                lambda model: (specs.get(model) or {}).get("size")
            ),
        }
    )
    table["hardware_tier"] = pd.Categorical(
        table["hardware_tier"], categories=PLOTTED_HARDWARE_TIERS, ordered=True
    )
    return table.sort_values(
        ["hardware_tier", "accuracy"], ascending=[True, False]
    ).reset_index(drop=True)


# One entry per tier table already written this run, so --model-size --hardware in the
# same invocation writes (and reports) each file once instead of repeating itself.
_WRITTEN_TIER_TABLES: set = set()


def write_hardware_tier_table(
    summary: pd.DataFrame, specs: dict, out_path: str, note: str = ""
) -> pd.DataFrame:
    """Write build_tier_table()'s output to `out_path` and print each tier's n, mean and
    median accuracy.

    This is the per-model audit trail behind the tier plots: the plots show one number
    per tier, so any tier statistic quoted in prose can only be checked against the
    underlying assignment.
    """
    table = build_tier_table(summary, specs)
    if out_path in _WRITTEN_TIER_TABLES:
        return table
    _WRITTEN_TIER_TABLES.add(out_path)

    table.to_csv(out_path, index=False)
    label = f" ({note})" if note else ""
    print(f"\nhardware tier accuracy{label}:")
    print(f"  {'tier':<30}{'n':>5}{'mean':>10}{'median':>10}{'by size':>9}{'band':>7}")
    for tier_label in PLOTTED_HARDWARE_TIERS:
        rows = table[table["hardware_tier"] == tier_label]
        if rows.empty:
            continue
        print(
            f"  {tier_label:<30}{len(rows):5d}"
            f"{rows['accuracy'].mean():9.1f}%{rows['accuracy'].median():9.1f}%"
            f"{int(rows['placed_by'].str.startswith('capacity').sum()):9d}"
            f"{int(rows['precision_dependent'].sum()):7d}"
        )
    n_unplaced = int(table["hardware_tier"].isna().sum())
    if n_unplaced:
        print(f"  {'(no tier -- excluded from plots)':<30}{n_unplaced:5d}")
    print(f"  wrote {out_path}")
    return table


def tier_medians(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per tier x language: median accuracy and the model count behind it.

    `tables` maps language -> build_tier_table() output. Rows keep PLOTTED_HARDWARE_TIERS
    order (smallest hardware first) so the figure and the CSV read the same way.
    """
    rows = []
    for tier in PLOTTED_HARDWARE_TIERS:
        for language, table in tables.items():
            in_tier = table[table["hardware_tier"] == tier]
            if in_tier.empty:
                continue
            rows.append(
                {
                    "hardware_tier": tier,
                    "language": language,
                    "n_models": len(in_tier),
                    "median_accuracy": round(in_tier["accuracy"].median(), 2),
                    "n_placed_by_capacity": int(
                        in_tier["placed_by"].str.startswith("capacity").sum()
                    ),
                    "n_precision_dependent": int(in_tier["precision_dependent"].sum()),
                }
            )
    return pd.DataFrame(rows)


def tier_top_n_means(tables: dict[str, pd.DataFrame], top_n: int) -> pd.DataFrame:
    """One row per tier x language: the MEAN of that tier's `top_n` best models.

    A different question from tier_medians(): the median says what a typical model on
    this hardware scores, this says what the BEST available options score -- which is
    what someone choosing a model for a device actually faces, since they would pick
    from the top of the tier rather than draw from it at random. It is also far less
    sensitive to how many weak models happen to sit in a tier.

    Where a tier holds fewer than `top_n` models every model is used, so `n_used` can be
    below `top_n`; both counts are kept in the output because the two are not the same
    claim and a tier of 4 should not be read as a top-10.
    """
    rows = []
    for tier in PLOTTED_HARDWARE_TIERS:
        for language, table in tables.items():
            in_tier = table[table["hardware_tier"] == tier]
            if in_tier.empty:
                continue
            top = in_tier.nlargest(top_n, "accuracy")
            rows.append(
                {
                    "hardware_tier": tier,
                    "language": language,
                    "n_available": len(in_tier),
                    "n_used": len(top),
                    "top_n_mean_accuracy": round(top["accuracy"].mean(), 2),
                    "top_n_min_accuracy": round(top["accuracy"].min(), 2),
                    "top_n_max_accuracy": round(top["accuracy"].max(), 2),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def plot_tier_dumbbell(
    stats: pd.DataFrame,
    out: str,
    title: str,
    subtitle: str,
    value_col: str = "median_accuracy",
    xlabel: str = "Median accuracy (%)",
    tier_display: dict | None = None,
) -> None:
    """One accuracy statistic per hardware tier, both languages, as a dumbbell plot.

    `value_col` selects what is plotted, so the same figure serves tier_medians() and
    tier_top_n_means() -- the layout is identical and only the statistic changes.

    One row per tier with the two languages joined by a connector: the quantity the
    reader is here for is the GAP between them at each tier, and a connector states that
    gap directly where grouped bars would make it a subtraction. Tiers run
    largest-hardware-first down the page so the reader descends from what is achievable
    to what is deployable.

    Both marks are directly labelled, so identity never rests on colour alone, and the
    values are readable in greyscale or print.
    """
    medians = stats
    # PLOTTED_HARDWARE_TIERS runs smallest hardware first, and matplotlib's y axis
    # increases upward, so plotting in that order puts Cloud at the top of the page --
    # the reader descends from what is achievable to what is deployable.
    tiers = [t for t in PLOTTED_HARDWARE_TIERS if t in set(medians["hardware_tier"])]

    fig, ax = plt.subplots(figsize=(10, 1.15 * len(tiers) + 2.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for y, tier in enumerate(tiers):
        rows = medians[medians["hardware_tier"] == tier]
        points = dict(zip(rows["language"], rows[value_col]))
        if len(points) == 2:
            lo, hi = min(points.values()), max(points.values())
            # Connector stops short of each marker so the dots stay distinct shapes
            # rather than reading as one capsule.
            ax.plot([lo + 0.6, hi - 0.6], [y, y], color=CONNECTOR, lw=2, zorder=1)
        for language, value in points.items():
            colour = SERIES[language]["color"]
            ax.scatter(
                value,
                y,
                s=170,
                color=colour,
                zorder=3,
                edgecolors=SURFACE,  # 2px surface ring keeps overlapping marks legible
                linewidths=2,
            )
            # Label outward from the pair, so the two never collide at close medians.
            others = [v for k, v in points.items() if k != language]
            outward_left = bool(others) and value <= min(others)
            ax.annotate(
                f"{value:.0f}%",
                (value, y),
                xytext=(-14 if outward_left else 14, 0),
                textcoords="offset points",
                ha="right" if outward_left else "left",
                va="center",
                fontsize=15,
                color=INK_PRIMARY,
            )

    ax.set_yticks(range(len(tiers)))
    ax.set_yticklabels([])
    for y, tier in enumerate(tiers):
        name, memory = (tier_display or TIER_DISPLAY).get(tier, (tier, ""))
        # Offsets in POINTS, not data units: the name/sub-label pair must stay the same
        # distance apart whether the figure has two rows or four. In data units the gap
        # would grow as rows get taller, and the two lines would stop reading as one
        # label.
        for text, dy, size, colour in (
            (name, 7, 16, INK_PRIMARY),
            (memory, -9, 12, INK_MUTED),
        ):
            ax.annotate(
                text,
                xy=(-0.02, y),
                xycoords=("axes fraction", "data"),
                xytext=(0, dy),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=size,
                color=colour,
            )

    # Headroom past 100 so the rightmost value label sits inside the figure rather than
    # against its edge; the ticks still stop at the real maximum of the scale.
    ax.set_xlim(0, 108)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_ylim(-0.55, len(tiers) - 0.45)
    ax.set_xlabel(xlabel, fontsize=12, color=INK_MUTED)
    ax.tick_params(axis="x", labelsize=12, colors=INK_MUTED, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRIDLINE, lw=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRIDLINE)

    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=11,
            markerfacecolor=SERIES[lang]["color"],
            markeredgecolor=SURFACE,
            label=SERIES[lang]["label"],
        )
        for lang in ("kiswahili", "english")
    ]
    # Explicit margins rather than tight_layout: the tier labels are drawn outside the
    # axes, which tight_layout cannot measure, and the header (title, subtitle, legend)
    # needs a fixed band that does not shift when the tier count changes.
    fig.subplots_adjust(left=0.24, right=0.97, top=0.62, bottom=0.15)
    fig.text(
        0.02,
        0.96,
        title,
        ha="left",
        va="top",
        fontsize=20,
        fontweight="bold",
        color=INK_PRIMARY,
        linespacing=1.3,
    )
    fig.text(
        0.02,
        0.80,
        subtitle,
        ha="left",
        va="top",
        fontsize=12.5,
        color=INK_SECONDARY,
        linespacing=1.4,
    )
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.015, 0.70),
        ncol=2,
        frameon=False,
        fontsize=13,
        handletextpad=0.4,
        columnspacing=1.6,
        labelcolor=INK_SECONDARY,
    )
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    for language in LANGUAGES:
        ap.add_argument(
            f"--experiment-id-{language}",
            default=DEFAULT_EXPERIMENT_ID[language],
            help=f"{language} experiment-id whose summary CSV to read "
            f"(default: {DEFAULT_EXPERIMENT_ID[language]})",
        )
        ap.add_argument(
            f"--summary-{language}",
            default=None,
            help=f"exact {language} summary CSV, overriding --experiment-id-{language}",
        )
    ap.add_argument("--models-csv", default=str(MODELS_CSV))
    ap.add_argument("--outdir", default=os.path.join(root, "results", "analysis"))
    ap.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    ap.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="how many of a tier's best models the top-N-mean figure averages "
        "(default: 10); a tier holding fewer uses all of them",
    )
    ap.add_argument(
        "--title",
        default="Do AI models know the structure of languages\nto teach foundational literacy?",
    )
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    specs = load_model_hardware(args.models_csv)

    _section("CONFIG")
    print(f"  models.csv        : {args.models_csv} ({len(specs)} models)")

    tables, labels = {}, {}
    for language in LANGUAGES:
        summary_path = getattr(args, f"summary_{language}")
        experiment_id = getattr(args, f"experiment_id_{language}")
        if summary_path is None:
            summary_path = os.path.join(
                root,
                "results",
                f"summary_linguistic-knowledge_{language}_{experiment_id}.csv",
            )
        if not os.path.exists(summary_path):
            raise SystemExit(
                f"Missing {language} summary CSV: {summary_path}\n"
                f"Point at it with --summary-{language}, or pass the "
                f"--experiment-id-{language} that produced it."
            )
        print(f"  {language:<18}: {summary_path}")
        summary = pd.read_csv(summary_path)
        summary = summary[summary["model"].notna()]
        labels[language] = f"{language}_{experiment_id}"
        report_hardware_tiers(summary["model"].tolist(), specs, note=language)
        tables[language] = write_hardware_tier_table(
            summary,
            specs,
            os.path.join(args.outdir, f"hardware_tiers_{labels[language]}.csv"),
            note=language,
        )

    title = args.title.replace("\\n", "\n")

    # Two views of the same tiers, deliberately not interchangeable. The median says
    # what a TYPICAL model on this hardware scores; the top-N mean says what the BEST
    # available options score, which is the number someone choosing a model for a device
    # actually faces. The second is also far less sensitive to how many weak models
    # happen to have been evaluated in a tier.
    medians = tier_medians(tables)
    medians_csv = os.path.join(args.outdir, "hardware_tiers_medians.csv")
    medians.to_csv(medians_csv, index=False)
    medians_figure = os.path.join(args.outdir, f"hardware_tiers_medians.{args.format}")
    plot_tier_dumbbell(
        medians,
        medians_figure,
        title,
        "Median accuracy on the Linguistic Knowledge Benchmark by hardware tier.\n"
        "What bounds each tier shown in grey.",
        value_col="median_accuracy",
        xlabel="Median accuracy (%)",
    )

    top = tier_top_n_means(tables, args.top_n)
    top_csv = os.path.join(args.outdir, f"hardware_tiers_top{args.top_n}_mean.csv")
    top.to_csv(top_csv, index=False)
    top_figure = os.path.join(
        args.outdir, f"hardware_tiers_top{args.top_n}_mean.{args.format}"
    )
    plot_tier_dumbbell(
        top,
        top_figure,
        title,
        f"Mean accuracy of the {args.top_n} best models in each hardware tier.",
        value_col="top_n_mean_accuracy",
        xlabel=f"Mean accuracy of the top {args.top_n} (%)",
    )

    # Slide cut of the same top-N statistic: two rows, plain-language labels, no
    # hardware detail. Same numbers as the figure above -- only the framing changes.
    slide = top[top["hardware_tier"].isin(SLIDE_TIERS)].reset_index(drop=True)
    slide_csv = os.path.join(
        args.outdir, f"hardware_tiers_top{args.top_n}_mean_slide.csv"
    )
    slide.to_csv(slide_csv, index=False)
    slide_figure = os.path.join(
        args.outdir, f"hardware_tiers_top{args.top_n}_mean_slide.{args.format}"
    )
    plot_tier_dumbbell(
        slide,
        slide_figure,
        title,
        f"Average accuracy of the {args.top_n} best models in each group.",
        value_col="top_n_mean_accuracy",
        xlabel="Average accuracy (%)",
        tier_display=SLIDE_TIER_DISPLAY,
    )

    _section("TIER MEDIANS (all models in the tier)")
    print(medians.to_string(index=False))
    _section(f"TIER TOP-{args.top_n} MEANS (the best models in the tier)")
    print(top.to_string(index=False))
    short = top[top["n_used"] < args.top_n]["hardware_tier"].unique()
    if len(short):
        print(
            f"\n  NOTE: {', '.join(short)} holds fewer than {args.top_n} models, so its "
            "figure is\n  the mean of every model in the tier -- not comparable to a "
            "true top-10."
        )
    _section("OUTPUT FILES")
    for path in [
        *(
            os.path.join(args.outdir, f"hardware_tiers_{label}.csv")
            for label in labels.values()
        ),
        medians_csv,
        medians_figure,
        top_csv,
        top_figure,
        slide_csv,
        slide_figure,
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
