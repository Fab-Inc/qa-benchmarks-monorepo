"""Word-level failure analysis for Kiswahili

Change `BASE` path to point to a different run.

Every score here is adjusted balanced accuracy = recall - FPR.

Two tables are used for plotting:
    marginals    one row per (model, feature, level)
    combinations one row per (model, cell) where cell is the combination of features
"""

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = (
    Path(__file__).resolve().parents[3] if "__file__" in globals() else Path.cwd()
)
BASE = (
    REPO_ROOT
    / "outputs/phonics-mixture-corrected-20260921/selections/beta1-8-provisional"
)
REPORT = BASE / "kiswahili/report"

# Palette and furniture copied from plot_report_f2_leaderboards.py on the
# add-phonics-local-merge branch, so these figures sit alongside the F2
# leaderboards without a visual jump.
SURFACE, INK_PRIMARY, INK_SECONDARY = "#fcfcfb", "#0b0b0b", "#52514e"
INK_MUTED, GRIDLINE, BASELINE = "#898781", "#e1e0d9", "#c3c2b7"
TIER_COLORS = {"Cloud": "#1f4e79", "Mobile + Tablet": "#b5651d"}
FEATURES = [
    "is_known_word",
    "grapheme_bin",
    "has_multigraph",
    "has_cluster",
    # "has_nonnasal_cluster",
    # "has_nasal_cluster",
]
TAGS = {
    "is_known_word": "known",
    "has_multigraph": "multi",
    "has_cluster": "CLUST",
    # "has_nonnasal_cluster": "clust",
    # "has_nasal_cluster": "NASAL",
}
GRAPHEME_BINS = pd.IntervalIndex.from_tuples([(1, 5), (5, 9), (9, 14)])
# Level order for plotting. `level` is stored as a label rather than a mix of
# Interval and bool, because pandas cannot sort those two types against each
# other once all four features are stacked into one long table.
LEVEL_ORDER = {
    "is_known_word": ["False", "True"],
    "grapheme_bin": ["2-5", "6-9", "10-14"],
    "has_multigraph": ["False", "True"],
    "has_cluster": ["False", "True"],
    # "has_nonnasal_cluster": ["False", "True"],
    # "has_nasal_cluster": ["False", "True"],
}


def is_nasal(cluster):
    """Nasal-initial cluster, excluding ng/ng'/ny, which are single sounds."""
    return cluster[:1] in "mn" and not cluster.startswith(("ng", "ng'", "ny"))


# --------------------------------------------------------------------------
# Prepare data
# --------------------------------------------------------------------------
def build():
    """Explode items to words and attach decodability features.

    The two library calls have different keys, so they are deduplicated
    separately. decompose_word depends only on spelling, so it runs once per
    unique word. analyse_word_kiswahili also needs the student profile, which
    is fixed within a benchmark item, so it runs once per (word, profile) pair
    rather than once per row across all models.
    """
    from decodability.kiswahili import KiswahiliStudentKnowledge, analyse_word_kiswahili
    from decodability.kiswahili.segment import decompose_word

    df = pd.read_csv(REPORT / "results_with_scoring.csv")
    tiers = pd.read_csv(BASE / "hardware_tiers/pooled_f2/hardware_tiers_kiswahili.csv")
    # Keep only models with hardware tiers.
    df = df.merge(tiers[["model", "hardware_tier"]], on="model", how="inner")
    for col in ["vocab_set", "response_parsed", "nondecodable_words"]:
        df[col] = df[col].apply(eval)

    item_words = (
        df[
            [
                "benchmark_item_id",
                "model",
                "hardware_tier",
                "vocab_set",
                "student_knowledge",
                "response_parsed",
                "nondecodable_words",
            ]
        ]
        .explode("vocab_set")
        .rename(columns={"vocab_set": "word"})
    )

    words = pd.DataFrame({"word": item_words.word.unique()})
    parts = [decompose_word(w) for w in words.word]
    pairs = item_words[["word", "student_knowledge"]].drop_duplicates()
    profiles = {
        s: KiswahiliStudentKnowledge.model_validate(eval(s))
        for s in pairs.student_knowledge.unique()
    }
    pairs["is_known_word"] = [
        analyse_word_kiswahili(w, profiles[s]).known_whole_word
        for w, s in zip(pairs.word, pairs.student_knowledge)
    ]
    item_words = item_words.merge(pairs, on=["word", "student_knowledge"])

    words["n_graphemes"] = [len(p.graphemes) for p in parts]
    words["grapheme_bin"] = pd.cut(
        words.n_graphemes, GRAPHEME_BINS, include_lowest=True
    )
    words["has_multigraph"] = [any(len(g.text) > 1 for g in p.graphemes) for p in parts]
    words["has_nonnasal_cluster"] = [
        any(not is_nasal(c.text) for c in p.clusters) for p in parts
    ]
    words["has_nasal_cluster"] = [
        any(is_nasal(c.text) for c in p.clusters) for p in parts
    ]
    words["has_cluster"] = [len(p.clusters) > 0 for p in parts]
    item_words = item_words.merge(words, on="word")

    item_words["predicted"] = [
        w.lower() in r for w, r in zip(item_words.word, item_words.response_parsed)
    ]
    item_words["is_nondecodable"] = [
        w.lower() in n for w, n in zip(item_words.word, item_words.nondecodable_words)
    ]
    return item_words.drop(
        columns=["student_knowledge", "response_parsed", "nondecodable_words"]
    )


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def tally(words, keys, min_pos=50, min_neg=50):
    """Counts and Youden's J per group.

    J is blanked, not the row dropped, where a group holds too few actual
    positives or negatives for a stable rate.
    """
    counts = (
        words.assign(
            tp=words.predicted & words.is_nondecodable,
            fp=words.predicted & ~words.is_nondecodable,
            pos=words.is_nondecodable,
            neg=~words.is_nondecodable,
        )
        .groupby(keys, observed=True)[["tp", "fp", "pos", "neg"]]
        .sum()
    )
    counts["n"] = counts.pos + counts.neg
    counts["recall"] = counts.tp / counts.pos
    counts["fpr"] = counts.fp / counts.neg
    counts["j"] = counts.recall - counts.fpr
    counts.loc[(counts.pos < min_pos) | (counts.neg < min_neg), "j"] = np.nan
    return counts.reset_index()


def marginal_table(words, **kwargs):
    """One row per (model, feature, level), marginalising over other features."""
    frames = [
        tally(words, ["hardware_tier", "model", feature], **kwargs)
        .rename(columns={feature: "level"})
        .assign(feature=feature, level=lambda d: d.level.map(level_label))
        for feature in FEATURES
    ]
    cols = [
        "hardware_tier",
        "model",
        "feature",
        "level",
        "n",
        "pos",
        "neg",
        "recall",
        "fpr",
        "j",
    ]
    return pd.concat(frames, ignore_index=True)[cols]


def combination_table(words, **kwargs):
    """One row per (model, cell) over the full cross product of the features."""
    combos = tally(words, ["hardware_tier", "model"] + FEATURES, **kwargs)
    combos["cell"] = combos.apply(cell_label, axis=1)
    combos["group"] = combos.apply(cell_group, axis=1)
    return combos


def cell_group(row):
    """Reading order for the heatmap columns.

    Known words first, then words with neither multigraph nor cluster, then
    multigraphs, then non-nasal clusters, then nasal clusters. A word carrying
    both cluster kinds lands in the nasal group, and a multigraph alongside a
    cluster is ordered by the cluster, so every cell appears exactly once.
    """
    if row.is_known_word:
        return 0
    if row.has_cluster:
        return 3
    # if row.has_nasal_cluster:
    #     return 4
    # if row.has_nonnasal_cluster:
    #     return 3
    if row.has_multigraph:
        return 2
    return 1


def ordered_cells(combinations):
    """Cell names in reading order: group, then word length, then multigraph."""
    spec = combinations[
        ["cell", "group", "grapheme_bin", "has_multigraph"]
    ].drop_duplicates()
    return spec.sort_values(["group", "grapheme_bin", "has_multigraph"]).cell.tolist()


def cell_label(row):
    span = row.grapheme_bin
    return " ".join(
        [
            f"{int(span.left) + 1}-{int(span.right)}g",
            *[tag for col, tag in TAGS.items() if row[col]],
        ]
    )


def level_label(value):
    if isinstance(value, pd.Interval):
        return f"{int(value.left) + 1}-{int(value.right)}"
    return str(value)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
PLOT_FEATURES = [f for f in FEATURES if f != "is_known_word"]


def overall_ranking(words):
    """Models ordered by Youden's J over every word, best first.

    This is J on the pooled confusion matrix, not the average of the per-level
    J values shown in the panels. Averaging levels would give a level holding
    400 words the same weight as one holding 400,000.
    """
    return (
        tally(words, ["model"]).set_index("model").j.sort_values(ascending=False).index
    )


def display_names():
    """Model ids mapped to the display names the F2 leaderboards use."""
    tiers = pd.read_csv(BASE / "hardware_tiers/pooled_f2/hardware_tiers_kiswahili.csv")
    return dict(zip(tiers.model, tiers.display_name.fillna(tiers.model)))


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, length=3, width=0.8, labelsize=9)


def plot_marginals(marginals, fig, models):
    """One heatmap per feature, models shared down the y axis.

    Marginals put every model at every level, so these panels have no gaps.
    Each square is annotated because nine columns leave room for it, and a
    printed number beats reading a value off a colour ramp.
    """
    grids = []
    for feature in PLOT_FEATURES:
        panel = marginals[marginals.feature == feature]
        grid = panel.pivot_table(index="model", columns="level", values="j")
        grids.append(
            grid.reindex(columns=[lv for lv in LEVEL_ORDER[feature] if lv in grid])
        )

    names, tiers = (
        display_names(),
        marginals.drop_duplicates("model").set_index("model").hardware_tier,
    )
    fig.set_facecolor(SURFACE)
    axes = fig.subplots(
        1, len(grids), sharey=True, width_ratios=[g.shape[1] for g in grids]
    )

    for ax, feature, grid in zip(axes, PLOT_FEATURES, grids):
        grid = grid.loc[models]
        # Missing values get their own neutral, so a cell nobody can score is
        # not mistaken for a cell everybody scores badly.
        cmap = plt.get_cmap("YlGnBu").with_extremes(bad=GRIDLINE)
        im = ax.imshow(grid.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for row, col in np.ndindex(grid.shape):
            value = grid.values[row, col]
            if not np.isnan(value):
                ax.text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=SURFACE if value > 0.55 else INK_PRIMARY,
                )
        ax.set_xticks(range(grid.shape[1]), grid.columns, fontsize=8)
        ax.set_title(
            feature.replace("has_", "").replace("_", " "),
            fontsize=10,
            fontweight="semibold",
            color=INK_PRIMARY,
        )
        style(ax)

    axes[0].set_yticks(
        range(len(models)), [names.get(m, m) for m in models], fontsize=9
    )
    for label, model in zip(axes[0].get_yticklabels(), models):
        label.set_color(TIER_COLORS[tiers[model]])
    fig.colorbar(
        im, ax=axes, label="Adj. Balanced Accuracy (0 = chance)", shrink=0.5, pad=0.02
    )
    return models


# Dumbbell furniture ported from plot_dumbbell in analyze_phonics_hardware_tiers.py
# on add-phonics-local-merge. The two series are the levels of a word feature
# rather than the two languages, since everything here is Kiswahili.
TIER_ORDER = ["Mobile + Tablet", "Cloud"]
TIER_LABELS = {
    "Cloud": ("Cloud", "Datacenter or API-only"),
    "Mobile + Tablet": ("Mobile + tablet", "Up to 12GB RAM, INT4"),
}
# Green and dark pink rather than the language charts' blue and orange, so a
# feature split and a language split stay distinguishable side by side.
LEVEL_COLORS = {"False": "#63b7bb", "True": "#0b4f52"}
TOP_N = 10
FEATURE_PLOTS = {
    "has_multigraph": ("multigraphs", "No multigraph", "Has a multigraph"),
    "has_cluster": ("consonant clusters", "No cluster", "Has a cluster"),
    # "has_nonnasal_cluster": (
    #     "non-nasal consonant clusters",
    #     "No non-nasal cluster",
    #     "Has a non-nasal cluster",
    # ),
    # "has_nasal_cluster": (
    #     "nasal consonant clusters",
    #     "No nasal cluster",
    #     "Has a nasal cluster",
    # ),
}


def top_n_stats(marginals, feature):
    """Mean Adj. balanced accuracy of the best TOP_N models per (tier, level)
    for one feature.

    Top-N is picked within each (tier, level), as summarize_tiers does for
    languages. The two dots on a row can therefore rest on different models:
    the chart asks what the best available models achieve on each kind of
    word, not how one fixed roster moves between them.
    """
    panel = marginals[marginals.feature == feature].dropna(subset=["j"])
    rows = []
    for (tier, level), group in panel.groupby(
        ["hardware_tier", "level"], observed=True
    ):
        top = group.sort_values(["j", "model"], ascending=[False, True]).head(TOP_N)
        rows.append(
            {
                "hardware_tier": tier,
                "level": level,
                "median_j": top.j.mean(),
                "n_models": len(top),
                "n_available": len(group),
                "words": top.n.median(),
            }
        )
    return pd.DataFrame(rows)


def draw_dumbbell(ax, y, points, size=170, fontsize=15):
    """One row: two dots joined by a connector, each labelled with its value.

    Y is categorical, so near-equal values are nudged apart vertically instead
    of overprinting.
    """
    close = max(points.values()) - min(points.values()) < 3
    point_y = {
        lv: y + ({"False": -0.12, "True": 0.12}[lv] if close else 0) for lv in points
    }
    low, high = min(points.values()), max(points.values())
    if close:
        ax.plot(
            list(points.values()),
            list(point_y.values()),
            color=BASELINE,
            lw=2,
            zorder=1,
        )
    elif high - low > 1.2:
        ax.plot([low + 0.6, high - 0.6], [y, y], color=BASELINE, lw=2, zorder=1)
    for level, score in points.items():
        ax.scatter(
            score,
            point_y[level],
            s=size,
            color=LEVEL_COLORS[level],
            zorder=3,
            edgecolors=SURFACE,
            linewidths=2,
        )
        left = score <= min(v for lv, v in points.items() if lv != level)
        ax.annotate(
            f"{score:.1f}%",
            (score, point_y[level]),
            xytext=(-12 if left else 12, 0),
            textcoords="offset points",
            ha="right" if left else "left",
            va="center",
            fontsize=fontsize,
            color=INK_PRIMARY,
        )


def level_legend(fig, off_label, on_label, anchor):
    fig.legend(
        handles=[
            plt.Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=11,
                markerfacecolor=LEVEL_COLORS[lv],
                markeredgecolor=SURFACE,
                label=label,
            )
            for lv, label in [("False", off_label), ("True", on_label)]
        ],
        loc="upper left",
        bbox_to_anchor=anchor,
        ncol=2,
        frameon=False,
        fontsize=12,
        labelcolor=INK_SECONDARY,
    )


def plot_feature_summary_dumbell(marginals):
    """All three features on one axis: a block of two tier rows per feature.

    Each block is the single-feature chart repeated, so the eye compares tiers
    within a feature and features within a tier without switching charts. The
    per-row model and word counts are dropped here because six rows cannot
    carry them; they stay on the single-feature charts and in the CSV. The
    right-hand column is the gap in percentage points, which is what the
    connector length already encodes.
    """
    features = list(FEATURE_PLOTS)
    # y grows upward, so build bottom-up and the first feature lands on top.
    layout = [(f, t) for f in reversed(features) for t in TIER_ORDER]

    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    fig.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for y, (feature, tier) in enumerate(layout):
        stats = top_n_stats(marginals, feature)
        group = stats[stats.hardware_tier.eq(tier)]
        points = dict(zip(group.level, group.median_j * 100))
        draw_dumbbell(ax, y, points, size=130, fontsize=12)
        ax.annotate(
            TIER_LABELS[tier][0],
            (-0.02, y),
            xycoords=("axes fraction", "data"),
            ha="right",
            va="center",
            fontsize=12,
            color=INK_SECONDARY,
        )

    # Feature heading sits between its two tier rows; a rule separates blocks.
    for block, feature in enumerate(reversed(features)):
        ax.annotate(
            textwrap.fill(FEATURE_PLOTS[feature][0], 19),
            (-0.02, 2 * block + 0.5),
            xycoords=("axes fraction", "data"),
            xytext=(-118, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=13,
            fontweight="semibold",
            color=INK_PRIMARY,
            linespacing=1.3,
        )
        if block:
            ax.axhline(2 * block - 0.5, color=GRIDLINE, lw=1, zorder=0)

    ax.set_xlim(-14, 112)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_ylim(-0.7, len(layout) - 0.3)
    ax.set_yticks([])
    ax.set_xlabel(
        f"Mean adjusted balanced accuracy of top {TOP_N} models",
        fontsize=12,
        color=INK_MUTED,
    )
    ax.tick_params(axis="x", labelsize=12, colors=INK_MUTED, length=0)
    ax.grid(axis="x", color=GRIDLINE, lw=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRIDLINE)
    fig.subplots_adjust(left=0.38, right=0.96, top=0.78, bottom=0.14)

    fig.text(
        0.025,
        0.975,
        "How well do models handle Kiswahili's multigraphs and consonant clusters?",
        va="top",
        fontsize=18,
        fontweight="bold",
        color=INK_PRIMARY,
    )
    fig.text(
        0.025,
        0.915,
        f"Mean adjusted balanced accuracy of the top {TOP_N} eligible models per group,"
        "by word feature.",
        va="top",
        fontsize=11,
        color=INK_SECONDARY,
    )
    level_legend(fig, "Without the feature", "With the feature", (0.025, 0.855))
    fig.text(
        0.025,
        0.02,
        "Kiswahili only; Eligibility: >80% coverage; minimum scored "
        "items."
        "Device tiers assume quantized weights; "
        "benchmark scores do not establish quality after quantization.",
        va="bottom",
        fontsize=8.5,
        color=INK_SECONDARY,
        linespacing=1.55,
    )
    return fig


def plot_combinations(combinations, ax, models):
    """Models against the full cross product of features, plain to clustered.

    A cell no model can score is dropped from the axis rather than left as a
    blank column. That removes the thin combinations (a multigraph alongside a
    cluster, and most 10-14 grapheme clusters) and the known-word cells, whose
    J is undefined because every known word is decodable.
    """
    grid = combinations.pivot_table(index="model", columns="cell", values="j")
    order = [cell for cell in ordered_cells(combinations) if cell in grid]
    grid = grid[order].loc[models]

    cmap = plt.get_cmap("YlGnBu").with_extremes(bad=GRIDLINE)
    im = ax.imshow(grid.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for row, col in np.ndindex(grid.shape):
        value = grid.values[row, col]
        if not np.isnan(value):
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color=SURFACE if value > 0.55 else INK_PRIMARY,
            )
    names = display_names()
    tiers = combinations.drop_duplicates("model").set_index("model").hardware_tier
    ax.set_xticks(
        range(grid.shape[1]), grid.columns, rotation=45, ha="right", fontsize=8
    )
    ax.set_yticks(range(len(grid)), [names.get(m, m) for m in grid.index], fontsize=9)
    for label, model in zip(ax.get_yticklabels(), grid.index):
        label.set_color(TIER_COLORS[tiers[model]])
    ax.set_title(
        "Adj. BA by word-type combination, ordered plain to clustered",
        fontsize=12,
        fontweight="semibold",
        color=INK_PRIMARY,
    )
    style(ax)
    ax.figure.colorbar(im, ax=ax, label="Adj. BA  (0 = chance)", shrink=0.5, pad=0.02)
    return grid


def plot_known_word_alarms(words, ax, models):
    """Known whole words are decodable by definition, so J is undefined there.

    What remains is how often a model flags a word the student already knows,
    which is a prompt-following failure rather than a phonics one.
    """
    names = display_names()
    # Same model order as the marginals panel, so the two figures line up row
    # for row, rather than each sorting by its own quantity.
    rate = (
        words[words.is_known_word]
        .groupby("model", observed=True)
        .predicted.mean()
        .reindex(models)
    )
    tiers = words.drop_duplicates("model").set_index("model").hardware_tier
    ax.figure.set_facecolor(SURFACE)
    ax.barh(
        range(len(rate)),
        rate.values,
        height=0.65,
        zorder=3,
        color=[TIER_COLORS[tiers[m]] for m in rate.index],
    )
    ax.set_yticks(range(len(rate)), [names.get(m, m) for m in rate.index], fontsize=9)
    ax.set_ylim(len(rate) - 0.3, -0.8)
    ax.set_xlabel("share of known words falsely flagged", color=INK_SECONDARY)
    ax.set_title(
        "False alarms on words the student already knows",
        fontsize=12,
        fontweight="semibold",
        color=INK_PRIMARY,
    )
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8)
    style(ax)


if __name__ == "__main__":
    words = build()
    marginals = marginal_table(words, min_pos=20, min_neg=20)
    combinations = combination_table(words)
    order = {cell: i for i, cell in enumerate(ordered_cells(combinations))}
    combinations = combinations.sort_values(
        ["model", "cell"], key=lambda c: c.map(order) if c.name == "cell" else c
    )
    marginals.to_csv(REPORT / "word_feature_marginals.csv", index=False)
    combinations.to_csv(REPORT / "word_feature_combinations.csv", index=False)

    # Height tracks the model count, as the F2 leaderboards do.
    n = words.model.nunique()
    ranking = overall_ranking(words)
    fig = plt.figure(figsize=(11, max(5, 2.0 + 0.34 * n)), layout="constrained")
    plot_marginals(marginals, fig, ranking)
    fig.suptitle(
        "Marginal model performance by word feature",
        fontsize=12,
        fontweight="semibold",
        color=INK_PRIMARY,
    )
    fig.savefig(
        REPORT / "word_feature_marginals.png",
        dpi=300,
        bbox_inches="tight",
        facecolor=SURFACE,
    )

    fig, ax = plt.subplots(figsize=(12, max(5, 2.0 + 0.34 * n)), layout="constrained")
    shown = plot_combinations(combinations, ax, ranking)
    fig.savefig(
        REPORT / "word_type_heatmap.png",
        dpi=300,
        bbox_inches="tight",
        facecolor=SURFACE,
    )

    fig, ax = plt.subplots(figsize=(9, max(5, 2.0 + 0.34 * n)), layout="constrained")
    plot_known_word_alarms(words, ax, ranking)
    fig.savefig(
        REPORT / "known_word_false_alarms.png",
        dpi=300,
        bbox_inches="tight",
        facecolor=SURFACE,
    )

    summary = plot_feature_summary_dumbell(marginals)
    summary.savefig(
        REPORT / "tier_medians_all_features.png", dpi=300, facecolor=SURFACE
    )

    dropped = sorted(set(combinations.cell) - set(shown.columns))
    sizes = combinations.groupby("cell").n.median()
    print(
        f"models: {n}   marginal rows: {len(marginals)}"
        f"   cells: {combinations.cell.nunique()}"
    )
    print(f"heatmap shows {shown.shape[1]}; dropped {len(dropped)} no model can score:")
    print("  " + ", ".join(f"{cell} (median n={int(sizes[cell])})" for cell in dropped))
