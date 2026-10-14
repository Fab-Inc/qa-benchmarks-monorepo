"""Rescale saved BA plots and CIs to adjusted BA points (0 chance, 100 perfect)."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, StrMethodFormatter


def plot_adjusted_leaderboard(table, colors, language, required, gold_note):
    """Plot saved adjusted-BA points, retaining negative scores and CI tails."""
    from plot_report_f2_leaderboards import plot_leaderboard

    plot_table = table.copy()
    plot_table[["score", "ci_low", "ci_high"]] /= 100
    fig = plot_leaderboard(plot_table, colors, language)
    ax = fig.axes[0]
    ax.set_xlim(min(-0.025, float(table.ci_low.min()) / 100 - 0.01), 1)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value * 100:.0f}"))
    ax.set_xlabel("Adjusted balanced accuracy (0 = chance; 100 = perfect)")
    ax.set_title(
        f"{language.title()} decodability · Adjusted balanced "
        f"accuracy\nLeaderboard · >90% coverage · ≥{required:,} evaluated items",
        loc="left",
        fontsize=12,
        fontweight="semibold",
        pad=35,
    )
    labels = [
        label
        for label in ax.texts
        if label.get_position()[0] == 1.02 and label.get_text() != "F2"
    ]
    assert len(labels) == len(table)
    for label, value in zip(labels, table.score, strict=True):
        label.set_text(f"{value:.1f}")
    for label in ax.texts:
        if label.get_text() == "F2":
            label.set_text("Adj. BA")
        elif label.get_text().startswith("Items = scorable"):
            label.set_text(
                "Adjusted BA = 100 × (2 × BA − 1). Dashed line = chance / "
                "flag-all / flag-none (0); negative values are retained."
            )
    ax.axvline(0, color="#454545", ls="--", lw=1, zorder=5)
    ax.annotate(
        gold_note,
        (0, 0),
        xytext=(0, -57),
        xycoords="axes fraction",
        textcoords="offset points",
        fontsize=8,
        color="#52514e",
        va="top",
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    scripts = Path(__file__).resolve().parent
    import analyze_phonics_hardware_tiers as hardware
    from plot_report_f2_leaderboards import SURFACE, digest, plot_leaderboard, read_csv
    from run_f2_analysis import inspect_dataset, verify_merge

    merged_before = verify_merge(root)
    datasets = {lang: inspect_dataset(root / lang) for lang in ("english", "kiswahili")}
    source = root / "balanced_accuracy"
    source_hardware = root / "hardware_tiers/balanced_accuracy/coverage_gt90"
    inputs = [
        p
        for directory in (source, source_hardware)
        for p in directory.rglob("*")
        if p.is_file()
    ]
    before = {str(p.relative_to(root)): digest(p) for p in inputs}
    out = root / "adjusted_balanced_accuracy"
    tier_out = root / "hardware_tiers/adjusted_balanced_accuracy/coverage_gt90"
    out.mkdir(exist_ok=True)
    tier_out.mkdir(parents=True, exist_ok=True)
    colors = (
        read_csv(root / "plot_metadata/providers.csv")
        .set_index("provider")
        .color.to_dict()
    )
    details, audits, outputs = {}, [], []
    metric = "adjusted_balanced_accuracy"

    for lang, (size, gold_note) in datasets.items():
        raw = read_csv(
            source / f"{lang}_balanced_accuracy_leaderboard_coverage_gt90.csv"
        )
        table = raw.copy()
        for generic, saved in (
            ("score", metric),
            ("ci_low", f"{metric}_ci_low"),
            ("ci_high", f"{metric}_ci_high"),
        ):
            np.testing.assert_allclose(
                raw[saved], 100 * (2 * raw[generic] - 1), rtol=0, atol=1e-11
            )
            table[generic] = raw[saved]
        assert table.score.is_monotonic_decreasing
        assert table.ci_low.ge(-100).all() and table.ci_high.le(100).all()
        assert table.adjusted_balanced_accuracy_scale.eq(100).all()
        np.testing.assert_allclose(
            table.score,
            100 * (table.recall - table.false_positive_rate),
            rtol=0,
            atol=1e-11,
        )
        required = hardware.minimum_scorable_items(size, 700, 0.9)
        assert table.n_items.ge(required).all()
        fig = plot_adjusted_leaderboard(table, colors, lang, required, gold_note)
        stem = out / f"{lang}_adjusted_balanced_accuracy_leaderboard_coverage_gt90"
        for ext in ("png", "pdf"):
            path = stem.with_suffix(f".{ext}")
            fig.savefig(path, dpi=250, bbox_inches="tight", facecolor=SURFACE)
            outputs.append(path)
        plt.close(fig)
        table.to_csv(stem.with_suffix(".csv"), index=False)
        outputs.append(stem.with_suffix(".csv"))
        pd.testing.assert_frame_equal(
            read_csv(stem.with_suffix(".csv"))[["score", "ci_low", "ci_high"]],
            table[["score", "ci_low", "ci_high"]],
            check_exact=True,
        )
        details[lang] = dict(
            n_models=len(table),
            target_items=size,
            minimum_scorable_items=required,
            below_or_at_zero=int(table.score.le(0).sum()),
            ci_contains_zero=int((table.ci_low.le(0) & table.ci_high.ge(0)).sum()),
        )
        print(lang, details[lang], flush=True)

        audit = read_csv(source_hardware / f"hardware_tiers_{lang}_audit.csv")
        for column in (
            "balanced_accuracy",
            "balanced_accuracy_ci_low",
            "balanced_accuracy_ci_high",
        ):
            audit[column.replace("balanced_accuracy", metric)] = 100 * (
                2 * audit[column] - 1
            )
        audit["metric"] = metric
        audit["adjusted_balanced_accuracy_scale"] = 100
        audit.to_csv(tier_out / f"hardware_tiers_{lang}_audit.csv", index=False)
        audit[audit.included].to_csv(
            tier_out / f"hardware_tiers_{lang}.csv", index=False
        )
        assert not audit.loc[audit.included, "hardware_tier"].eq("Workstation").any()
        audits.append(audit)

    medians, means, selected = hardware.summarize_tiers(
        pd.concat(audits, ignore_index=True), 10, metric
    )
    prior_means = read_csv(source_hardware / "hardware_tiers_top10_mean.csv")
    check = means.merge(
        prior_means,
        on=["hardware_tier", "language"],
        validate="one_to_one",
        suffixes=("", "_previous"),
    )
    np.testing.assert_allclose(
        check[f"top_n_mean_{metric}"],
        100 * (2 * check.top_n_mean_balanced_accuracy - 1),
        rtol=0,
        atol=1e-11,
    )
    assert check.models_used.eq(check.models_used_previous).all()
    medians.to_csv(tier_out / "hardware_tiers_medians.csv", index=False)
    means.to_csv(tier_out / "hardware_tiers_top10_mean.csv", index=False)
    selected.to_csv(tier_out / "hardware_tiers_top10_models.csv", index=False)
    slide = means[means.hardware_tier.isin(["Cloud", "Mobile + Tablet"])]
    slide.to_csv(tier_out / "hardware_tiers_top10_mean_slide.csv", index=False)
    for stats, value, name, slide_view in (
        (medians, f"median_{metric}", "hardware_tiers_medians", False),
        (means, f"top_n_mean_{metric}", "hardware_tiers_top10_mean", False),
        (slide, f"top_n_mean_{metric}", "hardware_tiers_top10_mean_slide", True),
    ):
        aggregation = "Median" if value.startswith("median") else "Mean"
        subtitle = (
            "Median adjusted balanced accuracy of eligible models per tier."
            if aggregation == "Median"
            else (
                "Mean adjusted balanced accuracy of up to 10 "
                "highest-scoring eligible models per group."
            )
        )
        plot_stats = stats.copy()
        plot_stats[value] /= 100
        fig = hardware.plot_dumbbell(
            plot_stats,
            value,
            "How well do models identify\nnon-decodable words?",
            subtitle,
            {lang: size for lang, (size, _) in datasets.items()},
            slide=slide_view,
            metric_label="Adjusted balanced accuracy",
            gold_note=datasets["english"][1],
            min_items=700,
            min_coverage=0.9,
        )
        ax = fig.axes[0]
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
        ax.set_xlabel(
            f"{aggregation} adjusted balanced accuracy (0 = chance; 100 = perfect)"
        )
        for label in ax.texts:
            if re.fullmatch(r"-?\d+\.\d+%", label.get_text()):
                label.set_text(f"{label.xy[0]:.1f}")
        ax.axvline(0, color="#454545", ls="--", lw=1, zorder=2)
        for ext in ("png", "pdf"):
            fig.savefig(tier_out / f"{name}.{ext}", dpi=250, facecolor=SURFACE)
        plt.close(fig)

    assert {relative: digest(root / relative) for relative in before} == before
    assert verify_merge(root) == merged_before
    archive = out / "reproduction"
    archive.mkdir(exist_ok=True)
    for path in (
        Path(__file__),
        scripts / "plot_report_f2_leaderboards.py",
        scripts / "analyze_phonics_hardware_tiers.py",
    ):
        dest = archive / path.name
        if path.resolve() != dest.resolve():
            shutil.copy2(path, dest)
    hardware_link = (
        "../hardware_tiers/adjusted_balanced_accuracy/coverage_gt90/"
        "hardware_tiers_top10_mean_slide.png"
    )
    command = (
        "uv run --frozen --package phonics-benchmark python "
        f"benchmarks/phonics_benchmark/scripts/{Path(__file__).name} "
        f'--root "{root}"'
    )
    (out / "README.md").write_text(f"""# Adjusted balanced accuracy

Adjusted BA points = 100 × (2 × BA − 1) = 100 × Youden's J.
Chance and constant predictions score 0; perfect predictions score 100. Negative
scores and CI tails are retained. These plots rescale the preceding corrected-gold
BA analysis; model ranks, coverage filters, data and bootstrap samples are unchanged.

- [English leaderboard][english]
- [Kiswahili leaderboard][kiswahili]
- [Hardware comparison][hardware]

[english]: english_adjusted_balanced_accuracy_leaderboard_coverage_gt90.png
[kiswahili]: kiswahili_adjusted_balanced_accuracy_leaderboard_coverage_gt90.png
[hardware]: {hardware_link}

PNG/PDF/CSV outputs. Models require >90% coverage and at least 700 scored items:
{details["english"]["minimum_scorable_items"]}/{datasets["english"][0]} English and
{details["kiswahili"]["minimum_scorable_items"]}/{datasets["kiswahili"][0]}
Kiswahili items.
Leaderboard scores and 95% CI bounds
come from the saved adjusted BA summary columns (1,000 item-bootstrap resamples,
seed 0), verified against 100×(2×BA−1). All adjusted-BA score and CI columns
are in points (chance 0; perfect 100), retaining negative values.
No new bootstrap was needed.

Hardware plots retain the same groups and selected models, exclude Workstation,
and show means of up to ten best models per group. Group CIs are not calculated.
Model counts and selected models are recorded in the hardware CSVs.
Counts at/below zero and intervals containing zero are in manifest.json.

Reproduce from the isolated worktree:

```sh
{command}
```

All input BA outputs and merged data were verified unchanged.
""")
    manifest = dict(
        metric=metric,
        transformation="100 * (2 * balanced_accuracy - 1)",
        score_scale=100,
        datasets=details,
        scores_recomputed=False,
        bootstrap_recomputed=False,
        rankings_unchanged=True,
        hardware_selected_models_unchanged=True,
        workstation_excluded=True,
        negative_values_retained=True,
        input_sha256=before,
        outputs={
            str(p.relative_to(root)): digest(p)
            for directory in (out, tier_out)
            for p in directory.rglob("*")
            if p.is_file() and p.name != "manifest.json"
        },
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(means.drop(columns="models_used").to_string(index=False))


if __name__ == "__main__":
    main()
