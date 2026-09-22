"""Recreate corrected-snapshot BA leaderboards, hardware plots and baseline counts."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_report_f2_leaderboards import load_plot_table, read_csv


def load_balanced_table(root, language):
    metadata, colors = load_plot_table(root / language, root / "plot_metadata")
    metadata = metadata.drop(
        columns=["score", "ci_low", "ci_high", "n_bootstrap", "ci_level"]
    )
    saved = read_csv(root / "balanced_accuracy" / f"{language}_summary.csv")
    if set(metadata.model) != set(saved.model):
        raise ValueError(
            "Model IDs differ between balanced accuracy and original report"
        )
    for column in ["n_bootstrap", "ci_level", "expected_items"]:
        if saved[column].nunique() != 1:
            raise ValueError(f"Inconsistent {column}")
    table = metadata.merge(
        saved.drop(columns="expected_items"), on="model", validate="one_to_one"
    )
    if not table.n_items.eq(table.n_scorable).all():
        raise ValueError("Scorable counts differ between reports")
    np.testing.assert_allclose(
        table.balanced_accuracy,
        (table.recall + table.specificity) / 2,
        rtol=0,
        atol=1e-14,
    )
    table = table.rename(
        columns={
            "balanced_accuracy": "score",
            "balanced_accuracy_ci_low": "ci_low",
            "balanced_accuracy_ci_high": "ci_high",
        }
    )
    if (
        not table[["score", "ci_low", "ci_high"]]
        .apply(lambda c: c.between(0, 1))
        .all()
        .all()
    ):
        raise ValueError("Undefined or out-of-range scores or CIs")
    if not table.ci_low.le(table.ci_high).all():
        raise ValueError("Reversed confidence interval")
    denominator = int(table.expected_items.iloc[0])
    gate = table.n_scorable * 5 > denominator * 4
    if not gate.eq(table.passes_previous_80_percent_gate).all():
        raise ValueError("Coverage gate differs from saved results")
    return table.sort_values(["score", "model"], ascending=[False, True]).reset_index(
        drop=True
    ), colors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    scripts = Path(__file__).resolve().parent
    import analyze_balanced_accuracy as ba
    import analyze_phonics_hardware_tiers as hardware
    from analyze_balanced_accuracy import load_items
    from plot_report_f2_leaderboards import SURFACE, digest, plot_leaderboard, read_csv
    from run_f2_analysis import inspect_dataset, verify_merge

    merge_before = verify_merge(root)
    datasets = {lang: inspect_dataset(root / lang) for lang in ("english", "kiswahili")}
    protected = []
    for directory in (
        root / "english/report",
        root / "kiswahili/report",
        root / "hardware_tiers/pooled_f1",
        root / "hardware_tiers/pooled_f2",
        root / "audit",
        root / "plot_metadata",
    ):
        protected.extend(p for p in directory.rglob("*") if p.is_file())
    before = {str(p.relative_to(root)): digest(p) for p in protected}
    output = root / "balanced_accuracy"
    output.mkdir(exist_ok=True)
    inputs, details, summaries, eligibility, baseline_rows, model_rows = (
        [],
        {},
        [],
        [],
        [],
        [],
    )
    for lang, (size, gold_note) in datasets.items():
        item_path = root / lang / "benchmark_items.jsonl"
        scored_path = root / lang / "report/results_with_scoring.csv"
        inputs.extend([item_path, scored_path, root / lang / "coverage_by_model.csv"])
        items = load_items(item_path, "en" if lang == "english" else "sw")
        scored = read_csv(scored_path)[
            [
                "model",
                "benchmark_item_id",
                "tp_count",
                "fp_count",
                "fn_count",
                "tn_count",
            ]
        ]
        coverage = read_csv(root / lang / "coverage_by_model.csv").set_index("model")
        actual = scored.groupby("model").size().reindex(coverage.index, fill_value=0)
        assert actual.equals(coverage.scorable_items)
        result, counts = ba.summarize(scored, items, n_bootstrap=1000, seed=0)
        result.insert(0, "dataset", lang)
        result["passes_previous_80_percent_gate"] = result.n_scorable * 5 > size * 4
        result.to_csv(output / f"{lang}_summary.csv", index=False)
        counts[["model", "benchmark_item_id", *ba.COUNTS]].to_csv(
            output / f"{lang}_item_counts.csv", index=False
        )
        summaries.append(result)
        full_table, colors = load_balanced_table(root, lang)
        required = hardware.minimum_scorable_items(size, 700, 0.8)
        full_table["minimum_scorable_items"] = required
        full_table["passes_minimum_items"] = full_table.n_scorable.ge(required)
        eligibility.append(full_table.assign(language=lang))
        table = full_table[full_table.passes_minimum_items].reset_index(drop=True)
        f1 = read_csv(root / lang / "report/pooled_f1_leaderboard_coverage_gt80.csv")
        assert set(table.model) == set(f1.model)
        np.testing.assert_allclose(
            table.score, (table.recall + table.specificity) / 2, rtol=0, atol=1e-14
        )
        assert table.n_valid_bootstrap.eq(1000).all()
        fig = plot_leaderboard(table, colors, lang)
        ax = fig.axes[0]
        ax.set_xlabel("Balanced accuracy")
        ax.set_title(
            f"{lang.title()} decodability · Balanced accuracy\nLeaderboard · "
            f">80% coverage · ≥{required:,} evaluated items",
            loc="left",
            fontsize=12,
            fontweight="semibold",
            pad=35,
        )
        for label in ax.texts:
            if label.get_text() == "F2":
                label.set_text("BA")
            elif label.get_text().startswith("Items = scorable"):
                label.set_text(
                    "Items = scorable responses; coverage varies. Dashed line "
                    "= 50% flag-all / flag-none baseline."
                )
        ax.axvline(0.5, color="#454545", ls="--", lw=1, zorder=5)
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
        stem = output / f"{lang}_balanced_accuracy_leaderboard_coverage_gt80"
        for extension in ("png", "pdf"):
            fig.savefig(
                stem.with_suffix(f".{extension}"),
                dpi=250,
                bbox_inches="tight",
                facecolor=SURFACE,
            )
        plt.close(fig)
        table.to_csv(stem.with_suffix(".csv"), index=False)
        pd.testing.assert_frame_equal(
            read_csv(stem.with_suffix(".csv"))[["score", "ci_low", "ci_high"]],
            table[["score", "ci_low", "ci_high"]],
            check_exact=True,
        )
        # Use exact integer confusion products for below/equal classification.
        determinant = table.tp_count.astype("int64") * table.tn_count.astype(
            "int64"
        ) - table.fp_in_vocabulary.astype("int64") * table.fn_count.astype("int64")
        assert np.array_equal(determinant.lt(0), table.score.lt(0.5))
        assert np.array_equal(determinant.eq(0), table.score.eq(0.5))
        row = dict(
            language=lang,
            models=len(table),
            below_50=int(determinant.lt(0).sum()),
            at_50=int(determinant.eq(0).sum()),
            below_or_at_50=int(determinant.le(0).sum()),
            entire_ci_below_or_at_50=int(table.ci_high.le(0.5).sum()),
            ci_contains_50=int((table.ci_low.le(0.5) & table.ci_high.ge(0.5)).sum()),
            entire_ci_above_50=int(table.ci_low.gt(0.5).sum()),
        )
        baseline_rows.append(row)
        model_rows.append(
            table.assign(
                language=lang,
                point_below_50=determinant.lt(0),
                point_at_50=determinant.eq(0),
                entire_ci_below_or_at_50=table.ci_high.le(0.5),
            )
        )
        details[lang] = dict(
            target_items=size,
            minimum_scorable_items=required,
            n_models=len(table),
            gold_note=gold_note,
        )
        print(json.dumps(row), flush=True)

    pd.concat(summaries, ignore_index=True).to_csv(
        output / "summary_all_datasets.csv", index=False
    )
    pd.concat(eligibility, ignore_index=True).to_csv(
        output / "model_eligibility_coverage_gt80.csv", index=False
    )
    pd.DataFrame(baseline_rows).to_csv(
        output / "baseline_counts_coverage_gt80.csv", index=False
    )
    pd.concat(model_rows, ignore_index=True).to_csv(
        output / "baseline_comparison_coverage_gt80.csv", index=False
    )
    hardware.main(
        root,
        top_n=10,
        denominators={lang: size for lang, (size, _) in datasets.items()},
        metric="balanced_accuracy",
        gold_note=datasets["english"][1],
        min_items=700,
        min_coverage=0.8,
    )
    hardware_dir = root / "hardware_tiers/balanced_accuracy/coverage_gt80"
    for lang in datasets:
        current = read_csv(hardware_dir / f"hardware_tiers_{lang}.csv")
        previous = read_csv(
            root / f"hardware_tiers/pooled_f1/coverage_gt80/hardware_tiers_{lang}.csv"
        )
        assert set(current.model) == set(previous.model)
        assert not current.hardware_tier.eq("Workstation").any()
    after = {relative: digest(root / relative) for relative in before}
    assert before == after, "Previously saved reports changed"
    assert verify_merge(root) == merge_before
    archive = output / "reproduction"
    archive.mkdir(exist_ok=True)
    for source in [
        Path(__file__),
        *(
            scripts / name
            for name in (
                "analyze_balanced_accuracy.py",
                "analyze_phonics_hardware_tiers.py",
                "plot_report_f2_leaderboards.py",
                "run_f2_analysis.py",
            )
        ),
    ]:
        target = archive / source.name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
    hardware_link = (
        "../hardware_tiers/balanced_accuracy/coverage_gt80/"
        "hardware_tiers_top10_mean_slide.png"
    )
    command = (
        "uv run --frozen --package phonics-benchmark python "
        f"benchmarks/phonics_benchmark/scripts/{Path(__file__).name} "
        f'--root "{root}"'
    )
    (output / "README.md").write_text(f"""# Corrected-gold balanced accuracy reports

{datasets["english"][0]:,} English and
{datasets["kiswahili"][0]:,} Kiswahili target items.
Models must have >80% successful, parsed and scored responses: at least
{details["english"]["minimum_scorable_items"]:,} English or
{details["kiswahili"]["minimum_scorable_items"]:,} Kiswahili items.
The 700-item minimum also applies. The eligible model sets match the F1/F2 plots.

- [English leaderboard](english_balanced_accuracy_leaderboard_coverage_gt80.png)
- [Kiswahili leaderboard](kiswahili_balanced_accuracy_leaderboard_coverage_gt80.png)
- [Hardware comparison][hardware]

[hardware]: {hardware_link}

- [Counts below/at the 50% baseline](baseline_counts_coverage_gt80.csv)
- [Per-model baseline comparisons](baseline_comparison_coverage_gt80.csv)

PDF and CSV versions accompany the figures. Pooled balanced accuracy is the mean
of pooled recall and specificity across unique vocabulary words within each item.
Out-of-text predictions are audited separately, as in the preceding BA analyses.

95% percentile CIs use 1,000 item-bootstrap samples, seed 0, with shared item draws
across models. Missing responses contribute no counts. The 50% constant-classifier
baseline holds for every replicate containing both gold classes. Counts against
the baseline use point estimates and separately the upper 95% CI endpoint;
these are individual intervals without adjustment for multiple comparisons.

Hardware summaries take the mean of up to ten highest-BA models per group,
reranked by BA, excluding Workstation. They retain the previous point-estimate
format; group CIs are not calculated. Per-model CIs remain in the audit CSVs.

Run from the isolated worktree:

```sh
{command}
```

The driver and supporting scripts are archived in reproduction/. Saved responses,
gold labels and all pre-existing F1/F2 reports were verified unchanged.
""")
    manifest = dict(
        metric="balanced_accuracy",
        datasets=details,
        bootstrap=dict(
            n_bootstrap=1000,
            seed=0,
            ci_level=0.95,
            unit="item; shared draws across models",
        ),
        inputs={str(p.relative_to(root)): digest(p) for p in inputs},
        previous_report_hashes=before,
        previous_reports_unchanged=True,
        merged_inputs_verified=True,
        leaderboard_models_match_f1=True,
        hardware_models_match_f1=True,
        workstation_excluded=True,
        baseline_counts=baseline_rows,
        outputs={
            str(p.relative_to(root)): digest(p)
            for directory in (output, hardware_dir)
            for p in directory.rglob("*")
            if p.is_file() and p.name != "manifest.json"
        },
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Report: {output / 'README.md'}", flush=True)


if __name__ == "__main__":
    main()
