# Scoring and plotting benchmark runs

Two commands handle ongoing reporting. Both work locally, make no model calls, and write
no cloud objects. Historical merging and gold refreshes are separate workflows; a normal
runner run needs only its config, target dataset and response files.

## Score a run and generate its report

From the repository root:

```sh
uv run --frozen --package phonics-benchmark python \
  benchmarks/phonics_benchmark/scripts/report_on_benchmark_run.py \
  --run-id YOUR_RUN_ID --inspect-sample-size 10
```

The existing `--run-id` and `--output-root` interface is preserved. The default root is
`benchmarks/phonics_benchmark/outputs`. Alternatively, pass `--run-dir PATH` directly.
Reports default to `RUN/report/`; `--output-dir PATH` selects a separate destination.
Original config, item and response files are read only and checked unchanged.

The report contains:

- `results_with_scoring.csv` and `summary_by_model.csv`: per-item scoring and the
  existing per-model metrics, including pooled F1 and F2.
- `bootstrap_ci.csv`: accuracy and pooled precision/recall/F1/F2 intervals.
- `balanced_accuracy_summary.csv` and `balanced_accuracy_item_counts.csv`:
  fixed-vocabulary BA, adjusted BA and out-of-text prediction counts.
- `coverage_by_model.csv`: scorable, failed, unparseable and missing responses for every
  configured model, including those with no responses.
- `model_summary_all_scored.csv` and `model_summary_eligible.csv`: combined metrics,
  confidence bounds and the coverage gate.
- F1, F2, BA and adjusted-BA leaderboards in PNG, PDF and CSV, plus accuracy against
  item difficulty and a `README.md` index.
- `baseline_comparison.csv` and `baseline_counts.csv`: eligible models below, at or
  above the BA chance baseline, with interval comparisons.
- `report_config.json`: input hashes, bootstrap settings and coverage policy.
- `sampled_items_for_inspection.csv` when requested.

All scored models enter the summary. By default, figures require at least one scorable
item. For the September leaderboard policy, use `--min-items 700 --min-coverage 0.9`:
the count floor is inclusive and the coverage threshold is strictly greater than 90%.
That means 892/991 English or 901/1,000 Kiswahili items. Models failing the gate remain
in the all-model summary and coverage table. Undefined metrics have no plotted bar.

## Bootstrap semantics and units

All metrics resample **whole target items with replacement**, using the same sorted item
IDs and seeded draws across models and metrics. Every draw has the original number of
target items. Missing model/item responses contribute no confusion counts; they are not
scored as incorrect. Undefined bootstrap replicates are skipped when computing
percentile bounds. The default is 1,000 draws, seed 0 and 95% intervals;
`--n-bootstrap`, `--seed` and `--ci-level` override those settings.

Earlier F1/F2 reports sampled the union of observed item IDs, whereas BA used the full
target dataset. The consolidated reporter uses the full target dataset for both. This
leaves point scores unchanged but can change intervals when an item has no scorable
response from any model. The low-level `bootstrap_metric_cis` API still supports omitted
`item_ids` for callers that only have an observed score table; reporting always supplies
the full target IDs.

BA averages pooled recall and specificity over each item's unique vocabulary.
Out-of-text predictions contribute to F1/F2 false positives and are audited separately
for BA. **Adjusted BA = 100 × (2 × BA − 1)**: chance is 0, perfect is 100 and inverted
predictions score −100. Adjusted scores and intervals are saved in points; F1/F2/BA use
fractions from 0 to 1. Negative adjusted scores and interval tails remain visible.

`--n-bootstrap 0` produces point estimates and plots without intervals. `--no-plots`
scores and saves tables only; it does not disable bootstrapping.

## Multiple runs and optional hardware comparisons

Repeat `--run-dir` or `--run-id` to process multiple runs. With `--output-dir`, each run
gets its own subdirectory and a collection-level index and combined summary:

```sh
uv run --frozen --package phonics-benchmark python \
  benchmarks/phonics_benchmark/scripts/report_on_benchmark_run.py \
  --run-dir outputs/MY_SNAPSHOT/english \
  --run-dir outputs/MY_SNAPSHOT/kiswahili \
  --output-dir outputs/MY_SNAPSHOT/reports \
  --metadata benchmarks/phonics_benchmark/scripts/f2_plot_metadata \
  --min-items 700 --min-coverage 0.9 --hardware-tiers
```

Display metadata is optional: without it, model IDs are labels and a default colour is
used. The metadata folder contains `models.csv`, `providers.csv`, and optionally
`display_overrides.json`. The historical `f2_plot_metadata` bundle preserves the
September labels, colours and model specifications; it is not required for new runs.

`--hardware-tiers` requires metadata and an explicit output directory. Hardware
comparisons support one English and/or one Kiswahili report with a common coverage
policy. They retain the access/parameter-size classification, exclude Workstation, and
show medians and means of up to `--top-n` models (default 10) per group. Group means
have no group confidence intervals. Unknown model specifications remain in the
classification audit but do not enter a hardware group. Outputs are under
`OUTPUT/hardware_tiers/METRIC/`.

## Redraw figures without rescoring

```sh
uv run --frozen --package phonics-benchmark python \
  benchmarks/phonics_benchmark/scripts/plot_benchmark_report.py \
  --report-dir outputs/MY_SNAPSHOT/reports/english
```

This reads the saved combined tables and configuration, with no response parsing,
scoring or bootstrapping. Repeat `--report-dir` for multiple reports. `--output-dir`
keeps the figures separate from their source reports. Optional
`--hardware-metadata PATH/TO/models.csv` and `--output-dir` recreate hardware
comparisons from saved tables.

## Code and review guides

Scripts only dispatch to `phonics_benchmark` modules. `scoring.py` and
`balanced_accuracy.py` define metrics, `bootstrap.py` owns item resampling, `plots.py`
renders leaderboards, `hardware.py` handles hardware comparisons, and `reporting.py`
loads runs and exports reports. `report_cli.py` owns command options. No package module
imports a script.

The separate F1/F2/BA/adjusted-BA drivers are superseded. Two import-only compatibility
files (`analyze_balanced_accuracy.py` and `plot_report_f2_leaderboards.py`) remain for
historical analysis consumers; they delegate to package modules and are not reporting
commands. Existing item-distribution plots remain supported.

- [Metric examples and plotting checks](validation.md)
- [Refactor scope and local comparison results](reporting-refactor-review.md)
- [Historical gold refresh and comparison](historical-report-refresh.md)
- [Historical merge workflow](local-merged-analysis.md)
- [Unified-run packaging and equality checks](unified-leaderboard-runs.md)
