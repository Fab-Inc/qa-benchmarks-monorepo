# F2 reports: commit scope and run guide

For the current F1/F2/BA/adjusted-BA workflow, gold refresh and combined summaries, see
[current-metric-reports.md](current-metric-reports.md). It also lists the current report
files to include in a commit.

Prepare this commit on `add-phonics-local-merge`. The scorer, bootstrap implementation,
dependencies, downloader, full merger and subset selector are already committed. Add
these ten files (paths below are relative to `benchmarks/phonics_benchmark/`):

- `scripts/run_f2_analysis.py`
- `scripts/plot_report_f2_leaderboards.py`
- `scripts/analyze_phonics_hardware_tiers.py`
- `scripts/f2_plot_metadata/models.csv`
- `scripts/f2_plot_metadata/providers.csv`
- `scripts/f2_plot_metadata/display_overrides.json`
- `scripts/f2_plot_metadata/provenance.json`
- `tests/test_f2_analysis.py`
- `tests/test_phonics_hardware_tiers.py`
- `docs/f2-reports-slack.md`

The metadata bundle supplies the saved model names, provider colours and total parameter
counts. Its provenance records the original LKB style/hardware sources and the new Qwen
8B alias. It contains no benchmark texts or model responses. Astra has display metadata
but no hardware specification in the original metadata; its tier stays unplaced. It is
below 700 scored items in both current subsets.

The entry point replaces the one-off `/tmp` wrapper. It has no dependency on older
output folders or a locally downloaded LKB source file. Dataset sizes and gold-status
notes come from the input files; the configurable minimum scored-item count defaults to
700\. It checks merged-input hashes before and after reporting, verifies coverage against
scoring, independently checks pooled F2 from confusion totals and preserves the saved
bootstrap bounds in the plots.

Generated reports, datasets, responses and the temporary wrapper stay outside this
commit. The other untracked metric scripts are not needed for these pooled F2 plots. No
dependency or lockfile changes are needed.

Validation command from the repository root:

```sh
uv run --frozen --package phonics-benchmark pytest \
  benchmarks/phonics_benchmark/tests/test_bootstrap.py \
  benchmarks/phonics_benchmark/tests/test_phonics_hardware_tiers.py \
  benchmarks/phonics_benchmark/tests/test_f2_analysis.py -q
```

## Copy into Slack

*Generate the updated decodability F2 leaderboards and small-vs-large model plot*

Use a checkout containing the report changes on `add-phonics-local-merge`. Run from the
repository root. If you still need the merged/selected datasets, first follow
`benchmarks/phonics_benchmark/docs/local-merged-analysis-slack.md` (GCS reads only).

With the local datasets already built, run:

```sh
PHONICS_SCRIPTS=benchmarks/phonics_benchmark/scripts
PHONICS_SNAPSHOT=outputs/phonics-mixture-20260921
PHONICS_ANALYSIS="$PHONICS_SNAPSHOT/selections/beta1-8-provisional"

uv run --frozen --package phonics-benchmark python \
  "$PHONICS_SCRIPTS/run_f2_analysis.py" \
  --root "$PHONICS_ANALYSIS" --min-items 700
```

Set `PHONICS_SNAPSHOT` to your existing snapshot directory. For the local frozen run
used here, it is `outputs/phonics-merged-20260921`. To analyze the entire candidate
pool, use `--root "$PHONICS_SNAPSHOT/full"` instead. The analysis command reads local
data, makes no model calls and does not access GCS. Rerunning replaces its derived
reports and plots under that root; merged items and responses remain unchanged.

This runs the committed reporting/scoring code, then creates pooled F2 leaderboards with
95% bootstrap confidence intervals (1,000 shared item resamples, seed 0). It also
creates the English-vs-Kiswahili hardware comparison with Mobile + Tablet and Cloud;
*workstation models are excluded from the hardware comparison*.

Main outputs, relative to `PHONICS_ANALYSIS`:

```text
english/report/pooled_f2_leaderboard_min700.png
kiswahili/report/pooled_f2_leaderboard_min700.png
hardware_tiers/pooled_f2/hardware_tiers_top10_mean_slide.png
```

PDF and CSV versions sit beside the PNGs. Each language's `report/` also contains
per-model summaries, bootstrap bounds and per-item confusion counts. The default
`pooled_f2_leaderboard.png` also applies the minimum-item filter.
`pooled_f2_summary_by_language_and_model.csv` includes eligible models only;
`audit/f2_model_eligibility.csv` retains all scored models and their eligibility. The
underlying scoring tables retain all responses for auditing. Hardware outputs include
model inclusion/exclusion audits, selected top models, means and medians.
`audit/f2_analysis_run.json` records validation, input hashes, code hashes and settings.

*Coverage:* plots and the combined analysis summary require at least 700 successfully
parsed/scored items per model in either language. Exactly 700 qualifies; failed and
unparseable responses do not count. Change `--min-items` to choose a different cutoff.
To also require the current strict >90% coverage rule, add `--min-coverage 0.9`:

```sh
uv run --frozen --package phonics-benchmark python \
  "$PHONICS_SCRIPTS/run_f2_analysis.py" \
  --root "$PHONICS_ANALYSIS" --min-items 700 --min-coverage 0.9
```

The denominator is the actual number of target items in each selected dataset, including
items without responses, never the number of saved responses. With 991 English and 1,000
Kiswahili items this requires at least 892 and 901 scored items, respectively; exactly
90% is excluded. Both criteria apply; use `--min-items 1` if only the coverage rule is
desired for a smaller dataset.

The coverage option writes separate variants and preserves the ≥700 outputs:

```text
english/report/pooled_f2_leaderboard_coverage_gt90.png
kiswahili/report/pooled_f2_leaderboard_coverage_gt90.png
hardware_tiers/pooled_f2/coverage_gt90/hardware_tiers_top10_mean_slide.png
pooled_f2_summary_by_language_and_model_coverage_gt90.csv
audit/f2_model_eligibility_coverage_gt90.csv
audit/f2_analysis_run_coverage_gt90.json
```

The per-model score and bootstrap source tables are recomputed under `report/` with the
chosen seed/resample count, independently of which models pass the filter.

For the ≥700 rule, the current data admits 31 English and 39 Kiswahili models to the
leaderboards. The hardware plot averages the best ten eligible models per
group/language, or all eligible models when fewer than ten qualify (currently seven
small models in each language). These group means have no group-level confidence
intervals. The current >90% rule uses the actual dataset size; eligible model counts are
recorded in each report's audit and depend on the downloaded response snapshot.

*Gold labels:* the original `beta1-8-provisional` selection has 540 corrected English
items and 451 provisional extension items. The corrected fixed-ID snapshot is at
`outputs/phonics-merged-20260921-corrected-english/selections/fixed-991-en-1000-sw`; set
`PHONICS_ANALYSIS` to that directory to use all 991 corrected English items. All 1,000
Kiswahili items were marked corrected in that snapshot, but it predates the latest
Kiswahili overlays. Use a new download and merge with the current manifest for those
corrections. The plots read gold status from the input provenance. CIs describe
item-sampling uncertainty, not uncertainty in labels. Scores are pooled F2, without item
or ratio weights.

For exact September 21 results, reuse that frozen source snapshot. A fresh download may
contain newer responses. Rebuild a moved snapshot locally if its recorded absolute
dataset paths are unavailable. When replacing gold, update the full merge first. Keep
selection IDs fixed for a direct before/after comparison; regenerate the mixture
selection only if you intend changed gold ratios to change the selected item IDs.
