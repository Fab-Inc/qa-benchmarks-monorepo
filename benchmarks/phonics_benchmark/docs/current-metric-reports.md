# Current F1, F2 and balanced-accuracy reports

Run from the repository root on `add-phonics-local-merge`. These are the reusable
versions of the scripts used for the latest corrected-gold reports. They read local
merged data; they make no model calls or GCS writes. Generated datasets, reports and
historical reproduction copies remain under ignored `outputs/`.

## Refresh English gold while retaining the selected IDs

Start with a freshly downloaded, fully merged source snapshot and the previous selected
dataset. See `local-merged-analysis.md` for the downloader and merger. Download the
corrected gold separately, or include it as a download-only dataset in your source
manifest. Do not add it alongside conflicting older gold overlays in the merger: this
step applies the new file as the authoritative replacement.

```sh
PHONICS_SCRIPTS=benchmarks/phonics_benchmark/scripts
PHONICS_DOWNLOAD=outputs/NEW_DOWNLOAD
PHONICS_PREVIOUS=outputs/PREVIOUS_ANALYSIS/selections/fixed-991-en-1000-sw
PHONICS_SNAPSHOT=outputs/NEW_ANALYSIS
PHONICS_GOLD="$PHONICS_DOWNLOAD/sources/datasets/phonics-datasets-english_991_with_corrected_gold.jsonl"

uv run --frozen --package phonics-benchmark python \
  "$PHONICS_SCRIPTS/apply_refreshed_gold.py" \
  --parent-full "$PHONICS_DOWNLOAD/full" \
  --parent-selection "$PHONICS_PREVIOUS" \
  --gold "$PHONICS_GOLD" \
  --gold-uri gs://phonics-benchmark/phonics/datasets/english/phonics-datasets-english_991_with_corrected_gold.jsonl \
  --selection-name fixed-991-en-1000-sw \
  --output-root "$PHONICS_SNAPSHOT"

PHONICS_ANALYSIS="$PHONICS_SNAPSHOT/selections/fixed-991-en-1000-sw"
```

Replace the placeholder paths with your local snapshots. `--output-root` must be new.
The gold file must cover exactly the previous English selection with unique IDs and
unchanged text/knowledge. Gold is applied to the full pool first; the previous English
and Kiswahili IDs are then selected from it, keeping the fresh responses. There is no
redistribution by the updated gold ratios. The selection name is configurable and
defaults to the previous directory name; counts are read from the datasets, not from the
name.

## Create the latest reports

If you already have corrected, merged datasets, set `PHONICS_ANALYSIS` to their selected
root and run only the following commands, in order:

```sh
uv run --frozen --package phonics-benchmark python \
  "$PHONICS_SCRIPTS/run_f2_analysis.py" \
  --root "$PHONICS_ANALYSIS" --min-items 700 --min-coverage 0.8

uv run --frozen --package phonics-benchmark python \
  "$PHONICS_SCRIPTS/plot_report_f1_leaderboards.py" \
  --root "$PHONICS_ANALYSIS" --min-items 700 --min-coverage 0.8 --hardware-tiers

uv run --frozen --package phonics-benchmark python \
  "$PHONICS_SCRIPTS/run_balanced_accuracy_reports.py" --root "$PHONICS_ANALYSIS"

uv run --frozen --package phonics-benchmark python \
  "$PHONICS_SCRIPTS/plot_adjusted_balanced_accuracy.py" --root "$PHONICS_ANALYSIS"

uv run --frozen --package phonics-benchmark python \
  "$PHONICS_SCRIPTS/finalize_phonics_refresh.py" --root "$PHONICS_ANALYSIS"
```

The BA drivers intentionally use the latest report policy: **strictly >80% of target
items and at least 700 successfully scored items**. This requires 793/991 English and
801/1,000 Kiswahili items. The denominator includes unanswered items. They check that
the eligible models match the F1/F2 reports. Hardware comparisons exclude Workstation
and average up to ten highest-scoring models per group. Group means have no group-level
CIs.

BA averages pooled recall and specificity over each item's unique vocabulary.
Out-of-text predictions are audited separately. Its 95% CIs use 1,000 shared
item-bootstrap draws, seed 0. **Adjusted BA = 100 × (2 × BA − 1)**, with chance at 0,
perfect at 100 and inverted predictions at −100. All adjusted scores and CIs in CSVs use
points; negative values are retained. F1, F2 and ordinary BA retain their 0–1 units.
Adjusted BA preserves BA rankings and transforms the saved CIs.

The finalizer writes `model_summary_all_scored.csv` and
`model_summary_coverage_gt80.csv`, combining F1/F2/BA/adjusted-BA scores and CIs. It
also creates a report index linking the leaderboards and hardware plots. PNG, PDF and
CSV outputs are under each language's `report/`, `balanced_accuracy/`,
`adjusted_balanced_accuracy/` and `hardware_tiers/METRIC/coverage_gt80/`. Rerunning
replaces derived reports, while merged inputs are checked unchanged.

For a before/after comparison, replace the final command above with:

```sh
uv run --frozen --package phonics-benchmark python \
  "$PHONICS_SCRIPTS/finalize_phonics_refresh.py" \
  --root "$PHONICS_ANALYSIS" \
  --previous-root "$PHONICS_PREVIOUS" \
  --output-dir "$PHONICS_SNAPSHOT" \
  --download-root "$PHONICS_DOWNLOAD" \
  --gold "$PHONICS_GOLD"
```

This also creates `model_metric_changes.csv` and audits changes to labels and responses.
The comparison requires matching item IDs. It accepts previous adjusted scores in either
fractions or points, converting them to points before subtracting. Changes combine gold
edits and any refreshed responses; they do not isolate the effect of gold alone. No run
count, model name, generation, date or number of changed labels is hardcoded.

Optionally save `gcloud storage ls -L GOLD_URI` output and pass that file using
`--gold-metadata`. The finalizer verifies its MD5 against `--gold` and reports whether
its update date is today/yesterday in London. This is a check of saved metadata;
rerunning the local report does not check the latest remote generation.

## Source files for the current report workflow

New files needed for a commit, relative to `benchmarks/phonics_benchmark/`:

- `scripts/apply_refreshed_gold.py`
- `scripts/run_balanced_accuracy_reports.py`
- `scripts/plot_adjusted_balanced_accuracy.py`
- `scripts/finalize_phonics_refresh.py`
- `scripts/analyze_balanced_accuracy.py`
- `scripts/plot_report_f1_leaderboards.py`
- `tests/test_balanced_accuracy.py`
- `tests/test_current_report_scripts.py`
- `docs/current-metric-reports.md`

Include the pending current-report changes to `scripts/run_f2_analysis.py`,
`scripts/plot_report_f2_leaderboards.py`, `scripts/analyze_phonics_hardware_tiers.py`,
`tests/test_f2_analysis.py`, `tests/test_phonics_hardware_tiers.py` and
`docs/f2-reports-slack.md`. The existing metadata bundle, scorer, merger and selector
are already tracked. Source-manifest updates are separate from these report changes.

The current workflow does not import the older MCC, baseline, item-weighting,
ratio-band, `finalize_merged_reports.py` or `plot_balanced_accuracy_leaderboards.py`
scripts. Those historical experiments are not needed for this commit.

```sh
uv run --frozen --package phonics-benchmark pytest \
  benchmarks/phonics_benchmark/tests/test_balanced_accuracy.py \
  benchmarks/phonics_benchmark/tests/test_current_report_scripts.py \
  benchmarks/phonics_benchmark/tests/test_f2_analysis.py \
  benchmarks/phonics_benchmark/tests/test_phonics_hardware_tiers.py -q
```
