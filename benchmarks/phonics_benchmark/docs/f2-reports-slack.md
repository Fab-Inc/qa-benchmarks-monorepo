# F2 reports: commit scope and run guide

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
below 80% coverage in both current subsets.

The entry point replaces the one-off `/tmp` wrapper. It has no dependency on older
output folders or a locally downloaded LKB source file. Dataset sizes, coverage cutoffs
and gold-status notes come from the input files. It checks merged-input hashes before
and after reporting, verifies coverage against scoring, independently checks pooled F2
from confusion totals and preserves the saved bootstrap bounds in the plots.

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
  --root "$PHONICS_ANALYSIS"
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
english/report/pooled_f2_leaderboard_coverage_gt80.png
kiswahili/report/pooled_f2_leaderboard_coverage_gt80.png
hardware_tiers/pooled_f2/hardware_tiers_top10_mean_slide.png
```

PDF and CSV versions sit beside the PNGs. Each language's `report/` also contains
per-model summaries, bootstrap bounds, per-item confusion counts and an all-scored-model
leaderboard. Hardware outputs include model inclusion/exclusion audits, selected top
models, means and medians. `audit/f2_analysis_run.json` records validation, input
hashes, code hashes and settings.

*Coverage:* headline plots require strictly more than 80% successfully parsed/scored
items. With the current 991 English / 1,000 Kiswahili subsets, that means at least 793 /
801 items. These cutoffs are calculated from the dataset, not hardcoded. The hardware
plot averages the best ten eligible models per group/language, or all eligible models
when fewer than ten qualify (currently six English and seven Kiswahili small models).
These group means have no group-level confidence intervals.

*Gold labels:* current English uses 540 corrected items and 451 provisional extension
items; all 1,000 Kiswahili items have corrected gold. The plots show gold status. CIs
describe item-sampling uncertainty, not uncertainty in provisional labels. Scores are
pooled F2, without item or ratio weights.

For exact September 21 results, reuse that frozen source snapshot. A fresh download may
contain newer responses. Rebuild a moved snapshot locally if its recorded absolute
dataset paths are unavailable. When English corrections arrive, rebuild the full merge
and subset before reporting because changed ratios can change selected item IDs.
