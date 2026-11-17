# Historical gold refresh and snapshot comparison

These are optional historical operations, separate from reporting on a normal run. See
[current reporting](current-metric-reports.md) for the supported reporting commands.

## Refresh English gold while retaining the selected IDs

Start with a freshly downloaded, fully merged source snapshot and the previous selected
dataset. See `local-merged-analysis.md` for the downloader and merger. Download the
corrected gold separately, or include it as a download-only dataset in your source
manifest. Do not add it alongside conflicting older gold overlays in the merger: this
step applies the new file as the authoritative replacement.

The current source manifest applies corrected Kiswahili gold to the full pool during the
merge. To update an older snapshot, download and merge into a new root first; rerunning
reports alone cannot update its labels. The English refresh below retains the corrected
Kiswahili labels from that new full pool while selecting the previous IDs.

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

## Compare with an earlier snapshot

Generate current reports into `$PHONICS_ANALYSIS/reports` with the consolidated
reporter, then run the historical audit:

```sh
uv run --frozen --package phonics-benchmark python \
  "$PHONICS_SCRIPTS/finalize_phonics_refresh.py" \
  --root "$PHONICS_ANALYSIS" \
  --report-root "$PHONICS_ANALYSIS/reports" \
  --previous-root "$PHONICS_PREVIOUS" \
  --output-dir "$PHONICS_SNAPSHOT/refresh-audit" \
  --download-root "$PHONICS_DOWNLOAD" \
  --gold "$PHONICS_GOLD"
```

This checks the sealed inputs and fixed item IDs, records gold and response changes, and
writes `model_metric_changes.csv` and `refresh_validation.json`. The previous snapshot
uses the archived September report layout. Adjusted BA comparisons accept its saved
fractional or point scale. Differences combine label edits and refreshed responses; they
do not isolate the effect of gold alone.

Optional `--gold-metadata` accepts saved `gcloud storage ls -L` output and verifies its
MD5 against the local gold file. It does not check the latest remote generation. Normal
report tables and the report index are now written by the consolidated reporter.
