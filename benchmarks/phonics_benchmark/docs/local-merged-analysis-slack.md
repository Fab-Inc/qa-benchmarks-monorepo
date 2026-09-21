*Generate the 991 English / 1,000 Kiswahili decodability datasets*

Use a checkout containing the latest merge/selection changes on
`add-phonics-local-merge`. Run from the repository root. You need `uv` and `gcloud`
authenticated with read access to `gs://phonics-benchmark/`. Everything is written
locally: no GCS uploads and no new model calls.

The pipeline downloads the 32 source run folders, merges the complete item and response
pools, then selects a separate mixture: *10% fully decodable + 70% Beta(1,8) + 20%
uniform*, using ten-percentage-point bins and seed 42.

Choose a new output directory; existing snapshots and selected datasets are not
overwritten. Change `PHONICS_SNAPSHOT` below if that directory already exists.

```sh
PHONICS_SCRIPTS=benchmarks/phonics_benchmark/scripts
PHONICS_SNAPSHOT=outputs/phonics-mixture-20260921

# 1. Read GCS sources into a new local snapshot.
uv run --frozen --package phonics-benchmark python \
  "$PHONICS_SCRIPTS/download_benchmark_snapshot.py" \
  --manifest "$PHONICS_SCRIPTS/merged_sources_20260921.json" \
  --root "$PHONICS_SNAPSHOT"

# 2. Merge the full pool: 1,610 English / 2,592 Kiswahili items.
uv run --frozen --package phonics-benchmark python \
  "$PHONICS_SCRIPTS/merge_local_benchmark_runs.py" \
  --manifest "$PHONICS_SCRIPTS/merged_sources_20260921.json" \
  --root "$PHONICS_SNAPSHOT"

# 3. Select the 991 English / 1,000 Kiswahili mixture.
uv run --frozen --package phonics-benchmark python \
  "$PHONICS_SCRIPTS/select_local_benchmark_subset.py" \
  --full-root "$PHONICS_SNAPSHOT/full" \
  --selection "$PHONICS_SCRIPTS/selection_beta1_8_20260921.json"
```

The selected items are here:

```text
outputs/phonics-mixture-20260921/selections/beta1-8-provisional/english/benchmark_items.jsonl
outputs/phonics-mixture-20260921/selections/beta1-8-provisional/kiswahili/benchmark_items.jsonl
```

Each language folder also contains `responses/merged.jsonl`, `config.json`,
`coverage_by_model.csv`, coverage by ratio band, and item/response provenance. Raw
responses include errors; use the coverage tables for successfully parsed response
counts. Coverage denominators use the selected dataset's actual size. The parent
folder's `audit/selection_summary.json` records quotas and shortages. The full datasets
remain available under `$PHONICS_SNAPSHOT/full/`.

*Why 991 English?* The intended size is 1,000, but the current pool has only 11 of the
required 20 items above 90% non-decodable. We retain the nine-item shortfall instead of
replacing them from another bin. Kiswahili meets all quotas.

*English is still provisional:* the new English items have not yet received corrected
gold labels. Once corrected, rebuild the gold-labelled full dataset and rerun selection
because items may change ratio bins. This selection samples the full pool; it does not
reuse the earlier current-items-first draft's exact IDs.

If you already have a full local merge, run only step 3 with a new selection name/output
location. Fresh downloads collect the responses currently in GCS; use the frozen
September 21 local snapshot for exact reproduction of that run.
