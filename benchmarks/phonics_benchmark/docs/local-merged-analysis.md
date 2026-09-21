# Full local merge and optional subsets

Work in the isolated `add-phonics-local-merge` worktree. The September 17/18 snapshots
and the original checkout remain unchanged. All commands below run from the worktree
root. Only the downloader accesses GCS, using GCS-to-local copies; the merger and
selector are offline and make no model calls.

## 1. Download a new snapshot

```sh
uv run --frozen --package phonics-benchmark python \
  benchmarks/phonics_benchmark/scripts/download_benchmark_snapshot.py \
  --manifest benchmarks/phonics_benchmark/scripts/merged_sources_20260921.json \
  --root outputs/NEW-SNAPSHOT
```

The September 21 manifest contains the 18 previous destinations plus 14 extension
destinations. Each group has an explicit language. Whole source folders, their
generation datasets, and declared gold overlays are downloaded and checksummed. Existing
snapshot roots are refused. Dataset URIs sharing a filename are refused instead of
silently overwriting one another.

## 2. Merge the complete source pool

```sh
uv run --frozen --package phonics-benchmark python \
  benchmarks/phonics_benchmark/scripts/merge_local_benchmark_runs.py \
  --manifest benchmarks/phonics_benchmark/scripts/merged_sources_20260921.json \
  --root outputs/NEW-SNAPSHOT
```

This always produces `outputs/NEW-SNAPSHOT/full/english/` and `full/kiswahili/` before
any optional selection. The September 21 sources contain 1,610 English and 2,592
Kiswahili unique items. These are candidate pools, not the Beta mixture. Every source
item is retained, including items with missing or failed responses. Gold-overlay files
update matching IDs but do not add unrun items to the pool.

The merger validates downloaded hashes, destination dataset hashes, item inputs and
rendered prompts. It deduplicates by `model_call_id`, preferring success, then the
newest timestamp; parsing and scores do not influence selection. Original response IDs,
run IDs, model aliases and raw response records are preserved. Multiple distinct calls
for one model/item are refused, rather than choosing between different generation
settings silently.

`gold_overlays` declares corrected datasets by language. Each overlay must retain the
same text and student knowledge; word counts and ratios are recomputed using the
scorer's unique-word vocabulary. Overlapping overlays must agree. Source labels remain
in effect for IDs without overlays. English's 532 extension items are explicitly
**provisional**, whereas all current Kiswahili pool labels have corrected provenance.
Correcting English labels later requires a new version of the full dataset and a fresh
subset selection because ratio bins can change.

Each language directory contains:

- `benchmark_items.jsonl`, `responses/merged.jsonl`, and a report-compatible config.
- `provenance.jsonl`: every retained call's selected and duplicate source locations.
- `item_provenance.jsonl`: generation dataset(s), scoring-gold source(s) and status.
- `coverage_by_model.csv` and `coverage_by_model_and_ratio_band.csv`.
- `excluded_responses.jsonl`: failed calls and unparseable outputs, kept in raw data.

Coverage includes models with no responses and always divides by the actual unique item
count of that output dataset. Strict >80% eligibility is
`scorable_items * 5 > expected_items * 4`. Errors, parse failures and missing responses
are counted separately. The companion analysis scripts likewise derive coverage
denominators from the dataset; hardware-tier historical overrides remain available
explicitly.

The full folder contains source, duplicate and gold-change audits plus an
`artifact_manifest.json`. Publishing is atomic after validation; an existing full folder
is never overwritten. The old snapshot-specific scripts remain archived under the
earlier outputs' `reproduction/` directories.

## 3. Optionally select a separate dataset

Skip this step to analyze the complete pool. To use the proposed mixture:

```sh
uv run --frozen --package phonics-benchmark python \
  benchmarks/phonics_benchmark/scripts/select_local_benchmark_subset.py \
  --full-root outputs/NEW-SNAPSHOT/full \
  --selection benchmarks/phonics_benchmark/scripts/selection_beta1_8_20260921.json
```

Output: `outputs/NEW-SNAPSHOT/selections/beta1-8-provisional/`, with the same
English/Kiswahili run structure as `full/`. The full dataset is hash-verified and left
unchanged; existing subset folders are refused. `--output-root` can select another new
destination outside the full folder.

The example specifies 1,000 items per language: 10% exactly zero non-decodable, 70%
Beta(1,8) and 20% uniform over positive ratios. These are **binned quotas**, not an
exact continuous beta sample. Ten-point positive-ratio quotas use largest-remainder
rounding. Items are chosen without replacement by a seeded SHA-256 ordering within each
bin, independently of model results or coverage. There is no preference for items in the
earlier analysis; selected IDs may differ from the earlier current-items-first
provisional plan.

Current labels yield all 1,000 Kiswahili items. English has only 11 of the desired 20
items above 90%, so the example explicitly allows `keep_available`: it selects 991 items
and records the nine-item shortfall. No other bin is used to fill it. The actual
denominator is 991, and the achieved mixture differs slightly from the requested
mixture. English labels and its mixture remain provisional. The default shortfall policy
is `error`.

For a fixed saved selection instead of sampling, use an ID list or a JSONL file:

```json
{
  "name": "my-fixed-selection",
  "languages": {
    "en": {"items_file": "selected_english_items.jsonl"},
    "sw": {"item_ids": ["example-item-id"]}
  }
}
```

`items_file` is relative to the selection spec (absolute paths are accepted). Records
need `benchmark_item_id`; if text/knowledge are supplied, they must match. Gold in a
selection file does not override the full dataset's labels. Duplicate or unknown
requested IDs fail explicitly. Languages can be selected separately. Each subset saves
its selection spec, quotas/shortages, exact selected items, filtered response
provenance, gold status and recomputed coverage.

For a fresh snapshot, the merger also accepts `--selection SPEC.json` to perform both
stages sequentially. The full dataset is still published first and remains available if
the optional selection fails. Later subsets need only the selector.

## 4. Analyze either output

For the entire pool, pass `--output-root outputs/NEW-SNAPSHOT/full`; for the mixture,
pass `--output-root outputs/NEW-SNAPSHOT/selections/beta1-8-provisional`:

```sh
uv run --frozen --package phonics-benchmark python \
  benchmarks/phonics_benchmark/scripts/report_on_benchmark_run.py \
  --output-root outputs/NEW-SNAPSHOT/full --run-id english
```

Repeat with `--run-id kiswahili`. The reporting script reads that run's scoring dataset
and configuration. The other analysis scripts take the same parent via `--root`; their
report and plotting-metadata prerequisites still apply. No new model calls are needed to
select or score any already collected response.
