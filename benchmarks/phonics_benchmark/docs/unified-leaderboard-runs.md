# Unified runs for the 22 September leaderboard

The afternoon 22 September report at
`outputs/phonics-reports-20260922-coverage90/selections/fixed-991-en-1000-sw` is the
source of these two analysis-only runs:

| Language  | Run ID                         | Items | Raw responses | Scored models | Leaderboard models |
| --------- | ------------------------------ | ----: | ------------: | ------------: | -----------------: |
| English   | `en-gold-leaderboard-20260922` |   991 |        36,444 |            42 |                 30 |
| Kiswahili | `sw-gold-leaderboard-20260922` | 1,000 |        40,220 |            48 |                 32 |

The September 23 Astra refresh is deliberately excluded. The selected item files and
every retained response row match the September 22 report byte for byte. Failed and
unparseable calls remain in `responses/merged.jsonl`; scoring excludes them as usual.
Original `run_id`, `result_id`, `model_call_id`, and `config_id` values remain on
response rows. `provenance.jsonl` traces each row to its source. The bundle's
`config.json` identifies the **analysis run**, with the corrected item file, its SHA-256
digest, model list, and GCS item URI. It does not claim that one generation config
produced all model calls. Its `gold_sources` counts come from the run's own
`item_provenance.jsonl`. An item can list more than one source, so the Kiswahili counts
add up to more than 1,000. Other selection metadata stays in `source_config.json`, which
is a verbatim copy of the selected config. That includes local paths and the English
gold-update record that the 22 September refresh attached to both languages. The builder
refuses to write a `config.json` that contains a local path.

Build from the sealed September 22 snapshot and selected report:

```sh
uv run --frozen --package phonics-benchmark python \
  benchmarks/phonics_benchmark/scripts/build_unified_leaderboard_runs.py
```

The output is `outputs/phonics-unified-leaderboard-20260922/`. Locally, the two run
directories are named `english/` and `kiswahili/` for compatibility with the analysis
scripts. Their `resolved_config.run_id` values are the versioned IDs in the table.
`items_path` is relative to each run directory, so the bundle remains usable after being
copied elsewhere. The root artifact manifest records hashes for all immutable inputs.
The builder refuses to replace an existing output.

Reparse the bundled raw responses, score them against the bundled gold, and compare all
adjusted balanced accuracy values and 95% intervals with the September 22 report:

```sh
uv run --frozen --package phonics-benchmark python \
  benchmarks/phonics_benchmark/scripts/analyze_unified_leaderboard_runs.py
```

This writes `analysis/english_adjusted_balanced_accuracy.csv`,
`analysis/kiswahili_adjusted_balanced_accuracy.csv`, the two eligible-model tables, and
`analysis/comparison.json`. It uses the same 1,000 item-bootstrap draws with seed 0 and
the same strict >90% coverage and 700-item minimum. The analysis refuses to publish its
output if any model or metric differs beyond `1e-10`. It also compares eligible models
directly with the combined September 22 `model_summary_coverage_gt90.csv`. The current
comparison has **zero difference** across all 42 English and 48 Kiswahili scored models,
including the confidence bounds, and across both eligible-model tables.

Publish the sealed runs after the comparison succeeds:

```sh
uv run --frozen --package phonics-benchmark python \
  benchmarks/phonics_benchmark/scripts/publish_unified_leaderboard_runs.py
```

The destinations are `gs://phonics-benchmark/phonics/runs/<run-id>/`. The publisher
uploads only the files listed in the bundle's seal. It stops if a run directory holds
anything else, such as a `report/` folder or `.DS_Store`. It checks both runs before
writing anything. It creates objects only when absent, accepts existing identical
objects, verifies their CRC32C after upload, and refuses conflicting or extra objects.
It never deletes or overwrites the original source runs.

To correct a published file, rebuild and re-analyze the bundle, then name the
run-relative path with `--replace`. Check the plan with `--dry-run` first:

```sh
uv run --frozen --package phonics-benchmark python \
  benchmarks/phonics_benchmark/scripts/publish_unified_leaderboard_runs.py \
  --replace config.json --dry-run
```

A replacement is uploaded only if the object still has the generation that the publisher
just checked. The output records the previous generation and CRC32C. The bucket has no
object versioning, and the generations replaced on 24 September did not appear under
soft delete. Keep the previous local bundle, because it may be the only copy of a
replaced file.

On 24 September both `config.json` objects were replaced. The first versions carried the
selection's English gold-update record, attached to both languages, and local paths. The
replaced files and the publisher's record are kept locally in
`outputs/phonics-unified-leaderboard-20260922-superseded-20260924/`.

These are scoring bundles, not runner input configs. To score a downloaded bundle with
the normal reporter, place it under an output root using its run ID as the directory
name, then pass that run ID to `report_on_benchmark_run.py`. The reporter resolves the
bundled relative item path and reads `responses/merged.jsonl`.
