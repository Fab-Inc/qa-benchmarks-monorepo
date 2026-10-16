# Unified runs for the 22 September leaderboard

The afternoon 22 September report at
`outputs/phonics-reports-20260922-coverage90/selections/fixed-991-en-1000-sw` is the
source of two normal runner runs:

| Language  | Run ID                         | Items | Models | Responses | Leaderboard models | Missing calls |
| --------- | ------------------------------ | ----: | -----: | --------: | -----------------: | ------------: |
| English   | `en-gold-leaderboard-20260922` |   991 |     40 |    35,910 |                 31 |         3,730 |
| Kiswahili | `sw-gold-leaderboard-20260922` | 1,000 |     46 |    40,524 |                 33 |         5,476 |

On 28 September the runs gained the `gpt-6-astra-medium` responses that finished by the
23 September refresh (`outputs/phonics-reports-20260923-astra`), giving Astra all 991
and 1,000 items. The other rows that refresh added are not included yet: 122 English
`mimo-v2.5` rows and 223 English `mimo-v2.5-pro` rows. Each run is laid out as
`launch-run.sh` lays out a run:

- `gs://phonics-benchmark/phonics/datasets/<run-id>.jsonl` holds the corrected items.
- `gs://phonics-benchmark/phonics/run-configs/<run-id>.json` holds the source config.
- `gs://phonics-benchmark/phonics/runs/<run-id>/` holds the runner's `config.json`
  snapshot and one `responses/responses_<model>.jsonl` file per model.

The run folder also keeps the merge records: `benchmark_items.jsonl`,
`item_provenance.jsonl`, `provenance.jsonl`, both coverage tables and
`excluded_responses.jsonl`. `source_config.json` is a verbatim copy of the selection's
config, local paths included. Response rows keep their original `run_id`, `result_id`,
`model_call_id` and `config_id`, and `provenance.jsonl` traces each one to its source
file. Failed and unparseable calls stay in the response files, and scoring excludes them
as usual.

Every kept response has the call ID the runner computes for the run config, so a later
run with the same config skips it. The builder refuses any response that fails this
check. The config lists only the models with responses. `qwen-3-14b` and
`qwen3-8b-alibaba` are dropped because they ran with `max_tokens` 16,384 and 8,192, and
no 30,000-token config reproduces their call IDs. Both are below the coverage threshold
in both languages, so no leaderboard changes.

The snapshot's `model_configs` record the September configuration of each model. Future
runs use the shared `custom_models.yaml`. As of 24 September its entries match every
kept model except `kimi-k2.5` and `qwen3-235b-a22b-thinking-2507` in Kiswahili. A run
that includes either model redoes its calls (320 and 222) and replaces its September
rows.

## Add models or responses

Add the model to the source config's `models`, keep `language`, `prompt_name` and
`max_tokens` unchanged, and run it with `results_dest` set to the run folder. The runner
checks that the config ID matches the snapshot, skips every call it already has, and
writes `responses/responses_<new-model>.jsonl`. A run of the full config also fills the
missing calls of existing models. On 24 September a local test with the real runner and
the shared model configs skipped all 35,542 English responses, kept each one byte for
byte, made 991 calls for one new model and made 4,098 gap-filling calls for 9 existing
models.

## Build, check and publish

The build needs the run's items in GCS first, as `launch-run.sh` uploads items before a
run. On 24 September they were copied from the run folders with
`gcloud storage cp --if-generation-match=0`. Then build from the sealed September 22
snapshot and selected report:

```sh
uv run --frozen --package phonics-benchmark python \
  benchmarks/phonics_benchmark/scripts/build_unified_leaderboard_runs.py
```

The builder loads each source config with the runner's own loader, as `run-cloud` does,
and writes the snapshot that `run_benchmark` writes before its first call. It uses a
workspace under `/tmp/phonics-gold-leaderboard-workspace`, so those neutral paths appear
in the snapshot. No model is called. The output is
`outputs/phonics-unified-leaderboard-20260922/`, with the run folders named `english/`
and `kiswahili/` and the source configs in `run-configs/`. The root artifact manifest
records hashes for every file. The builder refuses to replace an existing output.

Reparse the per-model responses, score them against the bundled gold, and compare all
adjusted balanced accuracy values and 95% intervals with the September 22 report:

```sh
uv run --frozen --package phonics-benchmark python \
  benchmarks/phonics_benchmark/scripts/analyze_unified_leaderboard_runs.py
```

This writes the adjusted balanced accuracy tables, the eligible-model tables and
`analysis/comparison.json`. It uses 1,000 item-bootstrap draws with seed 0, the strict
coverage threshold of more than 90% and the 700-item minimum. The analysis refuses to
write its output if any kept model's metric differs beyond `1e-10`. It also compares
eligible models with the reference report's combined `model_summary_coverage_gt90.csv`.
`--reference` selects the report, and `--skip-model LANGUAGE:MODEL` leaves out a model
whose rows differ from it on purpose. With the Astra responses added, the comparison
with the 23 September report shows **zero difference** for all 46 Kiswahili models and
for 39 English models, skipping `mimo-v2.5` and `mimo-v2.5-pro`. That includes the
confidence bounds and both eligible-model tables:

```sh
uv run --frozen --package phonics-benchmark python \
  benchmarks/phonics_benchmark/scripts/analyze_unified_leaderboard_runs.py \
  --reference outputs/phonics-reports-20260923-astra/selections/fixed-991-en-1000-sw \
  --skip-model english:mimo-v2.5 --skip-model english:mimo-v2.5-pro
```

Publish after the comparison succeeds, and check the plan with `--dry-run` first:

```sh
uv run --frozen --package phonics-benchmark python \
  benchmarks/phonics_benchmark/scripts/publish_unified_leaderboard_runs.py --dry-run
```

The publisher uploads only the files listed in the bundle's seal. It stops if a run
folder holds anything else, such as a `report/` folder or `.DS_Store`. It checks both
runs before writing anything, and it checks that the published items match each run. It
creates objects only when absent, accepts existing identical objects and refuses
conflicting or extra ones. `--replace PATH` overwrites a named run-relative object, and
`--delete PATH` removes a named object that is not in the seal. Each is conditional on
the generation just checked. The output records previous generations and CRC32C values.
The bucket has no object versioning, and replaced generations did not appear under soft
delete, so keep the previous local bundle.

History of the published runs:

- **24 September.** First published with a single `responses/merged.jsonl`. Both
  `config.json` objects were then replaced, because they carried the English gold-update
  record on both languages and local paths. The previous files are in
  `outputs/phonics-unified-leaderboard-20260922-superseded-20260924/`.

- **28 September.** Republished as runner runs:

  - added the per-model response files and the `run-configs/` objects
  - replaced `config.json`, both coverage tables, `provenance.jsonl`, and the English
    `excluded_responses.jsonl`
  - deleted both `responses/merged.jsonl` objects

  The previous files and the publisher's record are in
  `outputs/phonics-unified-leaderboard-20260922-superseded-20260928/`.

- **28 September, later.** Added the 23 September Astra responses: 368 English and 452
  Kiswahili. Each run's Astra response file, both coverage tables and `provenance.jsonl`
  were replaced. The previous files and the publisher's record are in
  `outputs/phonics-unified-leaderboard-20260922-superseded-20260928-astra/`.

## Reproduce the 22 September reports from GCS

The report scripts read one root with `english/` and `kiswahili/` analysis runs, an
`audit/` directory and an `artifact_manifest.json` over the inputs. The local merge and
selection steps used to create that layout. For the published runs, the download script
creates it:

```sh
uv run --frozen --package phonics-benchmark python \
  benchmarks/phonics_benchmark/scripts/download_unified_leaderboard_runs.py \
  --root outputs/phonics-gold-leaderboard-20260922-from-gcs
```

It downloads each object at the generation it lists and checks every CRC32C. It refuses
a run with missing, unknown or corrupt objects, or with any file under `responses/` that
is not a per-model response file. The items come from each run's `items_path`, at the
generation the run recorded. Each analysis run is rebuilt with the merge code, so
coverage is recomputed from the responses and includes models added later. The
downloaded runs are kept under `gcs/`. The manifest covers the downloaded and rebuilt
files. It does not link back to the 22 September selection. The download time is written
outside the seal, so the same generations give the same manifest.

Generate reports from the rebuilt runs with the consolidated reporter:

```sh
PHONICS_ROOT=outputs/phonics-gold-leaderboard-20260922-from-gcs
uv run --frozen --package phonics-benchmark python \
  benchmarks/phonics_benchmark/scripts/report_on_benchmark_run.py \
  --run-dir "$PHONICS_ROOT/english" --run-dir "$PHONICS_ROOT/kiswahili" \
  --output-dir "$PHONICS_ROOT/reports" \
  --metadata benchmarks/phonics_benchmark/scripts/f2_plot_metadata \
  --min-items 700 --min-coverage 0.9 --hardware-tiers
```

The supplied metadata preserves September display names, provider colours and hardware
classifications. See [current reporting](current-metric-reports.md) for the new output
layout and [refactor review notes](reporting-refactor-review.md) for its comparison with
saved results. The following records the original September workflow and its historical
file layout.

On 28 September this ran on the runs with the Astra responses, using this metadata, and
was compared with the 23 September selection. Of its 80 CSV files, 30 were
byte-identical, including every F1, F2, balanced accuracy and adjusted balanced accuracy
leaderboard and the eligible-model summaries. Another 39 differ only because rows are
gone for the two dropped models, for models that never responded, and for the two
English `mimo` models, whose rows differ on purpose. Every remaining row is identical,
and the set-valued columns of `results_with_scoring.csv` hold the same members. The
other 11 were not produced. Four are the
`adjusted_balanced_accuracy/*_all_scored_models.csv` and `*_summary_coverage_gt90.csv`
files from a validation script. Seven are Kiswahili hardware-flagging diagnostics from
`analyze_hardware_flagging.py`. Neither script is committed. 42 of the 44 figures were
produced; the two others are the same diagnostic figure. The previous-snapshot
comparison in `finalize_phonics_refresh.py` needs local inputs that were not published.
