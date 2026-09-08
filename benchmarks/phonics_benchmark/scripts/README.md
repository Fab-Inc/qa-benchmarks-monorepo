# Scripts to run the benchmark

## Creating a benchmark run dataset

These are the steps to creating a benchmark run dataset.

### Prerequisites

- Set up your data according to [`data/README.md`](../data/README.md). The data files
  are not in git, so you need the expert workbook in `data/<language>/raw/` before you
  start — ask the team for it. Every step below writes to `data/<language>/`, which is
  gitignored.

### Steps

1. Ingest a language benchmark data-entry workbook into JSONL data files.

   ```shell
    uv run --package phonics-benchmark python \
        benchmarks/phonics_benchmark/scripts/ingest_benchmark_data.py \
        --language kiswahili \
        --workbook ./benchmarks/phonics_benchmark/data/kiswahili/raw/benchmark_data_tusome_ke.xlsx
   ```

1. Cross the distinct student profiles with the generation scenarios to build the item
   sets.

   ```shell
    uv run --package phonics-benchmark python \
        benchmarks/phonics_benchmark/scripts/build_generation_benchmark_items.py \
        --language kiswahili \
        --profiles-file student_knowledge_profiles.jsonl
   ```

   `--profiles-file` names any JSONL in the language data folder whose records carry
   `student_knowledge` — either the ingested profiles file above, or a legacy
   labelled-items file, whose text and gold-label columns are ignored.

   This writes to `data/<language>/`:

   - `benchmark_items_generation.jsonl`: the full set (every distinct profile x every
     scenario)
   - `benchmark_items_generation_dev.jsonl`: a smaller stratified sample of profiles,
     for fast iteration
   - `generation_items_summary.json`: item counts and composition per difficulty bucket
   - `generation_items_by_difficulty.png`: the same composition as a bar chart

   Profiles are deduplicated on the *content* of `student_knowledge` rather than on
   `student_knowledge_id`, and each is assigned a difficulty bucket from the number of
   graphemes it teaches. Profiles with no cluster knowledge get a documented placeholder
   — see the caveat in [`../README.md`](../README.md).

## Running the benchmark

See [`../README.md`](../README.md) to learn how to set up the run configuration and run
the benchmark.

## Generate benchmark report

Generate a report on benchmark run by running

```shell
uv run --package phonics-benchmark python \
   benchmarks/phonics_benchmark/scripts/report_on_benchmark_run.py \
   --run-id $YOUR_RUN_ID \
   --inspect-sample-size 10
```

Providing a number for the `--inspect-sample-size` flag will generate a sample of
benchmark items and each model's responses for you to inspect manually, with only
relevant columns.

Only responses whose provider call failed (`call_status` is not `success`) are excluded
from scoring; these can be re-attempted with `run-generation --retry-errors` (see
[`../README.md`](../README.md)).

Responses that came back but omitted the `<text>` tags are **not** excluded — they are
scored zero as format failures and reported separately as `format_failure_rate`, so a
model cannot dodge a hard item by replying in the wrong shape. See the Scoring section
of [`../README.md`](../README.md).
