# Scripts to run the benchmark

## Creating a benchmark run dataset

These are the steps to creating a benchmark run dataset.

### Prerequisites

- `data/<language>/student_knowledge_profiles.jsonl` must be present. This is the
  canonical profile list maintained by the data team and is the single source of truth
  for the benchmark — ask them for it. It is gitignored, like everything else under
  `data/` except this README and the blank workbooks in `templates/`. See
  [`data/README.md`](../data/README.md).

### Steps

Cross the canonical student profiles with the generation scenarios to build the item
sets:

```shell
 uv run --package phonics-benchmark python \
     benchmarks/phonics_benchmark/scripts/build_benchmark_items.py \
     --language kiswahili
```

This writes to `data/<language>/`:

- `benchmark_items.jsonl`: the full set — every profile x every scenario in
  `src/phonics_benchmark/scenarios.yaml` (156 x 4 = 624 items for Kiswahili)
- `benchmark_items_dev.jsonl`: a stratified sample of 30 profiles x 4 scenarios = 120
  items, for fast iteration
- `benchmark_items_summary.json`: item counts and composition per difficulty bucket
- `benchmark_items_by_difficulty.png`: the same composition as a bar chart

Every profile's `student_knowledge` is validated against the language's model in
`decodability` before any items are written, so a bad grapheme or a cluster that
overlaps a known cluster pattern fails here rather than mid-run. Each item is stamped
with its `difficulty_bucket`, `n_graphemes_known`, `n_clusters_known` and
`n_cluster_patterns_known` so the report can break scores down by them.

### Re-deriving profiles from a workbook (rarely needed)

Only if you are regenerating the canonical file rather than receiving it. Ingest a
language's data-entry workbook into `student_knowledge_profiles.jsonl`:

```shell
 uv run --package phonics-benchmark python \
     benchmarks/phonics_benchmark/scripts/ingest_benchmark_data.py \
     --language kiswahili \
     --workbook ./benchmarks/phonics_benchmark/data/kiswahili/raw/benchmark_data_tusome_ke.xlsx
```

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
