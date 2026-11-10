# Scripts to run the benchmark

## Creating a benchmark run dataset

These are the steps to creating a benchmark run dataset. Note that this may be further
curated after running the benchmark, e.g. using Item Response Theory.

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

1. Pair student profiles with decodable texts and analyse the pairs to decide on
   filtering.

   ```shell
    uv run --package phonics-benchmark python \
        benchmarks/phonics_benchmark/scripts/build_benchmark_item_candidates.py \
        --language kiswahili
   ```

   Inspect the following contents in `data/<language>/` to determine how you want to
   filter the candidates:

   - `benchmark_item_candidates.jsonl`: `benchmark_item_id` and decodability analysis
   - `nondecodable_ratio_histogram_for_benchmark_candidates.png`: a histogram of the
     non-decodable ratio distribution
   - `nondecodable_summary_for_benchmark_candidates.json`: a summary of nondecodable
     word counts and ratios across the full cross-product

1. Cut the benchmark item candidates down to the set the benchmark runs.

   Run from the repository root with your _own filtering parameters_:

   ```shell
    uv run --package phonics-benchmark python \
        benchmarks/phonics_benchmark/scripts/filter_benchmark_items.py \
        --language kiswahili \
        --max-zero-nondecodable 6 \
        --max-one-nondecodable 10 \
        --max-nondecodable-ratio 0.2 \
        --sample-size-other 16 \
        --random-state 42
   ```

## Running the benchmark

See [`../README.md`](../README.md) to learn how to set up the run configuration and run
the benchmark.

## Generate benchmark report

Generate a report on benchmark run by running

```shell
uv run --frozen --package phonics-benchmark python \
   benchmarks/phonics_benchmark/scripts/report_on_benchmark_run.py \
   --run-id $YOUR_RUN_ID \
   --inspect-sample-size 10
```

Providing a number for the `--inspect-sample-size` flag will generate a sample of
benchmark items and each model's responses for you to inspect manually, with only
relevant columns.

The following responses are excluded from scoring:

- API error (`call_status` is not `success`): These can be re-attempted with
  `run-decodability` with `--retry-errors` (see [`../README.md`](../README.md)).
- Parsing error (`parse_ok` is `False`)

## Updating labels for existing benchmark items

If you updated your decodability logic and hence need to update the labels for a
benchmark items dataset, run

```shell
uv run --frozen --package phonics_benchmark \
    benchmarks/phonics_benchmark/scripts/update_benchmark_item_labels.py \
    $EXISTING_ITEMS_PATH \
    $TARGET_ITEMS_PATH
```

It will save the re-labelled version of `$EXISTING_ITEMS_PATH` to `$TARGET_ITEMS_PATH`.

It will log a report on

- how many items were re-labeled
- how many and which words were
  - dropped (previously non-decodable but now decodable)
  - added (previously decodable but now non-decodable)
