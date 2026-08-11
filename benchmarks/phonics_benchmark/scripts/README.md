# Scripts to run the benchmark

## Creating a benchmark run dataset

These are the steps to creating a benchmark run dataset. Note that this may be further
curated after running the benchmark, e.g. using Item Response Theory.

### Prerequisites

- Set up your data according to [`data/README.md`](../data/README.md)

### Steps

1. Ingest a language benchmark data-entry workbook into JSONL data files.

   ```shell
    uv run --package phonics-benchmark python \
        benchmarks/phonics_benchmark/scripts/ingest_benchmark_data.py \
        --language kiswahili \
        --workbook path/to/benchmark_template.xlsx
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
   - `nondecodable_ratio_histogram.png`: a histogram of the non-decodable ratio
     distribution
   - `nondecodable_count_and_ratio_summary.json`: a summary of nondecodable word counts
     and ratios across the full cross-product

1. Cut the benchmark item candidates down to the set the benchmark runs.

   Run from the repository root with your _own filtering parameters_:

   ```shell
    uv run --package phonics-benchmark python \
        benchmarks/phonics_benchmark/scripts/filter_benchmark_items.py \
        --language kiswahili \
        --max-zero-nondecodable 20 \
        --max-one-nondecodable 40 \
        --max-nondecodable-ratio 0.2 \
        --sample-size-other 128
   ```

## Running the benchmark

See [`../README.md`](../README.md) to learn how to set up the run configuration and run
the benchmark.

## Analysing the benchmark

To do.
