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

### Generating other analysis results

Other than the main report, there are a number of scripts that generate additional
analyses. (TODO: consolidate into the main report generation step)

1. Inspect difficult items: find items or item-word pairs that all of the top 10 models
   get wrong. The script doesn't produce any plots, only tables of items or item-words
   that too many strong models get wrong that they should be manually reviewed. Set
   `$TARGET_DIRECTORY/$LANGUAGE/` should contain `results_with_scoring.csv`.

   ```shell
   uv run --frozen --package phonics-benchmark python \
     benchmarks/phonics_benchmark/scripts/review_difficult_items.py \
     --language $LANGUAGE --target $TARGET_DIRECTORY
   ```

1. A couple of analysis per hardware tier:

   1. Plot predicted vs. actual non-decodabe word ratio in items

      ```shell
      uv run --frozen --package phonics-benchmark python \
        benchmarks/phonics_benchmark/scripts/plot_calibration.py \
        --root outputs/phonics-relabelled-20261110 \
        --language kiswahili \
        --metric adjusted_balanced_accuracy
      ```

   1. Plot error rates by word length. `--error` can be `fp` or `fn`.

      ```shell
      uv run --frozen --package phonics-benchmark python \
        benchmarks/phonics_benchmark/scripts/plot_error_rate_by_word_length.py \
        --root outputs/phonics-relabelled-20261116 \
        --error $ERROR_TYPE \
        --language $LANGUAGE \
        --metric balanced_accuracy
      ```

   `--language` can be `kiswahili` or `english`. `--metric` is used to select the top
   models in each hardware tier.

1. Kiswahili only: break down each hardware tier's errors by failure mode, i.e. what
   kind of word the models get wrong given the item's student profile (untaught
   graphemes or clusters, syllabic nasals, and so on). Like the plots above, it takes
   the top models per hardware tier from the run folder, here ranked by adjusted
   balanced accuracy, so the run folder needs
   `reports/hardware_tiers/adjusted_balanced_accuracy/hardware_tiers_kiswahili.csv` and
   `hardware_tiers_top10_models.csv` in the same folder.

   ```shell
   uv run --frozen --package phonics-benchmark python \
     benchmarks/phonics_benchmark/scripts/kiswahili_word_analysis.py \
     --run outputs/phonics-relabelled-20261116
   ```

   The outputs go to `<run>/reports/kiswahili/error_analysis/` (change this with
   `--report-dir`):
