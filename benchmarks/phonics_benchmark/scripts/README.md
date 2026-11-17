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

## Generate benchmark reports

Use `report_on_benchmark_run.py` to score runs and generate F1/F2, balanced-accuracy and
adjusted-BA tables and figures. Use `plot_benchmark_report.py` to redraw saved reports
without rescoring.

```sh
uv run --frozen --package phonics-benchmark python \
  benchmarks/phonics_benchmark/scripts/report_on_benchmark_run.py \
  --run-id YOUR_RUN_ID --inspect-sample-size 10
```

See [current reporting](../docs/current-metric-reports.md) for output files, coverage
filters, bootstrap settings, multiple runs and optional hardware comparisons. Failed API
calls and unparseable responses are excluded from scores and counted in the coverage
report. The inspection sample includes every model's response for each sampled item.

## Historical consolidation and review

The September consolidation combined existing responses and checked them against
rescoring the resulting unified runs. These scripts are retained for provenance and
reproduction; they are not steps required to report on a new run.

| Task                                                          | Scripts                                                                                                                                                   | Details                                                                 |
| ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| Download source snapshots, merge responses and select items   | `download_benchmark_snapshot.py`, `merge_local_benchmark_runs.py`, `select_local_benchmark_subset.py`                                                     | [Historical merge workflow](../docs/local-merged-analysis.md)           |
| Refresh labels and compare fixed-ID snapshots                 | `apply_refreshed_gold.py`, `finalize_phonics_refresh.py`                                                                                                  | [Historical refresh](../docs/historical-report-refresh.md)              |
| Package, verify, publish and retrieve unified runs            | `build_unified_leaderboard_runs.py`, `analyze_unified_leaderboard_runs.py`, `publish_unified_leaderboard_runs.py`, `download_unified_leaderboard_runs.py` | [Unified runs](../docs/unified-leaderboard-runs.md)                     |
| Audit actual retained/dropped responses and source provenance | `audit_unified_preservation.py`                                                                                                                           | [Committed preservation tables](../docs/merge-audit-20260922/README.md) |
| Review metric equations, examples and plotting behavior       | `build_metric_examples.py`                                                                                                                                | [Validation and manual HTML generation](../docs/validation.md)          |

[Reporting refactor review notes](../docs/reporting-refactor-review.md) explain the
module boundaries and record the local comparison with existing reports. Detailed
procedures live in `docs/`; this README remains the entry point beside the scripts.
