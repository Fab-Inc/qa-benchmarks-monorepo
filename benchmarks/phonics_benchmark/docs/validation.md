# Phonics validation reports

Two reports support review of the historical data consolidation and the ongoing scoring
code:

- [September merge preservation audit](merge-audit-20260922/README.md): actual response
  counts by source run and model, dispositions, and verified input hashes.
- Executable metric examples: handwritten expectations, computed results, and production
  leaderboard figures in one standalone HTML file.

## Run the metric examples

From the repository root:

```sh
uv run --frozen --package phonics-benchmark python \
  benchmarks/phonics_benchmark/scripts/build_metric_examples.py \
  --output outputs/metric-examples.html
```

Open `outputs/metric-examples.html` in a browser. It includes its images and works
offline. No documentation framework or hosting is required.

The examples cover F1/F2 and balanced accuracy, adjusted points, out-of-text
predictions, repeated words, undefined denominators, pooling, missing responses, item
bootstrap intervals, coverage boundaries, and plotted values and intervals. Expected
values and explanations live alongside each example in
[`build_metric_examples.py`](../scripts/build_metric_examples.py). Review expected
answers independently; do not update them automatically from computed outputs.

The command exits nonzero if a check fails. Numerical failures are included in the
rendered report. Unexpected execution errors also fail the command, but may prevent the
report from being written. Plot checks inspect data and labels rather than pixel
snapshots; visual inspection remains useful for layout and readability.

The **Phonics validation examples** GitHub Actions workflow runs these examples on
relevant PRs, including PRs targeting branches other than `main`, and on `main` updates.
Download its **phonics-metric-examples** artifact to review the HTML from that commit.
Artifacts are retained for 30 days. The same command regenerates the report locally. The
workflow publishes no website and needs no GCS credentials.

## Reproduce the historical audit

The actual snapshots are ignored local data. With the archived folders present:

```sh
python benchmarks/phonics_benchmark/scripts/audit_unified_preservation.py \
  --report-dir outputs/merge-audit-review \
  --ledger outputs/merge-audit-review-rows.csv.gz
```

Use a new report directory and ledger path on each run. `--outputs-root` allows the
archives to live elsewhere; manifest paths are resolved relative to that root. The audit
checks the actual source rows independently of the merge implementation, and writes
source/model tables, an input-hash inventory, a verification record, and a compressed
row-level ledger. It makes no cloud or model calls.

The small historical tables are checked into this branch. The row ledger remains under
`outputs/` because it describes every source occurrence. Normal CI exercises the audit's
rejection checks on small local inputs; it does not rerun the historical audit or
download the benchmark corpus.
