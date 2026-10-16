# Phonics validation reports

Two reports support review of the historical data consolidation and the ongoing scoring
code:

- [September merge preservation audit](merge-audit-20260922/README.md): actual response
  counts by source run and model, dispositions, and verified input hashes.
- Executable metric examples: handwritten expectations, Python code, computed results,
  and production leaderboard figures in one standalone HTML file.

## Create the metric examples HTML locally

From the repository root:

```sh
uv run --frozen --package phonics-benchmark python \
  benchmarks/phonics_benchmark/scripts/build_metric_examples.py \
  --output benchmarks/phonics_benchmark/outputs/metric-examples.html
```

Open `benchmarks/phonics_benchmark/outputs/metric-examples.html` in a browser. It
includes its images and works offline. No documentation framework or hosting is
required.

Omitting `--output` uses this same benchmark-local path regardless of the working
directory. Supply `--output` to write the report elsewhere.

Start with the count definitions and metric equations, then compare changes to a perfect
prediction. Each worked case shows hand-counted inputs, substituted equations, and
expected versus computed scores. The pooling section explains how item sizes and missing
responses affect the reported score and coverage.

Bootstrap resampling appears in the plotting section. A two-item example enumerates all
possible resamples, then passes production bootstrap estimates and intervals to the
plots. Production reporting samples the full target item set for every metric, including
items with no scorable response from any model, using shared seeded draws. Additional
plots use supplied intervals to check sorting, eligibility, units, and negative values
independently. These visible examples use intervals containing their scores. Separate
regression tests check bounds that exclude the score, which the renderer must preserve
rather than recenter. Both check that intervals line up with the bars in data and
rendered coordinates.

The main code blocks are expanded directly before the tables and figures they produce.
They include the scoring calls and the row construction, so each displayed value can be
traced to its input. The opening walkthrough computes the words in each confusion group;
handwritten expectations remain separately identified. Longer shared definitions and
repeated per-case table code are expandable.

The displayed code is extracted from the same script that runs the examples; there is no
separate copy to maintain. These are readable source excerpts, not interactive notebook
cells, and require no Jupyter dependencies.

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

The normal test suite runs the same numerical and plotting checks without writing HTML
or image files. To run just these tests locally:

```sh
uv run --frozen --package phonics-benchmark pytest -q \
  benchmarks/phonics_benchmark/tests/test_metric_examples.py
```

CI runs the tests through the existing test workflow. Generate the HTML manually with
the `build_metric_examples.py` command above; CI does not create or upload the report.

## Reproduce the historical audit

The actual snapshots are ignored local data. With the archived folders present:

```sh
uv run python benchmarks/phonics_benchmark/scripts/audit_unified_preservation.py \
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
