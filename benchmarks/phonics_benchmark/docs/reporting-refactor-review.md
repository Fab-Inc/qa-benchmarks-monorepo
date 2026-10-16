# Reporting refactor: review scope and local verification

Ongoing reporting now has two entry points: `report_on_benchmark_run.py` scores runs and
creates reports; `plot_benchmark_report.py` renders saved tables. Their implementations
import package modules, and package modules never import scripts.

| Previous responsibility                                                | Current implementation                            |
| ---------------------------------------------------------------------- | ------------------------------------------------- |
| Run loading and inspection in `report_on_benchmark_run.py`             | `reporting.py`                                    |
| `analyze_balanced_accuracy.py` calculations                            | `balanced_accuracy.py` and `bootstrap.py`         |
| F1/F2/BA/adjusted-BA leaderboard rendering                             | Shared metric-aware renderer in `plots.py`        |
| `analyze_phonics_hardware_tiers.py`                                    | `hardware.py`, optional in the report commands    |
| Combined summaries and report index from `finalize_phonics_refresh.py` | `reporting.py`                                    |
| Gold refresh and previous-snapshot comparison                          | Retained historical `finalize_phonics_refresh.py` |
| Historical input checks and legacy table reading                       | `reporting_inputs.py` and `historical_reports.py` |

The separate F1/F2/BA/adjusted-BA drivers have been removed. Two small import-only
compatibility files keep historical analysis consumers working without maintaining
another implementation. The merge, preservation-audit and unified-run packaging scripts
remain historical tooling. Exploratory, untracked analysis scripts were not incorporated
into the production workflow.

## Intentional bootstrap alignment

Both previous implementations already sampled whole items. F1/F2 previously used
observed item IDs; BA used every target item. Reporting now gives both implementations
the same sorted full target item list and shared seeded draw generator. Missing
responses contribute no counts. Metric definitions and point scores are unchanged.
Intervals can change when a target item has no scorable response from any model.

The normal tests include such an entirely unanswered item and verify that it remains in
the sampling population and that both routines receive identical item draws. The
handwritten metric examples and Matplotlib artist checks still exercise the production
implementations, now imported from the package.

## One-time comparison against saved reports

The refactor was exercised locally on the English and Kiswahili selections in
`outputs/phonics-reports-20260923-astra/selections/fixed-991-en-1000-sw`, with 1,000
bootstrap draws, seed 0, 95% intervals, the 700-item floor and strict >90% coverage. New
reports were written separately under `/tmp/phonics-report-refactor-after`; the saved
reports and inputs were retained unchanged.

| Check                                                 |     English |   Kiswahili |
| ----------------------------------------------------- | ----------: | ----------: |
| Target items                                          |         991 |       1,000 |
| Scored models                                         |          43 |          48 |
| Scorable responses                                    |      36,676 |      40,398 |
| Eligible leaderboard models                           |          31 |          33 |
| Largest F1/F2 summary or bootstrap difference         |           0 |           0 |
| Largest BA/adjusted-BA summary or interval difference | < 4 × 10⁻¹⁶ | < 3 × 10⁻¹⁵ |
| Coverage and failure/missing counts                   | Exact match | Exact match |
| Eligible model sets                                   | Exact match | Exact match |

Every target item in these snapshots had a scorable response from at least one model, so
aligning the bootstrap populations leaves their sampled item sets unchanged. The BA
differences are floating-point rounding; the numerical comparison used
`rtol=0, atol=1e-12`. Counts and model membership were compared exactly.

Hardware group membership, classifications, medians and top-ten means also match for all
four metrics. BA baseline counts match exactly in both languages.

The comparison program and generated report artifacts are local review evidence, not
committed tests or fixtures. The small synthetic checks remain in the normal test suite;
[validation.md](validation.md) explains how to generate their HTML.
