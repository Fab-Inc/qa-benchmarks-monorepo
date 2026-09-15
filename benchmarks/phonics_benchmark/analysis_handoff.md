# Kiswahili analysis and LKB plots: work in progress

This is a standalone analysis of saved responses, with LKB-style figures for average
perfect-match accuracy, pooled F1 and pooled F2. It is not a finished reporting
workflow. English support is deferred.

## Commit scope

Include these nine files, relative to `benchmarks/phonics_benchmark/`:

```text
analysis_handoff.md
scripts/analyze_finished_kiswahili_models.py
scripts/plot_model_metrics.py
scripts/plot_lkb_style_metrics.py
tests/test_finished_kiswahili_analysis.py
tests/test_plot_model_metrics.py
tests/test_lkb_style_metrics.py
data/model_families.csv
data/model_families.md
```

The LKB plotter imports snapshot helpers from the analysis script and CSV/CI helpers
from `plot_model_metrics.py`, so all three scripts must travel together. They use the
existing phonics parsing, labeling and scoring modules. The runner, CLI, deployment,
run-config, `pyproject.toml` and `uv.lock` changes elsewhere in the working tree are not
dependencies of this bundle. The F1 weighting audit and perfect-match failure diagnosis
are separate exploratory scripts and can remain outside this commit. Generated
`outputs/` directories are Git-ignored.

## Rerun from the repository root

Requires the repository's existing `uv` environment setup and Google Application Default
Credentials with read access to the phonics benchmark bucket. Analysis only reads cloud
objects; it makes no model calls or cloud writes. Scripts reject existing output
directories.

The analysis defaults select these runs:

- `gs://phonics-benchmark/phonics/runs/sw-tusome-clusters-20260905/`
- `gs://phonics-benchmark/phonics/runs/sw-tusome-clusters-20260906-remaining/`

Their original input is
`gs://phonics-benchmark/phonics/datasets/sw-tusome-clusters-20260905.jsonl`, SHA-256
`379b5b4afa60bb86772b692258d5193eac8062c1ceaf8ea1091c8c7e0bf852b1`. Keep this as the
original input; the corrected gold file is a separate override:

```bash
analysis_dir="benchmarks/phonics_benchmark/outputs/kiswahili-analysis-$(date -u +%Y%m%dT%H%M%SZ)"

uv run --frozen --package phonics-benchmark python \
  benchmarks/phonics_benchmark/scripts/analyze_finished_kiswahili_models.py \
  --gold-labels-source gs://phonics-benchmark/phonics/datasets/kiswahili-with-clusters/benchmark_items_syllabic_nasals_ungated.jsonl \
  --output-dir "$analysis_dir"
```

The override uses saved `nondecodable_words` directly, without recomputing decodability.
It checks that item IDs, texts and student profiles match the original dataset, and
retains the original run hash checks. On 15 September 2026, the corrected file had
SHA-256 `6124e245ed0cf2f6c915ed444cb7c928f521925d75005d9cc44977c73ce7de25`; labels
differed on 699 of 1,665 items. Every analysis snapshots its inputs and records hashes
and GCS generations. Omitting the override retains the earlier behaviour: recompute
Tusome labels and reject differences from saved labels.

Obtain the display-name, provider-colour, price and size metadata used for the figures
from the pinned LKB commit. This does not require switching branches:

```bash
git fetch origin qfb-194-implement-lkb-workflow
metadata_dir="$(mktemp -d)"
git show 5e67f9fb1836dff8d7b0a34695fca7c79ff4193b:benchmarks/linguistic_knowledge_benchmark/data/models.csv > "$metadata_dir/models.csv"
git show 5e67f9fb1836dff8d7b0a34695fca7c79ff4193b:benchmarks/linguistic_knowledge_benchmark/data/providers.csv > "$metadata_dir/providers.csv"

uv run --frozen --package phonics-benchmark python \
  benchmarks/phonics_benchmark/scripts/plot_lkb_style_metrics.py \
  --summary-csv "$analysis_dir/summary_by_model.csv" \
  --models-csv "$metadata_dir/models.csv" \
  --providers-csv "$metadata_dir/providers.csv" \
  --output-dir "${analysis_dir}-lkb-plots"
```

Both metadata arguments also accept `gs://` URIs. The family table defaults to the
committed `data/model_families.csv`. Plot outputs include PNGs, PDFs, plotted values as
CSVs, an input manifest and a README with links and captions. Each of the three metrics
gets a leaderboard, cost/size comparisons, hardware plots and a family comparison.

## Scoring and bootstrap conventions

- A finished model has a saved row for every benchmark item, including failures. Failed
  calls and unparseable responses are excluded from scores and counted separately.
  Partial models appear in the coverage CSV but are not scored.
- Words are lowercased and deduplicated within an item. Perfect-match accuracy averages
  exact equality of the predicted and gold word sets, including correct empty
  predictions. Pooled F1 and F2 are calculated from summed TP/FP/FN counts.
- Default CIs use 10,000 paired item resamples, seed 0, and 95% percentile bounds. An
  item is a student-profile/text combination. The same draws serve all models; undefined
  replicates are excluded and their valid counts are recorded.
- The LKB plotter uses the saved per-model CIs. Its separate hardware-tier mean plot
  uses LKB's bootstrap over models, not items; swarm boxes show quartiles.
- The analysis still exports item-averaged F1 for compatibility. The LKB figures default
  to the three metrics named above.

## Where to pick up

Family assignments currently cover 31 models. Extend the explicit table when newly
finished models lack a family. Model sizes and prices are inherited from the LKB
metadata snapshot; unpublished sizes remain unknown and blended prices are not measured
inference costs. The source commit and styling conventions are also documented in the
LKB plotting script.

The older failure-diagnosis script reconstructs Tusome rules and is not adapted to the
corrected gold override. It is not part of this handoff.

The 35 analysis tests and 24 plotting tests passed after the gold-label change:

```bash
uv run --frozen --package phonics-benchmark pytest \
  benchmarks/phonics_benchmark/tests/test_finished_kiswahili_analysis.py \
  benchmarks/phonics_benchmark/tests/test_plot_model_metrics.py \
  benchmarks/phonics_benchmark/tests/test_lkb_style_metrics.py
```
