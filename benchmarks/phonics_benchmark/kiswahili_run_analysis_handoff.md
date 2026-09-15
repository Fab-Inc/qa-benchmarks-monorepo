# Kiswahili decodability runs: analysis handoff

The original ten-model results were verified and updated on **15 September 2026, 09:14
UTC / 10:14 BST**: all ten models have 1,665 successful calls. The additional 44-model
section still reflects **10 September 2026, 09:01 UTC / 10:01 BST**; its counts are
historical confirmed minimums, not a fresh progress check.

## Shared benchmark items

- **Exact dataset used by both runs:**
  `gs://phonics-benchmark/phonics/datasets/sw-tusome-clusters-20260905.jsonl`
- **Original item artifact from which that snapshot was copied:**
  `gs://phonics-benchmark/phonics/datasets/kiswahili-with-clusters/benchmark_items.jsonl`
- Dataset SHA-256: `379b5b4afa60bb86772b692258d5193eac8062c1ceaf8ea1091c8c7e0bf852b1`
- 1,665 student-profile/text combinations, spanning 59 texts and 152 profiles.
- Language: `sw`. Prompt: `decodability_evaluation_clusters`, explicitly mentioning
  consonant clusters. Both runs use `max_tokens: 30000` and five-item checkpoints.

## Original 10-model run

**Run ID:** `sw-tusome-clusters-20260905`. The initial execution `phonics-run-f7z6m`
failed. Recovery execution `phonics-run-remaining-k6xfx` repaired the four Sonnet
errors, and resume `phonics-run-remaining-spskl` filled the remaining 480 Gemini calls,
finishing successfully on 14 September at 23:40 UTC / 15 September at 00:40 BST.

- **Results folder:** `gs://phonics-benchmark/phonics/runs/sw-tusome-clusters-20260905/`
- **Immutable source config:**
  `gs://phonics-benchmark/phonics/run-configs/sw-tusome-clusters-20260905-10models.json`
  ([local copy](configs/sw-tusome-clusters-20260905-10models.json)).
- **Pinned model/provider registry:**
  `gs://phonics-benchmark/phonics/model-configs/sw-tusome-clusters-20260905.yaml`

Confirmed totals at the original results folder: **16,650 / 16,650 rows**, all
successful API calls, with **zero recorded API errors**. Recovered results were copied
back on 15 September after a full backup and row-by-row verification.

Finished models:

- `claude-haiku-4-5-20251001`
- `claude-sonnet-4-6`
- `gemini-3-flash-preview`
- `gemini-3.1-flash-lite-preview`
- `gemini-3.1-pro-preview`
- `gemma-4-31b-it`
- `gpt-4.1`
- `gpt-5.4-nano-2026-03-17`
- `ministral-3-8b-2512`
- `mistral-small-2603`

Every model has 1,665 successful calls. All 16,166 previously successful rows were
preserved; the only changes were four repaired Sonnet rows and 480 new Gemini rows. The
original Sonnet and Gemini files were updated with generation preconditions, and the
recovery provenance was added as `log.json`.

Backup, object manifests, and verification reports:
`gs://phonics-benchmark/phonics/publish-audits/sw-tusome-clusters-20260905-publish-20260915T090538Z/`.
See [the recovery handoff](kiswahili_ten_model_retry_handoff.md) for the full audit.

## Additional 44-model run

**Run ID:** `sw-tusome-clusters-20260906-remaining`. Execution
`phonics-run-remaining-xtxwg` resumed this same run on 10 September, replacing the
cancelled `phonics-run-remaining-tgl46` execution.

- **Results folder (unchanged by the restart):**
  `gs://phonics-benchmark/phonics/runs/sw-tusome-clusters-20260906-remaining/`
- **Immutable source config / full selected model list:**
  `gs://phonics-benchmark/phonics/run-configs/sw-tusome-clusters-20260906-remaining-ready.json`
  ([local copy](configs/sw-tusome-clusters-20260906-remaining-ready.json)).
- **Pinned model/provider registry:**
  `gs://phonics-benchmark/phonics/model-configs/sw-tusome-clusters-20260906-remaining-v2.yaml`

Confirmed minimum: **30,105 / 73,260 rows**. The restart preserved all 30,015 previously
saved rows; Gemini Flash-Lite then added its remaining 90 successful responses. Counts
here do not include any subsequently added checkpoints or files.

Finished models — all 1,665 calls successful:

- `claude-fable-5`
- `claude-opus-5`
- `fw-gpt-oss-120b`
- `gemini-3.5-flash`
- `gemini-3.5-flash-lite`
- `gpt-4.1-mini`
- `gpt-4.1-nano`
- `gpt-5-nano-2025-08-07-medium`
- `gpt-5.4-2026-03-05-medium`
- `gpt-5.5-2026-04-23-medium`
- `gpt-5.6-luna`
- `gpt-5.6-sol`
- `gpt-5.6-terra`
- `grok-4.5`
- `o3-2025-04-16`
- `o4-mini-2025-04-16`

Other confirmed partial results:

| Model             | Saved rows | Successful calls |    Recorded errors |
| ----------------- | ---------: | ---------------: | -----------------: |
| `claude-sonnet-5` |      1,165 |            1,164 |     1 API overload |
| `deepseek-v4-pro` |      1,010 |            1,010 |                  0 |
| `kimi-k2.6`       |      1,290 |            1,254 | 36 empty responses |

Completion of the other 25 selected models has not been verified. Use the
`remaining-ready.json` config above, not the earlier 53-model planning config or the
smoke-test configs, to determine this run's selection.

## Reading the results

- Each results folder contains `config.json` (the resolved runtime snapshot, including
  model settings) and `responses/responses_<model>.jsonl` files. The immutable source
  config is outside that folder, at the URI given above.
- Join response rows to the shared dataset on `benchmark_item_id`, keeping `run_id` and
  `model` to distinguish evaluations. These items are profile/text combinations, not
  unique texts.
- Rows contain the saved `prompt`, `output`, `success` and `call_status`; failed rows
  also contain `error_type` and `error_message`. A successful API call is not
  necessarily a valid or correct benchmark answer. Do not treat failed calls as
  successful empty predictions (`[]`).
- Normal resume skips already saved rows, including failures. Recorded errors therefore
  still need a separate recovery pass; they were not repaired by the image restart. Copy
  a stable set of checkpoints for analysis because the live additional-run files
  continue changing.
- The additional run's replacement image is
  `europe-west1-docker.pkg.dev/fab-qaf-decodability/fab-benchmarks/phonics-benchmark@sha256:adf27af45d14ceea65b2b493f8d4fac3c0fd1ec53e41d2995e18104b3e9491c0`.
  It records `ReadError`, `JSONDecodeError`, `ValueError` and `IndexError` as failed
  calls instead of allowing them to stop the provider worker. Inputs, prompts and model
  settings were unchanged.
- The pre-resume config, execution details and checkpoint fingerprints are saved
  separately at
  `gs://phonics-benchmark/phonics/resume-audits/sw-tusome-clusters-20260906-remaining/20260910-call-errors-r1/`.
