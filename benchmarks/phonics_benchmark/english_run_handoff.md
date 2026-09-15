# English UFLI cloud run

**Latest progress check: 15 September 2026, 12:07 UTC / 13:07 BST.** The 8 GiB resume is
running with no task retries. It has **16,325 / 276,750 saved calls (5.90%)**, up 4,505
since the resume: 16,324 successes and the same one recorded Gemini error. GPT-4.1 Mini
and Nano are complete. Kimi K2.6 remains at 25 saved results with no checkpoint since
the restart.

**Latest action, 15 September 2026:** memory was increased from **4 to 8 GiB** and
English was resumed as `phonics-run-remaining-cntb9`, using the existing source config
and image. The pre-resume checkpoint contained **11,820 / 276,750 saved calls (4.27%)**,
including one recorded API error. Current progress must be read from this new execution
and its live results.

Prepared on 14 September 2026 from main commit
`4d5ac3793e3ea76aac90cca40c5ded30df1079fb`, including the existing local cloud
registry-pinning and response-error handling changes.

## Deployment

- Project: `fab-qaf-decodability`; region: `europe-west1`.
- Job: `phonics-run-remaining`.
- Image:
  `europe-west1-docker.pkg.dev/fab-qaf-decodability/fab-benchmarks/phonics-benchmark@sha256:a14d7236989e0b6d2703d8e26b9a44f5fc3eca61b0906ca873a2c3e5a8af9891`.
- Build: `bd8b9748-e1cb-462c-9f6f-dc0bd792fb72`.
- Initial resource settings: 2 CPUs, 4 GiB memory, 168-hour timeout per attempt, two
  retries, one task.
- The image-only update preserved credentials and execution settings. The existing
  Kiswahili execution `phonics-run-remaining-xtxwg` remains on its original
  `sha256:adf27af45d14ceea65b2b493f8d4fac3c0fd1ec53e41d2995e18104b3e9491c0` image.
- Validation: 242 local tests passed, followed by successful packaged English schwa
  prompt, GPC inventory, Kiswahili prompt, pinned-registry, and response-error checks in
  Cloud Build.

## Run selection

- Run ID: `en-ufli-schwa-20260914`.
- Cloud source config:
  `gs://phonics-benchmark/phonics/run-configs/en-ufli-schwa-20260914.json`.
- Local cloud config: `configs/en-ufli-schwa-20260914.json`.
- Prompt: `decodability_evaluation_schwa_rule`.
- Items: all 5,535 candidates, representing 45 texts and 123 student profiles.
- Dataset: `gs://phonics-benchmark/phonics/datasets/en-ufli-schwa-20260914.jsonl`.
- Dataset SHA-256: `89f70af31e1ee97f66f34771ac5ba4c9305cecebcff84bcce2a25d6846496a7e`.
- Registry: `gs://phonics-benchmark/phonics/model-configs/en-ufli-schwa-20260914.yaml`.
- Registry SHA-256: `c5a6222e6bdb2b044ebedad5f7cecbb50f322f1afcacb54aba5438f3cea0bc68`.
- Results: `gs://phonics-benchmark/phonics/runs/en-ufli-schwa-20260914/`.
- Execution settings: seven provider workers, checkpoints every five items, 30,000
  output tokens.

The 50 selected models are the entire 44-model remaining Kiswahili selection plus six
models from the original ten-model run that occur as active entries in
`~/Downloads/list_models.yaml`: `claude-haiku-4-5-20251001`, `gemma-4-31b-it`,
`gemini-3-flash-preview`, `gemini-3.1-pro-preview`, `ministral-3-8b-2512`, and
`mistral-small-2603`. The source file contains 60 active entries and has SHA-256
`b9c54a7f1507f5324ff043d5d9a104b77c9488c093ad5d86cb48c35013382687`. The full run plans
276,750 model calls.

The registry is a byte-identical copy of the Kiswahili remaining-v2 registry, retaining
its Kimi K2.5 and Qwen 3 235B routing fixes. All 50 names registered successfully, and
every English item rendered successfully before upload.

## Execution status

All 50 models passed the one-item English cloud smoke check. The first execution,
`phonics-run-remaining-hqfjt`, saved 49 successful calls and one upstream DeepInfra
rate-limit error for `qwen3.6-27b`. The targeted retry execution
`phonics-run-remaining-gcr74` succeeded, preserving the other 49 responses. The original
registry and provider routing were retained. Smoke results use the separate
`en-ufli-schwa-20260914-smoke/` prefix.

The full English run was launched as `phonics-run-remaining-wwglp` with the 50-model
cloud config and the deployed image above. Startup was verified against the uploaded run
snapshot: the exact 50-model selection, schwa prompt, dataset SHA-256, registry SHA-256,
results prefix, seven workers, and five-item checkpoints all match. The first checkpoint
saved five successful GPT-4.1 Mini responses. The original Kiswahili execution was still
running at this check.

Resuming uses the existing immutable source config:

```bash
gcloud run jobs execute phonics-run-remaining \
  --project=fab-qaf-decodability \
  --region=europe-west1 \
  --args=run-cloud,--config,gs://phonics-benchmark/phonics/run-configs/en-ufli-schwa-20260914.json \
  --async
```

Before resuming, verify that no other English execution is writing to the same results
prefix. Normal resumes preserve cached calls, including recorded errors. Use a separate
retry-errors pass only after every planned call is cached.

## Progress audit, 15 September 2026

Cloud Run task `phonics-run-remaining-wwglp-task0` is **Failed**, with
`The configured memory limit was reached` as the final reason and two task retries.
Completion: 14 September at 16:41:17 UTC (execution completion 16:41:19 UTC). The
subsequent executions on the job were the separate Kiswahili retry and resume; no later
English execution was found.

All eight response files were downloaded and counted. They contain **11,820 saved calls
/ 276,750 planned (4.27%)**: **11,819 successes and one error**. One model is complete,
seven are partial, and 42 have no saved results. There are 264,930 missing calls.

| Model              | Saved / 5,535 | Successful calls | Recorded errors |
| ------------------ | ------------: | ---------------: | --------------: |
| `gpt-4.1-mini`     |         5,535 |            5,535 |               0 |
| `gpt-4.1-nano`     |         3,120 |            3,120 |               0 |
| `claude-fable-5`   |         1,165 |            1,165 |               0 |
| `gemini-3.5-flash` |           885 |              884 |               1 |
| `fw-gpt-oss-120b`  |           750 |              750 |               0 |
| `grok-4.5`         |           185 |              185 |               0 |
| `deepseek-v4-pro`  |           155 |              155 |               0 |
| `kimi-k2.6`        |            25 |               25 |               0 |

The sole recorded error is a Gemini 3.5 Flash `ReadError`, connection reset by peer, on
item `ufli-en-l062_EN-UFLI-1G-L110`. This is distinct from the task's memory-limit
failure. The downloaded files contain unique result IDs and unique benchmark item IDs
within each model.

The checkpoints remain available for a normal resume, which will skip cached rows
including the one recorded error. Resolve the memory limit/usage issue before resuming.
No job settings, image, source config, or cloud results were changed by this progress
check.

Local audit: `/private/tmp/english-progress-20260915-3j94n3dl/summary.json`.

## Memory increase and normal resume, 15 September 2026

After the original execution reached its memory limit, the user requested at least
double the memory. The `phonics-run-remaining` job was updated from 4 to 8 GiB,
retaining 2 CPUs, seven provider workers, the 168-hour timeout, two task retries, and
the existing image. No image rebuild was required.

Execution: `phonics-run-remaining-cntb9`.
https://console.cloud.google.com/run/jobs/executions/details/europe-west1/phonics-run-remaining-cntb9?project=791041208163

It uses the original immutable source config and writes to the existing English results
prefix. Normal resume skips saved calls, including the one recorded Gemini error, which
remains for a later error-recovery pass. At the user's request, no additional local
cache-skip simulation or preflight was run.

The pre-resume runtime snapshot and all eight response files were backed up to:
`gs://phonics-benchmark/phonics/resume-audits/en-ufli-schwa-20260914-memory8-20260915-4vu5c2e/baseline/`.
The parent audit prefix stores the object manifest and `receipt.json`. Local audit:
`/private/tmp/english-memory-resume-4vu5_c2e/`.

## Progress after the 8 GiB resume, 15 September at 12:07 UTC

Execution `phonics-run-remaining-cntb9` remains Running with no reported task retries.
The latest downloaded snapshot contains 16,325 saved calls (5.90%): 16,324 successes and
the pre-existing Gemini connection-reset error. There are two complete models, seven
partial models and 41 with no saved results.

| Model                          | Saved / 5,535 | Added since resume |
| ------------------------------ | ------------: | -----------------: |
| `gpt-4.1-mini`                 |         5,535 |                  0 |
| `gpt-4.1-nano`                 |         5,535 |              2,415 |
| `claude-fable-5`               |         1,870 |                705 |
| `gemini-3.5-flash`             |         1,400 |                515 |
| `fw-gpt-oss-120b`              |         1,310 |                560 |
| `grok-4.5`                     |           320 |                135 |
| `deepseek-v4-pro`              |           245 |                 90 |
| `gpt-5-nano-2025-08-07-medium` |            85 |                 85 |
| `kimi-k2.6`                    |            25 |                  0 |

Kimi K2.6 has no new saved checkpoint since the restart; its latest response is still
dated 14 September at 16:14 UTC. Recent logs show an OpenAI-compatible SDK request
timeout at 11:16 UTC, but those log lines do not identify the model. Do not attribute
the timeout to Kimi without further evidence. Other partial models have newer
checkpoints, and the overall execution remains running.

Local audit: `/private/tmp/english-progress-8gi-n8e2b6b5/summary.json`. Counts are a
snapshot of files downloaded during an active run. This check did not change any cloud
settings or results.
