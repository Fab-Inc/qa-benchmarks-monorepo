# Kiswahili ten-model error recovery, 14 September 2026

**Current status, 15 September 2026:** the resume completed successfully and the
original results now contain **16,650 successful calls, zero recorded API errors**. The
history below describes the isolated retry and resume before the copy-back.

The ten-model run had 16,170 saved calls: 16,166 successes and four Claude Sonnet 4.6
API overload errors. Gemini 3.1 Pro had 480 missing calls, which require a separate
normal resume and are outside this error-only retry.

Execution: `phonics-run-remaining-k6xfx` (**Succeeded**, completed 14:47:45 UTC).

Cloud Run reported **4 executed, 1,661 skipped, 0 errors**. The recovery copy contains
**16,170 successful calls and zero saved API errors**; Gemini 3.1 Pro still has 480
missing calls.

- Original results, preserved:
  `gs://phonics-benchmark/phonics/runs/sw-tusome-clusters-20260905/`
- Separate recovery copy:
  `gs://phonics-benchmark/phonics/runs/sw-tusome-clusters-20260905-retry-errors-20260914T144345Z/`
- Retry source config:
  `gs://phonics-benchmark/phonics/run-configs/sw-tusome-clusters-20260905-retry-errors-20260914T144345Z.json`
- Audit records:
  `gs://phonics-benchmark/phonics/retry-audits/sw-tusome-clusters-20260905-retry-errors-20260914T144345Z/`
- Local working audit: `/private/tmp/kiswahili-ten-retry-mp1an1n6`

All 11 original artifacts (ten response files and the runtime config snapshot) were
downloaded and their MD5 checksums checked against GCS. They were uploaded to the new
recovery prefix using `--if-generation-match=0`. All 11 copied objects matched the
originals by path, size, MD5 and CRC32C before launch. The original snapshot and
original object generations/checksums were also saved under the separate audit prefix.

The retry config preserves the original run ID, dataset, cluster prompt, 30,000-token
limit and pinned model registry. It selects only `claude-sonnet-4-6`, sets
`retry_errors: true` and `force: false`, and directs writes exclusively to the recovery
prefix. The other nine model files remain available in that copy. Because the runtime
snapshot records only selected models, use the original snapshot saved in the audit
prefix for the full ten-model metadata.

A local simulation using the actual copied caches, registered model settings and
rendered dataset passed the resume identity and retry preflight checks. It called
exactly the four failed items and preserved all 16,166 successful rows across all ten
response files. Simulation results were kept separately and were never uploaded.

The execution uses the already deployed image
`europe-west1-docker.pkg.dev/fab-qaf-decodability/fab-benchmarks/phonics-benchmark@sha256:a14d7236989e0b6d2703d8e26b9a44f5fc3eca61b0906ca873a2c3e5a8af9891`
through `phonics-run-remaining`, with 2 CPUs, 4 GiB, a seven-day timeout and two task
retries. Only this execution's arguments were overridden; the job template and the
English/additional Kiswahili executions were not modified.

Final verification passed at 14:54 UTC:

- All 11 original objects retained the same generation, metadata generation, size, MD5
  and CRC32C.
- All 16,166 previously successful rows compare equal in the recovery copy.
- The same 16,170 result IDs remain, with no duplicates or additions.
- Exactly four previously failed rows changed, and all four are now successful.
- The runtime snapshot confirms the separate destination, retry-only mode,
  `force: false`, and the original dataset and registry hashes.

The before/after manifests, `verification.json`, and `receipt.json` are stored in the
audit prefix above. The recovered data remains separate; there is no automatic copy-back
to the original prefix.

## Normal resume for the 480 missing Gemini calls

On 14 September, execution `phonics-run-remaining-spskl` was submitted to resume the
same recovery prefix. The original results prefix remains untouched.

- Source config:
  `gs://phonics-benchmark/phonics/run-configs/sw-tusome-clusters-20260905-resume-20260914T145809Z.json`
- Execution:
  https://console.cloud.google.com/run/jobs/executions/details/europe-west1/phonics-run-remaining-spskl?project=791041208163
- Results:
  `gs://phonics-benchmark/phonics/runs/sw-tusome-clusters-20260905-retry-errors-20260914T144345Z/`
- Resume audit:
  `gs://phonics-benchmark/phonics/resume-audits/sw-tusome-clusters-20260905-resume-20260914T145809Z/`
- Complete pre-resume backup:
  `gs://phonics-benchmark/phonics/resume-audits/sw-tusome-clusters-20260905-resume-20260914T145809Z/baseline/`
- Local audit: `/private/tmp/kiswahili-gemini-resume-l0tnjy2d`

The resume config restores the full original ten-model list and sets
`retry_errors: false`, `force: false`. Before launch, all 16,170 saved call IDs matched
the original rendered prompts and pinned model configurations. Exactly 480 Gemini 3.1
Pro calls were missing; no other calls were pending. A local simulation executed those
480 and skipped all 16,170 existing successes while preserving every saved row,
including the four repaired Sonnet responses. Simulation output was not uploaded.

All 12 recovery artifacts were backed up with create-only GCS uploads and verified
against their original paths, sizes, MD5 and CRC32C. The execution uses the existing
image and job resources described above, with no deployment change. Its image digest and
config argument were checked after submission. Startup and first-checkpoint verification
passed. The task started at 15:06:29 UTC and remains running. At 15:16 UTC, Gemini had
**1,190 / 1,665** saved calls: the 1,185 existing successes plus **five new successful
calls**, leaving **475 missing**. All 1,185 existing Gemini rows were unchanged. The
other nine response objects and all 11 objects in the original run prefix retained their
earlier generations and checksums in the startup audit.

The runtime snapshot restored all ten model configurations and confirmed the expected
recovery destination, original prompt, dataset and registry hashes,
`retry_errors: false`, and `force: false`. `checkpoint-verification.json` and
`receipt.json` in the resume audit prefix record this first checkpoint; later progress
must be read from the live response files.

## Completed recovery copied to the original results, 15 September 2026

The Gemini resume `phonics-run-remaining-spskl` completed successfully at 23:40:35 UTC
on 14 September (00:40:35 BST on 15 September). Its final output added all 480 missing
Gemini results successfully. All ten models now have 1,665 successful calls; there are
no recorded API errors.

After the user's requested next-day check, the repaired Sonnet and completed Gemini
response files were copied to the original results prefix:
`gs://phonics-benchmark/phonics/runs/sw-tusome-clusters-20260905/`. The recovery
`log.json` was added for provenance. The original config snapshot retains the original
destination and the same verified ten-model settings. The complete recovery copy remains
available at
`gs://phonics-benchmark/phonics/runs/sw-tusome-clusters-20260905-retry-errors-20260914T144345Z/`.

Before any original object was updated, all 11 original artifacts were backed up and
verified by relative path, size, MD5 and CRC32C. The two response writes required the
exact original generations observed during the audit; adding the log required generation
zero. No automated follow-up was created.

Verification after publication passed at 09:14 UTC / 10:14 BST:

- All ten original response files match the completed recovery files by size, MD5 and
  CRC32C: 16,650 rows, all successful API calls.
- All 16,166 previously successful rows compare equal.
- Exactly four original error rows were repaired, and exactly 480 Gemini rows were
  added, with no duplicate result IDs or missing benchmark items.
- The eight other response files and original config snapshot retain their original
  generations and checksums.
- The recovery provenance log matches the completed recovery log.

Audit and full original backup:
`gs://phonics-benchmark/phonics/publish-audits/sw-tusome-clusters-20260905-publish-20260915T090538Z/`.
The audit contains `verification-before-publish.json`,
`verification-after-publish.json`, before/after object manifests, `backup-objects.json`,
the recovered runtime snapshot (`config.json`), and `receipt.json`. The full prior state
is under `original-backup/`. Local audit:
`/private/tmp/kiswahili-check-promote-no1exj51`.
