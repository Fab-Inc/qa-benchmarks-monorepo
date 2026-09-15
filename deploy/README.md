# Phonics benchmark Cloud Run Job

The `phonics.Dockerfile` image definition and scripts in this directory deploy one whole
phonics model run as one Cloud Run Job task. The container receives only a GCS config
URI; provider credentials come from Secret Manager environment variables and GCS
authentication uses the job's service account through Application Default Credentials.

## How the pieces fit together

There are two different configurations:

- The **Cloud Run job configuration** selects the Docker image, service account,
  secrets, timeout, and retry policy. `deploy/deploy-job.sh` creates or updates it.
- The **benchmark run configuration** selects the language, models, items, and other
  benchmark settings. `deploy/launch-run.sh` prepares it, uploads it to GCS, and passes
  its GCS URI to the job.

The usual workflow is:

1. Build the Docker image and deploy the job when the code or dependencies change.
1. Launch the already-deployed job for each benchmark run.
1. Launch the same run ID again when a stopped or failed run should resume.

New items, model selections, or benchmark settings do not require a new image. They
normally require only a new run ID and another call to `launch-run.sh`.

## One-time Google Cloud setup

Set the deployment values:

```bash
export PROJECT_ID="your-project"
export REGION="europe-west1"
export ARTIFACT_REPOSITORY="fab-benchmarks"
export RESULTS_BUCKET="gs://your-phonics-results-bucket"
export SERVICE_ACCOUNT="phonics-runner@${PROJECT_ID}.iam.gserviceaccount.com"
```

Use a current Google Cloud CLI. Older releases can send unsupported fields when
executing a job with argument overrides:

```bash
gcloud components update
gcloud version
```

Enable the APIs and create the Docker repository, bucket, and job identity. Run only the
resource-creation commands for resources that do not already exist:

```bash
gcloud services enable \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  secretmanager.googleapis.com \
  storage.googleapis.com \
  --project="${PROJECT_ID}"

gcloud artifacts repositories create "${ARTIFACT_REPOSITORY}" \
  --project="${PROJECT_ID}" \
  --location="${REGION}" \
  --repository-format=docker

gcloud storage buckets create "${RESULTS_BUCKET}" \
  --project="${PROJECT_ID}" \
  --location="${REGION}" \
  --uniform-bucket-level-access

gcloud iam service-accounts create phonics-runner \
  --project="${PROJECT_ID}" \
  --display-name="Phonics benchmark Cloud Run job"

gcloud storage buckets add-iam-policy-binding "${RESULTS_BUCKET}" \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/storage.objectUser"
```

The storage role must apply to objects inside the bucket. A project-level conditional
binding that matches only the exact bucket resource name does not authorize object reads
or writes; use the bucket-level binding above or ensure the condition also matches
`/objects/` descendants.

Cloud Build also needs permission to read its uploaded source archive and push the
resulting image. This is commonly granted automatically, but projects that disable
automatic service-account grants may need it explicitly:

```bash
export CLOUD_BUILD_SERVICE_ACCOUNT="$(
  gcloud builds get-default-service-account --project="${PROJECT_ID}"
)"

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${CLOUD_BUILD_SERVICE_ACCOUNT}" \
  --role="roles/cloudbuild.builds.builder" \
  --condition=None
```

`--condition=None` is explicit because `gcloud` requires a choice when the project IAM
policy already contains conditional bindings.

Create one Secret Manager secret for each provider key that the selected models need.
Grant `roles/secretmanager.secretAccessor` on those secrets to `SERVICE_ACCOUNT`. For
example:

```bash
printf '%s' "${GEMINI_API_KEY}" | gcloud secrets create gemini-api-key \
  --project="${PROJECT_ID}" \
  --replication-policy=automatic \
  --data-file=-

gcloud secrets add-iam-policy-binding gemini-api-key \
  --project="${PROJECT_ID}" \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/secretmanager.secretAccessor"
```

The deploying principal also needs Cloud Run Developer, Service Account User for the job
identity, and access to the image. See Google's
[Cloud Run Jobs deployment roles](https://cloud.google.com/run/docs/create-jobs#required-roles).

## Build and deploy

From the repository root, choose an image tag, build the image, and create or update the
job:

```bash
export IMAGE_TAG="$(git rev-parse --short HEAD)"
deploy/build-image.sh

export SECRET_MAPPINGS="GEMINI_API_KEY=gemini-api-key:latest"
deploy/deploy-job.sh
```

Add other provider keys to `SECRET_MAPPINGS` as a comma-separated list. For example,
`OPENAI_API_KEY=openai-api-key:latest`.

`build-image.sh` uploads the current source tree to Cloud Build, records the current
HEAD revision and whether the worktree is dirty in the image, and pushes the image to
Artifact Registry. The `.gcloudignore` and `.dockerignore` allowlists include package
source and build definitions, excluding local datasets, credentials, and unrelated
documents. Cloud Build checks the packaged cluster prompt before publishing the image.
Dirty worktrees are allowed. `deploy-job.sh` is idempotent: it creates the job if needed
or updates the existing job to use the selected image. Building a new image under an
existing tag is not enough by itself; run `deploy-job.sh` again so the job is updated.

The job runs one task with 1 CPU, 2 GiB of memory, and a 48-hour timeout by default, and
retries a failed task twice. Override `CPU`, `MEMORY`, `TASK_TIMEOUT`, or `MAX_RETRIES`
when needed. Deploying the job does not start a benchmark run.

## Launch or resume a run

### Resuming with a rebuilt image

Update the existing job with the new image digest using
`gcloud run jobs update JOB --image=IMAGE_DIGEST`, retaining its resources, secrets and
run arguments. A running execution and its automatic retries keep their old image; start
a new execution to use the fix. Stop any old execution writing to the same results
prefix and confirm it is no longer running before launching the replacement.

Reuse the existing immutable source config, dataset, pinned model registry and run ID,
without `force`. The runner resumes saved calls across image changes; only
uncheckpointed work may repeat. Normal resumes also skip recorded error rows; use a
separate `retry_errors` pass after the selected models have cached every item.

As a temporary safeguard for FDLLM response-formatting failures, the evaluator boundary
records `ReadError`, `ValueError` (including `JSONDecodeError`) and `IndexError` as
failed item rows, preserving the concrete exception type and message. Later items and
models in that provider group continue. This does not introduce additional per-call
retries, change FDLLM, or catch setup/checkpoint failures. Replace the broad formatting
exceptions with typed response errors once FDLLM supports them. The image build verifies
these cases without making provider calls.

### Launch command

Start from a normal local JSON config and its completed items JSONL:

```bash
deploy/launch-run.sh path/to/config.json path/to/items.jsonl my-run-id
```

For a new run, the script:

1. Uploads the items to `${RESULTS_BUCKET}/phonics/datasets/my-run-id.jsonl`.
1. Creates a cloud version of the config with that items URI, the requested run ID, and
   `${RESULTS_BUCKET}/phonics/runs/my-run-id/` as `results_dest`.
1. Uploads that immutable source config to
   `${RESULTS_BUCKET}/phonics/run-configs/my-run-id.json`.
1. Executes the existing Cloud Run job with the equivalent of:

```bash
phonics-benchmark run-cloud \
  --config "${RESULTS_BUCKET}/phonics/run-configs/my-run-id.json"
```

Inside the container, `run-cloud` downloads the config and items into the writable
`/workspace/inputs` directory, writes results beneath `/workspace/outputs`, and syncs
checkpoints back to `results_dest`. Provider keys are supplied by Secret Manager and GCS
access comes from the job's service account.

After the run, the main GCS objects are arranged as follows:

```text
phonics/
├── datasets/
│   └── my-run-id.jsonl
├── run-configs/
│   └── my-run-id.json
└── runs/
    └── my-run-id/
        ├── config.json  # resolved run snapshot
        ├── log.json
        └── responses/
```

If the source config already exists under `run-configs/`, `launch-run.sh` leaves it
unchanged and starts a resume. The supplied local config and items do not replace an
existing run. Use a new run ID for changed inputs or benchmark settings. The resolved
`runs/my-run-id/config.json` is a run artifact and never replaces the source config.

If the source config was removed but the run's items object remains, the launcher reuses
that object only when it is byte-for-byte identical to the supplied items file. A
mismatch fails before launching; use a new run ID or explicitly remove the old run
objects. The container downloads `custom_models.yaml` into its writable
`/workspace/inputs` directory, so the installed virtual environment can remain
read-only.

### Pin the model registry for a cloud run

To preserve provider settings across restarts, upload a copy of `custom_models.yaml`
under a unique GCS object name and add both fields to the source config:

```json
{
  "custom_models_uri": "gs://your-bucket/phonics/model-configs/my-run-id.yaml",
  "custom_models_sha256": "<64-character lowercase SHA-256 of the uploaded file>"
}
```

`run-cloud` downloads and verifies this registry before executing any model calls. It
does not refresh the public registry when a pinned URI is configured. The pinned
download is still required with `--skip-models-download`, since a fresh container has no
local copy. A checksum mismatch fails without replacing an existing local registry. Omit
both fields to retain the default public-registry refresh behavior.

The full Kiswahili cluster run is configured in
`benchmarks/phonics_benchmark/configs/sw-tusome-clusters-20260905-10models.json`. It
selects 1,665 items and the original ten-model selection, except that
`gemini-3-flash-preview` replaces the excluded `gemma-3-12b-it` (which repeatedly
returned empty responses in the smoke test). The replacement comes from the
user-provided expanded model list. The run uses the merged
`decodability_evaluation_clusters` prompt, four concurrent provider groups, and
checkpoints every five items per model. Its dataset and model registry are copied to
dedicated GCS objects. The original source config is retained unchanged; the revised
config uses the same, previously unstarted full-run ID and results prefix. Preparing
these objects and deploying the image do not execute the benchmark. Once ready to
launch, execute the uploaded source config:

```bash
gcloud run jobs execute phonics-run \
  --project=fab-qaf-decodability \
  --region=europe-west1 \
  --args=run-cloud,--config,gs://phonics-benchmark/phonics/run-configs/sw-tusome-clusters-20260905-10models.json
```

### Remaining models from the expanded Kiswahili list

`benchmarks/phonics_benchmark/configs/sw-tusome-clusters-20260906-models.yaml` preserves
the user-provided expanded list. Its 60 active entries overlap with six models in the
initial ten-model run. Excluding those six and the failing `gemma-3-12b-it` leaves 53
models (88,245 planned calls), configured in
`benchmarks/phonics_benchmark/configs/sw-tusome-clusters-20260906-remaining.json`.

The launch-ready source is
`benchmarks/phonics_benchmark/configs/sw-tusome-clusters-20260906-remaining-ready.json`.
It defers five paused Hugging Face endpoints and, at the user's request, the four models
whose smoke checks were pending: `qwen3.6-max-preview`, `qwen3.6-plus`, `qwen3.7-max`,
and `seed-1.6-flash`. This leaves 44 models and 73,260 planned calls. It retains the
pinned dataset, cluster prompt, and 30,000-token output limit from the initial run, with
a separate run ID and results prefix. The original 53-model config remains as a record
of the requested selection, not the launch-ready configuration.

The launch-ready registry is separately pinned under
`phonics/model-configs/sw-tusome-clusters-20260906-remaining-v2.yaml`. Only two model
definitions differ from the original registry: Kimi K2.5 uses OpenRouter because the
Moonshot account catalog no longer offers it, and Qwen 3 235B Thinking uses Alibaba
through OpenRouter because its DeepInfra endpoint disappeared. The named models are
unchanged; these routing overrides are recorded in the run metadata.

The remaining seven provider groups run concurrently with `model_workers: 7`; models and
items within each group remain sequential. Five-item checkpoints permit resuming after
interruption. Do not start overlapping executions against the same results prefix.

The separate `phonics-run-remaining` job uses the already-tested image digest, 2 CPUs, 4
GiB of memory, a 168-hour timeout per attempt, and two retries. It binds eight
version-pinned provider secrets: Anthropic, OpenAI, OpenRouter, Google, Fireworks,
xAI/Grok, Hugging Face, and Moonshot. The initial `phonics-run` job is unchanged.

After checking one response per included model (including separate Kimi and Qwen routing
smoke tests), launch or resume the remaining-model run with:

```bash
gcloud run jobs execute phonics-run-remaining \
  --project=fab-qaf-decodability \
  --region=europe-west1 \
  --args=run-cloud,--config,gs://phonics-benchmark/phonics/run-configs/sw-tusome-clusters-20260906-remaining-ready.json \
  --async
```

Fetch one model file after the run:

```bash
uv run --frozen --package phonics-benchmark phonics-benchmark fetch-results \
  --src "${RESULTS_BUCKET}/phonics/runs/my-run-id/" \
  --run-dir ./downloaded/my-run-id \
  --model gemini-3.1-pro-preview
```

You can also use `gcloud storage cp` directly against the object beneath the run's
`responses/` directory.
