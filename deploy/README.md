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
Artifact Registry. Dirty worktrees are allowed. `deploy-job.sh` is idempotent: it
creates the job if needed or updates the existing job to use the selected image.
Building a new image under an existing tag is not enough by itself; run `deploy-job.sh`
again so the job is updated.

The job runs one task, has a 24-hour timeout by default, and retries a failed task
twice. Deploying the job does not start a benchmark run.

## Launch or resume a run

Start from a normal local JSON config and its completed items JSONL:

```bash
deploy/launch-run.sh path/to/config.json path/to/items.jsonl my-run-id
```

For a new run, the script:

1. Uploads the items to `${RESULTS_BUCKET}/phonics/datasets/my-run-id.jsonl`.
1. Creates a cloud version of the config with that items URI, the requested run ID, and
   `${RESULTS_BUCKET}/phonics/runs/my-run-id/` as `results_dest`.
1. Uploads the config to `${RESULTS_BUCKET}/phonics/runs/my-run-id/config.json`.
1. Executes the existing Cloud Run job with the equivalent of:

```bash
phonics-benchmark run-cloud \
  --config "${RESULTS_BUCKET}/phonics/runs/my-run-id/config.json"
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
└── runs/
    └── my-run-id/
        ├── config.json
        ├── log.json
        └── responses/
```

If the run's remote `config.json` already exists, `launch-run.sh` leaves it unchanged
and starts a resume. The supplied local config and items do not replace an existing run.
Use a new run ID for changed inputs or benchmark settings.

If the config was removed but the run's items object remains, the launcher reuses that
object only when it is byte-for-byte identical to the supplied items file. A mismatch
fails before launching; use a new run ID or explicitly remove the old run objects. The
container downloads `custom_models.yaml` into its writable `/workspace/inputs`
directory, so the installed virtual environment can remain read-only.

Fetch one model file after the run:

```bash
uv run --frozen --package phonics-benchmark phonics-benchmark fetch-results \
  --src "${RESULTS_BUCKET}/phonics/runs/my-run-id/" \
  --run-dir ./downloaded/my-run-id \
  --model gemini-3.1-pro-preview
```

You can also use `gcloud storage cp` directly against the object beneath the run's
`responses/` directory.
