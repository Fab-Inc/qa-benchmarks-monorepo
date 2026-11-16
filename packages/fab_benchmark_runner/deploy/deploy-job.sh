#!/usr/bin/env bash
# Create or update the benchmark's Cloud Run job from the pushed image.
# From fab-benchmark-runner's deploy/ folder.
set -euo pipefail

# SET: the benchmark's name. It names the image, the Cloud Run job and the GCS
# prefix; each can also be overridden on its own.
BENCHMARK="${BENCHMARK:-my-benchmark}"

: "${PROJECT_ID:?Set PROJECT_ID to the Google Cloud project ID}"
: "${SERVICE_ACCOUNT:?Set SERVICE_ACCOUNT to the Cloud Run job service account}"

REGION="${REGION:-europe-west1}"
ARTIFACT_REPOSITORY="${ARTIFACT_REPOSITORY:-fab-benchmarks}"
IMAGE_NAME="${IMAGE_NAME:-${BENCHMARK}}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
JOB_NAME="${JOB_NAME:-${BENCHMARK}-run}"
CPU="${CPU:-1}"
MEMORY="${MEMORY:-2Gi}"
TASK_TIMEOUT="${TASK_TIMEOUT:-48h}"
MAX_RETRIES="${MAX_RETRIES:-2}"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}"

deploy_args=(
  gcloud run jobs deploy "${JOB_NAME}"
  --project="${PROJECT_ID}"
  --region="${REGION}"
  --image="${IMAGE_URI}"
  --service-account="${SERVICE_ACCOUNT}"
  --cpu="${CPU}"
  --memory="${MEMORY}"
  --task-timeout="${TASK_TIMEOUT}"
  --max-retries="${MAX_RETRIES}"
  --tasks=1
  --parallelism=1
  --args=run-cloud,--help
)

# Provider API keys, for example OPENAI_API_KEY=openai-api-key:latest,...
if [[ -n "${SECRET_MAPPINGS:-}" ]]; then
  deploy_args+=(--set-secrets="${SECRET_MAPPINGS}")
fi

"${deploy_args[@]}"
