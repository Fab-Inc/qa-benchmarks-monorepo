#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID to the Google Cloud project ID}"
: "${SERVICE_ACCOUNT:?Set SERVICE_ACCOUNT to the Cloud Run job service account}"

REGION="${REGION:-europe-west1}"
ARTIFACT_REPOSITORY="${ARTIFACT_REPOSITORY:-fab-benchmarks}"
IMAGE_NAME="${IMAGE_NAME:-phonics-benchmark}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
JOB_NAME="${JOB_NAME:-phonics-run}"
TASK_TIMEOUT="${TASK_TIMEOUT:-24h}"
MAX_RETRIES="${MAX_RETRIES:-2}"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}"

deploy_args=(
  gcloud run jobs deploy "${JOB_NAME}"
  --project="${PROJECT_ID}"
  --region="${REGION}"
  --image="${IMAGE_URI}"
  --service-account="${SERVICE_ACCOUNT}"
  --task-timeout="${TASK_TIMEOUT}"
  --max-retries="${MAX_RETRIES}"
  --tasks=1
  --args=run-cloud,--help
)

if [[ -n "${SECRET_MAPPINGS:-}" ]]; then
  deploy_args+=(--set-secrets="${SECRET_MAPPINGS}")
fi

"${deploy_args[@]}"
