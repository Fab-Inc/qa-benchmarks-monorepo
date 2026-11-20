#!/usr/bin/env bash
# Create or update the Cloud Run job that runs the shared fab-bench image. One job
# per project serves every benchmark and experiment there; each launch passes its
# own run config.
set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID to the project the job runs in}"
: "${IMAGE:?Set IMAGE to the image URI that build-image.sh printed}"
: "${SERVICE_ACCOUNT:?Set SERVICE_ACCOUNT to the Cloud Run job service account}"

REGION="${REGION:-europe-west1}"
JOB_NAME="${JOB_NAME:-fab-bench}"
CPU="${CPU:-1}"
MEMORY="${MEMORY:-2Gi}"
TASK_TIMEOUT="${TASK_TIMEOUT:-48h}"
MAX_RETRIES="${MAX_RETRIES:-2}"

# FAB_BENCH_IMAGE lets every run record the image it ran in. A retried task
# reruns the same config, which resumes from what was uploaded.
deploy_args=(
  gcloud run jobs deploy "${JOB_NAME}"
  --project="${PROJECT_ID}"
  --region="${REGION}"
  --image="${IMAGE}"
  --service-account="${SERVICE_ACCOUNT}"
  --cpu="${CPU}"
  --memory="${MEMORY}"
  --task-timeout="${TASK_TIMEOUT}"
  --max-retries="${MAX_RETRIES}"
  --tasks=1
  --parallelism=1
  --set-env-vars="FAB_BENCH_IMAGE=${IMAGE}"
  --args=run-cloud,--help
)

# Provider API keys, for example OPENAI_API_KEY=openai-api-key:latest,...
if [[ -n "${SECRET_MAPPINGS:-}" ]]; then
  deploy_args+=(--set-secrets="${SECRET_MAPPINGS}")
fi

"${deploy_args[@]}"
