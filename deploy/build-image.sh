#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID to the Google Cloud project ID}"

REGION="${REGION:-europe-west1}"
ARTIFACT_REPOSITORY="${ARTIFACT_REPOSITORY:-fab-benchmarks}"
IMAGE_NAME="${IMAGE_NAME:-phonics-benchmark}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}"

gcloud builds submit \
  --project="${PROJECT_ID}" \
  --tag="${IMAGE_URI}" \
  .

echo "Built and pushed ${IMAGE_URI}"
