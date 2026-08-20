#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID to the Google Cloud project ID}"

REGION="${REGION:-europe-west1}"
ARTIFACT_REPOSITORY="${ARTIFACT_REPOSITORY:-fab-benchmarks}"
IMAGE_NAME="${IMAGE_NAME:-phonics-benchmark}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}"
CODE_GIT_HASH="$(git rev-parse --short HEAD)"
CODE_GIT_DIRTY=false
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  CODE_GIT_DIRTY=true
fi

gcloud builds submit \
  --project="${PROJECT_ID}" \
  --config=deploy/cloudbuild.yaml \
  --substitutions="_CODE_GIT_DIRTY=${CODE_GIT_DIRTY},_CODE_GIT_HASH=${CODE_GIT_HASH},_IMAGE_URI=${IMAGE_URI}" \
  .

echo "Built and pushed ${IMAGE_URI} from git revision ${CODE_GIT_HASH} (dirty=${CODE_GIT_DIRTY})"
