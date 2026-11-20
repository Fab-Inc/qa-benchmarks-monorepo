#!/usr/bin/env bash
# Build the shared fab-bench image with Cloud Build and push it to Artifact
# Registry, tagged with the runner's git commit. Run it from a checkout of the
# runner; it builds from the runner package's folder.
set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID to the project whose Artifact Registry holds the image}"

REGION="${REGION:-europe-west1}"
ARTIFACT_REPOSITORY="${ARTIFACT_REPOSITORY:-fab-benchmarks}"
IMAGE_NAME="${IMAGE_NAME:-fab-bench}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# The tag names the commit, so the image must hold exactly that commit.
if [[ -n "$(git status --porcelain --untracked-files=normal -- .)" ]]; then
  echo "The runner package has uncommitted changes; commit them before building." >&2
  exit 1
fi
RUNNER_COMMIT="$(git rev-parse HEAD)"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPOSITORY}/${IMAGE_NAME}:${RUNNER_COMMIT:0:12}"

gcloud builds submit \
  --project="${PROJECT_ID}" \
  --config=deploy/cloudbuild.yaml \
  --substitutions="_IMAGE_URI=${IMAGE_URI},_RUNNER_COMMIT=${RUNNER_COMMIT}" \
  .

echo "Built and pushed ${IMAGE_URI}"
echo "Deploy it with: IMAGE=${IMAGE_URI} deploy/deploy-job.sh"
