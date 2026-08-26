#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 LOCAL_CONFIG LOCAL_ITEMS RUN_ID" >&2
  exit 2
fi

: "${PROJECT_ID:?Set PROJECT_ID to the Google Cloud project ID}"
: "${RESULTS_BUCKET:?Set RESULTS_BUCKET to a gs:// bucket URI}"

CONFIG_PATH="$1"
ITEMS_PATH="$2"
RUN_ID="$3"
REGION="${REGION:-europe-west1}"
JOB_NAME="${JOB_NAME:-phonics-run}"
BUCKET_URI="${RESULTS_BUCKET%/}"
ITEMS_URI="${BUCKET_URI}/phonics/datasets/${RUN_ID}.jsonl"
RUN_CONFIG_URI="${BUCKET_URI}/phonics/run-configs/${RUN_ID}.json"
RUN_PREFIX="${BUCKET_URI}/phonics/runs/${RUN_ID}/"

command -v jq >/dev/null || {
  echo "jq is required to prepare the cloud config" >&2
  exit 1
}

if gcloud storage ls "${RUN_CONFIG_URI}" >/dev/null 2>&1; then
  echo "Remote run config already exists; executing it as a resume: ${RUN_CONFIG_URI}"
else
  prepared_config="$(mktemp "${TMPDIR:-/tmp}/phonics-cloud-config.XXXXXX")"
  existing_items="$(mktemp "${TMPDIR:-/tmp}/phonics-cloud-items.XXXXXX")"
  trap 'rm -f "${prepared_config}" "${existing_items}"' EXIT

  jq \
    --arg run_id "${RUN_ID}" \
    --arg items_uri "${ITEMS_URI}" \
    --arg results_dest "${RUN_PREFIX}" \
    '.run_id = $run_id
      | .items_path = $items_uri
      | .results_dest = $results_dest
      | del(.output_dir)' \
    "${CONFIG_PATH}" >"${prepared_config}"

  if gcloud storage ls "${ITEMS_URI}" >/dev/null 2>&1; then
    gcloud storage cp "${ITEMS_URI}" "${existing_items}" >/dev/null
    if cmp -s "${ITEMS_PATH}" "${existing_items}"; then
      echo "Reusing the identical existing run items object: ${ITEMS_URI}"
    else
      echo "Remote items differ for reused RUN_ID '${RUN_ID}': ${ITEMS_URI}" >&2
      echo "Choose a new RUN_ID or explicitly remove the old run objects." >&2
      exit 1
    fi
  else
    gcloud storage cp \
      --if-generation-match=0 \
      "${ITEMS_PATH}" \
      "${ITEMS_URI}"
  fi
  gcloud storage cp \
    --if-generation-match=0 \
    "${prepared_config}" \
    "${RUN_CONFIG_URI}"
fi

gcloud run jobs execute "${JOB_NAME}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --args="run-cloud,--config,${RUN_CONFIG_URI}" \
  --wait
