#!/usr/bin/env bash
# Launch a run config on the fab-bench Cloud Run job. fab-bench checks the config
# as the job will read it and uploads it once, named by its content, so launching
# the same config again resumes it.
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 RUN_CONFIG" >&2
  exit 2
fi

: "${PROJECT_ID:?Set PROJECT_ID to the project the job runs in}"

REGION="${REGION:-europe-west1}"
JOB_NAME="${JOB_NAME:-fab-bench}"
# The fab-bench command, for example "uvx --from <runner> fab-bench".
read -r -a fab_bench <<<"${FAB_BENCH:-fab-bench}"

upload_args=(upload-run-config "$1")
# Where run configs go; by default run-configs/ in the experiment's bucket.
if [[ -n "${RUN_CONFIGS_URI:-}" ]]; then
  upload_args+=(--dest "${RUN_CONFIGS_URI}")
fi
config_uri="$("${fab_bench[@]}" "${upload_args[@]}")"
echo "Run config: ${config_uri}"

execute_args=(
  gcloud run jobs execute "${JOB_NAME}"
  --project="${PROJECT_ID}"
  --region="${REGION}"
  --args="run-cloud,--config,${config_uri}"
)
if [[ "${WAIT:-false}" == true ]]; then
  execute_args+=(--wait)
fi

"${execute_args[@]}"
