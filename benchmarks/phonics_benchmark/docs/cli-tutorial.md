# CLI tutorial

These flows exercise config and direct CLI runs, cache reuse, provider-error retries,
and adding a model. Run them from the repository root. They use real provider APIs;
commands expected to make calls are marked below.

Provider keys must be available in the repository-root `.env` or the environment. The
commands use the existing `custom_models.yaml` except for the first config-driven run,
which deliberately tests the default anonymous download.

## Prepare an isolated one-item run

Create a temporary output directory and copy one completed benchmark item from the
contract fixture:

```bash
PHONICS_MANUAL_OUTPUT_DIR="$(mktemp -d /tmp/phonics-manual.XXXXXX)"
export PHONICS_MANUAL_OUTPUT_DIR

head -n 1 \
  benchmarks/phonics_benchmark/tests/fixtures/completed_benchmark_items.jsonl \
  > "$PHONICS_MANUAL_OUTPUT_DIR/items.jsonl"

cat > "$PHONICS_MANUAL_OUTPUT_DIR/config.json" <<EOF
{
  "run_id": "config-smoke",
  "language": "sw",
  "items_path": "$PHONICS_MANUAL_OUTPUT_DIR/items.jsonl",
  "output_dir": "$PHONICS_MANUAL_OUTPUT_DIR",
  "models": ["gemini-3.1-pro-preview"],
  "max_tokens": 1000,
  "chunk_size": 1,
  "model_workers": 1
}
EOF
```

The output directory is unique, so the executed/skipped counts below are predictable.

## Config-driven run

The temporary config plans one Gemini call. This first command also exercises the
default anonymous refresh of the repository-root `custom_models.yaml`:

```bash
uv run --frozen --package phonics-benchmark phonics-benchmark run-generation \
  --config "$PHONICS_MANUAL_OUTPUT_DIR/config.json"
```

It should report `Executed: 1`, `Skipped: 0`, and `Errors: 0`. Check that the model
honoured the output contract — the response should wrap its text in `<text>` tags, which
is what the report script extracts:

```bash
jq -r '.output' \
  "$PHONICS_MANUAL_OUTPUT_DIR"/config-smoke/responses/*.jsonl
```

A response without those tags is scored zero as a format failure, so seeing them here
confirms the prompt's output contract is landing before you spend on a full run.

Repeat without refreshing the model registry to verify cache reuse:

```bash
uv run --frozen --package phonics-benchmark phonics-benchmark run-generation \
  --config "$PHONICS_MANUAL_OUTPUT_DIR/config.json" \
  --skip-models-download
```

It should now report `Executed: 0`, `Skipped: 1`, and `Errors: 0`. With a complete
cache, retry mode executes only error rows:

```bash
uv run --frozen --package phonics-benchmark phonics-benchmark run-generation \
  --config "$PHONICS_MANUAL_OUTPUT_DIR/config.json" \
  --skip-models-download \
  --retry-errors
```

## Direct CLI run and cache reuse

The first command makes one Gemini call:

```bash
uv run --frozen --package phonics-benchmark phonics-benchmark run-generation \
  --run-id direct-smoke \
  --language sw \
  --model gemini-3.1-pro-preview \
  --items-path "$PHONICS_MANUAL_OUTPUT_DIR/items.jsonl" \
  --output-dir "$PHONICS_MANUAL_OUTPUT_DIR" \
  --max-tokens 1000 \
  --skip-models-download
```

Repeat the same command. It should report `Executed: 0`, `Skipped: 1`, and `Errors: 0`.

## Save and retry a provider error

This command makes one call with a deliberately invalid key. It should save a Google
provider error row and exit with status 1:

```bash
GEMINI_API_KEY=definitely-invalid \
uv run --frozen --package phonics-benchmark phonics-benchmark run-generation \
  --run-id retry-error-smoke \
  --language sw \
  --model gemini-3.1-pro-preview \
  --items-path "$PHONICS_MANUAL_OUTPUT_DIR/items.jsonl" \
  --output-dir "$PHONICS_MANUAL_OUTPUT_DIR" \
  --max-tokens 1000 \
  --skip-models-download \
  --force
```

Retry that exact error using the real key from `.env` or the environment:

```bash
uv run --frozen --package phonics-benchmark phonics-benchmark run-generation \
  --run-id retry-error-smoke \
  --language sw \
  --model gemini-3.1-pro-preview \
  --items-path "$PHONICS_MANUAL_OUTPUT_DIR/items.jsonl" \
  --output-dir "$PHONICS_MANUAL_OUTPUT_DIR" \
  --max-tokens 1000 \
  --skip-models-download \
  --retry-errors
```

It should report one executed call and no remaining errors.

## Add a model to an existing run

Start with one Gemini call:

```bash
uv run --frozen --package phonics-benchmark phonics-benchmark run-generation \
  --run-id add-model-smoke \
  --language sw \
  --model gemini-3.1-pro-preview \
  --items-path "$PHONICS_MANUAL_OUTPUT_DIR/items.jsonl" \
  --output-dir "$PHONICS_MANUAL_OUTPUT_DIR" \
  --max-tokens 1000 \
  --skip-models-download
```

Adding Claude directly in retry mode should fail before making calls because Claude has
no cached row:

```bash
uv run --frozen --package phonics-benchmark phonics-benchmark run-generation \
  --run-id add-model-smoke \
  --language sw \
  --model gemini-3.1-pro-preview \
  --model claude-sonnet-4-6 \
  --items-path "$PHONICS_MANUAL_OUTPUT_DIR/items.jsonl" \
  --output-dir "$PHONICS_MANUAL_OUTPUT_DIR" \
  --max-tokens 1000 \
  --model-workers 2 \
  --skip-models-download \
  --retry-errors
```

Run normally with both models:

```bash
uv run --frozen --package phonics-benchmark phonics-benchmark run-generation \
  --run-id add-model-smoke \
  --language sw \
  --model gemini-3.1-pro-preview \
  --model claude-sonnet-4-6 \
  --items-path "$PHONICS_MANUAL_OUTPUT_DIR/items.jsonl" \
  --output-dir "$PHONICS_MANUAL_OUTPUT_DIR" \
  --max-tokens 1000 \
  --model-workers 2 \
  --skip-models-download
```

It should skip the cached Gemini row and execute only Claude.

## Inspect a run

`scripts/inspect_run.py` loads `config.json`, `log.json`, sanitized model
configurations, every per-model JSONL file, a status summary, and error rows:

```bash
uv run --frozen --package phonics-benchmark python \
  benchmarks/phonics_benchmark/scripts/inspect_run.py \
  "$PHONICS_MANUAL_OUTPUT_DIR/add-model-smoke"
```

The script prints the number of result files and rows plus a per-model status summary.
Change the final path to inspect any other run directory.

## Score a run

`scripts/report_on_benchmark_run.py` extracts each generated text, runs it through the
decodability checker, and writes the per-model summary and breakdowns:

```bash
uv run --frozen --package phonics-benchmark python \
  benchmarks/phonics_benchmark/scripts/report_on_benchmark_run.py \
  --run-id add-model-smoke \
  --output-root "$PHONICS_MANUAL_OUTPUT_DIR" \
  --inspect-sample-size 1
```

The summary prints to stdout and everything is written to `<run-dir>/report/`. Read
`sampled_items_for_inspection.csv` to check by eye that the words flagged non-decodable
really are — the fastest way to catch a mis-wired student profile.

## GCS-synced local smoke test

With Application Default Credentials configured and a writable scratch bucket, add a run
prefix to the temporary config:

```bash
PHONICS_SCRATCH_DEST="gs://your-scratch-bucket/phonics/runs/config-smoke/"
jq --arg dest "$PHONICS_SCRATCH_DEST" \
  '.results_dest = $dest' \
  "$PHONICS_MANUAL_OUTPUT_DIR/config.json" \
  > "$PHONICS_MANUAL_OUTPUT_DIR/config-synced.json"

uv run --frozen --package phonics-benchmark phonics-benchmark run-generation \
  --config "$PHONICS_MANUAL_OUTPUT_DIR/config-synced.json" \
  --skip-models-download
```

Watch the run prefix while the model call is in progress. `config.json` appears before
the first call, and the model response file is replaced after every local checkpoint.
Running the same command again pulls those artifacts before applying the normal result
cache rules.

Fetch only that model's responses into a separate local directory:

```bash
uv run --frozen --package phonics-benchmark phonics-benchmark fetch-results \
  --src "$PHONICS_SCRATCH_DEST" \
  --run-dir "$PHONICS_MANUAL_OUTPUT_DIR/fetched" \
  --model gemini-3.1-pro-preview
```

## Run the container locally

The container reads its config and items from GCS and writes its resolved config and
results to a separate GCS run prefix. Prepare and upload those inputs:

```bash
GOOGLE_CLOUD_PROJECT="your-project-id"
PHONICS_SCRATCH_BUCKET="gs://your-scratch-bucket"
CONTAINER_RUN_ID="container-smoke"
CONTAINER_ITEMS_URI="${PHONICS_SCRATCH_BUCKET}/phonics/datasets/${CONTAINER_RUN_ID}.jsonl"
CONTAINER_CONFIG_URI="${PHONICS_SCRATCH_BUCKET}/phonics/run-configs/${CONTAINER_RUN_ID}.json"
CONTAINER_RESULTS_DEST="${PHONICS_SCRATCH_BUCKET}/phonics/runs/${CONTAINER_RUN_ID}/"

jq -n \
  --arg run_id "$CONTAINER_RUN_ID" \
  --arg items_uri "$CONTAINER_ITEMS_URI" \
  --arg results_dest "$CONTAINER_RESULTS_DEST" \
  '{
    run_id: $run_id,
    language: "sw",
    items_path: $items_uri,
    results_dest: $results_dest,
    models: ["gemini-3.1-pro-preview"],
    max_tokens: 1000,
    chunk_size: 1,
    model_workers: 1
  }' > "$PHONICS_MANUAL_OUTPUT_DIR/container-config.json"

gcloud storage cp \
  "$PHONICS_MANUAL_OUTPUT_DIR/items.jsonl" \
  "$CONTAINER_ITEMS_URI"

gcloud storage cp \
  "$PHONICS_MANUAL_OUTPUT_DIR/container-config.json" \
  "$CONTAINER_CONFIG_URI"
```

The source config does not need an `output_dir`; `run-cloud` uses the container's
writable `/workspace/outputs` directory. Build the image, then run it with the provider
keys from the repository-root `.env`, the project used for GCS, and local Application
Default Credentials:

```bash
CODE_GIT_DIRTY=false
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  CODE_GIT_DIRTY=true
fi

docker build \
  --file deploy/phonics.Dockerfile \
  --build-arg CODE_GIT_HASH="$(git rev-parse --short HEAD)" \
  --build-arg CODE_GIT_DIRTY="$CODE_GIT_DIRTY" \
  -t phonics-benchmark:local \
  .

docker run --rm \
  --env-file .env \
  -e GOOGLE_CLOUD_PROJECT="$GOOGLE_CLOUD_PROJECT" \
  -e GOOGLE_APPLICATION_CREDENTIALS=/credentials/adc.json \
  -v "$HOME/.config/gcloud/application_default_credentials.json:/credentials/adc.json:ro" \
  phonics-benchmark:local \
  run-cloud --config "$CONTAINER_CONFIG_URI"
```

The image runs as a non-root user and writes only beneath `/workspace`. It contains no
provider secrets, `.env` files, benchmark datasets, or credentials; `--env-file` passes
the `.env` values to this container process at runtime without adding the file to the
image.
