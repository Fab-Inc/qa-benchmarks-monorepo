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
uv run --frozen --package phonics-benchmark phonics-benchmark run-decodability \
  --config "$PHONICS_MANUAL_OUTPUT_DIR/config.json"
```

It should report `Executed: 1`, `Skipped: 0`, and `Errors: 0`. Repeat without refreshing
the model registry to verify cache reuse:

```bash
uv run --frozen --package phonics-benchmark phonics-benchmark run-decodability \
  --config "$PHONICS_MANUAL_OUTPUT_DIR/config.json" \
  --skip-models-download
```

It should now report `Executed: 0`, `Skipped: 1`, and `Errors: 0`. With a complete
cache, retry mode executes only error rows:

```bash
uv run --frozen --package phonics-benchmark phonics-benchmark run-decodability \
  --config "$PHONICS_MANUAL_OUTPUT_DIR/config.json" \
  --skip-models-download \
  --retry-errors
```

## Direct CLI run and cache reuse

The first command makes one Gemini call:

```bash
uv run --frozen --package phonics-benchmark phonics-benchmark run-decodability \
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
uv run --frozen --package phonics-benchmark phonics-benchmark run-decodability \
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
uv run --frozen --package phonics-benchmark phonics-benchmark run-decodability \
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
uv run --frozen --package phonics-benchmark phonics-benchmark run-decodability \
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
uv run --frozen --package phonics-benchmark phonics-benchmark run-decodability \
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
uv run --frozen --package phonics-benchmark phonics-benchmark run-decodability \
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

## GCS-synced local smoke test

With Application Default Credentials configured and a writable scratch bucket, add a run
prefix to the temporary config:

```bash
PHONICS_SCRATCH_DEST="gs://your-scratch-bucket/phonics/runs/config-smoke/"
jq --arg dest "$PHONICS_SCRATCH_DEST" \
  '.results_dest = $dest' \
  "$PHONICS_MANUAL_OUTPUT_DIR/config.json" \
  > "$PHONICS_MANUAL_OUTPUT_DIR/config-synced.json"

uv run --frozen --package phonics-benchmark phonics-benchmark run-decodability \
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

Upload a config whose `items_path` is a GCS URI, then build and run the same contract
used by Cloud Run Jobs:

```bash
CODE_GIT_DIRTY=false
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  CODE_GIT_DIRTY=true
fi

docker build \
  --build-arg CODE_GIT_HASH="$(git rev-parse --short HEAD)" \
  --build-arg CODE_GIT_DIRTY="$CODE_GIT_DIRTY" \
  -t phonics-benchmark:local \
  .

docker run --rm \
  -e GEMINI_API_KEY \
  -e GOOGLE_APPLICATION_CREDENTIALS=/credentials/adc.json \
  -v "$HOME/.config/gcloud/application_default_credentials.json:/credentials/adc.json:ro" \
  phonics-benchmark:local \
  run-cloud --config "gs://your-scratch-bucket/phonics/runs/container-smoke/config.json"
```

The image runs as a non-root user and writes only beneath `/workspace`. It contains no
provider secrets, `.env` files, benchmark datasets, or credentials.
