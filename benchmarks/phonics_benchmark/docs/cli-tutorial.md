# CLI tutorial

These flows exercise config and direct CLI runs, cache reuse, provider-error retries,
and adding a model. Run them from the repository root. They use real provider APIs;
commands expected to make calls are marked below.

Provider keys must be available in the repository-root `.env` or the environment. The
commands use the existing `custom_models.yaml` except for the first config-driven run,
which deliberately tests the default anonymous download.

## Prepare an isolated one-item run

Create a temporary output directory and a one-row composed benchmark CSV:

```bash
PHONICS_MANUAL_OUTPUT_DIR="$(mktemp -d /tmp/phonics-manual.XXXXXX)"
export PHONICS_MANUAL_OUTPUT_DIR

head -n 2 \
  benchmarks/phonics_benchmark/data/kiswahili/tusome_kenya_llm_trial_cases.csv \
  > "$PHONICS_MANUAL_OUTPUT_DIR/items.csv"

cat > "$PHONICS_MANUAL_OUTPUT_DIR/config.json" <<EOF
{
  "run_id": "config-smoke",
  "language": "kiswahili",
  "items_path": "$PHONICS_MANUAL_OUTPUT_DIR/items.csv",
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
  --language kiswahili \
  --model gemini-3.1-pro-preview \
  --items-path "$PHONICS_MANUAL_OUTPUT_DIR/items.csv" \
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
  --language kiswahili \
  --model gemini-3.1-pro-preview \
  --items-path "$PHONICS_MANUAL_OUTPUT_DIR/items.csv" \
  --output-dir "$PHONICS_MANUAL_OUTPUT_DIR" \
  --max-tokens 1000 \
  --skip-models-download \
  --force
```

Retry that exact error using the real key from `.env` or the environment:

```bash
uv run --frozen --package phonics-benchmark phonics-benchmark run-decodability \
  --run-id retry-error-smoke \
  --language kiswahili \
  --model gemini-3.1-pro-preview \
  --items-path "$PHONICS_MANUAL_OUTPUT_DIR/items.csv" \
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
  --language kiswahili \
  --model gemini-3.1-pro-preview \
  --items-path "$PHONICS_MANUAL_OUTPUT_DIR/items.csv" \
  --output-dir "$PHONICS_MANUAL_OUTPUT_DIR" \
  --max-tokens 1000 \
  --skip-models-download
```

Adding Claude directly in retry mode should fail before making calls because Claude has
no cached row:

```bash
uv run --frozen --package phonics-benchmark phonics-benchmark run-decodability \
  --run-id add-model-smoke \
  --language kiswahili \
  --model gemini-3.1-pro-preview \
  --model claude-sonnet-4-6 \
  --items-path "$PHONICS_MANUAL_OUTPUT_DIR/items.csv" \
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
  --language kiswahili \
  --model gemini-3.1-pro-preview \
  --model claude-sonnet-4-6 \
  --items-path "$PHONICS_MANUAL_OUTPUT_DIR/items.csv" \
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
