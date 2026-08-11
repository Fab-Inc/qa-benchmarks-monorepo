# Phonics Benchmark

Config-driven benchmark harness for phonics experiments.

For an end-to-end walkthrough of config and direct runs, cache reuse, error retries, and
adding models, see the [CLI tutorial](docs/cli-tutorial.md).

For building the container, deploying the Cloud Run Job, and launching cloud runs, see
the [Cloud Run deployment guide](../../deploy/README.md).

## Run Decodability

```bash
phonics-benchmark run-decodability --config path/to/run.json
```

The same command can also be driven directly from CLI options:

```bash
phonics-benchmark run-decodability \
  --run-id decodability-smoke-test \
  --language sw \
  --models gpt-5-mini-2025-08-07-medium,claude-sonnet-4-5-20250929 \
  --items-path path/to/completed-items.jsonl \
  --output-dir results \
  --chunk-size 20 \
  --model-workers 4 \
  --env-path ../../.env \
  --custom-models-path ../../custom_models.yaml
```

When `--config` is supplied, values come from the config file. Without `--config`, the
required CLI options are `--language`, `--models`, and `--items-path`. `--run-id`
defaults to `default`; when supplied, it must use only alphanumeric characters and
hyphens. Use `--timestamped-output` to generate a UTC timestamp run ID for a
non-overwriting run. The command records the task as `decodability`; task selection is
not a config option.

`--run-id` and `--timestamped-output` apply only to direct CLI runs. With `--config`,
the `run_id` and `output_dir` come from the config file; the CLI execution overrides are
limited to `--force`, `--retry-errors`, and `--model-workers`. To create and retain a
fresh set of responses, use a new run ID in direct mode, or copy a config and change its
`run_id` or `output_dir`. A new run directory has no local cache, so all planned calls
execute even when their semantic model-call identities match another run. Use `--force`
to rerun calls in the existing run directory only when replacing its current responses
is intended.

Runs checkpoint after each `chunk_size` pending items per model. The default is `20`;
set `"chunk_size": null` in a config file to save once after each model. Items and
models within one top-level `custom_models.yaml` provider group run sequentially.
Different provider groups can run concurrently, and `model_workers` caps the number of
active groups; the default cap is `4`. Cached rows are reused when the benchmark item ID
and effective provider-call inputs are unchanged, even if provenance-only config fields,
operational settings, the dataset artifact, or the selected model list change. Unknown
future config fields are also treated as provenance. Changing `benchmark_item_id`, the
rendered prompt, model name, sanitized resolved model config, system prompt, or
effective generation parameters creates a new call identity. The default model-call
limit is 10,000 response tokens; set `max_tokens` in config or pass `--max-tokens` to
override it.

Use `--retry-errors` only after a normal run has cached every currently planned call.
Retry mode runs exactly the cached rows whose `call_status` is `error`; it fails before
starting any model workers if calls are missing or cache statuses are invalid. After
adding items or models, changing model-call inputs, or interrupting a run before every
call is recorded, run normally first to execute only the missing calls, then use
`--retry-errors`.

Before a run starts, the CLI anonymously downloads the latest public
`custom_models.yaml` from the `fab-open-configs` bucket. Use `--skip-models-download` to
skip this refresh. The CLI then loads provider keys from the repo-root `.env`, registers
the selected model file, and verifies that every configured model name is registered
with FDLLM. Use `--env-path` to override the environment file. Supplying
`--custom-models-path` uses that file as a pinned registry for the run, but still
refreshes the shared repo-root file unless `--skip-models-download` is also supplied.

The first-pass config format is JSON. Relative paths in a config file are resolved from
the config file's directory. Relative paths in CLI options are resolved from the current
working directory.

```json
{
  "run_id": "decodability-smoke-test",
  "language": "sw",
  "items_path": "path/to/completed-items.jsonl",
  "output_dir": "results",
  "chunk_size": 20,
  "model_workers": 4,
  "models": [
    "gpt-5-mini-2025-08-07-medium",
    "claude-sonnet-4-5-20250929"
  ],
  "env_path": "../../.env",
  "custom_models_path": "../../custom_models.yaml"
}
```

### Run the QFB Tusome trial

The checked-in `configs/tusome-llm-trial.json` records the intended model-running
configuration for the first Kiswahili run and points to the checked-in completed JSONL
version of the existing prepared trial data. This keeps the current trial runnable while
QFB-202 is updated to emit the same JSONL handoff contract. The runner stores only raw
model results and call metadata; it does not parse or score them.

From the repository root:

```bash
uv run --frozen --package phonics-benchmark phonics-benchmark run-decodability \
  --config benchmarks/phonics_benchmark/configs/tusome-llm-trial.json
```

The models can run concurrently when their `custom_models.yaml` entries belong to
different provider groups, while calls for each model remain sequential. Results are
written under `benchmarks/phonics_benchmark/outputs/tusome-llm-trial/responses/`.
Re-running the same command reuses successful cached rows. Set `"force": true` in a
copied config to rerun all calls, or add `--skip-models-download` to use the existing
repo-root `custom_models.yaml` without refreshing it.

The runner accepts completed JSONL artifacts selected through `items_path` in config or
`--items-path` with direct CLI options. It does not join or compose benchmark inputs.
Each line must be an object containing a non-empty `benchmark_item_id`, non-whitespace
`content`, and object-valued `student_knowledge`. Additional dataset-owned fields are
accepted and ignored. The selected language registry validates `student_knowledge`,
provides the display name, and controls its string representation in the shared prompt.

Each run writes:

- `results/<run_id>/config.json`: latest config snapshot with the exact raw config,
  resolved paths, exact-byte `items_sha256`, run-level code git hash, and sanitized
  model configs from FDLLM's registered model state.
- `results/<run_id>/responses/responses_<model-file-id>.jsonl`: one result per item for
  each model. The file ID is the model name when it is already filename-safe, or a
  sanitized hash-suffixed form otherwise.
- `results/<run_id>/log.json`: per-model execution provenance such as timestamp and git
  hash.

Each result row includes `result_id`, `run_id`, `config_id`, `model_call_id`,
`config_path`, `config_snapshot_path`, `model`, `benchmark_item_id`, `prompt`, raw
`output`, and call status fields. Successful model calls also include `max_tokens`,
completion-token count, QFB `timestamp_utc`, and `latency_seconds`. Error rows omit
timing fields. `model_call_id` is the cache identity for the response. Cached rows may
be reused across provenance-only changes because `config_id` and `model_call_id` remain
stable. Result rows identify the model by configured model name; the corresponding model
config is saved in the latest config snapshot, keyed by that name. Dataset-owned source
references, expected answers, annotations, and the normalized item object are not copied
into raw results.

The runner saves recognized provider SDK API and transport failures propagated by FDLLM
as error rows. It also saves an absent or empty model response as an error row. Other
exceptions, including FDLLM/model setup, response processing, and result persistence
errors, propagate and stop the affected run instead of being converted into call errors.
