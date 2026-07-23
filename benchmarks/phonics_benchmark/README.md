# Phonics Benchmark

Config-driven benchmark harness for phonics experiments.

## Run Decodability

```bash
phonics-benchmark run-decodability --config path/to/run.json
```

The same command can also be driven directly from CLI options:

```bash
phonics-benchmark run-decodability \
  --run-id decodability-smoke-test \
  --language kiswahili \
  --models gpt-5-mini-2025-08-07-medium,claude-sonnet-4-5-20250929 \
  --items-path path/to/composed-items.csv \
  --output-dir results \
  --chunk-size 100 \
  --model-workers 2 \
  --env-path ../../.env \
  --custom-models-path ../../custom_models.yaml
```

When `--config` is supplied, values come from the config file. Without `--config`, the
required CLI options are `--language`, `--models`, and `--items-path`. `--run-id`
defaults to `default`; when supplied, it must use only alphanumeric characters and
hyphens. Use `--timestamped-output` to generate a UTC timestamp run ID for a
non-overwriting run. The command records the task as `decodability`; task selection is
not a config option.

Runs checkpoint after each `chunk_size` pending items per model. The default is `100`;
set `"chunk_size": null` in a config file to save once after each model. Models run
sequentially by default. Set `model_workers` in config or pass `--model-workers` to run
multiple models concurrently; calls within each model remain sequential. Cached rows are
reused when the model-call inputs are unchanged, even if provenance-only config fields,
operational settings, or the model list change. Unknown future config fields are also
treated as provenance. Changing model-call inputs such as the item payload, prompt,
model name, model config, or max tokens creates new result rows for those calls. The
default model-call limit is 10,000 response tokens; set `max_tokens` in config or pass
`--max-tokens` to override it.

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
  "language": "kiswahili",
  "items_path": "path/to/composed-items.csv",
  "output_dir": "results",
  "chunk_size": 100,
  "model_workers": 2,
  "models": [
    "gpt-5-mini-2025-08-07-medium",
    "claude-sonnet-4-5-20250929"
  ],
  "env_path": "../../.env",
  "custom_models_path": "../../custom_models.yaml"
}
```

### Run the QFB Tusome trial

The checked-in `configs/tusome-llm-trial.json` reproduces the model-running part of
`scripts/llm_output_trial.py`: it runs all 57 composed Tusome cases with
`gemini-3.1-pro-preview` and `claude-sonnet-4-6`, using the script's 30,000-token limit.
It stores only raw model results and call metadata; it does not parse or score them.

From the repository root:

```bash
uv run --frozen --package phonics-benchmark phonics-benchmark run-decodability \
  --config benchmarks/phonics_benchmark/configs/tusome-llm-trial.json
```

The command makes 114 model calls on the first uncached run, running the two models
concurrently while keeping each model's 57 calls sequential. It writes one JSONL file
per model under `benchmarks/phonics_benchmark/outputs/tusome-llm-trial/answers/`.
Re-running the same command reuses successful cached rows. Set `"force": true` in a
copied config to rerun all calls, or add `--skip-models-download` to use the existing
repo-root `custom_models.yaml` without refreshing it.

The runner accepts only already-composed QFB-style CSV files, selected through
`items_path` in config or `--items-path` with direct CLI options. It does not join or
compose benchmark inputs. The CSV must contain `benchmark_item_id`, `content`,
`student_knowledge_id`, `graphemes`, and `whole_words`; the two knowledge columns must
be JSON arrays of strings. Each row is rendered with the packaged decodability prompt
and sent to FDLLM as the user message.

Each run writes:

- `results/<run_id>/config.json`: latest config snapshot with the exact raw config,
  resolved paths, run-level code git hash, and sanitized model configs from FDLLM's
  registered model state.
- `results/<run_id>/answers/resps_<model-file-id>.jsonl`: one result per item for each
  model. The file ID is the model name when it is already filename-safe, or a sanitized
  hash-suffixed form otherwise.
- `results/<run_id>/log.json`: per-model execution provenance such as timestamp and git
  hash.

Each result row includes `result_id`, `run_id`, `config_id`, `model_call_id`,
`config_path`, `config_snapshot_path`, `model`, `item_id`, `prompt`, raw `output`, and
call status fields. Successful model calls also include the effective `max_tokens`,
completion-token count, QFB `timestamp_utc`, and `latency_seconds`. Error rows omit
timing fields. `model_call_id` is the cache identity for the response. Cached rows may
keep an older `config_id` when reused across provenance-only config changes. Result rows
identify the model by configured model name; the corresponding model config is saved in
the latest config snapshot, keyed by that name.

The runner saves recognized provider SDK API and transport failures propagated by FDLLM
as error rows. It also saves an absent or empty model response as an error row. Other
exceptions, including FDLLM/model setup, response processing, and result persistence
errors, propagate and stop the affected run instead of being converted into call errors.
