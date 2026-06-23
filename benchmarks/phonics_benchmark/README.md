# Phonics Benchmark

Config-driven benchmark harness for phonics experiments.

## Run From Config

```bash
phonics-benchmark run --config path/to/run.json
```

The same command can also be driven directly from CLI options:

```bash
phonics-benchmark run \
  --run-id decodability-smoke-test \
  --language kiswahili \
  --models gpt-5-mini-2025-08-07-medium,claude-sonnet-4-5-20250929 \
  --data-root data \
  --output-dir results \
  --chunk-size 100 \
  --env-path ../../.env \
  --custom-models-path ../../custom_models.yaml
```

When `--config` is supplied, values come from the config file. Without `--config`, the
required CLI options are `--language` and `--models`. `--run-id` defaults to `default`;
when supplied, it must use only alphanumeric characters and hyphens. Use
`--timestamped-output` to generate a UTC timestamp run ID for a non-overwriting run.

Runs checkpoint after each `chunk_size` pending items per model. The default is `100`;
set `"chunk_size": null` in a config file to save once after each model.

Before a run starts, the CLI loads provider keys from the repo-root `.env` when present,
registers models from the repo-root `custom_models.yaml` when present, and verifies that
every configured model name is registered with FDLLM. Use `--env-path` or
`--custom-models-path` to override either path.

The first-pass config format is JSON. Relative paths in a config file are resolved from
the config file's directory. Relative paths in CLI options are resolved from the current
working directory. The CLI provides `--data-root` as the default benchmark data root
when a config file does not set `data_root`.

```json
{
  "run_id": "decodability-smoke-test",
  "language": "kiswahili",
  "task": "decodability",
  "data_root": "data",
  "output_dir": "results",
  "chunk_size": 100,
  "models": [
    "gpt-5-mini-2025-08-07-medium",
    "claude-sonnet-4-5-20250929"
  ],
  "env_path": "../../.env",
  "custom_models_path": "../../custom_models.yaml"
}
```

Benchmark items are loaded from `<data_root>/<language>/`. Each language directory
contains `decodable_texts.jsonl`, `student_knowledge_profiles.jsonl`, and
`benchmark_items.jsonl`; the item adapter joins those files into benchmark rows before
the temporary model-call placeholder.

Each run writes:

- `results/<run_id>/config.json`: exact raw config, resolved paths, run-level code git
  hash, and sanitized model configs from FDLLM's registered model state.
- `results/<run_id>/answers/resps_<model>.jsonl`: one result per item for each model.
- `results/<run_id>/log.json`: per-model execution provenance such as timestamp and git
  hash.

Each result row includes `result_id`, `run_id`, `config_id`, `config_path`,
`config_snapshot_path`, `model`, `item_id`, `prompt`, raw `output`, and call status
fields. Until the prompt/call implementation lands, the default prompt and output are
both recorded as `placeholder`. Result rows identify the model by configured model name;
the corresponding model config is saved once in `config.json`, keyed by that name.
