# fab-benchmark-runner

Shared run machinery for FAB benchmarks, extracted from the phonics benchmark:

- **Result IDs.** Every model call gets a `model_call_id` from its inputs and a
  `result_id` derived from it. A rerun reuses every result whose ID already exists.
- **Model registry.** `custom_models.yaml` comes from the public `fab-open-configs`
  bucket, or from a pinned GCS copy checked against its SHA-256.
- **Run configs.** One flat JSON file holds the shared keys and each benchmark's own
  params.
- **GCS.** Items can be read from GCS, and results are mirrored to a GCS run prefix with
  generation checks, so a restarted job resumes where it stopped.
- **Cloud Run.** A CLI with `run-cloud`, plus a Dockerfile, Cloud Build config and job
  scripts that a benchmark copies into its repository.
- **Dry run.** `--dry-run` reports which calls a run would make and which it would
  reuse, without calling any model.

## Install

The package lives in the monorepo on its own branch. Install it with uv from a git
subdirectory:

```toml
[project]
dependencies = ["fab-benchmark-runner"]

[tool.uv.sources]
fab-benchmark-runner = { git = "https://github.com/Fab-Inc/qa-benchmarks-monorepo", branch = "feat/benchmark-runner-package", subdirectory = "packages/fab_benchmark_runner" }
```

`uv.lock` pins the exact commit; `uv lock --upgrade-package fab-benchmark-runner` picks
up new ones. It supports Python 3.12 and later.

The package takes FDLLM (`fabdata-llm`) from the monorepo's git source, its `main`
branch. A project that pins another FDLLM branch gets a conflicting-URL error from uv;
choose one version for both with an override:

```toml
[tool.uv]
override-dependencies = ["fabdata-llm @ git+https://github.com/AI-for-Education/fabdata-llm.git@dev"]
```

## Plug in a benchmark

A benchmark describes itself with a `Benchmark` adapter and exposes the generated CLI as
its console script:

```python
from dataclasses import dataclass

from fab_benchmark_runner import Benchmark, build_app


@dataclass(frozen=True)
class MathParams:
    prompt_name: str = "level1"

    def __post_init__(self) -> None:
        if self.prompt_name not in PROMPTS:
            raise ValueError(f"Unknown prompt_name: {self.prompt_name}")


def build_prompt(item, params: MathParams) -> str:
    return PROMPTS[params.prompt_name].render(question=item["question"])


BENCHMARK = Benchmark(
    name="mathv",
    params_type=MathParams,
    build_prompt=build_prompt,
)
app = build_app(BENCHMARK)  # [project.scripts] fab-benchmarks-mathv = "mathv.cli:app"
```

- `params_type` is a dataclass whose fields are extra top-level keys of the run config.
  Fields without a default are required. Validate values in `__post_init__` and raise
  `ValueError`.
- `identity` (optional) selects the params that define what a run means. They go into
  the config ID and every result row. By default all params are included.
- `load_items` (optional) loads a local items file and returns the items with the
  SHA-256 of the bytes read. The default reads JSONL. Every item needs a unique
  `benchmark_item_id`. `file_sha256` helps with other formats.
- `system_prompt`, `default_env_path` and `default_custom_models_path` are optional.
  Paths default to `.env` and `custom_models.yaml` in the working directory.

The CLI has these commands:

| Command                            | What it does                                                  |
| ---------------------------------- | ------------------------------------------------------------- |
| `run --config run.json`            | Run from a local config; `items_path` may be a `gs://` URI    |
| `run-cloud --config gs://…`        | Download the config and items, run with results synced to GCS |
| `upload-results` / `fetch-results` | Copy a run's files to or from a GCS run prefix                |

`run` and `run-cloud` both accept `--dry-run` and `--skip-models-download`. `run` also
accepts `--force`, `--retry-errors`, `--model-workers` and `--results-dest`.

Library callers can use `load_local_run_config`, `execute_run` and `plan_run` directly.

## Run configs

```json
{
  "run_id": "level1-20261116",
  "models": ["gpt-4o", "claude-sonnet-4"],
  "items_path": "gs://bucket/mathv/datasets/level1.jsonl",
  "prompt_name": "level1",
  "max_tokens": 4096
}
```

The shared keys are listed in `CORE_CONFIG_KEYS`:

| Key                                         | Meaning                                                             |
| ------------------------------------------- | ------------------------------------------------------------------- |
| `run_id`                                    | Output folder name; letters, digits and hyphens (default `default`) |
| `models`                                    | Model names registered in `custom_models.yaml` (required)           |
| `items_path`                                | Local path or `gs://` URI of the items (required)                   |
| `output_dir`, `inputs_dir`, `env_path`      | Local paths, relative to the config file                            |
| `custom_models_path`                        | Use this registry file as it is                                     |
| `custom_models_uri`, `custom_models_sha256` | Download and verify a pinned registry                               |
| `max_tokens`                                | Response token limit (default 10,000)                               |
| `chunk_size`                                | Calls per model between checkpoints (default 20; `null` saves once) |
| `model_workers`                             | Models run in parallel (default 4)                                  |
| `force`, `retry_errors`                     | Rerun everything, or only failed calls                              |
| `results_dest`                              | GCS run prefix to mirror results to                                 |
| `metadata`                                  | Free-form object kept in the run snapshot                           |

Other keys belong to the benchmark's params; any left over are kept in
`RunConfig.extra`.

## Result IDs

- `model_call_id` is the SHA-256 of the item's `benchmark_item_id`, the model name, the
  model's registered config, the prompt, the system prompt and the effective
  `max_tokens`.
- `result_id` is the first 24 hex characters of a hash of the `model_call_id`.
- The config ID hashes the benchmark's identity params plus `max_tokens`. Resuming a run
  with a different config ID, or with different items, is refused.

Results are one JSONL file per model under `<output_dir>/<run_id>/responses/`. The run
also writes `config.json`, a snapshot with the resolved config, items hash, code
revision and model configs, and `log.json` with per-model provenance.

Within a model, a result is reused when its ID exists, unless `force` is set. The runner
looks only in this run's folder, which is first filled from the run's GCS copy when the
run has a `results_dest`; results under another `run_id` are not consulted. With
`retry_errors`, only rows whose call failed are rerun. Rows whose ID is no longer
planned are dropped at the next write. Provider SDK errors and empty responses become
error rows; any other exception stops the run.

## Model registry

- With `custom_models_uri` and `custom_models_sha256`, the pinned file is downloaded
  into the run's inputs folder and its SHA-256 checked before any model call. A mismatch
  stops the run. This happens even with `--skip-models-download`, because a fresh
  container has no local copy.
- With `custom_models_path`, that file is used as it is.
- Otherwise the public file is refreshed from `gs://fab-open-configs` into the default
  path, unless `--skip-models-download` is given.

Every model in a run must appear in exactly one top-level provider group of the
registry. FDLLM merges the groups, so a model listed twice would silently take one
group's settings.

Pin the registry for long cloud runs. Each model's registered config is part of its call
IDs, so a change to the public file between restarts would make the run redo that
model's calls.

## Cloud Run

The `deploy/` folder next to this README holds the files a benchmark copies into its own
repository, once:

| File              | Copy to                  | What it does                                                  |
| ----------------- | ------------------------ | ------------------------------------------------------------- |
| `Dockerfile`      | `deploy/Dockerfile`      | Installs the project with `uv sync --frozen` and runs its CLI |
| `cloudbuild.yaml` | `deploy/cloudbuild.yaml` | Builds the image, checks `run-cloud --help`, pushes it        |
| `build-image.sh`  | `deploy/build-image.sh`  | Submits the build with the git revision baked in              |
| `deploy-job.sh`   | `deploy/deploy-job.sh`   | Creates or updates the Cloud Run job                          |
| `launch-run.sh`   | `deploy/launch-run.sh`   | Uploads a local config and items, then executes the job       |
| `dockerignore`    | `.dockerignore`          | Keeps the build context to what the install needs             |
| `gcloudignore`    | `.gcloudignore`          | Uploads only those files to Cloud Build                       |

Then set the values marked `SET`:

- In the Dockerfile: `PYTHON_VERSION`, the console script in `ENTRYPOINT` and, in a uv
  workspace, `UV_PACKAGE`.
- In the three scripts: `BENCHMARK`, which names the image, the job (`<name>-run`) and
  the GCS prefix. Exporting `BENCHMARK` works too.

The ignore files are written for a single-project repository with a `src/` layout. In a
uv workspace, also allow each member's `pyproject.toml`, `README.md` and `src/`.

Run the scripts from the repository root:

```bash
PROJECT_ID=my-project deploy/build-image.sh
PROJECT_ID=my-project SERVICE_ACCOUNT=runner@my-project.iam.gserviceaccount.com \
  SECRET_MAPPINGS="OPENAI_API_KEY=openai-api-key:latest" deploy/deploy-job.sh
PROJECT_ID=my-project RESULTS_BUCKET=gs://my-bucket \
  deploy/launch-run.sh run.json items.jsonl my-run-id
```

Launching again with the same run ID resumes that run. The scripts also read `REGION`,
`IMAGE_NAME`, `JOB_NAME`, `GCS_PREFIX`, `CPU`, `MEMORY` and `TASK_TIMEOUT` from the
environment.

## Known limitations

- **Model configs come from FDLLM.** The config in each call ID is FDLLM's expanded view
  of the model: the registry entry merged with provider defaults and FDLLM's built-in
  model list. A change to provider defaults, or an FDLLM upgrade that changes its
  built-in list, changes the IDs of the affected models. Projects that override the
  FDLLM version can get different IDs for the same model.
- **Text only.** Prompts are single strings; images are not part of calls or IDs yet.
- **Calls within a model run one at a time.** Up to `model_workers` models run in
  parallel, so a run with one or two models gets no faster.
