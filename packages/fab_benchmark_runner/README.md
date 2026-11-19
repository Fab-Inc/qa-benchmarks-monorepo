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

## Experiments

An experiment stores a benchmark's rendered inputs once, so every model run on it, now
or later, gets identical input. Runs don't use experiments yet; that is the next step.

A benchmark writes an inputs file with its own code, one JSON object per line:

```json
{"benchmark_item_id": "sw-1", "prompt": "Evaluate …", "system_prompt": "You are …"}
{"benchmark_item_id": "1042", "prompt": "Which shape …?", "images": [{"uri": "gs://bucket/mathv/1042.jpg", "detail": "high"}]}
```

- `system_prompt` is optional and set per item. `images` go into the item's one user
  message, in order. An image `uri` is a `gs://` URI, an absolute path, or a path
  relative to the inputs file; `detail` is `low` or `high`.
- The inputs file can be local or on GCS. In a file on GCS, relative image paths resolve
  against its GCS folder, and local paths are refused.
- Unknown keys are refused, so a misspelled field can't silently drop part of an input.
- From Python, build `ModelInput` values and call `write_inputs`, or pass them straight
  to `create_experiment`.

Then store it:

```bash
fab-bench create-experiment --inputs inputs.jsonl \
  --dest gs://bucket/mathv/experiments/level1-v1 \
  --benchmark mathv --experiment-id level1-v1 --max-tokens 64000 \
  --label subset=level1 --dry-run
```

- `--dry-run` checks every row and image and shows the first inputs, writing nothing.
- Every image is read once, 32 at a time, to record its SHA-256. Images stay where they
  are unless `--copy-images` copies them into the experiment's `images/` folder, named
  by their hash. Images at local paths must be copied into an experiment on GCS.
- `--max-tokens` is required: a number, or `none` for no limit.
- `--experiment-id` names the experiment; it defaults to a short hash of the benchmark,
  inputs and `max_tokens`. It doesn't have to match the folder name.
- The folder gets `inputs.jsonl`, then `experiment.json`, written last so an interrupted
  creation never looks finished. Creating the same experiment again changes nothing; a
  folder holding a different experiment, or other files, is refused.

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
| `metadata`                                  | Free-form object kept in each model record's run entry              |

Other keys belong to the benchmark's params; any left over are kept in
`RunConfig.extra`.

## Result IDs

- `model_call_id` is the SHA-256 of the item's `benchmark_item_id`, the model name, the
  model's registered config, the prompt, the system prompt and the effective
  `max_tokens` (`null` when there is no limit).
- An item's images enter the ID in order, each by its file's SHA-256 and its detail
  level (`low` or `high`), not by where it is stored. A call without images hashes
  exactly as before images were supported.
- `result_id` is the first 24 hex characters of a hash of the `model_call_id`.
- The config ID hashes the benchmark's identity params plus `max_tokens`. A run folder
  refuses a run with a different config ID, or with different items, before any call.

Within a model, a result is reused when its ID exists, unless `force` is set. The runner
looks only in this run's folder, which is first filled from the run's GCS copy when the
run has a `results_dest`; results under another `run_id` are not consulted. With
`retry_errors`, only rows whose call failed are rerun. Rows whose ID is no longer
planned are dropped at the next write. Provider SDK errors and empty responses become
error rows; any other exception stops the run.

## Run folders

A run folder, `<output_dir>/<run_id>/` and its GCS copy at `results_dest`, holds:

| File                                | Written by                 | Holds                                                                                                                                      |
| ----------------------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `config.json`                       | The first job, once        | The run record: benchmark, config ID and its payload, items hash                                                                           |
| `responses/responses_<model>.jsonl` | The job running that model | One row per call                                                                                                                           |
| `models/<model>.json`               | The job running that model | The model's settings, registry and hash, and one entry per run that wrote its results: when, how many calls, code revision, the run config |

Later jobs check `config.json` against their own run and never rewrite it, so the folder
keeps describing the same run while models are added over time. The model list of a run
is the set of model records and response files.

No file is shared between jobs that run different models, and a job only uploads files
it wrote itself. So jobs running different models can write to the same folder at the
same time. Two jobs running the same model stop with a conflict instead of paying for
the same calls twice.

A run folder's `config.json` is not a run config: rerun the config the run was launched
with. Keep run configs outside their run folder; `launch-run.sh` stores them under
`run-configs/`.

Folders written by the phonics runner can be extended: their `config.json` is checked in
its old format and replaced by the new record, and their results are reused. Their
`log.json` is left as it was.

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
- **Images are not sent yet.** Call IDs already cover images, but runs still send only
  text, and `max_tokens` can't yet be set to no limit in a run config.
- **Calls within a model run one at a time.** Up to `model_workers` models run in
  parallel, so a run with one or two models gets no faster.
