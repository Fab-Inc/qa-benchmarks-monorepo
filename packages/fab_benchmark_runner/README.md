# fab-benchmark-runner

Shared run machinery for FAB benchmarks. A benchmark hands over its rendered inputs; the
runner stores them once as an experiment and runs any number of models on it, now or
later:

- **Experiments.** An experiment holds a benchmark's inputs, one row per item, and never
  changes after creation, except that its `max_tokens` can be raised. Every model gets
  identical input, however the benchmark's code changes later.
- **Result IDs.** Every model call gets a `model_call_id` from its input, model and
  registered config. A rerun reuses every result whose ID already exists.
- **One set of files per model.** Each model's results and record are written only by
  the job running that model, so jobs for different models can share an experiment.
- **GCS.** An experiment can live on GCS; jobs work on a local copy and upload with
  generation checks, so a restarted job resumes where it stopped.
- **Model registry.** `custom_models.yaml` comes from the public `fab-open-configs`
  bucket, or from a pinned GCS copy checked against its SHA-256.
- **Dry run.** `--dry-run` reports which calls a run would make and which it would
  reuse, without calling any model.

`tests/behaviour/` describes the intended behaviour, and the design decisions behind it,
as runnable scenarios; `tests/unit/` pins the details.

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

The package locks its own dependencies and takes FDLLM (`fabdata-llm`) from its `dev`
branch. A project that installs it without an FDLLM source of its own gets `dev` too. A
project that names another FDLLM source gets a conflicting-URL error from uv; use the
same source, or choose one version for both with an override:

```toml
[tool.uv]
override-dependencies = ["fabdata-llm @ git+https://github.com/AI-for-Education/fabdata-llm.git@main"]
```

An environment holds one FDLLM, so an override also decides what the runner runs on
there; it works with FDLLM's `main` and `dev` branches. A benchmark that only hands over
an inputs file doesn't need to import the package: running `fab-bench` from its own
environment keeps the two FDLLM versions apart.

```bash
uvx --from "git+https://github.com/Fab-Inc/qa-benchmarks-monorepo@feat/benchmark-runner-package#subdirectory=packages/fab_benchmark_runner" fab-bench --help
```

In the monorepo the package is not a member of the uv workspace, so its tests run from
its folder: `uv run --frozen pytest`.

## Experiments

An experiment stores a benchmark's rendered inputs once, so every model run on it, now
or later, gets identical input.

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
- `--max-tokens` is required: a number, or `none` for no limit. It can be raised later,
  never lowered; see [Token limits](#token-limits).
- `--experiment-id` names the experiment; it defaults to a short hash of the benchmark
  and inputs. It doesn't have to match the folder name.
- The folder gets `inputs.jsonl`, then `experiment.json`, written last so an interrupted
  creation never looks finished. Creating the same experiment again changes nothing; a
  folder holding a different experiment, or other files, is refused.

## Running models

A run config names the experiment, the models and how to run them:

```json
{
  "experiment": "gs://bucket/mathv/experiments/level1-v1",
  "models": ["gpt-4o", "claude-sonnet-4"],
  "custom_models_uri": "gs://bucket/model-configs/2026-11.yaml",
  "custom_models_sha256": "…"
}
```

| Key                                         | Meaning                                                             |
| ------------------------------------------- | ------------------------------------------------------------------- |
| `experiment`                                | Experiment folder or `gs://` URI (required)                         |
| `models`                                    | Model names registered in `custom_models.yaml` (required)           |
| `custom_models_path`                        | Use this registry file as it is                                     |
| `custom_models_uri`, `custom_models_sha256` | Download and verify a pinned registry                               |
| `env_path`                                  | A `.env` file with provider keys, for local runs                    |
| `chunk_size`                                | Calls per model between checkpoints (default 20; `null` saves once) |
| `model_workers`                             | Models run in parallel (default 4)                                  |
| `force`, `retry_errors`                     | Rerun everything, or only failed calls; they can't be combined      |

Local paths resolve against the config file's folder. Other keys are refused, so a
misspelled setting can't be silently ignored; `max_tokens` belongs to the experiment.
Adding a model means writing another run config; nothing shared is rewritten.

| Command                                       | What it does                                             |
| --------------------------------------------- | -------------------------------------------------------- |
| `fab-bench run --config run.json`             | Run a local or `gs://` config                            |
| `fab-bench run-cloud --config gs://…`         | The same, as a Cloud Run job does, in `/workspace`       |
| `fab-bench raise-max-tokens --experiment …`   | Raise an experiment's token limit                        |
| `fab-bench fetch-results --experiment gs://…` | Download inputs, records and results (`--model` for one) |
| `fab-bench upload-experiment <folder> gs://…` | Copy a local experiment, with its results, to GCS        |

`run` and `run-cloud` accept `--dry-run` and `--skip-models-download`; `run` also takes
`--force`, `--retry-errors`, `--model-workers` and `--workdir` (default `.fab-bench`).
Library callers can use `run_job` and `plan_job`, or `execute_run` and `plan_run` on a
local experiment folder.

## Experiment folders

| File                                | Written by                 | Holds                                                  |
| ----------------------------------- | -------------------------- | ------------------------------------------------------ |
| `experiment.json`                   | `create-experiment`        | What defines the experiment; only `max_tokens` changes |
| `inputs.jsonl`                      | `create-experiment`        | The stored inputs, each image with its SHA-256         |
| `images/`                           | `create-experiment`        | Only images copied in with `--copy-images`             |
| `responses/responses_<model>.jsonl` | The job running that model | One row per call                                       |
| `models/<model>.json`               | The job running that model | The model's settings, registry, and one entry per run  |

A row repeats the input (`system_prompt`, `prompt`, `images`) next to `result_id`,
`model_call_id`, `experiment_id`, `benchmark`, `benchmark_item_id`, `model`,
`max_tokens`, `output`, `call_status`, `error_type`, `error_message`, `stop_reason`,
token counts, `model_version`, `latency_seconds` and `timestamp_utc`, so each row reads
on its own.

A model's record lists every run that wrote its results: when, how many calls, the token
limit, the run config, and the versions of the runner, FDLLM, Pillow and Python, plus
the container image. Versions come from package metadata, not from the working
directory's git checkout. A run or dry run warns when the experiment already has results
from other runner, FDLLM or Pillow versions.

A job on an experiment on GCS works on a local copy in its workdir. It downloads the
experiment's own files, every model's record and the results of the models it runs, and
uploads only the files it writes, each checked against the GCS generation it last saw.
Jobs running different models never conflict; a second job on the same model stops
instead of overwriting the first one's results.

## Result IDs

- `model_call_id` is the SHA-256 of the item's `benchmark_item_id`, the model name, the
  model's registered config, the prompt and the system prompt.
- `max_tokens` is left out, so raising the limit keeps every result's ID and only calls
  that hit the old limit need rerunning. Every row, error rows included, records the
  `max_tokens` it ran with. These IDs differ from the phonics runner's, which hash
  `max_tokens`; its results need their IDs recomputed before they can be reused here.
- An item's images enter the ID in order, each by its file's SHA-256 and its detail
  level (`low` or `high`), not by where it is stored. A call without images hashes
  exactly as before images were supported.
- `result_id` is the first 24 hex characters of a hash of the `model_call_id`.

Within a model, a result is reused when its ID exists, unless `force` is set. With
`retry_errors`, only rows whose call failed are rerun; it refuses, before any call, to
run calls that have no row yet. Rows whose ID is no longer planned, because the model's
registered config changed, are dropped at the next write. Provider SDK errors, FDLLM's
response errors (from its `dev` branch: empty, filtered or malformed responses) and
empty responses become error rows; any other exception stops the run. An error row
carries the provider's `stop_reason` when FDLLM reports one; `length`, `max_tokens` or
`MAX_TOKENS` means the response ran out of tokens.

## Token limits

Every call gets the experiment's current `max_tokens`, and every row records it. When
some items hit the limit:

```bash
fab-bench raise-max-tokens --experiment gs://bucket/mathv/experiments/level1-v1 --max-tokens 128000
fab-bench run --config run.json --retry-errors
```

Results that finished within the old limit keep their IDs and stay; the error rows,
which include responses the limit left empty, rerun at the new limit. The limit can only
be raised, and `experiment.json` keeps each limit it had in `max_tokens_history`.
Creating the experiment again never changes it. A response cut off mid-answer counts as
an answer: FDLLM reports no stop reason for a response that isn't empty.

## Model registry

- With `custom_models_uri` and `custom_models_sha256`, the pinned file is downloaded
  into the workdir and its SHA-256 checked before any model call. A mismatch stops the
  run. This happens even with `--skip-models-download`, because a fresh container has no
  local copy.
- With `custom_models_path`, that file is used as it is.
- Otherwise the public file is refreshed from `gs://fab-open-configs` into
  `custom_models.yaml` in the working directory (the workdir for `run-cloud`), unless
  `--skip-models-download` is given.

Every model in a run must appear in exactly one top-level provider group of the
registry. FDLLM merges the groups, so a model listed twice would silently take one
group's settings.

Pin the registry for long cloud runs. Each model's registered config is part of its call
IDs, so a change to the public file between restarts would make the run redo that
model's calls.

## Cloud Run

The deploy files still build one image per benchmark and launch the old run configs;
they move to one shared runner image running `fab-bench run-cloud` in the next step.

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
- **Images and unlimited calls are not sent yet.** Experiments and call IDs cover images
  and `max_tokens: none`, but runs through FDLLM refuse them before any call.
- **Responses cut off mid-answer are scored as they are.** FDLLM reports no stop reason
  for a response that isn't empty, so they can't be told apart yet.
- **Calls within a model run one at a time.** Up to `model_workers` models run in
  parallel, so a run with one or two models gets no faster.
