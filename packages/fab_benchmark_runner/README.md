# fab-benchmark-runner

Shared run machinery for FAB benchmarks. A benchmark hands over its rendered inputs; the
runner stores them once as an experiment and runs any number of models on it, now or
later:

- **Experiments.** An experiment holds a benchmark's inputs, one row per item, and its
  token limit, and never changes after creation. Every model gets identical input,
  however the benchmark's code changes later.
- **Result IDs.** Every model call gets a `model_call_id` from its input, model and
  registered config. A rerun reuses every result whose ID already exists.
- **Images.** Inputs reference images by URI; the experiment records each file's
  SHA-256, and every run checks the images it sends against it before any call.
- **One set of files per model.** Each model's results and record are written only by
  the job running that model, so jobs for different models can share an experiment.
- **GCS.** An experiment can live on GCS; jobs work on a local copy and upload with
  generation checks, so a restarted job resumes where it stopped.
- **Model registry.** `custom_models.yaml` comes from the public `fab-open-configs`
  bucket, or from a pinned GCS copy checked against its SHA-256.
- **Dry run.** `--dry-run` reports which calls a run would make and which it would
  reuse, without calling any model.

To try it out, follow [WALKTHROUGH.md](WALKTHROUGH.md): four walkthroughs, from
everything local to everything on Cloud Run, with an example benchmark in
`examples/smoke/`.

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

A benchmark writes an inputs file with its own code, one JSON object per line. The full
format, what models receive and how it is stored are specified in
[INPUTS.md](INPUTS.md):

```json
{"benchmark_item_id": "sw-1", "prompt": "Evaluate …", "system_prompt": "You are …"}
{"benchmark_item_id": "1042", "prompt": "Which shape …?", "images": [{"uri": "gs://bucket/mathv/1042.jpg", "detail": "high"}]}
```

- `fab-bench check-inputs inputs.jsonl` lists every problem in a file, each with its
  line.
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
- `--max-tokens` is required: a number, or `none` for no limit. It never changes; see
  [Token limits](#token-limits).
- `--experiment-id` names the experiment; it defaults to a short hash of the benchmark,
  the inputs and the limit. It doesn't have to match the folder name.
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
| `fab-bench check-inputs inputs.jsonl`         | List every problem in an inputs file, with its line      |
| `fab-bench run --config run.json`             | Run a local or `gs://` config                            |
| `fab-bench run-cloud --config gs://…`         | The same, as a Cloud Run job does, in `/workspace`       |
| `fab-bench fetch-results --experiment gs://…` | Download inputs, records and results (`--model` for one) |
| `fab-bench upload-experiment <folder> gs://…` | Copy a local experiment, with its results, to GCS        |
| `fab-bench upload-run-config run.json`        | Check a config for Cloud Run, upload it, print its URI   |

`run` and `run-cloud` accept `--dry-run` and `--skip-models-download`; `run` also takes
`--force`, `--retry-errors`, `--model-workers` and `--workdir` (default `.fab-bench`).
Library callers can use `run_job` and `plan_job`, or `execute_run` and `plan_run` on a
local experiment folder.

## Images

Images stay where the inputs put them, usually on GCS, and creation records each file's
SHA-256. When a run sends them:

- Before its first call, a run fetches the images of the calls it still has to make,
  each once however many items and models use it, 32 at a time, into a temporary folder
  that goes when the run ends. A dry run fetches and checks them too, so a broken image
  shows up before a launch. On Cloud Run that folder is in memory.
- Each copy must match its recorded SHA-256 and open with Pillow. An image that changed,
  disappeared or can't be opened stops the run before any call, naming the image.
- Images copied into an experiment with `--copy-images` are read from its folder, local
  or on GCS.
- FDLLM gets a fresh image per call, loaded from the checked copy, in the item's one
  user message, in order and with its detail level. FDLLM resizes as it sends (`low` to
  512×512; `high` to at most 2048 on the long side and 768 on the short side) and puts
  images before the prompt text. Experiments store the originals; each run records
  FDLLM's commit and Pillow's version, which is enough to rebuild what was sent.
- FDLLM sends images only to models whose registry entry has `Vision: true`, so a run on
  an experiment with images refuses other models before any call.
- A call function of your own gets each image's `uri` pointing at its checked local
  copy, with `source_uri` saying where it came from.

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

Every call gets the experiment's `max_tokens`, and every row records it. With `none`,
FDLLM asks for what the model's context window leaves after the input, capped by the
registry's `Token_Limit_Completion`; set that for models whose output limit is below
their window. For Anthropic and Google models FDLLM counts the input with an extra API
call first.

An experiment's limit never changes, so every result in it ran under the same limit.
When answers hit it, create the experiment again in another folder with a higher limit,
or `none`, and run the models there:

```bash
fab-bench create-experiment --inputs inputs.jsonl \
  --dest gs://bucket/mathv/experiments/level1-v1-128k \
  --benchmark mathv --experiment-id level1-v1-128k --max-tokens 128000
```

The first experiment and its results stay as they were. Creating an experiment again in
its own folder with another limit is refused. A response cut off mid-answer counts as an
answer: FDLLM reports no stop reason for a response that isn't empty.

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

One image, built from this package, runs every benchmark: it holds the runner and its
locked dependencies, and no benchmark code. A Cloud Run job runs
`fab-bench run-cloud --config gs://…` on a run config whose experiment is on GCS. The
scripts are in this package's `deploy/` folder.

**Build the image**, after committing any change to the package:

```bash
PROJECT_ID=registry-project deploy/build-image.sh
```

Cloud Build builds it from this folder and pushes
`<region>-docker.pkg.dev/<project>/fab-benchmarks/fab-bench:<commit>`, tagged with the
runner's commit; the script refuses uncommitted changes, so the tag names what the image
holds. The build checks that the image starts and that its runs will record the runner's
and FDLLM's commits. The script also reads `REGION`, `ARTIFACT_REPOSITORY` and
`IMAGE_NAME`.

**Deploy the job**, once per project and again for each new image:

```bash
PROJECT_ID=my-project IMAGE=<the image URI build-image.sh printed> \
  SERVICE_ACCOUNT=runner@my-project.iam.gserviceaccount.com \
  SECRET_MAPPINGS="OPENAI_API_KEY=openai-api-key:latest" deploy/deploy-job.sh
```

- This creates or updates the job `fab-bench` (`JOB_NAME`) with the image, provider keys
  from Secret Manager, and `FAB_BENCH_IMAGE`, which every run records. It also reads
  `REGION`, `CPU`, `MEMORY` (default `2Gi`), `TASK_TIMEOUT` (`48h`) and `MAX_RETRIES`
  (`2`); a retried task reruns its config and resumes.
- The job's service account reads and writes the experiments' buckets.
- A job in another project than the image's needs read access to its registry: grant
  `roles/artifactregistry.reader` on the repository to that project's Cloud Run service
  agent, `service-<project number>@serverless-robot-prod.iam.gserviceaccount.com`.
- Images a job sends are held in a temporary folder, which on Cloud Run is memory; raise
  `MEMORY` for large image sets.

**Launch a run:**

1. Create the experiment on GCS with `create-experiment --dest gs://…`, or copy a local
   one with `upload-experiment`.

1. Write a run config naming that experiment. Pin the registry with `custom_models_uri`
   and `custom_models_sha256`, or leave it out for the public one; local paths are
   refused.

1. Launch it:

   ```bash
   PROJECT_ID=my-project deploy/launch-run.sh run.json
   ```

   The script runs `fab-bench upload-run-config`, which checks the config as the job
   will read it and uploads it to `run-configs/<sha256>.json` in the experiment's
   bucket, then starts the job on that URI. The same config always gets the same URI and
   is uploaded once, so launching it again resumes it. `FAB_BENCH` sets the command,
   such as `uvx --from "git+…" fab-bench`; `RUN_CONFIGS_URI` sets another folder;
   `WAIT=true` waits for the job to finish. The upload uses your application default
   credentials (`gcloud auth application-default login`).

1. Download the results with `fab-bench fetch-results --experiment gs://…`.

Jobs running different models can share an experiment at the same time; see
[Experiment folders](#experiment-folders).

## Known limitations

- **Model configs come from FDLLM.** The config in each call ID is FDLLM's expanded view
  of the model: the registry entry merged with provider defaults and FDLLM's built-in
  model list. A change to provider defaults, or an FDLLM upgrade that changes its
  built-in list, changes the IDs of the affected models. Projects that override the
  FDLLM version can get different IDs for the same model.
- **Responses cut off mid-answer are scored as they are.** FDLLM reports no stop reason
  for a response that isn't empty, so they can't be told apart yet.
- **Calls within a model run one at a time.** Up to `model_workers` models run in
  parallel, so a run with one or two models gets no faster.
