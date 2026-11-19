# Trying fab-bench

Four walkthroughs for anyone checking that fab-bench works for their benchmark. Each
starts from a tiny example benchmark, `examples/smoke/`, and says what to check; the
last section turns your own benchmark into an inputs file.

| Walkthrough                                                                                    | Inputs and images | Experiment      | Run config | Models run on |
| ---------------------------------------------------------------------------------------------- | ----------------- | --------------- | ---------- | ------------- |
| [1. Everything local](#walkthrough-1-everything-local)                                         | local             | local           | local      | your machine  |
| [2. Inputs and config on GCS](#walkthrough-2-inputs-and-config-on-gcs)                         | GCS               | local, then GCS | GCS        | your machine  |
| [3. New models on an existing experiment](#walkthrough-3-new-models-on-an-existing-experiment) | already stored    | from 1          | a new one  | your machine  |
| [4. Everything on Google Cloud](#walkthrough-4-everything-on-google-cloud)                     | GCS               | GCS             | GCS        | Cloud Run     |

Walkthroughs 2 and 4 need access to a Google Cloud project with a bucket and a deployed
fab-bench job, as in the README's Cloud Run section; ask its admin for the project, the
bucket and the job's name.

## Before you start

You need [uv](https://docs.astral.sh/uv/). Install `fab-bench` as a tool, so the command
and the launch script can find it, and get the example files from the branch:

```bash
uv tool install "git+https://github.com/Fab-Inc/qa-benchmarks-monorepo@feat/benchmark-runner-package#subdirectory=packages/fab_benchmark_runner"
git clone --depth 1 --branch feat/benchmark-runner-package \
  https://github.com/Fab-Inc/qa-benchmarks-monorepo.git
export RUNNER="$PWD/qa-benchmarks-monorepo/packages/fab_benchmark_runner"
fab-bench --help
```

To pick up later changes on the branch, run the `uv tool install` command again with
`--reinstall`.

The walkthroughs call two cheap Gemini models, `gemini-3.1-flash-lite` and
`gemini-3.5-flash-lite`, so you need a Gemini API key. fab-bench reads provider keys
from a `.env` file in the folder you run it from:

```bash
mkdir fab-bench-trial && cd fab-bench-trial
printf 'GEMINI_API_KEY=%s\n' "your-key" >.env
```

Any model in the shared registry works instead; items with images need a model whose
registry entry says `Vision: true`. After your first run, the registry is in
`custom_models.yaml` in this folder.

For walkthroughs 2 and 4, also install a current
[gcloud CLI](https://cloud.google.com/sdk/docs/install) and log in twice: once for
gcloud, and once for the application default credentials fab-bench uses to read and
write GCS.

```bash
gcloud components update
gcloud auth login
gcloud auth application-default login
export PROJECT_ID="the-playground-project"
export REGION="europe-west1"
export BUCKET="gs://the-playground-bucket"
export PREFIX="${BUCKET}/trials/${USER}"   # your own folder, so trials don't collide
```

## The inputs file

A benchmark hands fab-bench one file: one JSON object per line, one line per item, with
every prompt already rendered. fab-bench sends exactly what the file holds and never
builds a prompt. The example's file:

```json
{"benchmark_item_id": "mcq-1", "prompt": "Which word is a noun? A) run B) cat C) quickly", "system_prompt": "Answer with one letter."}
{"benchmark_item_id": "short-1", "prompt": "Name the noun in this sentence: The dog barked.", "system_prompt": "Answer with one word."}
{"benchmark_item_id": "count-1", "prompt": "How many sides does the shape in the picture have? Answer with a number.", "images": [{"uri": "images/triangle.png", "detail": "low"}]}
{"benchmark_item_id": "compare-1", "prompt": "Which shape has more sides, the first picture or the second? Answer with first or second.", "images": [{"uri": "images/triangle.png", "detail": "low"}, {"uri": "images/square.png", "detail": "high"}]}
```

| Key                 | Required | Meaning                                                                                                      |
| ------------------- | -------- | ------------------------------------------------------------------------------------------------------------ |
| `benchmark_item_id` | yes      | A string, unique in the file. Write numeric IDs as strings: `"1042"`, not `1042`.                            |
| `prompt`            | yes      | The user message, sent as it is. Must hold more than whitespace.                                             |
| `system_prompt`     | no       | The system message for this item. Leave it out, or write `null`, for none.                                   |
| `images`            | no       | A list of `{"uri", "detail"}`. They go into the item's one user message, in order, before the prompt's text. |

- An image's `uri` is a `gs://` URI, an absolute path, or a path relative to the inputs
  file. `detail` is `low` (FDLLM sends it at 512×512) or `high` (up to 2048 on the long
  side and 768 on the short side). Any file Pillow can open works.
- Any other key is refused, so a misspelled key fails instead of silently dropping part
  of an input. Answers stay with your benchmark: the example keeps them in
  `answers.jsonl`, and scoring joins them to results on `benchmark_item_id`.

`fab-bench check-inputs` checks a file and lists every problem, each with its line. It
also reads each image to check that it exists. The example passes:

```bash
fab-bench check-inputs "$RUNNER/examples/smoke/inputs.jsonl"
```

```text
Items: 4
Images: 2 files, used 3 times, all found
No problems in …/examples/smoke/inputs.jsonl.
```

The full specification is [INPUTS.md](INPUTS.md).

## Walkthrough 1: everything local

Store the example as an experiment in a local folder. `--copy-images` copies the images
into it, so the folder holds everything and can be uploaded later:

```bash
fab-bench create-experiment --inputs "$RUNNER/examples/smoke/inputs.jsonl" \
  --dest experiments/smoke-v1 --benchmark smoke --experiment-id smoke-v1 \
  --max-tokens 2048 --copy-images --dry-run
```

It prints the experiment's settings and its first rows. Run it again without `--dry-run`
to create it. Outside a git repository it warns that it can't record a git commit; that
is harmless.

Write a run config, `run.json`:

```json
{
  "experiment": "experiments/smoke-v1",
  "models": ["gemini-3.1-flash-lite"]
}
```

Check it, then run it:

```bash
fab-bench run --config run.json --dry-run
fab-bench run --config run.json
```

The dry run should print:

```text
Experiment: smoke-v1
Max tokens: 2048
Dry run: no model was called and nothing was written.
gemini-3.1-flash-lite: 4 planned, 0 reused, 4 to run
Total: 4 planned, 0 reused, 4 to run
Images: 2 to send, each fetched and checked against its SHA-256
```

What to check:

- [ ] The run ends with `Executed: 4`, `Reused: 0` and `Errors: 0`.

- [ ] `experiments/smoke-v1/` now also holds
  `responses/responses_gemini-3.1-flash-lite.jsonl`, with one row per item, and
  `models/gemini-3.1-flash-lite.json`, whose `runs` list has one entry with the versions
  it ran with.

- [ ] Each row repeats its input and has `"call_status": "success"`. Compare the outputs
  with the answers:

  ```bash
  python3 - "$RUNNER/examples/smoke/answers.jsonl" \
    experiments/smoke-v1/responses/responses_gemini-3.1-flash-lite.jsonl <<'EOF'
  import json, sys
  answers = {r["benchmark_item_id"]: r["answer"] for r in map(json.loads, open(sys.argv[1]))}
  for row in map(json.loads, open(sys.argv[2])):
      item = row["benchmark_item_id"]
      print(f"{item:10} {row['call_status']:8} {row['output'].strip()!r:12} expected {answers[item]!r}")
  EOF
  ```

  Small models get some answers wrong: both flash-lite models tend to answer `compare-1`
  with `first`. A wrong answer is the model's; what matters here is that every call
  succeeds.

- [ ] Running `fab-bench run --config run.json` again prints `Executed: 0` and
  `Reused: 4`: nothing is called twice.

Optionally, break things on purpose; each should stop with a message naming the problem:

- Add `"answer": "B"` to a line of a copy of the inputs file and create an experiment
  from it: refused, with the unknown key and the line.
- Edit a prompt in `experiments/smoke-v1/inputs.jsonl` and run: refused, because the
  inputs no longer match their hash. Undo the edit afterwards.
- Create the experiment again in `experiments/broken`, copy `square.png` over the
  triangle's copy, `experiments/broken/images/9de94300….png`, and run a config naming
  it: the run stops before any call, naming the image.

If a row has `"call_status": "error"`, its `error_type` and `error_message` say why. A
`stop_reason` of `MAX_TOKENS` or `length` means the answer ran out of tokens: raise the
limit with
`fab-bench raise-max-tokens --experiment experiments/smoke-v1 --max-tokens 8192` and
rerun with `--retry-errors`, which reruns only the error rows.

## Walkthrough 2: inputs and config on GCS

Here the inputs, images and finally the experiment and run config live on GCS, while the
models still run on your machine. Upload the example's inputs and images:

```bash
gcloud storage cp -r "$RUNNER/examples/smoke/inputs.jsonl" "$RUNNER/examples/smoke/images" \
  "${PREFIX}/smoke/"
```

**2a. Inputs on GCS, experiment local.** Relative image paths in an inputs file on GCS
resolve against its GCS folder:

```bash
fab-bench create-experiment --inputs "${PREFIX}/smoke/inputs.jsonl" \
  --dest experiments/smoke-gcs-inputs --benchmark smoke --max-tokens 2048
```

- [ ] `experiments/smoke-gcs-inputs/inputs.jsonl` references the images by their `gs://`
  URIs, and `experiment.json` records the inputs file's URI and GCS generation under
  `created.inputs_source`.
- [ ] A run config naming `experiments/smoke-gcs-inputs` runs as in walkthrough 1; the
  images are downloaded from GCS and checked before any call.
- [ ] Its rows have the same `result_id`s as walkthrough 1's: an image's location isn't
  part of a call's identity, only its file and detail level.

**2b. Experiment and run config on GCS.** Create the experiment on GCS:

```bash
fab-bench create-experiment --inputs "${PREFIX}/smoke/inputs.jsonl" \
  --dest "${PREFIX}/experiments/smoke-v1" --benchmark smoke --experiment-id smoke-v1 \
  --max-tokens 2048
```

Write `run-gcs.json`, with your own prefix spelled out:

```json
{
  "experiment": "gs://the-playground-bucket/trials/you/experiments/smoke-v1",
  "models": ["gemini-3.1-flash-lite"]
}
```

Upload it and run it from GCS. `upload-run-config` checks the config as a Cloud Run job
would read it and prints its URI, `run-configs/<sha256>.json` in the experiment's
bucket:

```bash
CONFIG_URI="$(fab-bench upload-run-config run-gcs.json)"
fab-bench run --config "${CONFIG_URI}" --dry-run
fab-bench run --config "${CONFIG_URI}"
```

- [ ] `gcloud storage ls -r "${PREFIX}/experiments/smoke-v1"` lists `experiment.json`,
  `inputs.jsonl`, `models/gemini-3.1-flash-lite.json` and
  `responses/responses_gemini-3.1-flash-lite.jsonl`.
- [ ] Your machine worked on a copy in `.fab-bench/experiments/`, and uploaded only the
  model's two files.
- [ ] Uploading the same config again prints the same URI, and running it again prints
  `Executed: 0`.
- [ ] A config on GCS can't point at local files: `fab-bench upload-run-config run.json`
  (walkthrough 1's, with a local experiment) is refused before anything is uploaded.

**2c. Move a local experiment to GCS.** Walkthrough 1's experiment, results included,
can continue on GCS:

```bash
fab-bench upload-experiment experiments/smoke-v1 "${PREFIX}/experiments/smoke-local"
```

- [ ] A run config naming `${PREFIX}/experiments/smoke-local` and
  `gemini-3.1-flash-lite` plans `4 reused, 0 to run`.

## Walkthrough 3: new models on an existing experiment

Adding models never changes an experiment's inputs or another model's files. Take
walkthrough 1's experiment, note its first model's files, and write `run-more.json`:

```bash
shasum experiments/smoke-v1/responses/* experiments/smoke-v1/models/* >before.txt
```

```json
{
  "experiment": "experiments/smoke-v1",
  "models": ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]
}
```

```bash
fab-bench run --config run-more.json --dry-run
fab-bench run --config run-more.json
```

- [ ] The dry run plans `gemini-3.1-flash-lite: 4 planned, 4 reused, 0 to run` and
  `gemini-3.5-flash-lite: 4 planned, 0 reused, 4 to run`.
- [ ] The run prints `Executed: 4` and `Reused: 4`.
- [ ] `shasum -c before.txt` reports every file as OK: the first model's results and
  record are untouched.
- [ ] `experiments/smoke-v1/` has one results file and one record per model.
- [ ] A config naming only the new model works the same way. Adding a model always means
  writing another run config.

If your `fab-bench` was updated between the two runs, the dry run also warns that the
experiment already has results from other runner, FDLLM or Pillow versions. The
experiment works on GCS in the same way; only the `experiment` value changes.

## Walkthrough 4: everything on Google Cloud

Here models run in the project's Cloud Run job, named `fab-bench` unless the admin says
otherwise. Your machine only creates the experiment, uploads the run config and starts
the job; Cloud Shell works as well.

Create a new experiment on GCS from the inputs uploaded in walkthrough 2:

```bash
fab-bench create-experiment --inputs "${PREFIX}/smoke/inputs.jsonl" \
  --dest "${PREFIX}/experiments/smoke-cloud-v1" --benchmark smoke \
  --experiment-id smoke-cloud-v1 --max-tokens 2048
```

Write `run-cloud.json` naming it, with `"models": ["gemini-3.1-flash-lite"]`, and launch
it. From a checkout of the runner, `deploy/launch-run.sh` does both steps:

```bash
PROJECT_ID="${PROJECT_ID}" "$RUNNER/deploy/launch-run.sh" run-cloud.json
```

or run them yourself:

```bash
CONFIG_URI="$(fab-bench upload-run-config run-cloud.json)"
gcloud run jobs execute fab-bench --project="${PROJECT_ID}" --region="${REGION}" \
  --args="run-cloud,--config,${CONFIG_URI}"
```

Follow the execution, and its logs:

```bash
gcloud run jobs executions list --job=fab-bench --project="${PROJECT_ID}" --region="${REGION}"
gcloud logging read 'resource.type="cloud_run_job" AND resource.labels.job_name="fab-bench"' \
  --project="${PROJECT_ID}" --freshness=1h --order=asc --format='value(textPayload)'
```

While it runs, a dry run on your machine shows its progress, as results are uploaded at
each checkpoint: `fab-bench run --config "${CONFIG_URI}" --dry-run`.

- [ ] The execution succeeds, and its log ends with `Executed: 4`, `Reused: 0` and
  `Errors: 0`.
- [ ] `fab-bench fetch-results --experiment "${PREFIX}/experiments/smoke-cloud-v1" --dest results/smoke-cloud-v1`
  downloads the inputs, the record and the results.
- [ ] In `results/smoke-cloud-v1/models/gemini-3.1-flash-lite.json`, the run's
  `environment` names the image it ran in and the runner's and FDLLM's commits, and
  `run_config_source` is the config's URI.
- [ ] Launching the same config again uploads nothing new, and the log shows
  `Executed: 0`.

**Adding models in a new run.** Write `run-cloud-more.json` naming the same experiment
with `"models": ["gemini-3.5-flash-lite"]`, and launch it the same way.

- [ ] Only the new model is called: the log shows `Executed: 4`.
- [ ] `gcloud storage ls -l "${PREFIX}/experiments/smoke-cloud-v1/responses/"` shows the
  first model's file with its old time, and the new model's file next to it.
- [ ] Two launches for different models on one experiment can run at the same time. Two
  launches for the same model at the same time can't both write: the one that uploads
  second stops with `GCS object changed since it was pulled`. Cloud Run then retries its
  task, which reuses the other's results, so both executions succeed and only the first
  attempt's log shows the error.

To try resuming, use an inputs file large enough to take a few minutes, set
`"chunk_size": 1` in the run config, cancel the execution part-way with
`gcloud run jobs executions cancel <execution>`, and launch the same config again: the
new execution reuses every result the first one uploaded.

## Your own benchmark

1. Write your benchmark's inputs file with its own code, following
   [the inputs file](#the-inputs-file). A plain script is enough; it doesn't need to
   import fab-bench:

   ```python
   import json

   with open("build/inputs.jsonl", "w", encoding="utf-8") as inputs:
       for item in items:  # your items, already selected
           row = {
               "benchmark_item_id": str(item["id"]),
               "prompt": render_prompt(item),  # your template
               "system_prompt": SYSTEM_PROMPT,  # or leave it out
               "images": [{"uri": uri, "detail": "high"} for uri in item.get("figures", [])],
           }
           inputs.write(json.dumps(row, ensure_ascii=False) + "\n")
   ```

1. Check it with `fab-bench check-inputs build/inputs.jsonl` and fix every problem it
   lists.

1. Create a small subset as its own experiment first, say 20 items, and run it as in
   walkthrough 1 or 4.

1. Then create the full experiment. Pick `--max-tokens` for your longest expected
   answer, or `none` to let FDLLM use what the model's context window leaves.

1. For long cloud runs, pin the model registry, so an edit to the shared file can't
   change call IDs between restarts. Upload a copy and name it in the run config:

   ```bash
   gcloud storage cp custom_models.yaml "${BUCKET}/registries/custom_models-$(date +%Y%m%d).yaml"
   shasum -a 256 custom_models.yaml
   ```

   ```json
   {
     "custom_models_uri": "gs://…/registries/custom_models-20261119.yaml",
     "custom_models_sha256": "<the hash shasum printed>"
   }
   ```

1. Score the rows' `output` against your answers, joined on `benchmark_item_id`. A
   better parser never needs new model calls.

## When something goes wrong

| Message                                                     | What to do                                                                                                 |
| ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `unknown keys ['answer']`                                   | Keep item data the model doesn't see out of the inputs file.                                               |
| `write the number 1042 as "1042"`                           | Item IDs are strings.                                                                                      |
| `Image … has changed since the experiment was created`      | The image's file changed after creation. Restore it, or create a new experiment.                           |
| `the registry doesn't mark … as taking them (Vision: true)` | Pick a model whose registry entry has `Vision: true`, or leave images out.                                 |
| `A run config on GCS … can't read the local experiment`     | Configs run from GCS need the experiment on GCS: `upload-experiment` it.                                   |
| `does not have api_key or Api_Key_Env_Var set`              | Put the model's key in `.env` in the current folder, or export it.                                         |
| `403` or `Permission denied` on `gs://…`                    | Ask the admin for `roles/storage.objectUser` on the bucket; check `gcloud auth application-default login`. |
| `PERMISSION_DENIED` on `run.jobs.runWithOverrides`          | Ask the admin for `roles/run.jobsExecutorWithOverrides` on the job.                                        |
| `changed since it was pulled`                               | Another job is running the same model on this experiment. Let it finish, then launch again.                |
