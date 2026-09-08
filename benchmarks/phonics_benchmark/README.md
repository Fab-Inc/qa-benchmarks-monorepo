# Phonics Benchmark

Config-driven benchmark harness for phonics experiments.

The benchmark measures whether a model can **write a decodable text**: given a student's
`StudentKnowledge` profile and a scenario, the model must produce a short text in which
every single word is decodable for that student. The model's text is then run through
the deterministic checker in [`packages/decodability`](../../packages/decodability),
which counts the decodable and non-decodable words. There is no reference answer — the
checker *is* the ground truth.

For an end-to-end walkthrough of config and direct runs, cache reuse, error retries, and
adding models, see the [CLI tutorial](docs/cli-tutorial.md).

For building the container, deploying the Cloud Run Job, and launching cloud runs, see
the [Cloud Run deployment guide](../../deploy/README.md).

## Scoring

Each item is scored on the share of its words the student could actually read:

```text
tokens        = extract_words(generated_text)   # every occurrence, not unique words
r             = non_decodable_tokens / total_tokens
graded_score  = (1 - r) ** 20                   # REFERENCE_LENGTH = 20
```

`graded_score` reads as *"the probability that a randomly drawn 20-word passage from
this text is fully decodable"* — a strict pass rate normalised to a reference sentence
length.

Normalising is what makes the metric measure the model rather than the text length.
Scoring raw error counts instead — a fixed penalty per bad word, or a strict
all-or-nothing pass — makes an identical model score 0.92 on a 15-word text and 0.19 on
a 200-word one, and rewards writing as little as possible. Reporting the raw
decodable-word rate has the opposite problem: it compresses every model into 0.80–1.00.
For per-word error rates of 0.3% to 20%, `graded_score` spreads models across 0.96 down
to 0.02.

Two scoring edge cases are deliberate:

- An **empty text** scores 0. Writing nothing is not a way to avoid non-decodable words.
- A response **missing its `<text>` tags** scores 0 as a *format failure*, and is
  neither dropped nor graded as-is. Grading the raw response would charge the model for
  its own preamble ("Sure! Here is a story…") as non-decodable words in the target
  language; dropping the row would let a model dodge hard items by malformatting.
  `format_failure_rate` is reported separately so a formatting problem stays
  distinguishable from poor phonics.

Note that for texts shorter than the 20-word reference length the score is high variance
(with 5 words, one bad word gives `(4/5)**20 = 0.01`). Text length is **not** penalised,
but `n_words`, `n_unique_words`, `unique_word_ratio` and `under_reference_length` are
all recorded so under-length or degenerately repetitive responses can be audited.

`scripts/report_on_benchmark_run.py` emits per model:

| Metric                                                                  | Meaning                                                                        |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `graded_score`                                                          | The headline score above, averaged over items                                  |
| `strict_pass_rate`                                                      | Share of texts with *zero* non-decodable words                                 |
| `word_decodable_rate`                                                   | Pooled token-level rate; `type_decodable_rate` is its unique-word variant      |
| `nondecodable_per_100_words`                                            | Error density, in the style of a word error rate                               |
| `format_failure_rate`                                                   | Share of responses that omitted the `<text>` tags                              |
| `pooled_n_fail_<scorer>`                                                | Which scorer rejected each failure, e.g. untaught grapheme vs untaught cluster |
| `mean_n_words`, `mean_unique_word_ratio`, `under_reference_length_rate` | Length/diversity diagnostics, never scored                                     |

All of these are also broken out by student difficulty bucket and by scenario.

## Known caveat: placeholder cluster knowledge

No profile in the Kiswahili dataset carries `clusters` or `cluster_patterns`. Under the
Kenya-Tusome aggregator that makes *any* word containing a consonant cluster
non-decodable for *every* student: for the most advanced profile (31 graphemes, the
whole inventory) only 4 corpus word types fail on graphemes while 299 fail purely on
clusters, putting everyday words like `mwalimu`, `ng'ombe` and `ndege` out of reach for
a student who has been taught everything.

`curation.apply_placeholder_cluster_knowledge` therefore marks all 11 native cluster
patterns as known and leaves `clusters` empty, so loan clusters (`bl`, `st`, `br`) stay
untaught. Beginners are not inflated by this — the grapheme scorer still gates them:

| profile              | graphemes | corpus decodable, as-is | with placeholder |
| -------------------- | --------- | ----------------------- | ---------------- |
| `tusome-ke-stage-02` | 4         | 1.6%                    | 1.8%             |
| `tusome-ke-j93`      | 18        | 31.9%                   | 40.1%            |
| `tusome-ke-g2-wk13`  | 31        | 74.1%                   | 96.8%            |

This is a stand-in for real per-stage cluster data, tracked as a follow-up with the data
team. Every item and run config records `cluster_knowledge_source` so results stay
traceable to it.

## Run the benchmark

```bash
phonics-benchmark run-generation --config path/to/run.json
```

Two configs are checked in: `configs/sw-generation-dev.json` (120 items — 30 stratified
profiles x 4 scenarios) for fast iteration, and `configs/sw-generation-full.json` (528
items — all 132 distinct profiles x 4 scenarios) for headline numbers.

The same command can also be driven directly from CLI options:

```bash
phonics-benchmark run-generation \
  --run-id generation-smoke-test \
  --language sw \
  --models gpt-5-mini-2025-08-07-medium,claude-sonnet-4-5-20250929 \
  --items-path path/to/benchmark-items.jsonl \
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
non-overwriting run. The command records the task as `decodable-text-generation`; task
selection is not a config option.

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

The config format is JSON. Relative paths in a config file are resolved from the config
file's directory. Relative paths in CLI options are resolved from the current working
directory.

```json
{
  "run_id": "generation-smoke-test",
  "language": "sw",
  "items_path": "path/to/benchmark-items.jsonl",
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

## Prompts and scenarios

The single prompt template lives in `src/phonics_benchmark/prompts.yaml` and states the
decoding rules explicitly — without them the benchmark would measure whether a model can
guess our definition of "decodable" rather than whether it can write to it. It injects
the language display name, the student knowledge block, and the scenario.

The four scenarios live in `src/phonics_benchmark/scenarios.yaml`, each stating its own
~50-word target: `story`, `informational`, `dialogue`, and `sentences`. Adding a
scenario means adding an entry there and rebuilding the item sets; nothing in the code
enumerates them.

The student knowledge block is rendered by the language's own pydantic model in
`decodability` (its `__str__` *is* the prompt text), so the block's shape follows the
language automatically.

## Items

The runner accepts completed JSONL artifacts selected through `items_path` in config or
`--items-path` with direct CLI options. It does not join or compose benchmark inputs.
Each line must be an object containing a non-empty `benchmark_item_id`, a non-empty
`scenario_id`, a non-whitespace `scenario`, and object-valued `student_knowledge`.
Additional dataset-owned fields (`difficulty_bucket`, `student_knowledge_id`,
`n_graphemes_known`, `cluster_knowledge_source`) are accepted and ignored by the runner;
the report script re-joins them from the same file, which the recorded `items_sha256`
proves is unchanged. The selected language registry validates `student_knowledge`,
provides the display name, and controls its string representation in the shared prompt.

Build the item sets with:

```bash
uv run --package phonics-benchmark python \
  benchmarks/phonics_benchmark/scripts/build_generation_benchmark_items.py \
  --language kiswahili \
  --profiles-file phonics_datasets_kiswahili-with-clusters_benchmark_items.jsonl
```

Profiles are deduplicated on the *content* of `student_knowledge`, not on
`student_knowledge_id`: the dataset reuses one knowledge blob across several ids, and
running the same prompt under each label would buy no information while skewing the
difficulty mix.

## Run outputs

Each run writes:

- `results/<run_id>/config.json`: latest config snapshot with the exact raw config,
  resolved paths, exact-byte `items_sha256`, run-level code git hash and dirty-worktree
  state, and sanitized model configs from FDLLM's registered model state.
- `results/<run_id>/responses/responses_<model-file-id>.jsonl`: one result per item for
  each model. The file ID is the model name when it is already filename-safe, or a
  sanitized hash-suffixed form otherwise.
- `results/<run_id>/log.json`: per-model execution provenance such as timestamp, git
  hash, and dirty-worktree state.

Each result row includes `result_id`, `run_id`, `config_id`, `model_call_id`,
`config_path`, `config_snapshot_path`, `model`, `benchmark_item_id`, `prompt`, raw
`output`, and call status fields. Successful model calls also include `max_tokens`,
completion-token count, QFB `timestamp_utc`, and `latency_seconds`. Error rows omit
timing fields. `model_call_id` is the cache identity for the response. Cached rows may
be reused across provenance-only changes because `config_id` and `model_call_id` remain
stable. Result rows identify the model by configured model name; the corresponding model
config is saved in the latest config snapshot, keyed by that name.

The runner stores only raw model results and call metadata; it does not extract or score
them. Scoring is a separate step:

```bash
uv run --package phonics-benchmark python \
  benchmarks/phonics_benchmark/scripts/report_on_benchmark_run.py \
  --run-id sw-generation-dev --inspect-sample-size 20
```

The report reads the run's language and item set from the run's own `config.json`
snapshot rather than from flags, so a run is always scored against the exact items it
was executed on, and writes `results_with_scoring.csv`, `summary_by_model.csv`,
`summary_by_difficulty.csv`, `summary_by_scenario.csv`, two plots, and (with
`--inspect-sample-size`) `sampled_items_for_inspection.csv`.

The runner saves recognized provider SDK API and transport failures propagated by FDLLM
as error rows. It also saves an absent or empty model response as an error row. Other
exceptions, including FDLLM/model setup, response processing, and result persistence
errors, propagate and stop the affected run instead of being converted into call errors.
Rows whose provider call failed are excluded from scoring and counted per model; format
failures are scored (as zero), not excluded.

## Language support

Language is a single ISO 639-1 config string (`"language": "sw"`) used as the key into
`decodability.DECODABILITY_REGISTRY`. It is validated at prompt-build time, which
happens for every item before any model call, so a bad key fails fast.

Kiswahili is the mature language. English is registered in `decodability` but has no
student profile dataset, so it cannot be run yet; the benchmark's own code paths carry
no language-specific branches, so adding English needs a dataset plus a
`labeling.DECODABILITY_CONFIGS` entry rather than code changes here.
