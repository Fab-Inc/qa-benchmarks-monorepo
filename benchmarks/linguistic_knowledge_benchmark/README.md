# Linguistic Knowledge Benchmark

A benchmark that measures how much an LLM *knows about the linguistics of a language* —
its phonics and orthography, morphology, syntax, vocabulary, and phonological awareness
— rather than its ability to converse in it.

**The medium of evaluation is English for every language.** Questions probing a specific
language are *written in English* (e.g. "What is the plural prefix for nouns describing
human beings in Luganda?"), not in the language under evaluation. Only the *subject* of
the questions changes from language to language; the prompt language stays English
throughout.

The goal is a **replicable pipeline across languages**. Every language plugs its own
*components* (source materials, question-domain and difficulty distributions, system
prompts, dataset) into a **shared, language-agnostic framework** — a common mapping of
question **domains** and **difficulty levels**. The framework stays fixed; only the
per-language components change.

This framing targets the **formal linguistic competence** of LLMs — getting the *form*
of a language right (spelling, morphology, grammar) — as distinct from **functional
linguistic competence**, using language to accomplish goals in the world. The two are
distinct cognitive skills.[^mahowald] We deliberately moved away from anchoring
difficulty to CEFR, which is built for second-language learners of (mostly European)
languages and leans toward functional competence.

\[^mahowald\]: Mahowald et al., *Dissociating language and thought in large language
models*. Formal linguistic competence (getting the form of language right) and
functional linguistic competence (using language to accomplish goals in the world) are
distinct cognitive skills.

The benchmark runs in **two stages**: first it collects model answers (`run-bench`),
then it scores them (`judge-bench`).

## Languages and their components

The shared framework (domain + difficulty-level mapping) is language-agnostic. Each
language supplies its own components, all evaluated through the same English-medium
prompts. Fill in a column as each language's components are finalized.

| Component                         | Luganda                                  | Swahili                                  | English |
| --------------------------------- | ---------------------------------------- | ---------------------------------------- | ------- |
| **Language key** (CLI `-l`)       | `luganda`                                | `swahili`                                | _TBD_   |
| **Dataset file** (`data/`)        | `linguistic_knowledge_eval_luganda.json` | `linguistic_knowledge_eval_swahili.json` | _TBD_   |
| **Question formats** (MCQ / open) | _TBD_                                    | _TBD_                                    | _TBD_   |
| **Status**                        | Active                                   | Active                                   | Planned |

Questions are written in English for every language (see above), so there is no
per-language "question language" column.

**Registering a language:** put its JSON dataset in [`data/`](data/) and add an entry to
the `DATASETS` dict in [`cmd.py`](src/linguistic_knowledge_benchmark/cmd.py) (language
key → dataset path and language name). The system prompts are English for every
language, so there is nothing else to configure.

## Dataset format

Datasets live in [`data/`](data/) as JSON arrays of question objects. Each question has
the following structure:

```json
{
  "question_id": 1,
  "domain": "D1",
  "subdomain": "metalanguage",
  "grade": "G1",
  "answer_type": "multiple_choice",
  "question": "Which pair of Latin letters is NOT used in writing standard Swahili?",
  "options": {
    "A": "f and h",
    "B": "q and x",
    "C": "c and j",
    "D": "v and z"
  },
  "expected_answers": ["B"],
  "linguistic_feature": "Letter inventory",
  "sources": [
    {
      "locator": "Swahili_Grammar_for_Introductory_and_Intermediate.pdf:Ch.2:The Alphabet",
      "tier": "other"
    }
  ],
  "grade_placement_evidence": "curriculum_anchored"
}
```

Short-form and true/false (TF) questions omit `options`. The `expected_answers` field is
always a list (multiple acceptable answers are supported for `short_form`).

There are two `answer_type`s in current datasets, each scored differently (see
[Scoring](#how-answers-are-scored)):

- `multiple_choice` (MCQ) — answer is a single letter `A`–`D`.
- `short_form` (open) — answer is 1–3 words.

> **Note:** this JSON schema is provisional and may change as the benchmark evolves. The
> judge ([`judge_bench.py`](src/linguistic_knowledge_benchmark/judge_bench.py)) supports
> three answer formats — MCQ, true/false (TF), and short-form — but current datasets use
> only MCQ and short-form. Any dataset must use one of the three per question.

## Key components

| Component                                                               | Purpose                                                                                                                                                                                     |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`run_bench.py`](src/linguistic_knowledge_benchmark/run_bench.py)       | **Stage 1.** Sends each question to a model (via `fdllm`) and saves raw answers. Holds the answer-type system prompts and the MCQ regex parser (`parse_answer`).                            |
| [`judge_bench.py`](src/linguistic_knowledge_benchmark/judge_bench.py)   | **Stage 2.** Scores raw answers — regex for MCQ/TF, an LLM judge for short-form — plus an optional LLM fallback for unparseable MCQ/TF rows, and accuracy summaries.                        |
| [`cmd.py`](src/linguistic_knowledge_benchmark/cmd.py)                   | Typer CLI exposing the `run-bench` and `judge-bench` commands. Loads `.env`, registers models, and holds the per-language `DATASETS` registry.                                              |
| [`shared_utils.py`](src/linguistic_knowledge_benchmark/shared_utils.py) | Paths and config: loads `EVAL_MODELS` from `list_models.yaml`, defines `BENCH_DIR` / `DATA_DIR`, lists rate-limited models. (The cache root `CACHE_BASE_DIR` is defined in `run_bench.py`.) |
| [`list_models.yaml`](list_models.yaml)                                  | The set of models to evaluate (key = model ID, value = display name).                                                                                                                       |
| [`custom_models.yaml`](../../custom_models.yaml)                        | `fdllm` provider/model registry: which API interface and `Api_Key_Env_Var` each model uses.                                                                                                 |
| [`.env`](.env)                                                          | Provider API keys (e.g. `ANTHROPIC_API_KEY`). Loaded automatically by the CLI.                                                                                                              |
| [`tests/`](tests/)                                                      | Unit tests for the CLI, runner, and judge.                                                                                                                                                  |

## Data and output folders

```text
benchmarks/linguistic_knowledge_benchmark/
  data/                              # input datasets (tracked in git)
    linguistic_knowledge_eval_luganda.json
    linguistic_knowledge_eval_swahili.json
  cache/                             # raw run artifacts (gitignored)
    <language>/                      # e.g. luganda/, swahili/
      runs/
        default/                     # outputs when no --run-id is given
          log.json                   # provenance: timestamp + git hash per model per stage
          answers/                   # stage 1: model answers, one CSV per model
            resps_<model>.csv
          answers_judged/            # stage 2: scored answers, one CSV per model
            resps_<model>_judged.csv
          answers_judged_fallback/   # stage 2 fallback (--judge-fallback only)
            resps_<model>_judged.csv
        <run-id>/                    # non-overwriting named or timestamped run
          log.json
          answers/
          answers_judged/
  results/                           # final per-language summaries (tracked in git)
    summary_linguistic-knowledge_<language>_<run-id>[_fallback].csv
```

**Filename convention for summaries:** fields are separated by `_`; spaces within a
field are replaced by `-`. The run ID is always present (defaulting to `default`), which
makes the filename machine-parseable into `{benchmark}_{language}_{run-id}[_{suffix}]`.

How outputs flow through the two stages:

1. **`run-bench` → `cache/<language>/runs/<run-id>/answers/`** — each model's raw
   answers are written to `resps_<model>.csv` (one row per question). `log.json` is
   updated with `answers_timestamp-utc` and `answers_git-hash` for that model (or the
   `answers_retry_*` variants when `--retry-errors` is used).
1. **`judge-bench` → `cache/<language>/runs/<run-id>/answers_judged/`** — those answers
   are scored and written to `resps_<model>_judged.csv` (the same rows plus `is_correct`
   / `justification`). With `--judge-fallback`, an extra `answers_judged_fallback/`
   subfolder is produced for rescued MC/TF rows. `log.json` is updated with
   `answers_judged_timestamp-utc` and `answers_judged_git-hash` (or the
   `answers_judged_fallback_*` variants).
1. **`judge-bench` → `results/`** — accuracy is aggregated across all judged models into
   `summary_linguistic-knowledge_<language>_<run-id>.csv`, with overall and
   per-answer-type accuracy. The summary also includes all 8 provenance fields from
   `log.json` so each row records exactly when and from which commit each model's
   answers were produced.

`cache/` holds reproducible run artifacts and is **gitignored**; `data/` and `results/`
are tracked. Delete `cache/` any time to force a clean re-run (or use `--force`).

## Setup

From the monorepo root, install dependencies:

```bash
uv sync --package linguistic-knowledge-benchmark
# or, for the whole workspace:
uv sync --all-packages
```

Add the API keys for the providers you want to evaluate to [`.env`](../../.env) file in
repo root (the CLI loads this file automatically):

```bash
# .env
ANTHROPIC_API_KEY="..."   # Claude models + default judge
OPENAI_API_KEY="..."      # GPT models
GEMINI_API_KEY="..."      # Gemini / Gemma models
```

Which env var a model needs is determined by its provider group in
[`custom_models.yaml`](../../custom_models.yaml) (`Api_Key_Env_Var`). A real environment
variable always takes precedence over the `.env` file.

## Running the benchmark

The CLI is invoked through `uv run`:

```bash
uv run --package linguistic-knowledge-benchmark linguistic-knowledge-benchmark <command> [options]
```

`--language` is **required** and accepts any registered language key (`luganda`,
`swahili` — see the [table above](#languages-and-their-components)).

### Stage 1 — collect answers (`run-bench`)

Query a model and cache its raw answers.

```bash
# One model on the Swahili dataset
uv run --package linguistic-knowledge-benchmark linguistic-knowledge-benchmark \
  run-bench --language swahili --model claude-haiku-4-5-20251001

# All models in list_models.yaml (omit --model)
uv run --package linguistic-knowledge-benchmark linguistic-knowledge-benchmark \
  run-bench --language luganda
```

Options:

- `-m, --model` — model ID (must be a key in `list_models.yaml`). Omit to run every
  model.
- `-l, --language` — registered language key (required).
- `-f, --force` — re-run even if cached answers exist.
- `--retry-errors` — retry only rows with `call_status=error` from an existing answers
  CSV.
- `--run-id` — write under `cache/<language>/runs/<run-id>/` instead of `runs/default/`.
  Must be alphanumeric and hyphens only (no underscores).
- `--timestamped-output` — generate a UTC timestamp run ID for a non-overwriting run.

Output: `cache/<language>/runs/default/answers/resps_<model>.csv` (or
`runs/<run-id>/answers/` when `--run-id` is set)

The stage-1 CSV keeps model output separate from infrastructure failures:

- `output` contains only the model's raw answer text.
- `call_status` is `success` or `error`.
- `error_type` and `error_message` are populated when the model call fails.
- `retry_count` counts how many times a row was rerun with `--retry-errors` (0 on the
  first successful run).

If a cached CSV contains any `call_status=error` rows, a later `run-bench` returns the
cached rows and prints a warning. Use `--retry-errors` to rerun only those failed rows.
Transient provider retries are handled by `fdllm`.

### Stage 2 — score answers (`judge-bench`)

Score the cached answers and write a per-model summary.

```bash
uv run --package linguistic-knowledge-benchmark linguistic-knowledge-benchmark \
  judge-bench --language luganda --model claude-haiku-4-5-20251001
```

Options:

- `-m, --model` — model whose answers to judge. Omit to judge all.
- `-l, --language` — registered language key (required).
- `-j, --judge-model` — judge model for short-form answers (default
  `claude-sonnet-4-5-20250929`).
- `-f, --force` — re-judge even if judged results exist.
- `--run-id` — read/write judged artifacts under `cache/<language>/runs/<run-id>/`. Must
  be alphanumeric and hyphens only (no underscores).
- `--judge-fallback` — re-evaluate MC/true-false rows the regex *couldn't* parse
  (`is_correct` is null) with the LLM judge. Reads from `answers_judged/`, writes to
  `answers_judged_fallback/`.

Outputs:

- Per-model judged CSV:
  `cache/<language>/runs/<run-id>/answers_judged/resps_<model>_judged.csv`
- Fallback CSV (with `--judge-fallback`):
  `cache/<language>/runs/<run-id>/answers_judged_fallback/resps_<model>_judged.csv`
- Summary across models: `results/summary_linguistic-knowledge_<language>_<run-id>.csv`
  (or `..._<run-id>_fallback.csv` with `--judge-fallback`)

### Typical end-to-end flow

```bash
# 1. Collect answers for all models
uv run --package linguistic-knowledge-benchmark linguistic-knowledge-benchmark run-bench -l luganda

# 2. Score them (regex + LLM judge)
uv run --package linguistic-knowledge-benchmark linguistic-knowledge-benchmark judge-bench -l luganda

# 3. (Optional) Rescue MC/TF answers the regex missed
uv run --package linguistic-knowledge-benchmark linguistic-knowledge-benchmark judge-bench -l luganda --judge-fallback
```

## How answers are scored

| Answer type       | Scoring method                                                                                                                                                   |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `multiple_choice` | Regex extracts the `A`–`D` letter (`parse_answer`) and compares to the expected letter. Unparseable answers are left `null` (unscored) until `--judge-fallback`. |
| `short_form`      | An LLM judge compares the answer to the accepted `expected_answers`, accepting only capitalization/hyphen/spacing variations — no synonyms or paraphrases.       |
| `true_false`      | Regex on the first word (`true`/`false`). Supported by the judge but not used in current datasets.                                                               |

Accuracy is computed only over **scored** rows; missing model outputs and still-unparsed
rows are reported as `unscored_count` rather than counted as wrong. When fallback
results exist, the summary uses the fallback verdict where available and the original
verdict otherwise.

Judged rows include provenance fields:

- `score_method` — `regex`, `llm_judge`, `llm_fallback`, or `unscored`.
- `parsed_answer` — the parsed MCQ/true-false answer when regex parsing succeeds.

This makes strict-format scores distinguishable from answers rescued by the LLM
fallback.

## Tests

```bash
# From the monorepo root
uv run pytest benchmarks/linguistic_knowledge_benchmark --cov
```
