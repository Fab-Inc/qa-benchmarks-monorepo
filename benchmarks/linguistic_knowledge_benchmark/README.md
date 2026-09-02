# Linguistic Knowledge Benchmark

A benchmark that measures how much an LLM *knows about the linguistics of a language* —
its phonics and orthography, morphology, syntax, vocabulary, and phonological awareness
— rather than its ability to converse in it.

**The medium of evaluation is English for every language.** Questions probing a specific
language are *written in English* (e.g. "What is the plural prefix for nouns describing
human beings in Kiswahili?"), not in the language under evaluation. Only the *subject*
of the questions changes from language to language; the prompt language stays English
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

The benchmark runs in **two stages**: first it collects model responses (`run-bench`),
then it scores them (`judge-bench`).

## Languages and their components

The shared framework (domain + difficulty-level mapping) is language-agnostic. Each
language supplies its own components, all evaluated through the same English-medium
prompts. Fill in a column as each language's components are finalized.

| Component                         | Kiswahili                                  | English                                  |
| --------------------------------- | ------------------------------------------ | ---------------------------------------- |
| **Language key** (CLI `-l`)       | `kiswahili`                                | `english`                                |
| **Dataset file** (`data/`)        | `linguistic_knowledge_eval_kiswahili.json` | `linguistic_knowledge_eval_english.json` |
| **Question formats** (MCQ / open) | `multiple_choice`, `short_form`            | `multiple_choice`, `short_form`          |
| **Status**                        | Active                                     | Active                                   |

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

Short-form questions omit `options`. The `expected_answers` field is always a list
(multiple acceptable answers are supported for `short_form`).

There are two `answer_type`s in current datasets, each scored differently (see
[Scoring](#how-responses-are-scored)):

- `multiple_choice` (MCQ) — answer is a single letter `A`–`D`.
- `short_form` (open) — answer is 1–3 words.

> **Note:** this JSON schema is provisional and may change as the benchmark evolves. The
> judge ([`judge_bench.py`](src/linguistic_knowledge_benchmark/judge_bench.py)) supports
> two answer formats — MCQ and short-form. Any dataset must use one of the two per
> question.

## Key components

| Component                                                               | Purpose                                                                                                                                                                                                                      |
| ----------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`run_bench.py`](src/linguistic_knowledge_benchmark/run_bench.py)       | **Stage 1.** Sends each question to a model (via `fdllm`) and saves raw responses. Holds the answer-type system prompts.                                                                                                     |
| [`judge_bench.py`](src/linguistic_knowledge_benchmark/judge_bench.py)   | **Stage 2.** Scores raw responses — regex for MCQ (`parse_answer_mcq`), an LLM judge (or ensemble of judges, majority vote) for short-form — plus an optional LLM fallback for unparseable MCQ rows, and accuracy summaries. |
| [`cmd.py`](src/linguistic_knowledge_benchmark/cmd.py)                   | Typer CLI exposing the `run-bench` and `judge-bench` commands. Loads `.env`, registers models, and holds the per-language `DATASETS` registry.                                                                               |
| [`shared_utils.py`](src/linguistic_knowledge_benchmark/shared_utils.py) | Paths and config: loads `EVAL_MODELS` from `list_models.yaml`, defines `BENCH_DIR` / `DATA_DIR` / `CACHE_BASE_DIR`, the shared `experiment_dir()` helper, and lists rate-limited models.                                     |
| [`list_models.yaml`](list_models.yaml)                                  | The set of models to evaluate (key = model ID, value = display name).                                                                                                                                                        |
| [`custom_models.yaml`](../../custom_models.yaml)                        | `fdllm` provider/model registry: which API interface and `Api_Key_Env_Var` each model uses.                                                                                                                                  |
| [`.env`](.env)                                                          | Provider API keys (e.g. `ANTHROPIC_API_KEY`). Loaded automatically by the CLI.                                                                                                                                               |
| [`tests/`](tests/)                                                      | Unit tests for the CLI, runner, and judge.                                                                                                                                                                                   |

## Data and output folders

```text
benchmarks/linguistic_knowledge_benchmark/
  data/                                 # input datasets (tracked in git)
    linguistic_knowledge_eval_kiswahili.json
    linguistic_knowledge_eval_english.json
  cache/                                # raw experiment artifacts (gitignored)
    <language>/                         # e.g. kiswahili/, english/
      experiments/
        default/                        # outputs when no --experiment-id is given
          log.json                      # provenance: timestamp + git hash per model per stage
          responses/                    # stage 1: model responses, one CSV per model
            responses_<model>.csv
          responses_judged/             # stage 2: scored responses, one CSV per model
            responses_judged_<model>.csv
          responses_judged_fallback/    # stage 2 fallback (--judge-fallback only)
            responses_judged_<model>.csv
        <experiment-id>/                # non-overwriting named or timestamped experiment
          log.json
          responses/
          responses_judged/
  results/                              # final per-language summaries (tracked in git)
    summary_linguistic-knowledge_<language>_<experiment-id>[_fallback].csv
```

**Filename convention for summaries:** fields are separated by `_`; spaces within a
field are replaced by `-`. The experiment ID is always present (defaulting to
`default`), which makes the filename machine-parseable into
`{benchmark}_{language}_{experiment-id}[_{suffix}]`.

How outputs flow through the two stages:

1. **`run-bench` → `cache/<language>/experiments/<experiment-id>/responses/`** — each
   model's raw responses are written to `responses_<model>.csv` (one row per question).
   `log.json` is updated with `responses_timestamp-utc` and `responses_git-hash` for
   that model (or the `responses_retry_*` variants when `--retry-errors` is used).
1. **`judge-bench` → `cache/<language>/experiments/<experiment-id>/responses_judged/`**
   — those responses are scored and written to `responses_judged_<model>.csv` (the same
   rows plus `is_correct` / `justification`). With `--judge-fallback`, an extra
   `responses_judged_fallback/` subfolder is produced for rescued MCQ rows. `log.json`
   is updated with `responses_judged_timestamp-utc` and `responses_judged_git-hash` (or
   the `responses_judged_fallback_*` variants).
1. **`judge-bench` → `results/`** — accuracy is aggregated across all judged models into
   `summary_linguistic-knowledge_<language>_<experiment-id>.csv`, with overall and
   per-answer-type accuracy. The summary also includes all 10 provenance fields from
   `log.json` (including `llm_judges`/`llm_judges_fallback`, the judge model(s) used) so
   each row records exactly when, from which commit, and by which judge(s) each model's
   responses were produced.

For a clean experiment, supply `--experiment-id` or `--timestamped-output` to write to a
fresh, non-overwriting directory. `cache/` holds experiment artifacts and is
**gitignored** (`data/` and `results/` are tracked) — delete it any time, or use
`--force`, if you'd rather force a clean re-run in place.

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

`--language` is **required** and accepts any registered language key (`kiswahili`,
`english` — see the [table above](#languages-and-their-components)).

### Stage 1 — collect responses (`run-bench`)

Query a model and cache its raw responses.

```bash
# One model on the Kiswahili dataset
uv run --package linguistic-knowledge-benchmark linguistic-knowledge-benchmark \
  run-bench --language kiswahili --model claude-haiku-4-5-20251001

# All models in list_models.yaml (omit --model)
uv run --package linguistic-knowledge-benchmark linguistic-knowledge-benchmark \
  run-bench --language english
```

Options:

- `-m, --model` — model ID (must be a key in `list_models.yaml`). Omit to run every
  model.
- `-l, --language` — registered language key (required).
- `-f, --force` — re-run even if cached responses exist.
- `--retry-errors` — retry only rows with `call_status=error` from an existing responses
  CSV.
- `--experiment-id` — write under `cache/<language>/experiments/<experiment-id>/`
  instead of `experiments/default/`. Must be alphanumeric and hyphens only (no
  underscores).
- `--timestamped-output` — generate a UTC timestamp experiment ID for a non-overwriting
  experiment.

Output: `cache/<language>/experiments/default/responses/responses_<model>.csv` (or
`experiments/<experiment-id>/responses/` when `--experiment-id` is set)

The stage-1 CSV keeps model output separate from infrastructure failures:

- `output` contains only the model's raw response text.
- `call_status` is `success` or `error`.
- `error_type` and `error_message` are populated when the model call fails.
- `retry_count` counts how many times a row was rerun with `--retry-errors` (0 on the
  first successful experiment).

If a cached CSV contains any `call_status=error` rows, a later `run-bench` returns the
cached rows and prints a warning. Use `--retry-errors` to rerun only those failed rows.
Transient provider retries are handled by `fdllm`.

### Stage 2 — score responses (`judge-bench`)

Score the cached responses and write a per-model summary.

```bash
uv run --package linguistic-knowledge-benchmark linguistic-knowledge-benchmark \
  judge-bench --language kiswahili --model claude-haiku-4-5-20251001
```

Options:

- `-m, --model` — model whose responses to judge. Omit to judge all.
- `-l, --language` — registered language key (required).
- `-j, --judge-model` — judge model to use for short-form responses; repeat for an
  explicit ensemble (e.g. `-j claude-sonnet-5 -j gpt-5.6-terra`). Overrides
  `--num-judges`. Defaults to the first `--num-judges` of `DEFAULT_JUDGE_MODELS`
  (`claude-sonnet-5`, `gpt-5.6-terra`, `gemini-3.6-flash`).
- `-n, --num-judges` — how many of `DEFAULT_JUDGE_MODELS` to use as independent judges
  when `--judge-model` isn't given. 1 (default) = single judge; up to 3 for an ensemble
  with majority-vote scoring plus agreement columns.
- `-f, --force` — re-judge even if judged results exist.
- `--experiment-id` — read/write judged artifacts under
  `cache/<language>/experiments/<experiment-id>/`. Must be alphanumeric and hyphens only
  (no underscores).
- `--judge-fallback` — re-evaluate MCQ rows the regex *couldn't* parse (`is_correct` is
  null) with the LLM judge. Reads from `responses_judged/`, writes to
  `responses_judged_fallback/`.

With more than one judge (`--num-judges` > 1 or multiple `--judge-model`), short-form
`is_correct`/`justification` reflect the majority verdict (ties are left unscored) and
the primary judge's rationale; `judge_agreement`/`judge_disagreement_flag` summarize
consensus, and each judge's own verdict lands in
`is_correct_<judge>`/`justification_<judge>`. Re-judging a model with a cached CSV
that's missing a requested judge's column (e.g. it was previously judged with a
different judge or a smaller ensemble) errors out asking for `--force` rather than
silently mixing judges.

Outputs:

- Per-model judged CSV:
  `cache/<language>/experiments/<experiment-id>/responses_judged/responses_judged_<model>.csv`
- Fallback CSV (with `--judge-fallback`):
  `cache/<language>/experiments/<experiment-id>/responses_judged_fallback/responses_judged_<model>.csv`
- Summary across models:
  `results/summary_linguistic-knowledge_<language>_<experiment-id>.csv` (or
  `..._<experiment-id>_fallback.csv` with `--judge-fallback`)
- Pairwise judge agreement (only with more than one judge):
  `results/summary_linguistic-knowledge_<language>_<experiment-id>[_fallback]_judge_agreement.csv`

### Typical end-to-end flow

```bash
# 1. Collect responses for all models
uv run --package linguistic-knowledge-benchmark linguistic-knowledge-benchmark run-bench -l kiswahili

# 2. Score them (regex + LLM judge)
uv run --package linguistic-knowledge-benchmark linguistic-knowledge-benchmark judge-bench -l kiswahili

# 3. (Optional) Rescue MCQ responses the regex missed
uv run --package linguistic-knowledge-benchmark linguistic-knowledge-benchmark judge-bench -l kiswahili --judge-fallback
```

## How responses are scored

| Answer type       | Scoring method                                                                                                                                                         |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `multiple_choice` | Regex extracts the `A`–`D` letter (`parse_answer_mcq`) and compares to the expected letter. Unparseable responses are left `null` (unscored) until `--judge-fallback`. |
| `short_form`      | An LLM judge compares the response to the accepted `expected_answers`, accepting only capitalization/hyphen/spacing variations — no synonyms or paraphrases.           |

Accuracy is computed only over **scored** rows; missing model outputs and still-unparsed
rows are reported as `unscored_count` rather than counted as wrong. When fallback
results exist, the summary uses the fallback verdict where available and the original
verdict otherwise.

Judged rows include provenance fields:

- `score_method` — `regex`, `llm_judge`, `llm_fallback`, or `unscored`.
- `parsed_answer` — the parsed MCQ response when regex parsing succeeds.

This makes strict-format scores distinguishable from responses rescued by the LLM
fallback.

## Tests

```bash
# From the monorepo root
uv run pytest benchmarks/linguistic_knowledge_benchmark --cov
```
