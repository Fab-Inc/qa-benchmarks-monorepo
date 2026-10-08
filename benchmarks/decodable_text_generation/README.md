# Decodable Text Generation Benchmarks

Two sibling benchmarks that ask models to write decodable text:

- **child** — A combination of curriculum and jagged profiles. Task name:
  `generation-child`.
- **curriculum** — only curriculum-aligned profiles that carry `target_knowledge` are
  used; the prompt also asks the model to practise the target. Task name:
  `generation-curriculum`.

Both reuse the `phonics_benchmark` run harness (config, checkpointed runner, GCS sync) —
see that package's README for the shared run/config semantics. The generation-specific
pieces live here: prompt templates, scenarios, per-language decoding-rules wording, the
item schema, and the data pipeline.

## Pipeline

Profiles are pre-built JSONL artifacts. Find them in GCS (or ask Justin) and place in
`data/<benchmark>/<language>/` (`child_adapted` / `curriculum_aligned`).

The following command converts them into benchmark items with prompts:

```bash
uv run --package decodable-text-generation python \
    benchmarks/decodable_text_generation/scripts/build_generation_items.py \
    --language english --benchmark child
```

This writes `benchmark_items.jsonl` next to the profiles.

- `--prompt-name` selects a non-default template key from `prompts.yaml`
- `--scenarios` limits which scenario ids get built (e.g. `-s story`)

## Running

```bash
decodable-text-generation run-generation-child --config path/to/run.json
decodable-text-generation run-generation-curriculum --config path/to/run.json
```

The same config format and execution flags as `phonics-benchmark run-decodability` apply
(see `benchmarks/phonics_benchmark/README.md`); `prompt_name` defaults to
`decodable_text_generation` / `decodable_text_generation_curriculum`.

## Prompt system

Packaged YAML resources under `src/decodable_text_generation/`:

- `prompts.yaml` — the generation templates. `{decoding_rules}`, `{language}`,
  `{student_knowledge}`, `{scenario}` placeholders in all of them; the `*_curriculum`
  variants add `{target}`.
- `scenarios.yaml` — `scenario_id` → TASK text (`words`, `sentences`, `story`).
- `decoding_rules.yaml` — per-language DECODING RULES wording + `word_list_name`.

## Item schema

Each line of `benchmark_items.jsonl` (English example):

```json
{
  "benchmark_item_id": "<student_knowledge_id>_<scenario_id>",
  "student_knowledge_id": "ufli-en-l008",
  "scenario_id": "story",
  "language_iso_code": "en",
  "student_knowledge": {"gpcs": [["a", "æ"]], "sight_words": ["the"]},
  "prompt": "CONTEXT\n...",
  "is_jagged": false,
  "target_knowledge": {"gpcs": [["i", "ɪ"]], "sight_words": []}
}
```

`prompt` is baked at build time. `target_knowledge` is required on curriculum items and
absent/ignored for child items.
