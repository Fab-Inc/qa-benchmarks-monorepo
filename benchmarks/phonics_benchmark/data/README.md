# Decodable-Text-Generation Benchmark Data

Benchmark data for the decodable-text-generation assessment, organised by language. Each
language gets its own subfolder (e.g. `kiswahili/`).

```text
data/
└── kiswahili/
    ├── raw/                                        (not in git)
    │   └── benchmark_data_tusome_ke.xlsx
    ├── templates/
    │   └── benchmark_data_template_kiswahili.xlsx
    ├── student_knowledge_profiles.jsonl            (not in git)
    ├── benchmark_items_generation.jsonl            (not in git)
    ├── benchmark_items_generation_dev.jsonl        (not in git)
    ├── generation_items_summary.json               (not in git)
    └── generation_items_by_difficulty.png          (not in git)
```

> [!IMPORTANT]\
> **None of the data files are committed** — only this README and the blank workbooks in
> `templates/` are. Two reasons: this repo is to be open sourced, and part of the
> benchmark item set has to stay private. The whole folder is gitignored apart from
> those two exceptions.
>
> To get the data:
>
> - **Expert workbooks (`raw/`)**: ask the team for the filled-in workbook and drop it
>   in `raw/` yourself.
> - **Everything else**: generate it locally by running the scripts in `../scripts/`
>   (see `scripts/README.md`). Everything below `raw/` is reproducible from the
>   workbook.
> - **Shared, validated item sets**: these live in GCS, not in git. A run config's
>   `items_path` can point straight at a `gs://...jsonl` URI.

## `raw/`

Holds the filled-in workbooks completed by language experts. These are the source data
from which `student_knowledge_profiles.jsonl` is generated. Not committed — see the note
above.

## `templates/`

Holds the blank Excel data-entry template(s) for a language. Each workbook has a
`Student Profiles` tab and a `Texts` tab, plus a `README` tab documenting which columns
are machine-read versus free-form expert context.

> [!NOTE]\
> The `Texts` tab is no longer read by this benchmark. The model now *writes* the text
> rather than being given one, so only the `Student Profiles` tab feeds the item set.
> The tab is left in the template because the workbooks are shared with other work.

> [!NOTE]\
> `student_knowledge_profiles.jsonl` is programmatically generated from a filled-in
> workbook in `raw/` using `scripts/ingest_benchmark_data.py` — don't hand-edit it
> directly, re-run the ingestion script against the workbook instead.

## `student_knowledge_profiles.jsonl`

One JSON object per line. Each record describes what knowledge a student has been taught
at a given point in their instruction — used to determine which words are decodable for
that student.

| Field                  | Type   | Description                                           |
| ---------------------- | ------ | ----------------------------------------------------- |
| `student_knowledge_id` | string | Unique identifier for the profile                     |
| `language_iso_code`    | string | ISO 639-1 language code                               |
| `student_knowledge`    | object | Phonics units the student has been taught (see below) |

> [!WARNING]\
> Define the fields for `student_knowledge` in the `STUDENT_KNOWLEDGE_FIELDS` registry
> in `scripts/ingest_benchmark_data.py`. Refer to that registry for the authoritative
> field list per language.

**Example (Kiswahili)**

```json
{
  "student_knowledge_id": "tusome-ke-stage-01",
  "language_iso_code": "sw",
  "student_knowledge": {
    "graphemes": ["a", "e", "i", "o", "u", "b", "k", "m"],
    "whole_words": ["mama", "paka", "ana"]
  }
}
```

## `benchmark_items_generation.jsonl` and `benchmark_items_generation_dev.jsonl`

One JSON object per line. Each record pairs a **generation scenario** with a **student
knowledge profile** — the model is asked to write a text of that kind in which every
word is decodable for that student. There is no reference answer: the deterministic
checker in `packages/decodability` grades whatever the model writes.

- `benchmark_items_generation.jsonl`: the full cross-product of every distinct student
  profile with every scenario in `src/phonics_benchmark/scenarios.yaml`
- `benchmark_items_generation_dev.jsonl`: the same, over a smaller stratified sample of
  profiles, for fast iteration and CI

| Field                      | Type   | Description                                                                |
| -------------------------- | ------ | -------------------------------------------------------------------------- |
| `benchmark_item_id`        | string | Unique identifier, `<scenario_id>_<student_knowledge_id>`                  |
| `scenario_id`              | string | References a key in `src/phonics_benchmark/scenarios.yaml`                 |
| `scenario`                 | string | The resolved TASK instruction injected into the prompt                     |
| `student_knowledge_id`     | string | References a `student_knowledge_id` in `student_knowledge_profiles.jsonl`  |
| `language_iso_code`        | string | ISO 639-1 language code                                                    |
| `student_knowledge`        | object | Phonics units the student has been taught                                  |
| `difficulty_bucket`        | string | `very-early` / `early` / `mid` / `late`, from `n_graphemes_known`          |
| `n_graphemes_known`        | int    | How many graphemes the profile teaches                                     |
| `cluster_knowledge_source` | string | `dataset` or `placeholder_all_patterns` — see the caveat in `../README.md` |

**Example**

```json
{
   "benchmark_item_id": "story_tusome-ke-j3",
   "scenario_id": "story",
   "scenario": "Write a short story of about 50 words for this student to read. ...",
   "student_knowledge_id": "tusome-ke-j3",
   "language_iso_code": "sw",
   "student_knowledge": {
      "graphemes": ["a","u","i","o","e","m","k","t","n","w","l","h","s","b","y","z","g","d","j","r","p","f","v"],
      "whole_words": ["mama","ama","anaona","na","paka"],
      "cluster_patterns": ["prenasalised", "labialised_glide", "palatalised_glide"]
   },
   "difficulty_bucket": "mid",
   "n_graphemes_known": 23,
   "cluster_knowledge_source": "placeholder_all_patterns"
}
```

Profiles are deduplicated on the *content* of `student_knowledge`, not on
`student_knowledge_id`: the dataset reuses one knowledge blob across several ids, and
running the same prompt under each label would buy no information while skewing the
difficulty mix.

## Difficulty buckets

Buckets are assigned from the number of graphemes taught, with inclusive upper bounds
(`curation.DIFFICULTY_BUCKETS`). They matter because the writable vocabulary varies by
more than an order of magnitude across them — measured against the 1,171-word corpus
vocabulary, with placeholder cluster knowledge applied:

| Bucket       | Graphemes | Profiles | Corpus decodable (min / median / max) |
| ------------ | --------- | -------- | ------------------------------------- |
| `very-early` | ≤10       | 12       | 1.8% / 7.3% / 13.4%                   |
| `early`      | 11–17     | 35       | 13.0% / 29.8% / 46.2%                 |
| `mid`        | 18–24     | 39       | 40.1% / 60.9% / 82.7%                 |
| `late`       | 25+       | 46       | 54.7% / 84.8% / 96.8%                 |

The `very-early` bucket sits close to an impossibility floor for 50-word prose, so
expect low scores there and read the per-bucket breakdown rather than only the pooled
score.

## Adding a new dataset

1. If this is in a new language, create a subfolder using the full language name in
   lowercase (e.g. `data/english/`).
1. Commit a blank data-entry template to `templates/`, put a filled-in copy in `raw/`
   (which stays local), then generate `student_knowledge_profiles.jsonl` by running
   `scripts/ingest_benchmark_data.py` against the filled-in workbook (see the `raw/`
   section above).
1. Register the language's student-knowledge fields in the `STUDENT_KNOWLEDGE_FIELDS`
   dict in `scripts/ingest_benchmark_data.py`.
1. Register the language's scorers and aggregation in the `DECODABILITY_CONFIGS` dict in
   `src/phonics_benchmark/labeling.py`.
1. Follow `scripts/README.md` to generate the benchmark item sets.
