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
    ├── benchmark_items.jsonl            (not in git)
    ├── benchmark_items_dev.jsonl        (not in git)
    ├── benchmark_items_summary.json               (not in git)
    └── benchmark_items_by_difficulty.png          (not in git)
```

> [!IMPORTANT]\
> **None of the data files are committed** — only this README and the blank workbooks in
> `templates/` are. Two reasons: this repo is to be open sourced, and part of the
> benchmark item set has to stay private. The whole folder is gitignored apart from
> those two exceptions.
>
> To get the data:
>
> - **`student_knowledge_profiles.jsonl`**: the canonical profile list, maintained by
>   the data team. Ask them for it — this is the single source of truth and is **not**
>   regenerated locally.
> - **Expert workbooks (`raw/`)**: ask the team for the filled-in workbook and drop it
>   in `raw/` yourself. Only needed if you are re-deriving profiles from scratch.
> - **The item files**: generate them locally by running
>   `scripts/build_benchmark_items.py` (see `scripts/README.md`).
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

This file is the **single source of truth** for the benchmark: 156 Kiswahili profiles,
each with a distinct `student_knowledge_id`. Every one is validated against the
language's model in `decodability` (`KiswahiliStudentKnowledge` for `sw`) at item-build
time, so an invalid grapheme, an uppercase whole word, or a cluster that overlaps a
known cluster pattern fails the build rather than the run.

For Kiswahili, `student_knowledge` carries four fields: `graphemes`, `clusters`,
`cluster_patterns` and `whole_words`. Clusters are taught in grade 2, so the early-stage
profiles legitimately carry empty `clusters` and `cluster_patterns` lists.

> [!WARNING]\
> If you are re-deriving profiles from a workbook, define the fields for
> `student_knowledge` in the `STUDENT_KNOWLEDGE_FIELDS` registry in
> `scripts/ingest_benchmark_data.py`. Refer to that registry for the authoritative field
> list per language.

**Example: an early grade-1 profile (no clusters taught yet)**

```json
{
  "student_knowledge_id": "tusome-ke-stage-01",
  "language_iso_code": "sw",
  "student_knowledge": {
    "graphemes": ["a", "m"],
    "clusters": [],
    "cluster_patterns": [],
    "whole_words": ["mama", "ama", "anaona", "na", "paka"]
  }
}
```

**Example: a grade-2 profile (clusters taught)**

```json
{
  "student_knowledge_id": "tusome-ke-g2-wk05",
  "language_iso_code": "sw",
  "student_knowledge": {
    "graphemes": ["a", "u", "i", "o", "e", "m", "k", "t", "n", "..."],
    "clusters": ["mb", "nd", "nz", "mk", "mt", "..."],
    "cluster_patterns": [],
    "whole_words": ["mama", "ama", "anaona", "na", "paka"]
  }
}
```

A profile may express cluster knowledge either as explicit `clusters` or as a
`cluster_patterns` entry (a generative rule such as `prenasalised`), but not both for
the same cluster — the language model rejects the overlap.

## `benchmark_items.jsonl` and `benchmark_items_dev.jsonl`

One JSON object per line. Each record pairs a **generation scenario** with a **student
knowledge profile** — the model is asked to write a text of that kind in which every
word is decodable for that student. There is no reference answer: the deterministic
checker in `packages/decodability` grades whatever the model writes.

- `benchmark_items.jsonl`: the full cross-product of every distinct student profile with
  every scenario in `src/phonics_benchmark/scenarios.yaml`
- `benchmark_items_dev.jsonl`: the same, over a smaller stratified sample of profiles,
  for fast iteration and CI

| Field                      | Type   | Description                                                               |
| -------------------------- | ------ | ------------------------------------------------------------------------- |
| `benchmark_item_id`        | string | Unique identifier, `<scenario_id>_<student_knowledge_id>`                 |
| `scenario_id`              | string | References a key in `src/phonics_benchmark/scenarios.yaml`                |
| `scenario`                 | string | The resolved TASK instruction injected into the prompt                    |
| `student_knowledge_id`     | string | References a `student_knowledge_id` in `student_knowledge_profiles.jsonl` |
| `language_iso_code`        | string | ISO 639-1 language code                                                   |
| `student_knowledge`        | object | Phonics units the student has been taught                                 |
| `difficulty_bucket`        | string | `very-early` / `early` / `mid` / `late`, from `n_graphemes_known`         |
| `n_graphemes_known`        | int    | How many graphemes the profile teaches                                    |
| `n_clusters_known`         | int    | How many explicit consonant clusters the profile teaches                  |
| `n_cluster_patterns_known` | int    | How many generative cluster patterns it teaches                           |

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
   "n_clusters_known": 0,
   "n_cluster_patterns_known": 0
}
```

Every canonical profile becomes items: all 156 ids are distinct and so are all 156
knowledge blobs, so there is nothing to deduplicate. 156 profiles x 4 scenarios = 624
items in the full set.

## Difficulty buckets

Buckets are assigned from the number of graphemes taught, with inclusive upper bounds
(`curation.DIFFICULTY_BUCKETS`). They matter because the writable vocabulary varies by
more than two orders of magnitude across them — measured against a 1,171-word reference
vocabulary:

| Bucket       | Graphemes | Profiles | Taught clusters | Corpus decodable (min / median / max) |
| ------------ | --------- | -------- | --------------- | ------------------------------------- |
| `very-early` | ≤10       | 17       | 0               | 0.3% / 5.2% / 12.0%                   |
| `early`      | 11–17     | 37       | 2               | 7.7% / 24.5% / 35.4%                  |
| `mid`        | 18–24     | 41       | 1               | 31.9% / 47.2% / 63.1%                 |
| `late`       | 25+       | 61       | 18              | 41.1% / 68.3% / 98.6%                 |

Cluster knowledge is the other axis and it is not captured by the bucket: it is taught
in grade 2, so it concentrates in `late`, and at a fixed 31 graphemes it swings the
readable vocabulary from 74.1% (`tusome-ke-stage-16`, grade-1 endpoint) to 98.6%
(`tusome-ke-g2-j10`). That is why every item records `n_clusters_known` and
`n_cluster_patterns_known` — so any metric can be split by it.

The `very-early` end is severe: `tusome-ke-stage-01` teaches two graphemes and five
sight words, leaving four usable words in the reference vocabulary. A 50-word text there
is necessarily repetitive, which the scenarios allow. Read the per-bucket breakdown
rather than only the pooled score.

## Adding a new dataset

1. If this is in a new language, create a subfolder using the full language name in
   lowercase (e.g. `data/english/`).
1. Obtain a canonical `student_knowledge_profiles.jsonl` for that language. Either the
   data team provides it directly, or you commit a blank data-entry template to
   `templates/`, put a filled-in copy in `raw/` (which stays local), and generate it
   with `scripts/ingest_benchmark_data.py`.
1. Register the language's student-knowledge fields in the `STUDENT_KNOWLEDGE_FIELDS`
   dict in `scripts/ingest_benchmark_data.py`.
1. Register the language's scorers and aggregation in the `DECODABILITY_CONFIGS` dict in
   `src/phonics_benchmark/labeling.py`.
1. Follow `scripts/README.md` to generate the benchmark item sets.
