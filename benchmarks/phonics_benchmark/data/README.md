# Decodability Benchmark Data

Benchmark data for the decodability assessment, organised by language. Each language
gets its own subfolder (e.g. `kiswahili/`).

```text
data/
└── kiswahili/
    ├── raw/                                       (not in git)
    │   └── benchmark_data_tusome_ke.xlsx
    ├── templates/
    │   └── benchmark_data_template_kiswahili.xlsx
    ├── benchmark_items_smoke.jsonl
    ├── decodable_texts.jsonl                      (not in git)
    ├── student_knowledge_profiles.jsonl           (not in git)
    ├── benchmark_item_candidates.jsonl            (not in git)
    └── benchmark_items.jsonl                      (not in git)
```

> [!IMPORTANT]\
> **None of the data files are committed** — only this README, the workbooks in
> `templates/`, and `benchmark_items_smoke.jsonl` are. The smoke data is the same in the
> workbook and can be used for testing.
>
> To get the data:
>
> - **Expert workbooks (`raw/`)**: ask the team for the filled-in workbook and drop it
>   in `raw/` yourself.
> - **Everything else**: generate it locally by running the three scripts in
>   `../scripts/` (see `scripts/README.md`).
> - **Shared, validated item sets**: these live in GCS, not in git. A run config's
>   `items_path` can point straight at a `gs://...jsonl` URI.

## `raw/`

Holds the filled-in workbooks completed by language experts. These are the source data
from which `decodable_texts.jsonl` and `student_knowledge_profiles.jsonl` are generated.
Not committed — see the note above.

## `templates/`

Holds the blank Excel data-entry template(s) for a language. Each workbook has a
`Student Profiles` tab and a `Texts` tab, plus a `README` tab documenting which columns
are machine-read versus free-form expert context.

> [!NOTE]\
> `decodable_texts.jsonl` and `student_knowledge_profiles.jsonl` are programmatically
> generated from a filled-in workbook in `raw/` using `scripts/ingest_benchmark_data.py`
> — don't hand-edit those two files directly, re-run the ingestion script against the
> workbook instead.

## `decodable_texts.jsonl`

One JSON object per line. Each record is a short reading passage used as an assessment
item.

| Field               | Type   | Description                                         |
| ------------------- | ------ | --------------------------------------------------- |
| `text_id`           | string | Unique identifier for the passage (e.g. `"sw-001"`) |
| `language_iso_code` | string | ISO 639-1 language code (e.g. `"sw"` for Kiswahili) |
| `content`           | string | Full text of the passage                            |

**Example**

```json
{
  "text_id": "sw-001",
  "language_iso_code": "sw",
  "content": "Tatu ni mtoto. Tatu ana dada. ..."
}
```

## `student_knowledge_profiles.jsonl`

One JSON object per line. Each record describes what knowledge a student has been taught
at a given point in their instruction — used to determine which words in a passage are
decodable for that student.

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
    "graphemes": ["m", "b", "y"],
    "clusters": ["mb"],
    "cluster_patterns": ["prenasalised_m", "labialized_glide"]
    "whole_words": ["mama", "paka", "ana"]
  }
}
```

## `benchmark_item_candidates.jsonl` and `benchmark_items.jsonl`

One JSON object per line. Each record pairs a decodable text with a student knowledge
profile (text, student_knowledge), forming a single benchmark evaluation unit. Each line
also contains the correct list of non-decodable words, and a few computed columns that
helps with benchmark item curation.

- `benchmark_item_candidates.jsonl`: the full cross-product of
  `student_knowledge_profiles.jsonl` and `decodable_texts.jsonl` plus the label and
  analysis columns
- `benchmark_item.jsonl`: a filtered subset of above, on which we run the benchmark

| Field                  | Type   | Description                                                               |
| ---------------------- | ------ | ------------------------------------------------------------------------- |
| `benchmark_item_id`    | string | Unique identifier for the item                                            |
| `text_id`              | string | References a `text_id` in `decodable_texts.jsonl`                         |
| `student_knowledge_id` | string | References a `student_knowledge_id` in `student_knowledge_profiles.jsonl` |

- other fields from the `decodable_texts.jsonl`, `student_knowledge_profile.jsonl`,
  `nondecodable_words`, and computed fields.

**Example**

```json
{
   "benchmark_item_id": "tusome-ke-j3_SW-KE-R-S01",
   "text_id": "SW-KE-R-S01",
   "student_knowledge_id": "tusome-ke-j3",
   "language_iso_code": "sw",
   "content": "Mama anaona uma na maua. Paka anaona ua.",
   "student_knowledge": {
      "graphemes": ["a","u","i","o","e","m","k","t","n","w","l","h","s","b","y","z","g","d","j","r","p","f","v"],
      "whole_words": ["mama","ama","anaona","na","paka"]
   },
   "nondecodable_words": [],
   "num_nondecodable_words": 0,
   "num_total_words": 7,
   "nondecodable_ratio": 0.0
}
```

## Adding a new dataset

1. If this is in a new language, create a subfolder using the full language name in
   lowercase (e.g. `data/english/`).
1. Commit a blank data-entry template to `templates/`, put a filled-in copy in `raw/`
   (which stays local), then generate `decodable_texts.jsonl` and
   `student_knowledge_profiles.jsonl` by running `scripts/ingest_benchmark_data.py`
   against the filled-in workbook (see the `raw/` section above).
1. Register the language's student-knowledge fields in the `STUDENT_KNOWLEDGE_FIELDS`
   dict in `scripts/ingest_benchmark_data.py`.
1. Follow `scripts/README.md` to generate the benchmark dataset.
