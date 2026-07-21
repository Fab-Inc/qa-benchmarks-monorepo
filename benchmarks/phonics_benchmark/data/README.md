# Decodability Benchmark Data

Benchmark data for the decodability assessment, organised by language. Each language
gets its own subfolder (e.g. `kiswahili/`).

```
data/
└── kiswahili/
    ├── raw/
    │   └── benchmark_data_tusome_ke.xlsx
    ├── templates/
    │   └── benchmark_data_template_kiswahili.xlsx
    ├── decodable_texts.jsonl
    ├── student_knowledge_profiles.jsonl
    └── benchmark_items.jsonl
```

______________________________________________________________________

## `raw/`

Holds the filled-in workbooks completed by language experts. These are the source data
from which `decodable_texts.jsonl` and `student_knowledge_profiles.jsonl` are generated.

______________________________________________________________________

## `templates/`

Holds the blank Excel data-entry template(s) for a language. Each workbook has a
`Student Profiles` tab and a `Texts` tab, plus a `README` tab documenting which columns
are machine-read versus free-form expert context.

> [!NOTE]\
> `decodable_texts.jsonl` and `student_knowledge_profiles.jsonl` are programmatically
> generated from a filled-in workbook in `raw/` using
> `scripts/ingest_benchmark_data_template.py` — don't hand-edit those two files
> directly, re-run the ingestion script against the workbook instead.

______________________________________________________________________

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

______________________________________________________________________

## `student_knowledge_profiles.jsonl`

One JSON object per line. Each record describes what knowledge a student has been taught
at a given point in their instruction — used to determine which words in a passage are
decodable for that student.

| Field                  | Type   | Description                                             |
| ---------------------- | ------ | ------------------------------------------------------- |
| `student_knowledge_id` | string | Unique identifier for the profile (e.g. `"stu-sw-001"`) |
| `language_iso_code`    | string | ISO 639-1 language code                                 |
| `student_knowledge`    | object | Phonics units the student has been taught (see below)   |

> [!WARNING]\
> The contents of the `student_knowledge` object are language-dependent and still
> evolving. Each language defines its own set of fields (e.g. graphemes, whole words,
> syllable patterns) in the `STUDENT_KNOWLEDGE_FIELDS` registry in
> `scripts/ingest_benchmark_data_template.py`. Refer to that registry for the
> authoritative field list per language.

**Example (Kiswahili)**

```json
{
  "student_knowledge_id": "tusome-ke-stage-01",
  "language_iso_code": "sw",
  "student_knowledge": {
    "graphemes": ["a", "e", "i", "o", "u", "ba", "be", "bi"],
    "whole_words": ["mama", "paka", "ana"]
  }
}
```

______________________________________________________________________

## `benchmark_items.jsonl`

One JSON object per line. Each record pairs a decodable text with a student knowledge
profile (text, student_knowledge), forming a single benchmark evaluation unit. The full
set of items for a language is the cross-product of all texts and all student profiles.

> [!WARNING]\
> This file will ideally be programmatically created from an outer join of decodable
> texts and student knowledge profiles. The current benchmark_items.jsonl file serves
> demonstration purposes of what that would look like!

| Field                  | Type   | Description                                                               |
| ---------------------- | ------ | ------------------------------------------------------------------------- |
| `benchmark_item_id`    | string | Unique identifier for the item (e.g. `"bi-sw-001-stu-sw-001"`)            |
| `text_id`              | string | References a `text_id` in `decodable_texts.jsonl`                         |
| `student_knowledge_id` | string | References a `student_knowledge_id` in `student_knowledge_profiles.jsonl` |

**Example**

```json
{
  "benchmark_item_id": "bi-sw-001-stu-sw-001",
  "text_id": "sw-001",
  "student_knowledge_id": "stu-sw-001"
}
```

> Referential integrity (i.e. that `text_id` and `student_knowledge_id` exist in the
> other files) is validated by `validate_benchmark_dataset()` in `data_io`.

______________________________________________________________________

## Adding a new language

1. Create a subfolder using the full language name in lowercase (e.g. `data/english/`).
1. Add a blank data-entry template to `templates/` and a filled-in copy to `raw/`, then
   generate `decodable_texts.jsonl` and `student_knowledge_profiles.jsonl` by running
   `scripts/ingest_benchmark_data_template.py` against the filled-in workbook (see the
   `raw/` section above).
1. Register the language's student-knowledge fields in the `STUDENT_KNOWLEDGE_FIELDS`
   dict in `scripts/ingest_benchmark_data_template.py`.
1. Add `benchmark_items.jsonl` following the schema above, using the appropriate ISO
   639-1 code in `language_iso_code`.
1. The `get_data_dir()` utility in `data_io` resolves language folders by name, so the
   folder name must match the string passed to the data-loading functions.
