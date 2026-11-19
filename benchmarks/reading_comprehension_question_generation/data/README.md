# Reading Comprehension Question Generation Data

Passages for the reading comprehension question generation benchmark, one workbook and
one folder per language.

```text
data/
└── english/
    ├── raw/                     (not in git)
    │   └── <workbook>.xlsx
    ├── templates/
    │   └── rcqg_texts_template.xlsx
    ├── texts.jsonl              (not in git)
    └── benchmark_items.jsonl    (not in git)
```

> [!IMPORTANT] ❗ \
> **Nothing here is committed except this README and the workbook in `templates/`.** No
> filled-in workbook, no `texts.jsonl`, and no real passage text goes into git. Copy the
> template, fill it in, and put your copy in `raw/` yourself, then generate the other
> two files with the scripts in `../scripts/` (see the package README).

## `templates/`

`rcqg_texts_template.xlsx` is the data-entry workbook: a `Field Definitions` sheet and a
`Texts` sheet with one column per field below. It is committed so everyone enters texts
in the same shape. It holds one example row (`ENG-G2-001`) to show the format. Ingestion
does not skip example rows, so delete or replace that row in your copy before you
ingest. Do not fill the committed template in place: work on a copy in `raw/`.

## Workbook

One workbook per language, with a sheet named `Texts` and one row per passage. Columns
are matched by header name, in any order. Extra columns are ignored.

### Required (used by the pipeline)

| Field               | What it holds                                                                     |
| ------------------- | --------------------------------------------------------------------------------- |
| `text_id`           | Unique id for the text, such as `ENG-G2-001`. It becomes the `benchmark_item_id`. |
| `text_content`      | The full passage, as the model will see it. Line breaks are kept.                 |
| `language_iso_code` | The language the text is written in (`en`, `sw`). One code per workbook.          |

A row with content but a blank value in any of these three is rejected. So are a
duplicate `text_id` and more than one distinct language code in one workbook.

### Descriptive (optional, stored as typed)

| Field                | What it holds                                                   |
| -------------------- | --------------------------------------------------------------- |
| `text_title`         | Short title for the passage                                     |
| `text_type`          | Narrative or informational                                      |
| `country_code`       | Country context the text is anchored to                         |
| `grade`              | Normalized grade level the text is intended for                 |
| `source`             | Corpus or process the text came from                            |
| `location`           | More specific place or region, if the source gives one          |
| `source_grade_label` | The grade or level exactly as the source labels it              |
| `is_target`          | Marks whether the passage is in the intended target context.    |
| `is_adaptation`      | Whether the source says the text is a translation or adaptation |
| `adaptation_source`  | The original title and language, if it is an adaptation         |
| `has_images`         | Whether the text has images                                     |

The workbook's own `Field Definitions` sheet gives the exact wording and the intended
values.

## `texts.jsonl`

Written by `scripts/ingest_texts.py`. One JSON object per text with the 14 fields above
as keys.

## `benchmark_items.jsonl`

Written by `scripts/build_benchmark_items.py` from `texts.jsonl`. One object per text
with exactly three keys: `benchmark_item_id` (the `text_id`), `text_content`, and
`language` (the language name used in the prompt; `en` becomes `English`, `sw` becomes
`Kiswahili`). This is the file a run reads. Rebuild it after `texts.jsonl` changes.
