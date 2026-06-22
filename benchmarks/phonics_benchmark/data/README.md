# Decodability Benchmark Data

Benchmark data for the decodability assessment, organised by language. Each language
gets its own subfolder (e.g. `kiswahili/`, `english/`). A subfolder without data files
is a placeholder for a language not yet populated.

```
data/
├── kiswahili/
│   ├── decodable_texts.jsonl
│   ├── student_knowledge_profiles.jsonl
│   └── benchmark_items.jsonl
└── english/                    # placeholder — no data files yet
```

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
| `taught_knowledge`     | object | Phonics units the student has been taught (see below)   |

### `taught_knowledge` fields

> [!WARNING]\
> Student knowledge representation is currently evolving and fields in the taught
> knowledge object are subject to change.

| Field                     | Type     | Description                                                   |
| ------------------------- | -------- | ------------------------------------------------------------- |
| `known_syllables`         | string[] | Syllables the student has been taught (vowels and CV/CCV etc) |
| `known_syllable_patterns` | string[] | Syllable structure patterns the student can decode            |
| `max_word_length`         | number   | Maximum word length (in characters) the student can decode    |

**Example**

```json
{
  "student_knowledge_id": "stu-sw-002",
  "language_iso_code": "sw",
  "taught_knowledge": {
    "known_syllables": ["a", "e", "i", "o", "u", "ba", "be", "bi", "bo", "bu",
                        "ma", "me", "mi", "mo", "mu", "m_", "n_", "st"],
    "known_syllable_patterns": ["CV", "CVCV", "CCV"],
    "max_word_length": 7
  }
}
```

______________________________________________________________________

## `benchmark_items.jsonl`

One JSON object per line. Each record pairs a decodable text with a student knowledge
profile (text, student_knowledge), forming a single benchmark evaluation unit. The full
set of items for a language is the cross-product of all texts and all student profiles.

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
1. Add `decodable_texts.jsonl`, `student_knowledge_profiles.jsonl`, and
   `benchmark_items.jsonl` following the schemas above, using the appropriate ISO 639-1
   code in `language_iso_code`.
1. The `get_data_dir()` utility in `data_io` resolves language folders by name, so the
   folder name must match the string passed to the data-loading functions.
