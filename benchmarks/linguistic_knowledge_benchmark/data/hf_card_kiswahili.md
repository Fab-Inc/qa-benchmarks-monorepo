______________________________________________________________________

language:

- en size_categories:
- n\<1K tags:
- linguistics
- education
- benchmark
- language-knowledge pretty_name: Kiswahili Linguistic Knowledge Benchmark

______________________________________________________________________

# Kiswahili Linguistic Knowledge Benchmark

A 20-item public sample of the Kiswahili Linguistic Knowledge Benchmark, measuring how
much an LLM *knows about the linguistics of Kiswahili* (phonics/ orthography,
morphology, syntax, vocabulary, and phonological awareness), rather than its ability to
converse in the language. Questions are written in English; only the linguistic
*subject* of the questions is Kiswahili.

This is a curated sample, not the full item set: the full 135-item benchmark (70
multiple_choice, 65 short_form) is kept private to avoid leaking into future models'
training data. The full dataset is available on request. Items here were selected to
spread across question domain, answer format, and difficulty.

## Domain and grade definitions

Domain (fixed, language-agnostic -- every language's items are filed against the same 9
domains) and grade (the content level a fact is *first introduced* at, not the
question's difficulty) definitions, side by side:

| Domain | Covers                   |     | Grade            | Content level            |
| ------ | ------------------------ | --- | ---------------- | ------------------------ |
| D1     | Phonics/Orthography      |     | `G0`             | Reception / below Year 1 |
| D2     | Phonology                |     | `G1`             | Year 1                   |
| D3     | Nominal morphology       |     | `G2`             | Year 2                   |
| D4     | Verbal morphology        |     | `G3`             | Year 3                   |
| D5     | Syntax                   |     | `G4`             | Year 4                   |
| D6     | Semantics/Lexicon        |     | `G5`             | Year 5                   |
| D7     | Pragmatics/Discourse     |     | `beyond_primary` | Above Year 5             |
| D8     | Sociolinguistic/Register |     |                  |                          |
| D9     | Metalinguistic reasoning |     |                  |                          |

## Columns

| Column                     | Description                                                                                                                                                                                           |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `question_id`              | Question identifier in the full benchmark                                                                                                                                                             |
| `domain`                   | Domain code (`D1`-`D9`) -- see Domain and grade definitions above                                                                                                                                     |
| `subdomain`                | Free-text sub-topic within the domain                                                                                                                                                                 |
| `grade`                    | Grade code (`G0`-`G5`, or `beyond_primary`) -- see Domain and grade definitions above                                                                                                                 |
| `answer_type`              | `multiple_choice` (answer is a single letter A-D) or `short_form` (1-3 words)                                                                                                                         |
| `question`                 | Question text (MCQ options are a separate `options` field)                                                                                                                                            |
| `options`                  | `{"A": ..., "B": ..., "C": ..., "D": ...}`, present for `multiple_choice` only                                                                                                                        |
| `expected_answers`         | List of acceptable answers (short_form can have more than one)                                                                                                                                        |
| `linguistic_feature`       | The specific linguistic point the question targets                                                                                                                                                    |
| `svr_strand`               | Simple View of Reading strand the item serves: `decoding`, `language_comprehension`, or `outside_flr_construct`                                                                                       |
| `geeap_component`          | GEEAP core instructional skill an error in this item would corrupt: `systematic_phonics`, `phonological_awareness`, `oral_language`, `reading_fluency`, `reading_comprehension`, `writing`, or `none` |
| `sources`                  | Source materials the question/answer is grounded in                                                                                                                                                   |
| `grade_placement_evidence` | How the `grade` assignment was justified in the pipeline                                                                                                                                              |

## Source & methodology

Built by
[The Quality Assurance Facility for AI in Education](https://www.fab-ai.org/initiatives/ai-for-education/edtech-quality/quality-assurance).
