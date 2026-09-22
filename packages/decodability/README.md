# Decodability

Lightweight scaffolding for language-specific decodability tooling.

## Architecture

This package intentionally keeps the shared interface small. Each language can implement
its own tokenization, student-knowledge representation, and internal scoring logic.

The package splits into two layers: a small, language-agnostic **core** and one
**subpackage per language**.

### Shared core

`annotate.py` holds everything that is reusable across languages and knows nothing about
any specific language's rules:

- `annotate_words(...)`: the main entry point. Scores each word with every scoring
  method, then applies each aggregation, returning the dataframe described above.
- DataFrame-level aggregators with call signature `pd.DataFrame -> pd.Series`, which can
  be
  - generic vectorized aggregations (`mean_score`, `weighted_average`), or
  - custom, via `rowwise(row_func)`, where `row_func` is the
    language/curriculum-specific row-wise aggregation function over various scores.

### Per-language subpackage

Language-specific modules go under `decodability/<language>/`.

| Module                 | Responsibility                                                                                                          |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `definitions.py`       | Language orthography definitions and any external decoding-tool setup                                                   |
| `student_knowledge.py` | The student-knowledge pydantic model                                                                                    |
| `segment.py`           | Language-specific word segmentation                                                                                     |
| `analyse.py`           | Language-specific word analyses based on a student knowledge profile for decodability                                   |
| `extract_words.py`     | Language-specific word extractor.                                                                                       |
| `scorers.py`           | (Deprecated) Per-word scoring functions mapping `(word, student_knowledge)` to a score in range [0.0, 1.0]              |
| `aggregators.py`       | (Deprecated) Functions that combine a word's per-measure scores into one final score.                                   |
| `__init__.py`          | Exposes the `SCORING_METHODS` and `AGGREGATIONS` registries (see [Config Shape](#config-shape)) and the public exports. |

The dependency structure is as follows:

```mermaid
flowchart
  analyse.py --> student_knowledge.py
  analyse.py --> segment.py
  analyse.py --> definitions.py
  student_knowledge.py --> definitions.py
  segment.py --> definitions.py
  extract_words.py --> definitions.py
```

A language may add more modules where the tooling requires it. English extends the
shared shape with three:

- `extract_phonemes.py` — a `Phoneme` value type (IPA symbol + stress bit) and helpers
  that turn CMUdict pronunciations into normalized `Phoneme` sequences.
- `gpc_parser.py` — a constrained aligner that returns every valid GPC decomposition of
  a `(word, phoneme sequence)` pair under a supplied inventory.
- `phoneme_overrides.py` — a lookup table for words where CMUdict's pronunciation should
  be replaced (dialect fixes, curriculum choices, OOV additions).

## Kiswahili Scoring

Kiswahili decodability follows Kenya's Tusome curriculum as well as expert-validated,
binary definition of decodability. Three binary scorers feed the final score:

| Scorer                                        | Returns 1.0 when                                                                                                                                    |
| --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `score_known_graphemes_kiswahili`             | every grapheme in the word has been taught                                                                                                          |
| `score_known_clusters_and_patterns_kiswahili` | every cluster in the word is known, directly or by pattern, or if the word contains no consonant clusters (no cluster knowledge required to decode) |
| `score_whole_words_kiswahili`                 | the word has been taught as a whole word                                                                                                            |

`aggregate_scores_kenya_tusome` combines them as a decision tree: a known whole word is
decodable regardless of anything else; otherwise an untaught grapheme _or_ an unknown
cluster makes the word non-decodable.

All three scorers must be listed in `scoring_methods`.

### Student knowledge

`KiswahiliStudentKnowledge` is defined by the following fields. We do not consider
uppercase letters as separate graphemes, so we enforce all lowercase for the fields.

- `graphemes`: taught graphemes, validated against the Kiswahili grapheme inventory.
- `clusters`: individually taught clusters, for example `"st"` or `"mbw"`. Every
  grapheme inside a listed cluster must also appear in `graphemes`.
- `cluster_patterns`: `ClusterPattern` codes such as `"prenasalised"` or
  `"labialised_glide"`, each standing for a whole family of clusters (see
  `VALID_CLUSTERS` in `definitions.py`). A cluster already covered by a taught pattern
  must not be repeated in `clusters`.
- `whole_words`: sight words.

## English Scoring

English decodability follows the UFLI Foundations scope and sequence with a binary
definition of decodability. Two binary scorers feed the final score:

| Scorer                      | Returns 1.0 when                                                                |
| --------------------------- | ------------------------------------------------------------------------------- |
| `score_known_gpcs_english`  | at least one valid parse of the word uses only GPCs the student has been taught |
| `score_sight_words_english` | the word (case-insensitive) has been taught as a sight word                     |

`aggregate_scores_ufli_curriculum` (registered as `ufli_curriculum_aggregator`) combines
them as a decision tree: a known sight word is decodable regardless of anything else;
otherwise the final score falls back to the GPC score.

Both scorers must be listed in `scoring_methods`.

Under the GPC scorer, reduced-vowel matches (parser-synthesized `REDUCED_*` GPCs from
unstressed positions) count as taught whenever the student knows _any_ GPC for the same
grapheme letter — vowel reduction is a universal spoken-English process, not a separate
correspondence.

### Student knowledge

`EnglishStudentKnowledge` is defined by the following fields.

- `gpcs`: taught grapheme-phoneme correspondences as `(grapheme, IPA)` tuples, validated
  against the UFLI inventory. The phoneme side is a single joined IPA string, so a
  multi-phoneme GPC reads as `("x", "ks")` or `("qu", "kw")`.
- `sight_words`: sight words, lowercased on ingest for case-insensitive matching.

Pronunciations come from CMUdict, with per-word overrides available via
`CMUDICT_OVERRIDES` in `phoneme_overrides.py` — used to force a specific CMU variant,
supply an OOV pronunciation, or correct one that doesn't match the curriculum.

## Examples

Two runnable examples ship with the package. Run either from the repository root.

### Kiswahili

Scoring based on Kenya's Tusome curriculum.

```bash
uv run --frozen --package decodability python \
  packages/decodability/scripts/kiswahili_example.py
```

### English

Scoring based on UFLI Foundations GPCs with CMUdict-derived pronunciations. The
`EnglishStudentKnowledge` set uses the `(grapheme, IPA)` shape — e.g. `("x", "ks")` —
and sight words are matched case-insensitively.

```bash
uv run --frozen --package decodability python \
  packages/decodability/scripts/english_example.py
```

## Config Shape

The runtime API expects Python callables. Config loading should be a thin layer that
translates stable config names into those callables.

Every scorer named in `scoring_methods` becomes a score column, so a `weighted_average`
aggregation needs a weight for each of them.

Example JSON-like config:

```json
{
  "language": "kiswahili",
  "scoring_methods": [
    "score_known_graphemes_kiswahili",
    "score_known_clusters_and_patterns_kiswahili",
    "score_whole_words_kiswahili"
  ],
  "aggregations": {
    "kenya_tusome_score": {
      "name": "kiswahili_kenya_tusome_aggregator"
    },
    "mean_score": {
      "name": "mean_score"
    },
    "weighted_score": {
      "name": "weighted_average",
      "params": {
        "weights": {
          "score_known_graphemes_kiswahili": 0.5,
          "score_known_clusters_and_patterns_kiswahili": 0.2,
          "score_whole_words_kiswahili": 0.3
        }
      }
    }
  }
}
```
