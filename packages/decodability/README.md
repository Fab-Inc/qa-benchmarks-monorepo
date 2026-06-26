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

| Module           | Responsibility                                                                                                          |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `config.py`      | Language constants and any external decoding-tool setup                                                                 |
| `models.py`      | The student-knowledge pydantic model                                                                                    |
| `scorers.py`     | Per-word scoring functions mapping `(word, student_knowledge)` to a score in range [0.0, 1.0]                           |
| `aggregators.py` | Functions that combine a word's per-measure scores into one final score.                                                |
| `tokenizer.py`   | Language-specific word tokenizer.                                                                                       |
| `__init__.py`    | Exposes the `SCORING_METHODS` and `AGGREGATIONS` registries (see [Config Shape](#config-shape)) and the public exports. |

The dependency structure is as follows:

```mermaid
flowchart
  models.py --for data validation--> config.py
  scorers.py --for scoring logic--> config.py
  tokenizer.py --for data validation*--> config.py
  aggregators.py --> scorers.py
```

_\*This linkage currently doesn't exist for Kiswahili but theoretically we may use a
language-wide config to extract valid tokens._

## Kiswahili Example

There is a runnable example at `packages/decodability/examples/kiswhaili_example.py`. It
uses decodability scoring based on Kenya's Tusome curriculum.

Run it from the repository root:

```bash
uv run --frozen --package decodability python \
  packages/decodability/examples/kiswahili_example.py
```

## Config Shape

The runtime API expects Python callables. Config loading should be a thin layer that
translates stable config names into those callables.

Example JSON-like config:

```json
{
  "language": "example_kiswahili",
  "scoring_methods": [
    "score_known_graphemes_kiswahili",
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
          "score_whole_words_kiswahili": 0.5
        }
      }
    }
  }
}
```
