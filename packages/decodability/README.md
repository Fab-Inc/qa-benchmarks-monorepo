# Decodability

Lightweight scaffolding for language-specific decodability tooling.

This package intentionally keeps the shared interface small. Each language can implement
its own tokenization, student-knowledge representation, and internal scoring logic. The
shared expectation is just:

```python
def scorer(word: str, student_knowledge) -> float:
    ...
```

Scorers should return a score between `0.0` and `1.0`. Language annotators return a
pandas dataframe with one row per word, one column per scoring method, and one or more
aggregation columns. By default, the aggregation is an equal-weight row-wise mean called
`mean_score`.

## English Example

There is a runnable toy example at `packages/decodability/examples/english_example.py`.
It uses arbitrary English scoring functions only to demonstrate the scaffold.

Run it from the repository root:

```bash
uv run --frozen --package decodability python \
  packages/decodability/examples/english_example.py
```

```python
import re
from functools import partial

from decodability import annotate_words, mean_score, weighted_average

text = "The cat sat on a thin mat."
student_knowledge = {"known_letters": set("acmnorst"), "max_easy_length": 4}


def known_letters_score(word, student_knowledge):
    letters = [character for character in word.casefold() if character.isalpha()]
    known_letters = student_knowledge["known_letters"]
    return sum(letter in known_letters for letter in letters) / len(letters)


def short_word_score(word, student_knowledge):
    if len(word) <= student_knowledge["max_easy_length"]:
        return 1.0
    return 0.5


words = re.findall(r"[A-Za-z]+", text)

df = annotate_words(
    words,
    scoring_methods=[known_letters_score, short_word_score],
    student_knowledge=student_knowledge,
)

print(df)
print(df["mean_score"].mean())

weighted_df = annotate_words(
    words,
    scoring_methods=[known_letters_score, short_word_score],
    student_knowledge=student_knowledge,
    aggregations={
        "weighted_score": partial(
            weighted_average,
            weights={
                "known_letters_score": 0.7,
                "short_word_score": 0.3,
            },
        ),
    },
)
```

## Config Shape

The runtime API expects Python callables. Config loading should be a thin layer that
translates stable config names into those callables.

Example JSON-like config:

```json
{
  "language": "example_english",
  "scoring_methods": [
    "known_letters_score",
    "short_word_score"
  ],
  "aggregations": {
    "mean_score": {
      "name": "mean_score"
    },
    "weighted_score": {
      "name": "weighted_average",
      "params": {
        "weights": {
          "known_letters_score": 0.7,
          "short_word_score": 0.3
        }
      }
    }
  }
}
```

A language package is expected to expose registries like this:

```python
from decodability import mean_score, weighted_average

SCORING_METHODS = {
    "known_letters_score": known_letters_score,
    "short_word_score": short_word_score,
}

AGGREGATIONS = {
    "mean_score": mean_score,
    "weighted_average": weighted_average,
}
```
