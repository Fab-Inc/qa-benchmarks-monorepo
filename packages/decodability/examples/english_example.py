"""Small runnable English example for the decodability scaffold.

Run from the repository root with:

    uv run --frozen --package decodability python \
        packages/decodability/examples/english_example.py
"""

from __future__ import annotations

import re
from functools import partial
from typing import Any

import pandas as pd
from decodability import annotate_words, mean_score, weighted_average

TEXT = "The cat sat on a thin mat."
WORD_RE = re.compile(r"[A-Za-z]+")

STUDENT_KNOWLEDGE = {
    "known_letters": set("acmnorst"),
    "max_easy_length": 4,
}


def known_letters_score(word: str, student_knowledge: dict[str, Any]) -> float:
    letters = [character for character in word.casefold() if character.isalpha()]
    known_letters = student_knowledge["known_letters"]
    return sum(letter in known_letters for letter in letters) / len(letters)


def short_word_score(word: str, student_knowledge: dict[str, Any]) -> float:
    if len(word) <= student_knowledge["max_easy_length"]:
        return 1.0
    return 0.5


def tokenize_english(text: str) -> list[str]:
    return WORD_RE.findall(text)


def build_dataframe() -> pd.DataFrame:
    return annotate_words(
        tokenize_english(TEXT),
        scoring_methods=[known_letters_score, short_word_score],
        student_knowledge=STUDENT_KNOWLEDGE,
        aggregations={
            "mean_score": mean_score,
            "weighted_score": partial(
                weighted_average,
                weights={
                    "known_letters_score": 0.7,
                    "short_word_score": 0.3,
                },
            ),
        },
    )


def main() -> None:
    print(build_dataframe().to_string(index=False))


if __name__ == "__main__":
    main()
