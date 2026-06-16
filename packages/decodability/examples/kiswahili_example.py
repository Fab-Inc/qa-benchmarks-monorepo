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
from decodability.kiswahili import (
    KiswahiliStudentKnowledge,
    score_known_syllables,
    score_word_length,
    score_syllable_pattern,
)


TEXT = "Juma ana baba na mama. Baba ana duka sokoni."
WORD_RE = re.compile(r"[A-Za-z']+")

STUDENT_KNOWLEDGE = KiswahiliStudentKnowledge(
    known_syllables={
        "a",
        "e",
        "i",
        "o",
        "u",
        "m_",
        "n_",
        "ba",
        "be",
        "bi",
        "bo",
        "bu",
        "ka",
        "ke",
        "ki",
        "ko",
        "ku",
        "ma",
        "me",
        "mi",
        "mo",
        "mu",
        "na",
        "ne",
        "ni",
        "no",
        "nu",
    },
    max_word_length=5,
    known_syllable_patterns={"CV", "CVCV", "CCV"},
)


def tokenize_swahili(text: str) -> list[str]:
    return WORD_RE.findall(text)


def build_dataframe() -> pd.DataFrame:
    return annotate_words(
        tokenize_swahili(TEXT),
        scoring_methods=[
            score_known_syllables,
            score_word_length,
            score_syllable_pattern,
        ],
        student_knowledge=STUDENT_KNOWLEDGE,
        aggregations={
            "mean_score": mean_score,
            "weighted_score": partial(
                weighted_average,
                weights={
                    "score_known_syllables": 0.5,
                    "score_word_length": 0.3,
                    "score_syllable_pattern": 0.2,
                },
            ),
        },
    )


def main() -> None:
    print(build_dataframe().to_string(index=False))


if __name__ == "__main__":
    main()
