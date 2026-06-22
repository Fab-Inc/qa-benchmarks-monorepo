"""Small runnable Kiswahili example for the decodability scaffold.

Run from the repository root with:

    uv run --frozen --package decodability python \
        packages/decodability/examples/kiswahili_example.py
"""

from __future__ import annotations

from functools import partial

import pandas as pd
from decodability import annotate_words, mean_score, weighted_average
from decodability.kiswahili import (
    KiswahiliStudentKnowledge,
    score_known_syllables_kiswahili,
    score_syllable_pattern_kiswahili,
    score_word_length_kiswahili,
    tokenize_kiswahili,
)

TEXT = "Juma ana baba na mama. Baba ana duka sokoni."

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
    },  # all vowels, CV syllables with b, k, m, n, and prenasalisation m_ and n_.
    max_word_length=5,
    known_syllable_patterns={"CV", "CVCV", "CCV"},
)


def build_dataframe() -> pd.DataFrame:
    return annotate_words(
        tokenize_kiswahili(TEXT),
        scoring_methods=[
            score_known_syllables_kiswahili,
            score_word_length_kiswahili,
            score_syllable_pattern_kiswahili,
        ],
        student_knowledge=STUDENT_KNOWLEDGE,
        aggregations={
            "mean_score": mean_score,
            "weighted_score": partial(
                weighted_average,
                weights={
                    "score_known_syllables_kiswahili": 0.5,
                    "score_word_length_kiswahili": 0.3,
                    "score_syllable_pattern_kiswahili": 0.2,
                },
            ),
        },
    )


def main() -> None:
    print(build_dataframe().to_string(index=False))


if __name__ == "__main__":
    main()
