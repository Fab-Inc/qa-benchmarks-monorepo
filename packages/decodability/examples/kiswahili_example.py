"""Small runnable Kiswahili example for the decodability scaffold.

Run from the repository root with:

    uv run --frozen --package decodability python \
        packages/decodability/examples/kiswahili_example.py
"""

from __future__ import annotations

from functools import partial

import pandas as pd
from decodability import annotate_words, mean_score, rowwise, weighted_average
from decodability.kiswahili import (
    KiswahiliStudentKnowledge,
    aggregate_scores_kenya_tusome,
    score_known_graphemes_kiswahili,
    score_whole_words_kiswahili,
    tokenize_kiswahili,
)

TEXT = """Tulisikia mayowe tukiwa sebuleni. Sote tukatoka.
Kutazama tukaona moto ukiwaka sana. Waya wa stima uliwaka moto.
Sote tukawa na wasiwasi sana. Wazima moto wakatokea.
Mioyo yetu ikatulia, tukaelekea sebuleni tena."""

STUDENT_KNOWLEDGE = KiswahiliStudentKnowledge(
    graphemes={
        "a",
        "u",
        "i",
        "o",
        "e",
        "m",
        "k",
        "t",
        "n",
        "s",
        "b",
        "y",
        "z",
        "g",
        "d",
        "j",
        "r",
    },
    whole_words={"mama", "ama", "anaona", "na", "paka"},
)


def build_dataframe() -> pd.DataFrame:
    return annotate_words(
        tokenize_kiswahili(TEXT),
        scoring_methods=[
            score_known_graphemes_kiswahili,
            score_whole_words_kiswahili,
        ],
        student_knowledge=STUDENT_KNOWLEDGE,
        aggregations={
            "mean_score": mean_score,
            "weighted_score": partial(
                weighted_average,
                weights={
                    score_known_graphemes_kiswahili.__name__: 0.2,
                    score_whole_words_kiswahili.__name__: 0.8,
                },
            ),
            "kenya_tusome": rowwise(aggregate_scores_kenya_tusome),
        },
    )


def main() -> None:
    print(build_dataframe().to_string(index=False))


if __name__ == "__main__":
    main()
