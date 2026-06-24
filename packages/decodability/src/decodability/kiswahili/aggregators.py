"""Language-specific aggregators that combine Kiswahili scores into a final score.

Unlike the generic helpers in :mod:`decodability.annotate`, these encode Kiswahili
decoding rules (e.g. gating on grapheme knowledge) and reference specific scorers by
name.
"""

from collections.abc import Sequence

import pandas as pd

from decodability.kiswahili.scorers import (
    score_known_graphemes_kiswahili,
    score_sight_words_kiswahili,
    score_syllable_pattern_kiswahili,
    score_word_length_kiswahili,
)

SIGHT_WORD_GATE_COLUMN = score_sight_words_kiswahili.__name__
GRAPHEME_GATE_COLUMN = score_known_graphemes_kiswahili.__name__
SYLLABLE_PATTERN_COLUMN = score_syllable_pattern_kiswahili.__name__
WORD_LENGTH_COLUMN = score_word_length_kiswahili.__name__


def kiswahili_kenya_tusome_aggregator(
    df: pd.DataFrame, score_columns: Sequence[str]
) -> pd.Series:
    """decision tree for kiswahili score based on the Tusome Kenya curriculum."""
    final_scores = df[SIGHT_WORD_GATE_COLUMN]
    final_scores = final_scores.where(final_scores == 0, df[GRAPHEME_GATE_COLUMN])
    return final_scores


def kiswahili_complex_aggregator(
    df: pd.DataFrame, score_columns: Sequence[str]
) -> pd.Series:
    """decision tree for kiswahili score based on the Tusome Kenya curriculum."""

    def func(row: pd.Series) -> float:
        if row[SIGHT_WORD_GATE_COLUMN] == 1.0:
            return 1.0
        elif row[GRAPHEME_GATE_COLUMN] == 0.0:
            return 0.0
        elif row[SYLLABLE_PATTERN_COLUMN] == 0.0:
            return 0.0
        else:
            return row[WORD_LENGTH_COLUMN]

    return df.apply(func, axis=1)
