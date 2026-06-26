"""Language-specific aggregators that combine Kiswahili scores into a final score.

Unlike the generic helpers in :mod:`decodability.annotate`, these encode Kiswahili
decoding rules (e.g. gating on grapheme knowledge) and reference specific scorers by
name.
"""

from collections.abc import Mapping

from decodability.kiswahili.scorers import (
    score_known_graphemes_kiswahili,
    score_whole_words_kiswahili,
)

WHOLE_WORD = score_whole_words_kiswahili.__name__
GRAPHEMES = score_known_graphemes_kiswahili.__name__


def aggregate_scores_kenya_tusome(scores: Mapping[str, float]) -> float:
    """Decision tree for a Kiswahili score based on the Kenya Tusome curriculum.

    If the student knows the whole word as a sight word, it's fully decodable (1.0).
    Otherwise the score falls back to whether the student knows all its graphemes.
    """
    if scores[WHOLE_WORD] == 1.0:
        return 1.0
    return scores[GRAPHEMES]
