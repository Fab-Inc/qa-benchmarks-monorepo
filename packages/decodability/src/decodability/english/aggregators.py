"""Language-specific aggregators that combine English scores into a final score.

Unlike the generic helpers in :mod:`decodability.annotate`, these encode English
decoding rules (e.g. gating on GPC knowledge) and reference specific scorers by
name.
"""

from collections.abc import Mapping

from decodability.english.scorers import (
    score_known_gpcs_english,
    score_sight_words_english,
)

SIGHT_WORD = score_sight_words_english.__name__
GPCS = score_known_gpcs_english.__name__


def aggregate_scores_ufli_curriculum(scores: Mapping[str, float]) -> float:
    """Decision tree for an English score based on the UFLI curriculum.

    If the student knows the word as a sight word, it's fully decodable (1.0).
    Otherwise the score falls back to whether the student knows all its GPCs.
    """
    if scores[SIGHT_WORD] == 1.0:
        return 1.0
    return scores[GPCS]
