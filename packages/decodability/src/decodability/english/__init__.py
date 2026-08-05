from decodability import mean_score, rowwise, weighted_average
from decodability.english.aggregators import (
    aggregate_scores_upl_curriculum,
)
from decodability.english.extract_words import extract_words_english
from decodability.english.models import EnglishStudentKnowledge
from decodability.english.scorers import (
    score_known_gpcs_english,
    score_whole_words_english,
)

SCORING_METHODS = {
    "score_known_gpcs_english": score_known_gpcs_english,
    "score_whole_words_english": score_whole_words_english,
}

AGGREGATIONS = {
    "upl_curriculum_aggregator": rowwise(aggregate_scores_upl_curriculum),
    "mean_score": mean_score,
    "weighted_average": weighted_average,
}

__all__ = [
    "SCORING_METHODS",
    "AGGREGATIONS",
    "aggregate_scores_upl_curriculum",
    "EnglishStudentKnowledge",
    "score_known_gpcs_english",
    "score_whole_words_english",
    "extract_words_english",
]
