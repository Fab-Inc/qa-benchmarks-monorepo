from decodability import mean_score, rowwise, weighted_average
from decodability.english.aggregators import (
    aggregate_scores_ufli_curriculum,
)
from decodability.english.analyse import (
    EnglishWordAnalysis,
    analyse_word_english,
    assess_decodability_ufli,
)
from decodability.english.extract_words import extract_words_english
from decodability.english.models import EnglishStudentKnowledge
from decodability.english.scorers import (
    score_known_gpcs_english,
    score_sight_words_english,
)

SCORING_METHODS = {
    "score_known_gpcs_english": score_known_gpcs_english,
    "score_sight_words_english": score_sight_words_english,
}

AGGREGATIONS = {
    "ufli_curriculum_aggregator": rowwise(aggregate_scores_ufli_curriculum),
    "mean_score": mean_score,
    "weighted_average": weighted_average,
}

__all__ = [
    "SCORING_METHODS",
    "AGGREGATIONS",
    "EnglishWordAnalysis",
    "aggregate_scores_ufli_curriculum",
    "analyse_word_english",
    "assess_decodability_ufli",
    "EnglishStudentKnowledge",
    "score_known_gpcs_english",
    "score_sight_words_english",
    "extract_words_english",
]
