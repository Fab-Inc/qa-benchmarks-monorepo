from decodability import mean_score, weighted_average
from decodability.kiswahili.models import KiswahiliStudentKnowledge
from decodability.kiswahili.scorers import (
    score_known_syllables,
    score_word_length,
    score_syllable_pattern,
)

STUDENT_KNOWLEDGE_MODEL = "decodability.kiswahili.models.KiswahiliStudentKnowledge"

SCORING_METHODS = {
    "kiswahili_score_known_syllables": score_known_syllables,
    "kiswahili_score_word_length": score_word_length,
    "kiswahili_score_syllable_pattern": score_syllable_pattern,
}

AGGREGATIONS = {
    "mean_score": mean_score,
    "weighted_average": weighted_average,
}

__all__ = [
    "STUDENT_KNOWLEDGE_MODEL",
    "SCORING_METHODS",
    "AGGREGATIONS",
    KiswahiliStudentKnowledge,
    score_known_syllables,
    score_word_length,
    score_syllable_pattern,
]
