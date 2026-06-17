from decodability import mean_score, weighted_average
from decodability.kiswahili.models import KiswahiliKnowledge
from decodability.kiswahili.scorers import (
    score_known_syllables,
    score_syllable_pattern,
    score_word_length,
)
from decodability.kiswahili.tokenizer import tokenize_kiswahili

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
    KiswahiliKnowledge,
    score_known_syllables,
    score_word_length,
    score_syllable_pattern,
    tokenize_kiswahili,
]
