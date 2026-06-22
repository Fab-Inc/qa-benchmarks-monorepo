from decodability import mean_score, weighted_average
from decodability.kiswahili.models import KiswahiliStudentKnowledge
from decodability.kiswahili.scorers import (
    score_known_syllables_kiswahili,
    score_syllable_pattern_kiswahili,
    score_word_length_kiswahili,
)
from decodability.kiswahili.tokenizer import tokenize_kiswahili

SCORING_METHODS = {
    "score_known_syllables_kiswahili": score_known_syllables_kiswahili,
    "score_word_length_kiswahili": score_word_length_kiswahili,
    "score_syllable_pattern_kiswahili": score_syllable_pattern_kiswahili,
}

AGGREGATIONS = {
    "mean_score": mean_score,
    "weighted_average": weighted_average,
}

__all__ = [
    "SCORING_METHODS",
    "AGGREGATIONS",
    "KiswahiliStudentKnowledge",
    "score_known_syllables_kiswahili",
    "score_word_length_kiswahili",
    "score_syllable_pattern_kiswahili",
    "tokenize_kiswahili",
]
