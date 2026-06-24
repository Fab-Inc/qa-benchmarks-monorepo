from decodability.kiswahili.aggregators import kiswahili_kenya_tusome_aggregator
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
    "kiswahili_kenya_tusome_aggregator": kiswahili_kenya_tusome_aggregator,
}

__all__ = [
    "SCORING_METHODS",
    "AGGREGATIONS",
    "kiswahili_kenya_tusome_aggregator",
    "KiswahiliStudentKnowledge",
    "score_known_syllables_kiswahili",
    "score_word_length_kiswahili",
    "score_syllable_pattern_kiswahili",
    "tokenize_kiswahili",
]
