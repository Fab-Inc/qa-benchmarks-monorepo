from functools import partial

from decodability import mean_score, rowwise, weighted_average
from decodability.kiswahili.aggregators import (
    aggregate_scores_kenya_tusome,
)
from decodability.kiswahili.models import KiswahiliStudentKnowledge
from decodability.kiswahili.scorers import (
    score_known_graphemes_kiswahili,
    score_whole_words_kiswahili,
)
from decodability.kiswahili.tokenizer import tokenize_kiswahili

SCORING_METHODS = {
    "score_known_graphemes_kiswahili": score_known_graphemes_kiswahili,
    "score_whole_words_kiswahili": score_whole_words_kiswahili,
}

AGGREGATIONS = {
    "kiswahili_kenya_tusome_aggregator": rowwise(aggregate_scores_kenya_tusome),
    # NOTE: the following are generic aggregators that are not Kiswahili-specific,
    # but we include them here for now to demonstrate potential usage with multiple
    # aggregation methods.
    "mean_score": mean_score,
    "weighted_average": partial(
        weighted_average,
        weights={
            score_known_graphemes_kiswahili.__name__: 0.8,
            score_whole_words_kiswahili.__name__: 0.2,
        },
    ),
}

__all__ = [
    "SCORING_METHODS",
    "AGGREGATIONS",
    "aggregate_scores_kenya_tusome",
    "KiswahiliStudentKnowledge",
    "score_known_graphemes_kiswahili",
    "score_whole_words_kiswahili",
    "tokenize_kiswahili",
]
