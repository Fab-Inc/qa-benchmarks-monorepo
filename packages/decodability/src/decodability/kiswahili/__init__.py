from decodability import mean_score, rowwise, weighted_average
from decodability.kiswahili.aggregators import (
    aggregate_scores_kenya_tusome,
)
from decodability.kiswahili.extract_words import extract_words_kiswahili
from decodability.kiswahili.models import KiswahiliStudentKnowledge
from decodability.kiswahili.scorers import (
    score_known_clusters_and_patterns_kiswahili,
    score_known_graphemes_kiswahili,
    score_whole_words_kiswahili,
)

cluster_scorer = score_known_clusters_and_patterns_kiswahili
SCORING_METHODS = {
    "score_known_graphemes_kiswahili": score_known_graphemes_kiswahili,
    "score_known_clusters_and_patterns_kiswahili": cluster_scorer,
    "score_whole_words_kiswahili": score_whole_words_kiswahili,
}

AGGREGATIONS = {
    "kiswahili_kenya_tusome_aggregator": rowwise(aggregate_scores_kenya_tusome),
    # NOTE: the following are generic aggregators that are not Kiswahili-specific,
    # but we include them here for now to demonstrate potential usage with multiple
    # aggregation methods.
    "mean_score": mean_score,
    "weighted_average": weighted_average,
}

__all__ = [
    "SCORING_METHODS",
    "AGGREGATIONS",
    "aggregate_scores_kenya_tusome",
    "KiswahiliStudentKnowledge",
    "score_known_graphemes_kiswahili",
    "score_known_clusters_and_patterns_kiswahili",
    "score_whole_words_kiswahili",
    "extract_words_kiswahili",
]
