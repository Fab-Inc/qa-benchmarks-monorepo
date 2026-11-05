from decodability import mean_score, rowwise, weighted_average
from decodability.kiswahili.aggregators import (
    aggregate_scores_kenya_tusome,
    aggregate_scores_kenya_tusome_lenient,
)
from decodability.kiswahili.analyse import (
    KiswahiliDecodabilityAnalysis,
    analyse_word_kiswahili,
    assess_decodability_kenya_tusome,
    assess_decodability_kenya_tusome_lenient,
)
from decodability.kiswahili.extract_words import extract_words_kiswahili
from decodability.kiswahili.scorers import (
    score_known_clusters_and_patterns_kiswahili,
    score_known_clusters_and_patterns_lenient_kiswahili,
    score_known_graphemes_kiswahili,
    score_whole_words_kiswahili,
)
from decodability.kiswahili.student_knowledge import KiswahiliStudentKnowledge

SCORING_METHODS = {
    "score_known_graphemes_kiswahili": score_known_graphemes_kiswahili,
    "score_known_clusters_and_patterns_kiswahili": (
        score_known_clusters_and_patterns_kiswahili
    ),
    "score_known_clusters_and_patterns_lenient_kiswahili": (
        score_known_clusters_and_patterns_lenient_kiswahili
    ),
    "score_whole_words_kiswahili": score_whole_words_kiswahili,
}

AGGREGATIONS = {
    "kiswahili_kenya_tusome_aggregator": rowwise(aggregate_scores_kenya_tusome),
    "kiswahili_kenya_tusome_lenient_aggregator": rowwise(
        aggregate_scores_kenya_tusome_lenient
    ),
    # NOTE: the following are generic aggregators that are not Kiswahili-specific,
    # but we include them here for now to demonstrate potential usage with multiple
    # aggregation methods.
    "mean_score": mean_score,
    "weighted_average": weighted_average,
}

__all__ = [
    "SCORING_METHODS",
    "AGGREGATIONS",
    "KiswahiliDecodabilityAnalysis",
    "aggregate_scores_kenya_tusome",
    "aggregate_scores_kenya_tusome_lenient",
    "analyse_word_kiswahili",
    "assess_decodability_kenya_tusome",
    "assess_decodability_kenya_tusome_lenient",
    "KiswahiliStudentKnowledge",
    "score_known_graphemes_kiswahili",
    "score_known_clusters_and_patterns_kiswahili",
    "score_known_clusters_and_patterns_lenient_kiswahili",
    "score_whole_words_kiswahili",
    "extract_words_kiswahili",
]
