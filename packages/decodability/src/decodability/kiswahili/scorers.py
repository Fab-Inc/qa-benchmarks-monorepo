from functools import reduce

from decodability.kiswahili.definitions import (
    VALID_CLUSTERS,
    get_clusters,
    get_graphemes,
)
from decodability.kiswahili.models import KiswahiliStudentKnowledge


def score_known_graphemes_kiswahili(
    word: str, student_knowledge: KiswahiliStudentKnowledge
) -> float:
    """Score a word based on whether the student knows all the graphemes in the word.

    The score is binary: if there's any grapheme that the student doesn't know, the
    score is 0.
    """
    taught_graphemes = student_knowledge.graphemes

    if not taught_graphemes:
        return 0.0

    if any(grapheme not in taught_graphemes for grapheme in get_graphemes(word)):
        return 0.0

    return 1.0


def score_known_clusters_and_patterns_kiswahili(
    word: str, student_knowledge: KiswahiliStudentKnowledge
) -> float:
    """Score a word based on whether the student knows all the clusters in the word.

    The score is binary: if there's any cluster that the student doesn't know, the
    score is 0.

    If the word doesn't contain any cluter, the score is 1.0, since the student
    doesn't need to know any clusters to decode it.
    """
    taught_cluster_patterns = student_knowledge.cluster_patterns
    taught_clusters_from_taught_patterns = reduce(
        set.union,
        (VALID_CLUSTERS[pattern] for pattern in taught_cluster_patterns),
        set(),
    )

    taught_clusters = student_knowledge.clusters | taught_clusters_from_taught_patterns

    for cluster in get_clusters(word):
        if cluster not in taught_clusters:
            return 0.0

    return 1.0


def score_whole_words_kiswahili(
    word: str, student_knowledge: KiswahiliStudentKnowledge
) -> float:
    """Score a word based on whether the student knows it as a whole word.

    The score is binary: if the student knows the word as a whole word, the score is 1.
    Otherwise, the score is 0.
    """
    taught_whole_words = student_knowledge.whole_words

    if not taught_whole_words:
        return 0.0

    if word.lower() in taught_whole_words:
        return 1.0

    return 0.0
