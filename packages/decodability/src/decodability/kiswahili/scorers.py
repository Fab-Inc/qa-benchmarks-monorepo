from decodability.kiswahili.segment import (
    Span,
    decompose_word,
    find_cluster_spans,
    find_clustering_variants,
    find_fused_cluster_spans,
    get_grapheme_symbols,
)
from decodability.kiswahili.student_knowledge import KiswahiliStudentKnowledge


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

    if any(grapheme not in taught_graphemes for grapheme in get_grapheme_symbols(word)):
        return 0.0

    return 1.0


def score_known_clusters_and_patterns_kiswahili(
    word: str, student_knowledge: KiswahiliStudentKnowledge
) -> float:
    """Score a word on whether the student knows all its clusters.

    The score is binary: if there's any cluster that the student doesn't know,
    the score is 0.

    Ambiguous nasal + consonant boundaries count as fused clusters, so the
    score is 1.0 only if every cluster under that resolution is known
    (directly or via a known cluster pattern). This is the strict reading; see
    :func:`score_known_clusters_and_patterns_lenient_kiswahili` for the
    lenient alternative.

    If the word doesn't contain any cluster, the score is 1.0, since the student
    doesn't need to know any clusters to decode it.
    """
    taught_clusters = student_knowledge.all_known_clusters

    graphemes = decompose_word(word)
    return float(
        all(
            span.normalised in taught_clusters for span in find_cluster_spans(graphemes)
        )
    )


def score_known_clusters_and_patterns_lenient_kiswahili(
    word: str, student_knowledge: KiswahiliStudentKnowledge
) -> float:
    """Score a word leniently on whether the student knows all its clusters.

    The score is 1.0 if any viable clustering of the word contains no cluster
    the student cannot decode. A cluster is decodable when the student knows it
    (directly or via a known cluster pattern) or when it can be decomposed —
    read as its component graphemes, all of which must be taught. Only the
    fused nasal onsets (see :func:`find_fused_cluster_spans`) cannot be
    decomposed; every other cluster — an ambiguous boundary read as fused, or a
    non-nasal stretch like "bl" or "st" — may be sounded out grapheme by
    grapheme.

    If the word doesn't contain any cluster, the score is 1.0, since the student
    doesn't need to know any clusters to decode it.
    """
    taught_clusters = student_knowledge.all_known_clusters
    taught_graphemes = student_knowledge.graphemes

    graphemes = decompose_word(word)
    fused_clusters = set(find_fused_cluster_spans(graphemes))

    def cluster_is_decodable(span: Span) -> bool:
        if span.normalised in taught_clusters:
            return True
        if span in fused_clusters:
            return False
        return all(
            symbol in taught_graphemes for symbol in get_grapheme_symbols(span.text)
        )

    for clustering in find_clustering_variants(graphemes):
        if all(cluster_is_decodable(span) for span in clustering):
            return 1.0

    return 0.0


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
