"""Per-word decodability analysis for Kiswahili.

- :func:`analyse_kiswahili_word` analyses a word based on a student knowledge profile,
    returning a Kiswahili-specific analysis object.
- :func:`assess_decodability_kenya_tusome` takes the analysis object and returns a
    boolean indicating decodability.
- :func:`is_decodable_kenya_tusome` composes the two, taking a word and a student
    knowledge profile and returning a boolean indicating decodability.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from decodability.kiswahili.definitions import (
    VALID_GRAPHEMES,
)
from decodability.kiswahili.segment import (
    Span,
    decompose_word,
    find_ambiguous_boundaries,
    find_clustering_variants,
    find_fused_cluster_spans,
    get_grapheme_symbols,
)
from decodability.kiswahili.student_knowledge import KiswahiliStudentKnowledge


class KiswahiliWordAnalysis(BaseModel):
    """Analysis of a Kiswahili word based on a student knowledge profile.

    ``unknown_clusters`` is the strict view: the unknown clusters under the
    fused resolution, in which every ambiguous nasal boundary is a cluster.
    ``lenient_unknown_clusters`` reports the clusters that still block the
    word under the lenient reading — clusters that are neither taught nor
    decomposable into taught graphemes — of the least-blocked viable
    clustering. ``ambiguous_boundaries`` locates the nasal + consonant spans
    the rules left unresolved, for annotation.
    """

    model_config = ConfigDict(frozen=True)

    word: str
    invalid_graphemes: list[Span] = []
    unknown_graphemes: list[Span] = []
    unknown_clusters: list[Span] = []
    lenient_unknown_clusters: list[Span] = []
    ambiguous_boundaries: list[Span] = []
    known_whole_word: bool = False


def analyse_word_kiswahili(
    word: str, student_knowledge: KiswahiliStudentKnowledge
) -> KiswahiliWordAnalysis:
    """Analyze a Kiswahili word based on a student knowledge profile.

    The word is decomposed once and both graphemes and clusters are derived from that
    single decomposition.
    """
    graphemes = decompose_word(word)
    taught_graphemes = student_knowledge.graphemes
    taught_clusters = student_knowledge.all_known_clusters

    invalid_graphemes: list[Span] = []
    unknown_graphemes: list[Span] = []
    for grapheme in graphemes:
        if grapheme.symbol not in VALID_GRAPHEMES:
            invalid_graphemes.append(grapheme.to_span())
        elif grapheme.symbol not in taught_graphemes:
            unknown_graphemes.append(grapheme.to_span())

    fused_clusters = set(find_fused_cluster_spans(graphemes))

    def blocks_leniently(span: Span) -> bool:
        """Whether a cluster blocks the lenient reading: unknown and not
        decomposable into graphemes the student knows."""
        if span.normalised in taught_clusters:
            return False
        if span in fused_clusters:
            return True
        return any(
            symbol not in taught_graphemes
            for symbol in get_grapheme_symbols(span.text)
        )

    clustering_variants = find_clustering_variants(graphemes)
    # Variants are ordered fused-first, so index 0 is the strict resolution.
    unknown_clusters = [
        span
        for span in clustering_variants[0]
        if span.normalised not in taught_clusters
    ]
    # The lenient view reports the least-blocked clustering; on ties the first
    # (most fused) variant wins, preferring the cluster a teacher recognises.
    lenient_unknown_clusters = min(
        (
            [span for span in variant if blocks_leniently(span)]
            for variant in clustering_variants
        ),
        key=len,
    )

    return KiswahiliWordAnalysis(
        word=word,
        invalid_graphemes=invalid_graphemes,
        unknown_graphemes=unknown_graphemes,
        unknown_clusters=unknown_clusters,
        lenient_unknown_clusters=lenient_unknown_clusters,
        ambiguous_boundaries=find_ambiguous_boundaries(graphemes),
        known_whole_word=word.lower() in student_knowledge.whole_words,
    )


def assess_decodability_kenya_tusome(word_analysis: KiswahiliWordAnalysis) -> bool:
    """Decide decodability of a word based on the analysis results and the
    Kenya Tusome curriculum.

    The rule: a known whole word is decodable regardless of anything else; otherwise
    an untaught grapheme _or_ an unknown cluster makes the word non-decodable.

    The cluster check is strict: ambiguous nasal + consonant boundaries count as
    fused clusters that the student must know.

    An empty word is considered decodable.
    """
    if word_analysis.known_whole_word:
        return True

    if word_analysis.invalid_graphemes:
        return False

    if word_analysis.unknown_graphemes:
        return False

    if word_analysis.unknown_clusters:
        return False

    return True


def assess_decodability_kenya_tusome_lenient(
    word_analysis: KiswahiliWordAnalysis,
) -> bool:
    """Lenient variant of the Kenya Tusome decodability rule.

    Identical to :func:`assess_decodability_kenya_tusome`, except ambiguous
    nasal + consonant boundaries can be read syllabically, and any cluster
    that is not a fused nasal onset may be decomposed into taught graphemes.
    A word is blocked only when every viable clustering contains a cluster
    that is neither taught nor decomposable.
    """
    if word_analysis.known_whole_word:
        return True

    if word_analysis.invalid_graphemes:
        return False

    if word_analysis.unknown_graphemes:
        return False

    if word_analysis.lenient_unknown_clusters:
        return False

    return True


def is_decodable_kenya_tusome(
    word: str, student_knowledge: KiswahiliStudentKnowledge
) -> bool:
    """Decide decodability of a word based on the Kenya Tusome curriculum.

    The rule: a known whole word is decodable regardless of anything else; otherwise
    an untaught grapheme _or_ an unknown cluster makes the word non-decodable.

    The cluster check is strict: ambiguous nasal + consonant boundaries count as
    fused clusters that the student must know.

    An empty word is considered decodable.
    """
    findings = analyse_word_kiswahili(word, student_knowledge)
    return assess_decodability_kenya_tusome(findings)


def is_decodable_kenya_tusome_lenient(
    word: str, student_knowledge: KiswahiliStudentKnowledge
) -> bool:
    """Decide decodability under the lenient Kenya Tusome rule.

    Identical to :func:`is_decodable_kenya_tusome`, except ambiguous nasal +
    consonant boundaries can be read syllabically, and any cluster that is not
    a fused nasal onset may be decomposed into taught graphemes.
    """
    findings = analyse_word_kiswahili(word, student_knowledge)
    return assess_decodability_kenya_tusome_lenient(findings)
