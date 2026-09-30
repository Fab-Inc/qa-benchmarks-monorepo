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
from decodability.kiswahili.segment import Span, decompose_word, find_cluster_spans
from decodability.kiswahili.student_knowledge import KiswahiliStudentKnowledge


class KiswahiliWordAnalysis(BaseModel):
    """Analysis of a Kiswahili word based on a student knowledge profile"""

    model_config = ConfigDict(frozen=True)

    word: str
    invalid_graphemes: list[Span] = []
    unknown_graphemes: list[Span] = []
    unknown_clusters: list[Span] = []
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

    unknown_clusters = [
        span
        for span in find_cluster_spans(graphemes)
        if span.normalised not in taught_clusters
    ]

    return KiswahiliWordAnalysis(
        word=word,
        invalid_graphemes=invalid_graphemes,
        unknown_graphemes=unknown_graphemes,
        unknown_clusters=unknown_clusters,
        known_whole_word=word.lower() in student_knowledge.whole_words,
    )


def assess_decodability_kenya_tusome(word_analysis: KiswahiliWordAnalysis) -> bool:
    """Decide decodability of a word based on the analysis results and the
    Kenya Tusome curriculum.

    The rule: a known whole word is decodable regardless of anything else; otherwise
    an untaught grapheme _or_ an unknown cluster makes the word non-decodable.

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


def is_decodable_kenya_tusome(
    word: str, student_knowledge: KiswahiliStudentKnowledge
) -> bool:
    """Decide decodability of a word based on the Kenya Tusome curriculum.

    The rule: a known whole word is decodable regardless of anything else; otherwise
    an untaught grapheme _or_ an unknown cluster makes the word non-decodable.

    An empty word is considered decodable.
    """
    findings = analyse_word_kiswahili(word, student_knowledge)
    return assess_decodability_kenya_tusome(findings)
