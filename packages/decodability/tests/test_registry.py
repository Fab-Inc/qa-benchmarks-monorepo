"""Tests for the shared language integration registry."""

from decodability import DECODABILITY_REGISTRY
from decodability.english import (
    AGGREGATIONS as ENGLISH_AGGREGATIONS,
)
from decodability.english import (
    SCORING_METHODS as ENGLISH_SCORING_METHODS,
)
from decodability.english import (
    EnglishStudentKnowledge,
    extract_words_english,
)
from decodability.kiswahili import (
    AGGREGATIONS as KISWAHILI_AGGREGATIONS,
)
from decodability.kiswahili import (
    SCORING_METHODS as KISWAHILI_SCORING_METHODS,
)
from decodability.kiswahili import (
    KiswahiliStudentKnowledge,
    extract_words_kiswahili,
)


def test_kiswahili_registry_entry_exposes_the_complete_integration() -> None:
    integration = DECODABILITY_REGISTRY["sw"]

    assert integration == {
        "name": "Kiswahili",
        "extract_words": extract_words_kiswahili,
        "student_knowledge_model": KiswahiliStudentKnowledge,
        # DEPRECATED, pending removal once consumers move to the analysis entries.
        "scoring_methods": KISWAHILI_SCORING_METHODS,
        "aggregations": KISWAHILI_AGGREGATIONS,
    }


def test_english_registry_entry_exposes_the_complete_integration() -> None:
    integration = DECODABILITY_REGISTRY["en"]

    assert integration == {
        "name": "English",
        "scoring_methods": ENGLISH_SCORING_METHODS,
        "aggregations": ENGLISH_AGGREGATIONS,
        "extract_words": extract_words_english,
        "student_knowledge_model": EnglishStudentKnowledge,
    }
