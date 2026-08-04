"""Registry of language-specific decodability integrations."""

from decodability.kiswahili import (
    AGGREGATIONS,
    SCORING_METHODS,
    KiswahiliStudentKnowledge,
    extract_words_kiswahili,
)

DECODABILITY_REGISTRY = {
    "sw": {
        "name": "Kiswahili",
        "scoring_methods": SCORING_METHODS,
        "aggregations": AGGREGATIONS,
        "extract_words": extract_words_kiswahili,
        "student_knowledge_model": KiswahiliStudentKnowledge,
    }
}

__all__ = ["DECODABILITY_REGISTRY"]
