"""Registry of language-specific decodability integrations."""

from decodability import english, kiswahili

DECODABILITY_REGISTRY = {
    "sw": {
        "name": "Kiswahili",
        "scoring_methods": kiswahili.SCORING_METHODS,
        "aggregations": kiswahili.AGGREGATIONS,
        "extract_words": kiswahili.extract_words_kiswahili,
        "student_knowledge_model": kiswahili.KiswahiliStudentKnowledge,
    },
    "en": {
        "name": "English",
        "scoring_methods": english.SCORING_METHODS,
        "aggregations": english.AGGREGATIONS,
        "extract_words": english.extract_words_english,
        "student_knowledge_model": english.EnglishStudentKnowledge,
    },
}

__all__ = ["DECODABILITY_REGISTRY"]
