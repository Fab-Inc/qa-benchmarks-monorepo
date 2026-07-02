from primerpro import Word

from decodability.kiswahili.definitions import primerpro_settings
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

    primerpro_word = Word(word, primerpro_settings)
    grapheme_symbols = [
        primerpro_word.get_grapheme(index).symbol.lower()
        for index in range(primerpro_word.grapheme_count())
    ]

    if any(grapheme not in taught_graphemes for grapheme in grapheme_symbols):
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
