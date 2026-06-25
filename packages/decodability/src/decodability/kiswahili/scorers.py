from primerpro import Word

from decodability.kiswahili.config import primerpro_settings
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

    primerpro_word = Word(word, primerpro_settings)
    display_word = primerpro_word.display_word.lower()

    if display_word in taught_whole_words:
        return 1.0

    return 0.0


def score_known_syllables_kiswahili(
    word: str, student_knowledge: KiswahiliStudentKnowledge
) -> float:
    """Score a word based on the number of known syllables and the total number of
    syllables in the word.

    The score is binary: if there's any syllable that the student doesn't know, the
    score is 0.
    """
    taught_syllables = student_knowledge.syllables

    if not taught_syllables:
        return 0.0

    primerpro_word = Word(word, primerpro_settings)
    syllable_symbols = [
        syllable.display_syllable.lower() for syllable in primerpro_word.syllables
    ]

    if any(syllable not in taught_syllables for syllable in syllable_symbols):
        return 0.0

    return 1.0


def score_word_length_kiswahili(
    word: str, student_knowledge: KiswahiliStudentKnowledge
) -> float:
    """Score a word based on its length and the maximum length of words the student
    can decode.

    The score is calculated as the maximum word length the student can decode divided
    by the length of the word, capped at 1.
    """
    taught_length = student_knowledge.max_word_length
    n = len(word)

    return min(1, taught_length / n)


def score_syllable_pattern_kiswahili(
    word: str, student_knowledge: KiswahiliStudentKnowledge
) -> float:
    """Score a word based on its syllable pattern and the student's knowledge.

    The score is calculated as the ratio of known syllable patterns to total syllables
    in the word.
    """
    taught_patterns = student_knowledge.syllable_patterns

    primerpro_word = Word(word, primerpro_settings)
    syllable_pattern = primerpro_word.cv_pattern

    if not taught_patterns:
        return 0.0

    is_known_pattern = syllable_pattern in taught_patterns

    return 1.0 if is_known_pattern else 0.0
