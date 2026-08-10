from decodability.english.definitions import UFLI_GPC_INVENTORY
from decodability.english.extract_phonemes import get_word_phonemes
from decodability.english.gpc_parser import GPC, parse_gpcs
from decodability.english.models import EnglishStudentKnowledge

GPC_INVENTORY = tuple(
    GPC(gpc_id, grapheme, phonemes) for gpc_id, grapheme, phonemes in UFLI_GPC_INVENTORY
)


def score_known_gpcs_english(
    word: str, student_knowledge: EnglishStudentKnowledge
) -> float:
    """Score a word based on whether the student knows all the grapheme - phoneme
      correspondences in the word.

    The score is binary: if there's any grapheme that the student doesn't know, the
    score is 0.
    """
    taught_gpcs = student_knowledge.gpcs

    if not taught_gpcs:
        return 0.0

    phonemes = get_word_phonemes(word.lower())
    parses = parse_gpcs(word.lower(), phonemes, GPC_INVENTORY)

    if not parses:
        return 0.0

    derived_gpc_pairs = [
        (match.gpc.grapheme, match.gpc.phonemes) for match in parses[0]
    ]

    taught_set = {(g, tuple(p)) for g, p in taught_gpcs}
    if any(gpc not in taught_set for gpc in derived_gpc_pairs):
        return 0.0

    return 1.0


def score_whole_words_english(
    word: str, student_knowledge: EnglishStudentKnowledge
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
