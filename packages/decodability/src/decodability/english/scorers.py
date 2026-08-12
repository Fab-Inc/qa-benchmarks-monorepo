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
    """Score a word based on whether the student can decode it.

    The score is 1.0 if at least one valid parse of the word uses only GPCs the
    student has been taught; 0.0 otherwise. Multiple parses can arise when prefix
    or suffix GPCs overlap with letter-by-letter GPCs.

    Reduced-vowel matches (parser-synthesized ``REDUCED_*`` GPCs from unstressed
    schwa/lax positions) count as taught whenever the student knows *any* GPC
    for the same grapheme letter — vowel reduction is a universal spoken-
    English process, not a separate correspondence the student needs taught.
    """
    taught_gpcs = student_knowledge.gpcs

    if not taught_gpcs:
        return 0.0

    phonemes = get_word_phonemes(word.lower())
    parses = parse_gpcs(word.lower(), phonemes, GPC_INVENTORY)

    if not parses:
        return 0.0

    taught_set = {(g, tuple(p)) for g, p in taught_gpcs}
    taught_graphemes = {g for g, _ in taught_gpcs}

    def is_taught(match) -> bool:
        if match.gpc.id.startswith("REDUCED_"):
            return match.gpc.grapheme in taught_graphemes
        return (match.gpc.grapheme, match.gpc.phonemes) in taught_set

    for parse in parses:
        if all(is_taught(match) for match in parse):
            return 1.0

    return 0.0


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
