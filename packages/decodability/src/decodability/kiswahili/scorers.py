from primerpro import Consonant, GraphemeInventory, Settings, Vowel, Word

from decodability.kiswahili.models import KiswahiliKnowledge

KISWAHILI_CONSONANT_GPCS = [
    ("b", "b"),
    ("ch", "tʃ"),
    ("d", "d"),
    ("dh", "ð"),
    ("f", "f"),
    ("g", "ɡ"),
    ("gh", "Ɣ"),
    ("h", "h"),
    ("j", "j"),
    ("k", "k"),
    ("l", "l"),
    ("m", "m"),
    ("n", "n"),
    ("ng", "ŋg"),
    ("ng'", "ŋ"),
    ("ny", "ɲ"),
    ("p", "p"),
    ("r", "r"),
    ("s", "s"),
    ("sh", "ʃ"),
    ("t", "t"),
    ("th", "θ"),
    ("v", "v"),
    ("w", "w"),
    ("y", "y"),
    ("z", "z"),
]

KISWAHILI_VOWEL_GPCS = [
    ("a", "a"),
    ("e", "ɛ"),
    ("i", "i"),
    ("o", "ɔ"),
    ("u", "u"),
]


def _get_primerpro_settings() -> Settings:
    settings = Settings()
    inventory = GraphemeInventory()

    for symbol, _ in KISWAHILI_CONSONANT_GPCS:
        inventory.add_consonant(Consonant(symbol))

    for symbol, _ in KISWAHILI_VOWEL_GPCS:
        inventory.add_vowel(Vowel(symbol))

    settings.grapheme_inventory = inventory
    return settings


settings: Settings = _get_primerpro_settings()


def score_known_syllables(word: str, student_knowledge: KiswahiliKnowledge) -> float:
    """Score a word based on the number of known syllables and the total number of
    syllables in the word.

    The score is binary: if there's any syllable that the student doesn't know, the
    score is 0.
    """
    taught_syllables = student_knowledge.known_syllables

    if not taught_syllables:
        return 0.0

    primerpro_word = Word(word, settings)
    syllable_symbols = [
        syllable.display_syllable.lower() for syllable in primerpro_word.syllables
    ]

    if any(syllable not in taught_syllables for syllable in syllable_symbols):
        return 0.0

    return 1.0


def score_word_length(word: str, student_knowledge: KiswahiliKnowledge) -> float:
    """Score a word based on its length and the maximum length of words the student
    can decode.

    The score is calculated as the maximum word length the student can decode divided
    by the length of the word, capped at 1.
    """
    taught_length = student_knowledge.max_word_length
    n = len(word)

    return min(1, taught_length / n)


def score_syllable_pattern(word: str, student_knowledge: KiswahiliKnowledge) -> float:
    """Score a word based on its syllable pattern and the student's knowledge.

    The score is calculated as the ratio of known syllable patterns to total syllables
    in the word.
    """
    taught_patterns = student_knowledge.known_syllable_patterns

    primerpro_word = Word(word, settings)
    syllable_pattern = primerpro_word.cv_pattern

    if not taught_patterns:
        return 0.0

    is_known_pattern = syllable_pattern in taught_patterns

    return 1.0 if is_known_pattern else 0.0
