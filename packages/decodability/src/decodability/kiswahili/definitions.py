"""Shared configurations for Kiswahili decodability analysis."""

from enum import Enum

from primerpro import Consonant, GraphemeInventory, Settings, Vowel, Word

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

VALID_CONSONANTS = {symbol for symbol, _ in KISWAHILI_CONSONANT_GPCS}

VALID_VOWELS = {symbol for symbol, _ in KISWAHILI_VOWEL_GPCS}

VALID_GRAPHEMES: set[str] = VALID_CONSONANTS | VALID_VOWELS


class ClusterPattern(Enum):
    """Enum for Kiswahili consonant cluster patterns."""

    PRENASALISED_M = '"m" + consonant'
    PRENASALISED_N = '"n" + consonant'
    LABIALISED_GLIDE = 'consonant + "w"'
    PALATALISED_GLIDE = 'consonant + "y"'
    # MIGHT NEED TO CREATE another enum that doesn't include the following
    M_LABIALISED_GLIDE = '"m" + consonant + "w"'
    N_LABIALISED_GLIDE = '"n" + consonant + "w"'
    M_PALATALISED_GLIDE = '"m" + consonant + "y"'
    N_PALATALISED_GLIDE = '"n" + consonant + "y"'


ILLEGAL_COMBINATIONS: dict[ClusterPattern, set[str]] = {
    ClusterPattern.PRENASALISED_M: set(),
    ClusterPattern.PRENASALISED_N: {"g", "y", "ng", "ng'", "ny"},
    ClusterPattern.LABIALISED_GLIDE: {"w"},
    ClusterPattern.PALATALISED_GLIDE: {"y", "ny", "n"},
}


VALID_CLUSTERS: dict[ClusterPattern, set[str]] = {
    ClusterPattern.PRENASALISED_M: {
        "m" + c
        for c in VALID_CONSONANTS
        if c not in ILLEGAL_COMBINATIONS[ClusterPattern.PRENASALISED_M]
    },
    ClusterPattern.PRENASALISED_N: {
        "n" + c
        for c in VALID_CONSONANTS
        if c not in ILLEGAL_COMBINATIONS[ClusterPattern.PRENASALISED_N]
    },
    ClusterPattern.LABIALISED_GLIDE: {
        c + "w"
        for c in VALID_CONSONANTS
        if c not in ILLEGAL_COMBINATIONS[ClusterPattern.LABIALISED_GLIDE]
    },
    ClusterPattern.PALATALISED_GLIDE: {
        c + "y"
        for c in VALID_CONSONANTS
        if c not in ILLEGAL_COMBINATIONS[ClusterPattern.PALATALISED_GLIDE]
    },
    ClusterPattern.M_LABIALISED_GLIDE: {
        "m" + c + "w"
        for c in VALID_CONSONANTS
        if c
        not in ILLEGAL_COMBINATIONS[ClusterPattern.PRENASALISED_M]
        | ILLEGAL_COMBINATIONS[ClusterPattern.LABIALISED_GLIDE]
    },
    ClusterPattern.N_LABIALISED_GLIDE: {
        "n" + c + "w"
        for c in VALID_CONSONANTS
        if c
        not in ILLEGAL_COMBINATIONS[ClusterPattern.PRENASALISED_N]
        | ILLEGAL_COMBINATIONS[ClusterPattern.LABIALISED_GLIDE]
    },
    ClusterPattern.M_PALATALISED_GLIDE: {
        "m" + c + "y"
        for c in VALID_CONSONANTS
        if c
        not in ILLEGAL_COMBINATIONS[ClusterPattern.PRENASALISED_M]
        | ILLEGAL_COMBINATIONS[ClusterPattern.PALATALISED_GLIDE]
    },
    ClusterPattern.N_PALATALISED_GLIDE: {
        "n" + c + "y"
        for c in VALID_CONSONANTS
        if c
        not in ILLEGAL_COMBINATIONS[ClusterPattern.PRENASALISED_N]
        | ILLEGAL_COMBINATIONS[ClusterPattern.PALATALISED_GLIDE]
    },
}


def _get_primerpro_settings() -> Settings:
    settings = Settings()
    settings.option_settings.max_size_grapheme = 3
    settings.option_settings.general_punct = ' ,:;-"'

    inventory = GraphemeInventory()

    for symbol, _ in KISWAHILI_CONSONANT_GPCS:
        inventory.add_consonant(Consonant(symbol))

    for symbol, _ in KISWAHILI_VOWEL_GPCS:
        inventory.add_vowel(Vowel(symbol))

    settings.grapheme_inventory = inventory
    return settings


primerpro_settings: Settings = _get_primerpro_settings()


def get_graphemes(word: str) -> list[str]:
    """Decompose a word into its constituent graphemes using PrimerPro settings.

    Symbols are lowercased: matched graphemes already come back as the
    inventory's lowercase symbol, but the fallback for characters outside the
    inventory preserves the input's case.
    """
    primerpro_word = Word(text=word, settings=primerpro_settings)
    return [grapheme.symbol.lower() for grapheme in primerpro_word.graphemes]


__all__ = [
    "ClusterPattern",
    "VALID_GRAPHEMES",
    "VALID_CONSONANTS",
    "VALID_CLUSTERS",
    "primerpro_settings",
]
