"""Definitions for Kiswahili orthography"""

from __future__ import annotations

from enum import Enum

from primerpro import Consonant, GraphemeInventory, Settings, Vowel

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
    """Kiswahili consonant cluster patterns.

    Each member carries two strings:
    - its ``value`` (a stable ``code``): what is stored in the data files
    - its ``label``: the human-readable form for display
    """

    # Declared for type checkers
    label: str

    PRENASALISED_M = ("prenasalised_m", '"m" + consonant')
    PRENASALISED_N = ("prenasalised_n", '"n" + consonant')
    PRENASALISED = ("prenasalised", '"m/n" + consonant')

    LABIALISED_GLIDE = ("labialised_glide", 'consonant + "w"')
    PALATALISED_GLIDE = ("palatalised_glide", 'consonant + "y"')

    M_LABIALISED_GLIDE = ("m_labialised_glide", '"m" + consonant + "w"')
    N_LABIALISED_GLIDE = ("n_labialised_glide", '"n" + consonant + "w"')
    PRENASALISED_LABIALISED_GLIDE = (
        "prenasalised_labialised_glide",
        '"m/n" + consonant + "w"',
    )

    M_PALATALISED_GLIDE = ("m_palatalised_glide", '"m" + consonant + "y"')
    N_PALATALISED_GLIDE = ("n_palatalised_glide", '"n" + consonant + "y"')
    PRENASALISED_PALATALISED_GLIDE = (
        "prenasalised_palatalised_glide",
        '"m/n" + consonant + "y"',
    )

    def __new__(cls, code: str, label: str) -> ClusterPattern:
        # Split the (code, label) tuple
        obj = object.__new__(cls)
        obj._value_ = code
        obj.label = label
        return obj


_EXCLUDED_CONSONANTS: dict[ClusterPattern, set[str]] = {
    # Excluded consonants for each cluster pattern in Kiswahili, either
    # because they are not valid in that cluster or because they are already
    # represented by a digraph.
    ClusterPattern.PRENASALISED_M: set(),
    ClusterPattern.PRENASALISED_N: {"g", "y", "ng", "ng'", "ny"},
    ClusterPattern.LABIALISED_GLIDE: {"w"},
    ClusterPattern.PALATALISED_GLIDE: {"y", "ny", "n"},
}


VALID_CLUSTERS: dict[ClusterPattern, set[str]] = {
    ClusterPattern.PRENASALISED_M: {
        "m" + c
        for c in VALID_CONSONANTS
        if c not in _EXCLUDED_CONSONANTS[ClusterPattern.PRENASALISED_M]
    },
    ClusterPattern.PRENASALISED_N: {
        "n" + c
        for c in VALID_CONSONANTS
        if c not in _EXCLUDED_CONSONANTS[ClusterPattern.PRENASALISED_N]
    },
    ClusterPattern.LABIALISED_GLIDE: {
        c + "w"
        for c in VALID_CONSONANTS
        if c not in _EXCLUDED_CONSONANTS[ClusterPattern.LABIALISED_GLIDE]
    },
    ClusterPattern.PALATALISED_GLIDE: {
        c + "y"
        for c in VALID_CONSONANTS
        if c not in _EXCLUDED_CONSONANTS[ClusterPattern.PALATALISED_GLIDE]
    },
    ClusterPattern.M_LABIALISED_GLIDE: {
        "m" + c + "w"
        for c in VALID_CONSONANTS
        if c
        not in _EXCLUDED_CONSONANTS[ClusterPattern.PRENASALISED_M]
        | _EXCLUDED_CONSONANTS[ClusterPattern.LABIALISED_GLIDE]
    },
    ClusterPattern.N_LABIALISED_GLIDE: {
        "n" + c + "w"
        for c in VALID_CONSONANTS
        if c
        not in _EXCLUDED_CONSONANTS[ClusterPattern.PRENASALISED_N]
        | _EXCLUDED_CONSONANTS[ClusterPattern.LABIALISED_GLIDE]
    },
    ClusterPattern.M_PALATALISED_GLIDE: {
        "m" + c + "y"
        for c in VALID_CONSONANTS
        if c
        not in _EXCLUDED_CONSONANTS[ClusterPattern.PRENASALISED_M]
        | _EXCLUDED_CONSONANTS[ClusterPattern.PALATALISED_GLIDE]
    },
    ClusterPattern.N_PALATALISED_GLIDE: {
        "n" + c + "y"
        for c in VALID_CONSONANTS
        if c
        not in _EXCLUDED_CONSONANTS[ClusterPattern.PRENASALISED_N]
        | _EXCLUDED_CONSONANTS[ClusterPattern.PALATALISED_GLIDE]
    },
}

VALID_CLUSTERS[ClusterPattern.PRENASALISED] = (
    VALID_CLUSTERS[ClusterPattern.PRENASALISED_M]
    | VALID_CLUSTERS[ClusterPattern.PRENASALISED_N]
)
VALID_CLUSTERS[ClusterPattern.PRENASALISED_LABIALISED_GLIDE] = (
    VALID_CLUSTERS[ClusterPattern.M_LABIALISED_GLIDE]
    | VALID_CLUSTERS[ClusterPattern.N_LABIALISED_GLIDE]
)
VALID_CLUSTERS[ClusterPattern.PRENASALISED_PALATALISED_GLIDE] = (
    VALID_CLUSTERS[ClusterPattern.M_PALATALISED_GLIDE]
    | VALID_CLUSTERS[ClusterPattern.N_PALATALISED_GLIDE]
)


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


__all__ = [
    "ClusterPattern",
    "VALID_GRAPHEMES",
    "VALID_CONSONANTS",
    "VALID_CLUSTERS",
    "primerpro_settings",
]
