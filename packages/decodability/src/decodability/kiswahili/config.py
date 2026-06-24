from primerpro import Consonant, GraphemeInventory, Settings, Vowel

# Grapheme-to-phoneme correspondences (GPCs) for Kiswahili. Each tuple is
# (orthographic grapheme, IPA phoneme).
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

VALID_GRAPHEMES: set[str] = {
    symbol.lower() for symbol, _ in KISWAHILI_CONSONANT_GPCS + KISWAHILI_VOWEL_GPCS
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
