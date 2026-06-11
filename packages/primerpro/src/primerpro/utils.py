"""Util functions
"""

from primerpro.model import GPCInventory, Word


def get_phoneme_annotation(
    word: Word, gpc_inventory: GPCInventory
) -> list[tuple[str, str | None]]:
    """Return ``[(grapheme, phoneme), ...]`` for each grapheme in ``word``.
    """

    return [(g.symbol, gpc_inventory.get_phoneme(g.symbol)) for g in word.graphemes]


def get_phonemic_form(word: Word, gpc_inventory: GPCInventory) -> str:
    """Return the contatenated phonemes for the word's grapheme sequence.
    
    Raises error if any grapheme is missing a phoneme in the GPC inventory.
    """
    phonemes = [gpc_inventory.get_phoneme(g.symbol) for g in word.graphemes]
    
    if not all(phonemes):
        raise ValueError(
            f"Word '{word.display_word}' contains graphemes with missing phonemes: "
            f"{[g.symbol for g, p in zip(word.graphemes, phonemes) if p is None]}"
        )
    
    return "".join(phonemes)
