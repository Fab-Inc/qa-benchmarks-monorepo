import os
from functools import cache

from decodability.english.definitions import ARPABET_TO_IPA, VOWEL_INITIALS


@cache
def load_g2p():
    # Lazy: G2p() loads a seq2seq model + NLTK data on first construction.
    os.environ.setdefault("NLTK_DISABLE_IMPORT_SECURITY", "1")
    from g2p_en import G2p

    return G2p()


def phoneme_arpabet_lookup(phoneme: str) -> str:
    """Lookup an ARPABET phoneme's IPA representation."""
    if phoneme in ARPABET_TO_IPA:
        return ARPABET_TO_IPA[phoneme]
    base = phoneme.rstrip("012")
    if base in ARPABET_TO_IPA:
        return ARPABET_TO_IPA[base]
    raise ValueError(f"Unknown ARPABET phoneme: {phoneme}")


def _normalize_phonemes(phonemes: tuple[str, ...]) -> tuple[str, ...]:
    """Bridge CMU's phonetic transcription to the UFLI inventory's categories.

    - /ɚ/ → /ə/ + /r/: CMU emits a single r-colored schwa for ``er, ir, ur,
      -ar (unstressed), -or (unstressed), -ture`` endings. The inventory
      decomposes these into ``ə + r``, so we split the phoneme to match.
    - /ɪ/ + /r/ → /i/ + /r/ when the /r/ is in coda position (word-final
      or before a consonant): undoes CMU's pre-r laxing so ``beer, cheer,
      deer, steer`` match ``ee → i``. /ɪ/+/r/ before a vowel (``mirror,
      spirit``) is preserved.
    """
    result: tuple[str, ...] = ()
    i = 0
    while i < len(phonemes):
        if phonemes[i] == "ɚ":
            result += ("ə", "r")
            i += 1
            continue
        if (
            phonemes[i] == "ɪ"
            and i + 1 < len(phonemes)
            and phonemes[i + 1] == "r"
            and (i + 2 == len(phonemes) or phonemes[i + 2][0] not in VOWEL_INITIALS)
        ):
            result += ("i",)
            i += 1
            continue
        result += (phonemes[i],)
        i += 1
    return result


def get_word_phonemes(word: str) -> tuple[str, ...]:
    """Return the IPA phonemes for a word, one phoneme per tuple item.

    Uses ``g2p_en`` — CMUdict for in-vocab words, a seq2seq fallback for
    OOV — and applies dialect normalizations expected by the UFLI
    inventory.
    """
    arpabet = load_g2p()(word)
    phonemes = tuple(phoneme_arpabet_lookup(token) for token in arpabet)
    return _normalize_phonemes(phonemes)
