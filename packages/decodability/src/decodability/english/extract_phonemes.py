import os
from functools import cache

from decodability.english.definitions import ARPABET_TO_IPA, VOWEL_INITIALS

_REQUIRED_NLTK_DATA = (
    # g2p_en calls nltk.pos_tag; nltk 3.9+ requires the "_eng" tagger.
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
)


@cache
def load_g2p():
    # Lazy: G2p() loads a seq2seq model + NLTK data on first construction.
    os.environ.setdefault("NLTK_DISABLE_IMPORT_SECURITY", "1")

    import nltk

    for resource_path, package in _REQUIRED_NLTK_DATA:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package, quiet=True)

    from g2p_en import G2p

    return G2p()


def phoneme_arpabet_lookup(phoneme: str) -> tuple[str, bool]:
    """Lookup an ARPABET phoneme's IPA representation."""

    is_unstressed = phoneme.endswith("0")

    if phoneme in ARPABET_TO_IPA:
        return ARPABET_TO_IPA[phoneme], is_unstressed
    base = phoneme.rstrip("012")
    if base in ARPABET_TO_IPA:
        return ARPABET_TO_IPA[base], is_unstressed
    raise ValueError(f"Unknown ARPABET phoneme: {phoneme}")


def _normalize_phonemes(
    phonemes: tuple[tuple[str, bool], ...],
) -> tuple[tuple[str, bool], ...]:
    """Bridge CMU's phonetic transcription to the UFLI inventory's categories.

    - /ɚ/ → /ə/ + /r/: CMU emits a single r-colored schwa for ``er, ir, ur,
      -ar (unstressed), -or (unstressed), -ture`` endings. The inventory
      decomposes these into ``ə + r``, so we split the phoneme to match.
    - /ɪ/ + /r/ → /i/ + /r/ when the /r/ is in coda position (word-final
      or before a consonant): undoes CMU's pre-r laxing so ``beer, cheer,
      deer, steer`` match ``ee → i``. /ɪ/+/r/ before a vowel (``mirror,
      spirit``) is preserved.
    """
    result: tuple[tuple[str, bool], ...] = ()
    i = 0
    while i < len(phonemes):
        ipa, unstressed = phonemes[i]
        if ipa == "ɚ":
            result += (("ə", unstressed), ("r", False))
            i += 1
            continue
        if (
            ipa == "ɪ"
            and i + 1 < len(phonemes)
            and phonemes[i + 1][0] == "r"
            and (i + 2 == len(phonemes) or phonemes[i + 2][0][0] not in VOWEL_INITIALS)
        ):
            result += (("i", unstressed),)
            i += 1
            continue
        result += ((ipa, unstressed),)
        i += 1
    return result


def get_word_phonemes(word: str) -> tuple[tuple[str, bool], ...]:
    """Return the IPA phonemes for a word, one phoneme per tuple item.

    Uses ``g2p_en`` — CMUdict for in-vocab words, a seq2seq fallback for
    OOV — and applies dialect normalizations expected by the UFLI
    inventory.
    """
    arpabet = load_g2p()(word)
    phonemes = tuple(phoneme_arpabet_lookup(token) for token in arpabet)
    return _normalize_phonemes(phonemes)


if __name__ == "__main__":
    print(get_word_phonemes("review"))
