from functools import cache

import cmudict

from decodability.english.definitions import ARPABET_TO_IPA, VOWEL_INITIALS


@cache
def _cmudict() -> dict[str, list[list[str]]]:
    return cmudict.dict()


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


def get_word_phonemes(word: str) -> tuple[tuple[tuple[str, bool], ...], ...]:
    """Return every CMUdict pronunciation variant for a word.

    Each outer element is a variant; each variant is a tuple of
    ``(ipa, is_unstressed)`` pairs. Dialect normalizations are applied to
    each variant independently. Raises ``KeyError`` if the word is not in
    CMUdict.

    CMUdict lists multiple variants for homographs (``read`` → present- vs
    past-tense) and stress alternations (``either`` → /iː/ vs /aɪ/). Callers
    should try every variant and accept a match on any one.
    """
    word_variants = _cmudict().get(word.lower())
    if not word_variants:
        raise KeyError(f"{word!r} not in CMUdict")

    return tuple(
        _normalize_phonemes(tuple(phoneme_arpabet_lookup(token) for token in arpabet))
        for arpabet in word_variants
    )
