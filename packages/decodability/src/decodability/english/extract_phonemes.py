from dataclasses import dataclass
from functools import cache

import cmudict

from decodability.english.definitions import ARPABET_TO_IPA, VOWEL_INITIALS


@dataclass(frozen=True)
class Phoneme:
    """One phonetic token: an IPA symbol plus whether CMU marked it stress-0."""

    ipa: str
    is_unstressed: bool


@cache
def _cmudict() -> dict[str, list[list[str]]]:
    return cmudict.dict()


def convert_arpabet_to_phoneme(phoneme: str) -> Phoneme:
    """Convert an ARPABET symobol to its IPA phoneme representation."""

    is_unstressed = phoneme.endswith("0")

    if phoneme in ARPABET_TO_IPA:
        return Phoneme(ARPABET_TO_IPA[phoneme], is_unstressed)
    base = phoneme.rstrip("012")
    if base in ARPABET_TO_IPA:
        return Phoneme(ARPABET_TO_IPA[base], is_unstressed)
    raise ValueError(f"Unknown ARPABET phoneme: {phoneme}")


def _normalize_phonemes(
    phonemes: tuple[Phoneme, ...],
) -> tuple[Phoneme, ...]:
    """Bridge CMU's phonetic transcription to the UFLI inventory's categories.

    - /ɚ/ → /ə/ + /r/: CMU emits a single r-colored schwa for ``er, ir, ur,
      -ar (unstressed), -or (unstressed), -ture`` endings. The inventory
      decomposes these into ``ə + r``, so we split the phoneme to match.
    - /ɪ/ + /r/ → /i/ + /r/ when the /r/ is in coda position (word-final
      or before a consonant): undoes CMU's pre-r laxing so ``beer, cheer,
      deer, steer`` match ``ee → i``. /ɪ/+/r/ before a vowel (``mirror,
      spirit``) is preserved.
    """
    result: tuple[Phoneme, ...] = ()
    i = 0
    while i < len(phonemes):
        current = phonemes[i]
        if current.ipa == "ɚ":
            result += (Phoneme("ə", current.is_unstressed), Phoneme("r", False))
            i += 1
            continue
        if (
            current.ipa == "ɪ"
            and i + 1 < len(phonemes)
            and phonemes[i + 1].ipa == "r"
            and (i + 2 == len(phonemes) or phonemes[i + 2].ipa[0] not in VOWEL_INITIALS)
        ):
            result += (Phoneme("i", current.is_unstressed),)
            i += 1
            continue
        result += (current,)
        i += 1
    return result


def get_word_phonemes(word: str) -> tuple[tuple[Phoneme, ...], ...]:
    """Return every CMUdict pronunciation variant for a word.

    Each outer element is a variant; each variant is a tuple of ``Phoneme``
    objects. Dialect normalizations are applied to each variant independently.
    Raises ``KeyError`` if the word is not in CMUdict.

    CMUdict lists multiple variants for homographs (``read`` → present- vs
    past-tense) and stress alternations (``either`` → /iː/ vs /aɪ/). Callers
    should try every variant and accept a match on any one.
    """
    word_variants = _cmudict().get(word.lower())
    if not word_variants:
        raise KeyError(f"{word!r} not in CMUdict")

    return tuple(
        _normalize_phonemes(
            tuple(convert_arpabet_to_phoneme(token) for token in arpabet)
        )
        for arpabet in word_variants
    )


if __name__ == "__main__":
    test_words = ["about", "sofa", "until"]
    for word in test_words:
        print(f"{word}: {get_word_phonemes(word)}")
