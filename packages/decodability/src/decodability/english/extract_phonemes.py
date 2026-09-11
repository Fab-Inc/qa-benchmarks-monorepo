from dataclasses import dataclass
from functools import cache

import cmudict

from decodability.english.definitions import ARPABET_TO_IPA, VOWEL_INITIALS
from decodability.english.phoneme_overrides import CMUDICT_OVERRIDES


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


def _suffixed_s_variants(key: str) -> list[list[str]] | None:
    """Pronunciations for an out-of-vocabulary word ending in ``s``/``'s``.

    Strips the suffix, looks up the base in ``CMUDICT_OVERRIDES``/CMUdict, and
    appends the English plural/possessive allomorphs to each of its variants —
    both ARPABET ``S`` and ``Z`` — so ``haps`` and ``hap's`` inherit ``hap``'s
    pronunciation plus /s/ or /z/. Returns ``None`` when the word has no such
    suffix or the base is also unknown.
    """
    if key.endswith("'s"):
        base = key[:-2]
    elif key.endswith("s"):
        base = key[:-1]
    else:
        return None
    if not base:
        return None
    base_variants = CMUDICT_OVERRIDES.get(base) or _cmudict().get(base)
    if not base_variants:
        return None
    return [variant + [suffix] for variant in base_variants for suffix in ("S", "Z")]


def get_word_phonemes(word: str) -> tuple[tuple[Phoneme, ...], ...]:
    """Return every pronunciation variant for a word.

    Each outer element is a variant; each variant is a tuple of ``Phoneme``
    objects. Manual entries in ``CMUDICT_OVERRIDES`` take priority; otherwise
    CMUdict is consulted. Dialect normalizations are applied to each variant
    independently. Raises ``KeyError`` if the word appears in neither source.

    Words missing from both sources get one fallback: a trailing ``s`` or
    ``'s`` is stripped and the base's variants are returned with /s/ and /z/
    appended (``haps`` → ``hap`` + /s/ or /z/).

    CMUdict lists multiple variants for homographs (``read`` → present- vs
    past-tense) and stress alternations (``either`` → /iː/ vs /aɪ/). Callers
    should try every variant and accept a match on any one.
    """
    key = word.lower()
    word_variants = (
        CMUDICT_OVERRIDES.get(key) or _cmudict().get(key) or _suffixed_s_variants(key)
    )
    if not word_variants:
        raise KeyError(f"{word!r} not in CMUdict")

    normalized_phonemes: list[tuple[Phoneme, ...]] = []
    for arpabet_variant in word_variants:
        phonemes = tuple(convert_arpabet_to_phoneme(t) for t in arpabet_variant)
        normalized_phonemes.append(_normalize_phonemes(phonemes))
    return tuple(normalized_phonemes)
