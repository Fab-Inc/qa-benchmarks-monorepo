from dataclasses import dataclass
from functools import cache

import cmudict

from decodability.english.definitions import ARPABET_TO_IPA, VOWEL_INITIALS
from decodability.english.phoneme_overrides import CMUDICT_OVERRIDES

STRESS_DIGITS = "012"


@dataclass(frozen=True)
class Phoneme:
    """One phonetic token: an IPA symbol plus whether CMU marked it stress-0."""

    ipa: str
    is_unstressed: bool


@dataclass(frozen=True)
class Pronunciation:
    """One pronunciation variant: its phonemes plus word-shape facts.

    ``is_monosyllabic`` is a property of the variant, not of the word: CMU
    records both careful and syncopated readings. Some words (``aged``,
    ``aisle``, ``cache``) are one syllable in one variant and two in another.
    It is counted from ARPABET before conversion, because ``Phoneme`` keeps
    only ``is_unstressed`` and the syllable count cannot be recovered later.
    """

    phonemes: tuple[Phoneme, ...]
    is_monosyllabic: bool


def count_syllables(arpabet_variant: list[str]) -> int:
    """Count syllables in one ARPABET variant.

    ARPABET marks every vowel — and only vowels — with a stress digit, so the
    number of stress-bearing tokens is the number of syllable nuclei. Note
    this counts all stress levels, not just primary: ``banana`` is
    ``B AH0 N AE1 N AH0``, three vowels and three syllables, even though only
    ``AE1`` carries primary stress.
    """
    return sum(1 for token in arpabet_variant if token[-1] in STRESS_DIGITS)


@cache
def _cmudict() -> dict[str, list[list[str]]]:
    return cmudict.dict()


def convert_arpabet_to_phoneme(phoneme: str) -> Phoneme:
    """Convert an ARPABET symobol to its IPA phoneme representation."""

    is_unstressed = phoneme.endswith("0")

    if phoneme in ARPABET_TO_IPA:
        return Phoneme(ARPABET_TO_IPA[phoneme], is_unstressed)
    base = phoneme.rstrip(STRESS_DIGITS)
    if base in ARPABET_TO_IPA:
        return Phoneme(ARPABET_TO_IPA[base], is_unstressed)
    raise ValueError(f"Unknown ARPABET phoneme: {phoneme}")


def _add_cot_caught_variants(
    arpabet_variants: list[list[str]],
) -> list[list[str]]:
    """Add an /ɑ/ reading for every variant CMU transcribes with /ɔ/.

    CMUdict keeps AO (``caught``, /ɔ/) distinct from AA (``cot``, /ɑ/), but
    most North American speakers merge the two. The inventory provides
    both readings of the same grapheme — ``au``/``augh``/``ough`` and ``o``
    each appear with /ɔ/ and with /ɑ/ — so emitting the AA counterpart as an
    extra pronunciation variant lets a student taught only the merged reading
    still match the word.

    Runs on ARPABET, before ``convert_arpabet_to_phoneme``, so the new variant
    goes through the same stress handling and normalization as CMU's own.
    Stress digits are preserved (``AO1`` -> ``AA1``); every AO in a variant is
    swapped; duplicates collapse in first-seen order.
    """
    expanded: list[list[str]] = []
    for variant in arpabet_variants:
        expanded.append(list(variant))
        if any(token.rstrip(STRESS_DIGITS) == "AO" for token in variant):
            expanded.append(
                [
                    "AA" + token[2:] if token.rstrip(STRESS_DIGITS) == "AO" else token
                    for token in variant
                ]
            )
    return [list(v) for v in dict.fromkeys(tuple(v) for v in expanded)]


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


def get_word_phonemes(word: str) -> tuple[Pronunciation, ...]:
    """Return every pronunciation variant for a word.

    Each element is a ``Pronunciation``: its ``phonemes`` tuple plus
    ``is_monosyllabic``. Manual entries in ``CMUDICT_OVERRIDES`` take priority;
    otherwise CMUdict is consulted. Dialect normalizations are applied to each
    variant independently. Raises ``KeyError`` if the word appears in neither
    source.

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

    pronunciations: list[Pronunciation] = []
    for arpabet_variant in _add_cot_caught_variants(word_variants):
        phonemes = tuple(convert_arpabet_to_phoneme(t) for t in arpabet_variant)
        pronunciations.append(
            Pronunciation(
                phonemes=_normalize_phonemes(phonemes),
                is_monosyllabic=count_syllables(arpabet_variant) == 1,
            )
        )
    return tuple(pronunciations)
