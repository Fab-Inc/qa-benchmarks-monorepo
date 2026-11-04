"""Decompose words into the knowledge units a profile's story exercises.

Units are tuples:

- ``("gpc", grapheme_symbol, phoneme_string)`` — a grapheme-phoneme
  correspondence used by a parse of the word. Symbols are the parser's
  canonical forms: edge markers stripped, so the ``-er`` unit appears as
  ``("gpc", "er", "ər")``.
- ``("sight", word)`` — the word itself, or an affix-stripped stem of it,
  when that form is one of the profile's known sight words.

GPC units are emitted only from parses whose matches are all taught — a
word the student cannot decode is not evidence the story exercises its
correspondences.
"""

from __future__ import annotations

from dataclasses import dataclass

from decodability.english.extract_phonemes import get_word_phonemes
from decodability.english.gpc_parser import parse_gpcs
from decodability.english.models import EnglishStudentKnowledge
from decodability.english.scorers import (
    GPC_INVENTORY,
    _is_taught_match,
    _sight_word_stem,
    normalize_taught_gpcs,
)


@dataclass(frozen=True)
class EnglishUnitsProfile:
    taught_set: frozenset[tuple[str, str]]
    taught_graphemes: frozenset[str]
    sight_words: frozenset[str]
    affixes: frozenset[str]


def build_english_units_profile(
    student_knowledge: EnglishStudentKnowledge,
) -> EnglishUnitsProfile:
    taught_set, taught_graphemes = normalize_taught_gpcs(student_knowledge.gpcs)
    return EnglishUnitsProfile(
        frozenset(taught_set),
        frozenset(taught_graphemes),
        frozenset(student_knowledge.sight_words),
        frozenset(student_knowledge.affixes),
    )


_ES_STEM_ENDINGS = ("s", "x", "z", "sh", "ch")
_ES_PHONEMES = ({"ə", "ɪ"}, {"z"})


def _stem_phonemes_prefix(word: str, stem: str) -> bool:
    """Some pronunciation of ``word`` = a pronunciation of ``stem`` + əz/ɪz."""
    try:
        stem_variants = get_word_phonemes(stem)
        word_variants = get_word_phonemes(word)
    except KeyError:
        return False
    stem_forms = {tuple(p.ipa for p in v.phonemes) for v in stem_variants}
    for variant in word_variants:
        phonemes = variant.phonemes
        if len(phonemes) < 3:
            continue
        if phonemes[-2].ipa in _ES_PHONEMES[0] and phonemes[-1].ipa in _ES_PHONEMES[1]:
            if tuple(p.ipa for p in phonemes[:-2]) in stem_forms:
                return True
    return False


def _has_real_stem(word: str, suffix_start: int, symbol: str) -> bool:
    """A suffix unit is only real when the word minus the suffix is a word.

    The stem must be a word itself (``buses`` → ``bus``, ``yes`` → ``y``
    fails), with one spelling convention allowed: an un-doubled final
    consonant (``busses`` → ``bus``). Silent-e stems like ``horses`` →
    ``hors`` don't count — those are ``-s`` on an e-final stem, not
    ``-es`` words. For ``-es`` specifically the stem must also end in
    s/x/z/sh/ch — the sounds that take the ``-es`` form — and sound like
    the word minus /əz/: ``fuses`` fails because its stem is ``fuse``
    /fjuz/, not the dictionary word ``fus`` /fʌs/.
    """
    stem = word[:suffix_start]
    if len(stem) < 2:
        return False
    candidates = [stem]
    if stem[-1] == stem[-2]:
        candidates.append(stem[:-1])
    for candidate in candidates:
        if symbol == "es" and not candidate.endswith(_ES_STEM_ENDINGS):
            continue
        try:
            get_word_phonemes(candidate)
        except KeyError:
            continue
        if symbol != "es" or _stem_phonemes_prefix(word, candidate):
            return True
    return False


def word_units_english(word: str, profile: EnglishUnitsProfile) -> set[tuple]:
    """Units exercised by ``word`` under ``profile``'s knowledge."""
    lowered = word.lower()
    units: set[tuple] = set()
    if lowered in profile.sight_words:
        units.add(("sight", lowered))
    stem = _sight_word_stem(
        lowered, profile.sight_words, profile.affixes, profile.taught_set
    )
    if stem is not None:
        units.add(("sight", stem))

    try:
        variants = get_word_phonemes(lowered)
    except KeyError:
        return units
    for variant in variants:
        for parse in parse_gpcs(lowered, variant, GPC_INVENTORY):
            if all(
                _is_taught_match(match, profile.taught_set, profile.taught_graphemes)
                for match in parse
            ):
                for match in parse:
                    # A suffix unit must attach to a real stem: "yes" is not
                    # the -es plural of "y".
                    if match.gpc.is_suffix and not _has_real_stem(
                        lowered, match.letter_positions[0], match.gpc.symbol
                    ):
                        continue
                    units.add(("gpc", match.gpc.symbol, "".join(match.gpc.phonemes)))
    return units


PROFILE_BUILDERS = {"en": build_english_units_profile}
WORD_UNITS = {"en": word_units_english}
