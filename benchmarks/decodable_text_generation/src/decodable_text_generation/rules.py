"""Orthographic rule targets, registered per language.

Some lessons teach a spelling *rule* rather than a new grapheme (double the
consonant before -ed/-ing, drop the -e before a vowel suffix). Profiles
declare these in ``target_knowledge.rules``; each rule id maps to a word
predicate answering "does this word exercise the rule?".

Adding a language means adding entries to ``RULE_CHECKERS[language]`` —
callers never touch language internals.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import cache

from decodability.english.extract_phonemes import get_word_phonemes

# language -> rule id -> does this word exercise the rule
RULE_CHECKERS: dict[str, dict[str, Callable[[str], bool]]] = {}


def rule_ids(language: str = "en") -> frozenset[str]:
    """Rule ids the language knows how to check."""
    return frozenset(RULE_CHECKERS.get(language, ()))


def rule_exercised(rule: str, word: str, language: str = "en") -> bool:
    """Whether ``word`` exercises ``rule`` under ``language``'s checks."""
    check = RULE_CHECKERS.get(language, {}).get(rule)
    return check is not None and check(word.lower())


# --- English -----------------------------------------------------------------

_C = "b-df-hj-np-tv-z"  # consonant letters, for VC/CVC word shapes
_VC_CVC_RE = re.compile(rf"^([aeiou][{_C}]|[{_C}][aeiou][{_C}])$")

# The drop-e lesson's suffixes: -ed, -er, -est, -ly. Other vowel-initial
# suffixes (plural -s/-es, -ing) are different lessons.
_DROP_E_SUFFIXES = ("est", "ed", "er", "ly")
_DOUBLING_SUFFIXES = {
    "doubling_ed_ing": ("ing", "ed"),
    "doubling_er_est": ("est", "er"),
}

_DOUBLING_CONSONANTS = "bcdfghjklmnpqrstvwxz"


@cache
def _is_english_word(word: str) -> bool:
    try:
        get_word_phonemes(word)
        return True
    except KeyError:
        return False


def _fold_phonemes(ipa: tuple[str, ...]) -> tuple[str, ...]:
    """Normalize a pronunciation for stem/halves comparison.

    ``ʌ`` folds to ``ə`` so reduced halves compare equal (``upon`` =
    up /ʌp/ + on); unstressed ``i`` folds to ``ɪ`` (any+body ==
    /ɛnɪbɑdi/); adjacent identical phonemes collapse (can+not ==
    /kænɑt/, not /kænnɑt/)."""
    folded = ["ə" if p == "ʌ" else "ɪ" if p == "i" else p for p in ipa]
    return tuple(p for i, p in enumerate(folded) if i == 0 or p != folded[i - 1])


@cache
def _pronunciations(word: str) -> set[tuple[str, ...]]:
    """IPA shapes for ``word``; empty when absent from the dictionary."""
    try:
        return {
            _fold_phonemes(tuple(p.ipa for p in variant.phonemes))
            for variant in get_word_phonemes(word)
        }
    except KeyError:
        return set()


# Suffix allomorphs as IPA tails. A word pronounced as ``stem + allomorph``
# is regular inflection — used both to reject would-be drop-e bases that
# are only dictionary junk (``walke`` is a homophone of ``walk``, so
# ``walked`` is walk + /t/, not walke minus e + -ed) and to keep junk stems
# (``toda`` the acronym) from making compounds like ``today`` = to + day
# look inflected.
_SUFFIX_ALLOMORPHS: dict[str, tuple[tuple[str, ...], ...]] = {
    "ed": (("t",), ("d",), ("ɪ", "d"), ("ə", "d")),
    "er": (("ə", "r"), ("ɚ",)),
    "es": (("ə", "z"), ("ɪ", "z"), ("s",), ("z",)),
    "est": (("ɪ", "s", "t"), ("ə", "s", "t"), ("ɛ", "s", "t")),
    "ing": (("ɪ", "ŋ"), ("ə", "ŋ"), ("i", "ŋ")),
    "ly": (("l", "i"), ("l", "ɪ")),
    "s": (("s",), ("z",), ("ə", "z"), ("ɪ", "z")),
    "y": (("i",), ("ɪ",)),
}


def _plain_inflection(word: str, base: str, suffix: str) -> bool:
    """``word`` can be pronounced as ``base`` + a suffix allomorph."""
    base_prons = _pronunciations(base)
    if not base_prons:
        return False
    return any(
        word_pron == base_pron + tail
        for word_pron in _pronunciations(word)
        for base_pron in base_prons
        for tail in _SUFFIX_ALLOMORPHS[suffix]
    )


def _is_inflected_form(word: str, stem: str, suffix: str) -> bool:
    """``stem`` is a plausible word and ``word`` sounds like ``stem`` + suffix.

    A dictionary hit alone isn't proof — when the word has a pronunciation,
    require it to decompose into ``stem`` + a suffix allomorph. Words with
    no dictionary entry keep the old orthographic verdict.
    """
    if not _is_compound_half(stem):
        return False
    if not _pronunciations(word):
        return True
    return _plain_inflection(word, stem, suffix)


def _doubling_match(word: str, suffixes: tuple[str, ...]) -> bool:
    """word = stem + suffix where the stem ends in a doubled consonant.

    Any doubled consonant counts — lexical (``passed`` <- pass + -ed) or
    suffixation-derived (``stopped`` <- stop + -ed): at the surface level the
    story is still exercising doubled-consonant-plus-suffix forms.
    """
    for sfx in suffixes:
        if not word.endswith(sfx):
            continue
        stem = word[: -len(sfx)]
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] in _DOUBLING_CONSONANTS:
            return True
    return False


def _drop_e_match(word: str) -> bool:
    """word = base minus silent e + vowel suffix (``baking`` <- ``bake``).

    The base must end in a consonant (so ``being`` <- be doesn't count) and
    ``base + e`` must be a dictionary word (so ``cats`` doesn't). A word
    pronounced as plain ``base + suffix`` is regular inflection, not a
    dropped silent e — ``walked`` is walk + /t/ even though ``walke`` is a
    dictionary entry.
    """
    for sfx in _DROP_E_SUFFIXES:
        if not word.endswith(sfx) or len(word) <= len(sfx) + 1:
            continue
        base = word[: -len(sfx)]
        if base[-1] not in _DOUBLING_CONSONANTS:
            continue
        if _is_english_word(base + "e") and not _plain_inflection(word, base, sfx):
            return True
    return False


# Function words short enough to appear as compound halves (into, today,
# outside, someone) but not themselves in junk-heavy dictionary lookups.
# "be" and "as" are excluded on purpose: nearly every be-/as- split is a
# prefix word, not a compound (began, behind, because, asleep).
_SHORT_COMPOUND_HALVES = frozenset(
    {
        "in",
        "on",
        "up",
        "to",
        "of",
        "go",
        "no",
        "so",
        "my",
        "by",
        "do",
        "we",
        "he",
        "me",
        "at",
        "is",
        "it",
        "if",
        "am",
    }
)


# Dictionary junk that passes the sound check anyway (``almost`` splits
# alm+ost, /ɑlm/ + /oʊst/ = /ɑlmoʊst/). These are acronym/proper-noun
# entries, not words a child would read as a compound half.
_JUNK_COMPOUND_HALVES = frozenset(
    {
        "alm",
        "ost",
        "aga",
        "mon",
        "dow",
        "ain",
        "gan",
        "ide",
        "ins",
        "ote",
        "cam",
        "pus",
        "imp",
        "ort",
        "tux",
        "edo",
        "org",
        "ob",
        "unt",
        "bel",
        "ong",
        "beco",
        "dise",
    }
)


def _is_compound_half(chunk: str) -> bool:
    """A chunk is a compound half if it's a dictionary word of >= 3 letters
    (smaller entries are too noisy: ``ba``/``th``/``ed`` are all in CMUdict)
    or a known short function word."""
    return (
        len(chunk) >= 3
        and chunk not in _JUNK_COMPOUND_HALVES
        and _is_english_word(chunk)
    ) or chunk in _SHORT_COMPOUND_HALVES


def _inflection_stems(word: str) -> list[str]:
    """Stems that make ``word`` a grammatical inflection (want+ed,
    hot+test, bake+s).

    A stem can hide a doubled consonant (``hott``→``hot``) or a dropped
    silent e (``bak``→``bake``). When the word has a dictionary
    pronunciation it must also sound like ``stem + suffix``: CMU lists
    junk like the acronym ``toda``, which alone would make ``today``
    look inflected rather than to + day.
    """
    stems = []
    for sfx in ("ing", "est", "ed", "er", "es", "ly", "y", "s"):
        if not word.endswith(sfx) or len(word) <= len(sfx) + 2:
            continue
        stem = word[: -len(sfx)]
        if _is_inflected_form(word, stem, sfx):
            stems.append(stem)
        if (
            len(stem) >= 2
            and stem[-1] == stem[-2]
            and _is_inflected_form(word, stem[:-1], sfx)
        ):
            stems.append(stem[:-1])  # doubling: hottest <- hot
        if stem[-1] in _DOUBLING_CONSONANTS and _is_inflected_form(
            word, stem + "e", sfx
        ):
            stems.append(stem + "e")  # drop-e: baked <- bake
    return stems


_VOWEL_CHARS = frozenset("aeiouæɑɔəɜɛɪʊʌɚɝ")
# Vowels that can be reduced further in an unstressed compound half.
_REDUCED_VOWELS = frozenset({"ə", "ɪ", "ʊ", "ɚ"})


def _halves_pron(
    word_pron: tuple[str, ...], a_pron: tuple[str, ...], b_pron: tuple[str, ...]
) -> bool:
    """One pronunciation triple matches, modulo natural compound drift.

    Exact match after folding — or vowel drift confined to the *second*
    half, where compounds deaccent (``fireman`` /mən/ != man, ``anybody``
    /ə/ != body). First-half vowels carry the compound's main stress, so
    drift there means it isn't a compound (``betray`` /bɪ/ != bet)."""
    combo = _fold_phonemes(a_pron + b_pron)
    if word_pron == combo or len(word_pron) != len(combo):
        return word_pron == combo
    # A boundary geminate absorbs one phoneme off ``a``'s end.
    cut = len(a_pron) - (a_pron[-1:] == b_pron[:1])
    diffs = [i for i, (w, c) in enumerate(zip(word_pron, combo)) if w != c]
    return all(
        i >= cut and word_pron[i] in _REDUCED_VOWELS and set(combo[i]) & _VOWEL_CHARS
        for i in diffs
    )


def _sounds_like_compound(word: str, a: str, b: str) -> bool:
    """``word`` can be pronounced as ``a`` + ``b``.

    Orthographic splits that don't sound out are not compounds —
    ``betray`` is /bɪtreɪ/, not /bɛt/ + /reɪ/. When the word or a half has
    no dictionary pronunciation, keep the orthographic verdict (same
    convention as ``_is_inflected_form``)."""
    word_prons = _pronunciations(word)
    if not word_prons:
        return True
    a_prons, b_prons = _pronunciations(a), _pronunciations(b)
    if not a_prons or not b_prons:
        return True
    return any(
        _halves_pron(wp, ap, bp)
        for wp in word_prons
        for ap in a_prons
        for bp in b_prons
    )


def _compound_match(word: str) -> bool:
    """word = two real words joined (``sunshine`` = sun + shine).

    Both halves must be plausible words, the word must not just be an
    inflected stem (``hottest`` = hot+test is a letter-level coincidence),
    and the halves must sound out as the word (``betray`` != bet + ray).
    """
    stems = _inflection_stems(word)
    if stems:
        # An inflected surface form still exercises the compound rule
        # when its stem is itself a compound: sometimes = sometime + -s.
        return any(_compound_match(stem) for stem in stems)
    return any(
        _is_compound_half(word[:i])
        and _is_compound_half(word[i:])
        and _sounds_like_compound(word, word[:i], word[i:])
        for i in range(2, len(word) - 1)
    )


RULE_CHECKERS["en"] = {
    "vc_cvc": lambda w: bool(_VC_CVC_RE.fullmatch(w)),
    "drop_e": _drop_e_match,
    "compound": _compound_match,
    **{
        rule: (lambda sfxs: lambda w: _doubling_match(w, sfxs))(suffixes)
        for rule, suffixes in _DOUBLING_SUFFIXES.items()
    },
}
