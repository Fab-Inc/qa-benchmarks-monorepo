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


def _is_english_word(word: str) -> bool:
    try:
        get_word_phonemes(word)
        return True
    except KeyError:
        return False


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
    ``base + e`` must be a dictionary word (so ``cats`` doesn't).
    """
    for sfx in _DROP_E_SUFFIXES:
        if not word.endswith(sfx) or len(word) <= len(sfx) + 1:
            continue
        base = word[: -len(sfx)]
        if base[-1] in _DOUBLING_CONSONANTS and _is_english_word(base + "e"):
            return True
    return False


# Function words short enough to appear as compound halves (into, today,
# outside, someone) but not themselves in junk-heavy dictionary lookups.
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
        "be",
        "we",
        "he",
        "me",
        "at",
        "is",
        "it",
        "as",
        "if",
        "us",
        "am",
        "an",
    }
)


def _is_compound_half(chunk: str) -> bool:
    """A chunk is a compound half if it's a dictionary word of >= 3 letters
    (smaller entries are too noisy: ``ba``/``th``/``ed`` are all in CMUdict)
    or a known short function word."""
    return (
        len(chunk) >= 3 and _is_english_word(chunk)
    ) or chunk in _SHORT_COMPOUND_HALVES


def _looks_inflected(word: str) -> bool:
    """word = stem + grammatical suffix (want+ed, hot+test, bake+s).

    True when stripping a suffix leaves a word — with doubling (``hott``→
    ``hot``) or drop-e restoration (``bak``→``bake``) — so inflected forms
    aren't mistaken for compounds.
    """
    for sfx in ("ing", "est", "ed", "er", "es", "ly", "y", "s"):
        if not word.endswith(sfx) or len(word) <= len(sfx) + 2:
            continue
        stem = word[: -len(sfx)]
        if _is_compound_half(stem):
            return True
        if len(stem) >= 2 and stem[-1] == stem[-2] and _is_compound_half(stem[:-1]):
            return True  # doubling: hottest <- hot
        if stem[-1] in _DOUBLING_CONSONANTS and _is_compound_half(stem + "e"):
            return True  # drop-e: baked <- bake
    return False


def _compound_match(word: str) -> bool:
    """word = two real words joined (``sunshine`` = sun + shine).

    Both halves must be plausible words and the word must not just be an
    inflected stem (``hottest`` = hot+test is a letter-level coincidence).
    """
    if _looks_inflected(word):
        return False
    return any(
        _is_compound_half(word[:i]) and _is_compound_half(word[i:])
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
