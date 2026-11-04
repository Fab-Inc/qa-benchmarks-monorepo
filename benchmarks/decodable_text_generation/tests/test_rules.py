"""Tests for orthographic rule targets (rules.py)."""

import pytest
from decodable_text_generation.rules import rule_exercised, rule_ids


def test_rule_ids_english() -> None:
    assert rule_ids("en") == {
        "vc_cvc",
        "drop_e",
        "compound",
        "doubling_ed_ing",
        "doubling_er_est",
    }


def test_rule_ids_other_languages_empty() -> None:
    assert rule_ids("sw") == frozenset()
    assert rule_ids("xx") == frozenset()


def test_rule_exercised_unknown_rule_and_language() -> None:
    assert not rule_exercised("not_a_rule", "stopped")
    assert not rule_exercised("vc_cvc", "at", "sw")


def test_rule_exercised_is_case_insensitive() -> None:
    assert rule_exercised("doubling_ed_ing", "STOPPED")


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        # VC shape
        ("at", True),
        ("it", True),
        ("up", True),
        ("on", True),
        ("am", True),
        # CVC shape
        ("cat", True),
        ("sit", True),
        ("hen", True),
        ("bus", True),
        ("yes", True),
        # CV shape is not VC/CVC
        ("be", False),
        ("go", False),
        ("to", False),
        ("we", False),
        # CCV and longer shapes are out
        ("she", False),
        ("the", False),
        ("cats", False),
        ("chat", False),
        ("a", False),
    ],
)
def test_vc_cvc(word: str, expected: bool) -> None:
    assert rule_exercised("vc_cvc", word) is expected


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        # Suffixation-derived doubling
        ("stopped", True),
        ("stopping", True),
        ("hopping", True),
        ("running", True),
        ("swimming", True),
        ("hugged", True),
        ("planned", True),
        ("batted", True),
        # Lexical doubling counts too (pass + -ed)
        ("passed", True),
        ("kissed", True),
        # No doubled consonant at the stem's end
        ("walked", False),
        ("played", False),
        ("reading", False),
        # -est/-er doubling is a different lesson
        ("hottest", False),
        ("hotter", False),
        # "ing" alone leaves a one-letter stem
        ("sing", False),
        ("king", False),
    ],
)
def test_doubling_ed_ing(word: str, expected: bool) -> None:
    assert rule_exercised("doubling_ed_ing", word) is expected


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("hotter", True),
        ("hottest", True),
        ("bigger", True),
        ("biggest", True),
        ("madder", True),
        # Lexical double at the stem boundary
        ("smaller", True),
        # No doubled consonant
        ("faster", False),
        ("later", False),
        # -ed/-ing doubling is a different lesson
        ("stopped", False),
    ],
)
def test_doubling_er_est(word: str, expected: bool) -> None:
    assert rule_exercised("doubling_er_est", word) is expected


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        # base + vowel suffix where base + e is a real word
        ("baked", True),
        ("baker", True),
        ("nicer", True),
        ("nicest", True),
        ("hiked", True),
        ("timed", True),
        ("loved", True),
        ("ruler", True),
        ("diner", True),
        # -ing is not one of the drop-e lesson's suffixes
        ("baking", False),
        ("being", False),
        # base + e is not a real word
        ("jumped", False),
        ("dinner", False),
        # Pronounced as plain base + suffix: regular inflection, not drop-e —
        # "walke"/"wante" are homophone/name junk entries in CMUdict
        ("walked", False),
        ("wanted", False),
        ("hottest", False),
        # base ends in a vowel, not a consonant
        ("namely", False),
        ("likely", False),
        ("holy", False),
        # Too short to carry the suffix
        ("red", False),
        ("bed", False),
        # -s is not a drop-e suffix
        ("cats", False),
    ],
)
def test_drop_e(word: str, expected: bool) -> None:
    assert rule_exercised("drop_e", word) is expected


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("sunshine", True),
        ("bedtime", True),
        ("outside", True),
        ("inside", True),
        ("sunset", True),
        ("into", True),
        # "toda" is a CMU entry, but "today" doesn't sound like toda + -y
        ("today", True),
        ("anybody", True),
        ("yesterday", True),
        # Inflected forms are not compounds
        ("hottest", False),
        ("walked", False),
        ("cats", False),
        ("baking", False),
        ("makes", False),
        ("horses", False),
        ("sunny", False),
        # Prefix words, not compounds — be-/as- splits are excluded halves
        ("before", False),
        ("because", False),
        ("belong", False),
        ("began", False),
        ("asleep", False),
        # Dictionary junk halves (acronyms/proper nouns), not real compounds
        ("almost", False),
        ("tuxedo", False),
        ("import", False),
        # Single words with no real-word split
        ("cat", False),
    ],
)
def test_compound(word: str, expected: bool) -> None:
    assert rule_exercised("compound", word) is expected
