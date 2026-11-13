"""Tests for the labeling module."""

from hashlib import sha256
from importlib.resources import files

import pandas as pd
import pytest
from decodability.english.definitions import gpc_pairs
from phonics_benchmark.labeling import (
    KNOWLEDGE_INVENTORY,
    get_decodability_config,
    get_grapheme_breakdown,
    knowledge_inventory,
    label_nondecodable_words,
    parse_student_knowledge,
    resolve_decodability_config,
)

# Checksums of the frozen inventory files; a change here re-labels and can
# reject stored items, so it has to be deliberate.
FROZEN_INVENTORY_SHA256 = {
    "en": "30f09372faed1372a4167e358fbb3765b658e44ac0da8cfc1466e38cbe857634",
}

CONFIG = {
    "scoring_methods": [
        "score_known_graphemes_kiswahili",
        "score_known_clusters_and_patterns_kiswahili",
        "score_whole_words_kiswahili",
    ],
    "aggregations": {
        "kenya_tusome_score": {"name": "kiswahili_kenya_tusome_aggregator"},
    },
    "score_column": "kenya_tusome_score",
}


def test_resolve_decodability_config_returns_callables() -> None:
    """Config names resolve to three scorers and one aggregation callable."""
    scoring_methods, aggregations = resolve_decodability_config(CONFIG, "sw")
    assert len(scoring_methods) == 3
    assert set(aggregations) == {"kenya_tusome_score"}
    assert all(callable(method) for method in scoring_methods)


def test_label_nondecodable_words_returns_sorted_lowercased_unique_words() -> None:
    """The returned words are lowercased and unique."""
    items = pd.DataFrame(
        {
            "content": [
                "Ofisi huwa na vifaa vya kufanyia kazi mbalimbali. Vifaa hivi hufanya "
                "kazi kuwa rahisi."
            ],
            "student_knowledge": [
                {
                    "graphemes": [
                        "a",
                        "u",
                        "i",
                        "o",
                        "e",
                        "m",
                        "k",
                        "t",
                        "n",
                        "w",
                        "l",
                        "h",
                        "s",
                        "b",
                        "y",
                        "z",
                        "g",
                        "d",
                        "r",
                        "p",
                        "v",
                        "sh",
                    ],
                    "whole_words": ["mama", "ama", "anaona", "na", "paka"],
                }
            ],
        }
    )
    labels = label_nondecodable_words(
        items, CONFIG, "sw", score_column=CONFIG["score_column"]
    )

    # The "f" words are flagged for an untaught grapheme; "mbalimbali" ("mb") and
    # "vya" ("vy") for clusters this student knows neither directly nor by pattern.
    assert labels.iloc[0] == [
        "hufanya",
        "kufanyia",
        "mbalimbali",
        "ofisi",
        "vifaa",
        "vya",
    ]


def test_label_nondecodable_words_flags_untaught_word() -> None:
    """Words that are neither sight words nor all-taught-graphemes are flagged.

    With only graphemes {a, m} taught plus the listed sight words, 'kuku' (k, u)
    is the only non-decodable word in the passage.
    """
    items = pd.DataFrame(
        {
            "content": ["Mama anaona kuku na paka."],
            "student_knowledge": [
                {
                    "graphemes": ["a", "m"],
                    "whole_words": ["mama", "ama", "anaona", "na", "paka"],
                }
            ],
        }
    )

    labels = label_nondecodable_words(
        items, CONFIG, "sw", score_column=CONFIG["score_column"]
    )

    assert labels.iloc[0] == ["kuku"]


@pytest.mark.parametrize("language", sorted(KNOWLEDGE_INVENTORY))
def test_knowledge_inventory_file_is_frozen(language: str) -> None:
    path = files("phonics_benchmark").joinpath(KNOWLEDGE_INVENTORY[language])
    # Hash the text, as it is loaded: a Windows checkout has CRLF line endings.
    text = path.read_text(encoding="utf-8")
    assert sha256(text.encode("utf-8")).hexdigest() == FROZEN_INVENTORY_SHA256[language]


@pytest.mark.parametrize("language", sorted(KNOWLEDGE_INVENTORY))
def test_parse_student_knowledge_uses_the_frozen_inventory(language: str) -> None:
    profile = parse_student_knowledge({}, language)
    assert profile.gpc_inventory is knowledge_inventory(language)


def test_english_inventory_spells_uni_as_the_stored_profiles_do() -> None:
    """Stored profiles spell uni- /juni/; the copy keeps that spelling."""
    pairs = gpc_pairs(knowledge_inventory("en"))
    assert ("uni", "juni") in pairs
    assert ("uni", "junɪ") not in pairs


def test_label_nondecodable_words_parses_with_the_frozen_inventory() -> None:
    """A stored /juni/ profile still matches uni- in 'unit' when labelling."""
    items = pd.DataFrame(
        {
            "content": ["Unit."],
            "student_knowledge": [{"gpcs": [["uni", "juni"], ["t", "t"]]}],
        }
    )
    config = get_decodability_config("en")

    labels = label_nondecodable_words(
        items, config, "en", score_column=config["score_column"]
    )

    assert labels.iloc[0] == []


def test_label_nondecodable_words_requires_expected_columns() -> None:
    """A missing content/student_knowledge column raises a clear error."""
    items = pd.DataFrame({"content": ["mama"]})
    with pytest.raises(ValueError, match="student_knowledge"):
        label_nondecodable_words(
            items, CONFIG, "sw", score_column=CONFIG["score_column"]
        )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Ng'ombe ni mnyama", "ng'/o/m/b/e n/i m/ny/a/m/a"),
        (
            "Baba ana wembe, mimi nina embe",
            "b/a/b/a a/n/a w/e/m/b/e, m/i/m/i n/i/n/a e/m/b/e",
        ),
    ],
)
def test_grapheme_breakdown_matches_expected(text: str, expected: str) -> None:
    """The grapheme breakdown of a Kiswahili sentence is as expected."""
    breakdown = get_grapheme_breakdown(text, "sw")
    assert breakdown == expected
