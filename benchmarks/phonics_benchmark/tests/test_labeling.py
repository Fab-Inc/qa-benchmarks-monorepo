"""Tests for the labeling module."""

import pandas as pd
import pytest
from phonics_benchmark.labeling import (
    label_nondecodable_words,
    resolve_decodability_config,
)

CONFIG = {
    "scoring_methods": [
        "score_known_graphemes_kiswahili",
        "score_whole_words_kiswahili",
    ],
    "aggregations": {
        "kenya_tusome_score": {"name": "kiswahili_kenya_tusome_aggregator"},
    },
}


def test_resolve_decodability_config_returns_callables() -> None:
    """Config names resolve to two scorers and one aggregation callable."""
    scoring_methods, aggregations = resolve_decodability_config(CONFIG, "sw")
    assert len(scoring_methods) == 2
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

    labels = label_nondecodable_words(items, CONFIG, "sw")

    assert labels.iloc[0] == ["hufanya", "kufanyia", "ofisi", "vifaa"]


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

    labels = label_nondecodable_words(items, CONFIG, "sw")

    assert labels.iloc[0] == ["kuku"]


def test_label_nondecodable_words_requires_expected_columns() -> None:
    """A missing content/student_knowledge column raises a clear error."""
    items = pd.DataFrame({"content": ["mama"]})
    with pytest.raises(ValueError, match="student_knowledge"):
        label_nondecodable_words(items, CONFIG, "sw")
