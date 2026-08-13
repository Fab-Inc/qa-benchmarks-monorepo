"""Tests for the dataset module."""

import pandas as pd
import pytest
from phonics_benchmark.curation import (
    add_labels_and_analysis,
    filter_benchmark_items,
    get_nondecodable_word_analysis,
)
from phonics_benchmark.data_io import build_benchmark_item_candidates_dataframe
from phonics_benchmark.models import DecodableText, StudentKnowledge

PROFILES = [
    StudentKnowledge(
        student_knowledge_id="p1",
        language_iso_code="sw",
        student_knowledge={"graphemes": ["a"], "whole_words": ["mama"]},
    ),
    StudentKnowledge(
        student_knowledge_id="p2",
        language_iso_code="sw",
        student_knowledge={"graphemes": ["a", "m"], "whole_words": ["na"]},
    ),
]
TEXTS = [
    DecodableText(text_id="t1", language_iso_code="sw", content="mama"),
    DecodableText(text_id="t2", language_iso_code="sw", content="na"),
    DecodableText(text_id="t3", language_iso_code="sw", content="paka"),
]
DECODABILITY_CONFIGS: dict[str, dict] = {
    "sw": {
        "scoring_methods": [
            "score_known_graphemes_kiswahili",
            "score_whole_words_kiswahili",
        ],
        "aggregations": {
            "kenya_tusome_score": {"name": "kiswahili_kenya_tusome_aggregator"},
        },
    },
}


def test_build_benchmark_item_candidates_is_full_cross_product() -> None:
    """Every profile is paired with every text, with a composite id."""
    items = build_benchmark_item_candidates_dataframe(PROFILES, TEXTS)

    assert len(items) == len(PROFILES) * len(TEXTS)
    assert "p1_t1" in set(items["benchmark_item_id"])
    assert "p2_t3" in set(items["benchmark_item_id"])


def test_build_benchmark_item_candidates_keeps_one_language_column() -> None:
    """Both inputs carry language_iso_code; the join must not split it in two."""
    items = build_benchmark_item_candidates_dataframe(PROFILES, TEXTS)

    assert "language_iso_code" in items.columns
    assert not {"language_iso_code_x", "language_iso_code_y"} & set(items.columns)


def test_build_benchmark_item_candidates_rejects_mismatched_languages() -> None:
    """The pipeline handles one language at a time, so a mix is an error."""
    english_text = DecodableText(text_id="t9", language_iso_code="en", content="cat")

    with pytest.raises(ValueError, match="Expected one language_iso_code"):
        build_benchmark_item_candidates_dataframe(PROFILES, [english_text])


def test_build_benchmark_item_candidates_rejects_mixed_languages_within_texts() -> None:
    """A stray second language among otherwise matching records is caught too."""
    english_text = DecodableText(text_id="t9", language_iso_code="en", content="cat")

    with pytest.raises(ValueError, match="Expected one language_iso_code"):
        build_benchmark_item_candidates_dataframe(PROFILES, [*TEXTS, english_text])


@pytest.mark.parametrize(
    ("profiles", "texts"),
    [
        ([], TEXTS),
        (PROFILES, []),
        ([], []),
    ],
)
def test_build_benchmark_item_candidates_rejects_empty_inputs(
    profiles: list[StudentKnowledge], texts: list[DecodableText]
) -> None:
    """A cross-product needs at least one profile and one text."""
    with pytest.raises(ValueError, match="No benchmark items"):
        build_benchmark_item_candidates_dataframe(profiles, texts)


def test_add_analysis_columns_computes_ratio() -> None:
    """Analysis columns count words and derive the non-decodable ratio."""
    # `nondecodable_words` is not supplied: add_labels_and_analysis derives it
    # from the student knowledge. "mama" and "na" are known whole words, "paka"
    # is decodable from p/a/k, and "soko" needs the untaught s/o, so exactly one
    # of the four words is non-decodable.
    items = pd.DataFrame(
        {
            "content": ["mama na paka soko"],  # 4 words
            "student_knowledge": [
                {"graphemes": ["a", "p", "k"], "whole_words": ["mama", "na"]}
            ],
        }
    )

    enriched = add_labels_and_analysis(items, DECODABILITY_CONFIGS["sw"], "sw")

    assert enriched.loc[0, "nondecodable_words"] == ["soko"]

    assert enriched.loc[0, "num_nondecodable_words"] == 1
    assert enriched.loc[0, "num_total_words"] == 4
    assert enriched.loc[0, "nondecodable_ratio"] == 0.25


def test_add_analysis_columns_drops_items_with_no_words() -> None:
    """An empty vocab_set would give a 0/0 ratio, so the item is dropped.

    A NaN ratio satisfies neither `== 0.0` nor `> max_nondecodable_ratio`, so such
    an item would otherwise drift silently into the catch-all sampling bucket.
    """
    items = pd.DataFrame(
        {
            "benchmark_item_id": ["keep-me", "no-words"],
            "content": ["mama na paka soko", "..."],
            "student_knowledge": [
                {"graphemes": ["a"], "whole_words": ["mama", "na"]},
                {"graphemes": ["a"], "whole_words": []},
            ],
            "nondecodable_words": [["soko"], []],
            "vocab_set": [{"mama", "na", "paka", "soko"}, set()],
        }
    )

    with pytest.warns(UserWarning, match="no-words"):
        enriched = add_labels_and_analysis(items, DECODABILITY_CONFIGS["sw"], "sw")

    assert list(enriched["benchmark_item_id"]) == ["keep-me"]
    assert not enriched["nondecodable_ratio"].isna().any()


def test_nondecodable_word_analysis_counts_small_nonzero_ratios() -> None:
    """Items just above 0.0 belong in the first bin, not dropped from the histogram."""
    items = pd.DataFrame(
        {
            "nondecodable_ratio": [0.0, 0.03, 0.04, 0.12],
            "num_nondecodable_words": [0, 1, 1, 3],
        }
    )

    analysis = get_nondecodable_word_analysis(items)
    histogram = analysis["Non-decodable ratio histogram"]

    assert histogram["(0.00, 0.05]"] == 2
    assert histogram["(0.10, 0.15]"] == 1
    # Zero-ratio items are reported on their own line, not in the histogram.
    assert analysis["No non-decodable words"] == 1
    assert sum(histogram.values()) == len(items) - 1


def test_filter_benchmark_items_respects_thresholds() -> None:
    """Sampling caps and the max-ratio drop are all applied."""
    items = pd.DataFrame(
        {
            "nondecodable_ratio": (
                [0.0] * 30  # 30 zero-ratio items -> capped
                + [0.05] * 50  # 50 one-word items -> capped
                + [0.5] * 5  # 5 too-hard items -> dropped
            ),
            "num_nondecodable_words": [0] * 30 + [1] * 50 + [10] * 5,
        }
    )

    filtered = filter_benchmark_items(
        items,
        max_zero_nondecodable=20,
        max_one_nondecodable=40,
        max_nondecodable_ratio=0.2,
        sample_size_other=None,
        random_state=42,
    )

    assert (filtered["nondecodable_ratio"] == 0.0).sum() == 20
    assert (filtered["num_nondecodable_words"] == 1).sum() == 40
    assert (filtered["nondecodable_ratio"] > 0.2).sum() == 0
    assert len(filtered) == 60


def test_filter_benchmark_items_is_reproducible() -> None:
    """The same random_state yields the same sampled subset."""
    items = pd.DataFrame(
        {
            "nondecodable_ratio": [0.0] * 100,
            "num_nondecodable_words": [0] * 100,
        }
    )
    a = filter_benchmark_items(items, random_state=7)
    b = filter_benchmark_items(items, random_state=7)
    assert list(a.index) == list(b.index)
