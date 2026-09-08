"""Tests for scoring generated texts on decodability."""

import math

import pandas as pd
import pytest
from phonics_benchmark.scoring import (
    REFERENCE_LENGTH,
    add_per_item_decodability,
    graded_score,
    summarize_by_model,
    summarize_by_phase,
    summarize_by_scenario,
)

# A student taught most of the inventory but no consonant clusters, so
# "mwalimu" (mw) and "mbuzi" (mb) are non-decodable while "mama" is not.
KNOWLEDGE = {
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
        "g",
        "z",
        "y",
        "d",
        "j",
        "r",
        "p",
        "f",
        "v",
        "sh",
        "ch",
        "ng",
    ],
    "whole_words": ["mama", "ama", "anaona", "na", "paka"],
}


def build_row(text: str, *, delimiter_found: bool = True, **extra: object) -> dict:
    return {
        "model": "m1",
        "benchmark_item_id": extra.pop("benchmark_item_id", "item-1"),
        "generated_text": text,
        "delimiter_found": delimiter_found,
        "student_knowledge": KNOWLEDGE,
        **extra,
    }


def score_one(text: str, *, delimiter_found: bool = True) -> pd.Series:
    frame = pd.DataFrame([build_row(text, delimiter_found=delimiter_found)])
    return add_per_item_decodability(frame, "sw").iloc[0]


class TestGradedScore:
    """The length-normalised per-item score."""

    def test_perfect_text_scores_one(self) -> None:
        """No non-decodable words means a perfect score, at any length."""
        assert graded_score(60, 0) == 1.0
        assert graded_score(5, 0) == 1.0

    def test_empty_text_scores_zero(self) -> None:
        """Writing nothing is not a way to avoid non-decodable words."""
        assert graded_score(0, 0) == 0.0

    def test_all_words_nondecodable_scores_zero(self) -> None:
        """A wholly undecodable text bottoms out at zero."""
        assert graded_score(10, 10) == 0.0

    def test_matches_the_documented_formula(self) -> None:
        """The score is (1 - nondecodable_rate) ** REFERENCE_LENGTH exactly."""
        assert graded_score(50, 1) == pytest.approx((49 / 50) ** REFERENCE_LENGTH)

    def test_is_length_normalised(self) -> None:
        """The same error *rate* scores the same regardless of text length.

        This is the property that makes the metric measure the model rather
        than how much it chose to write.
        """
        assert graded_score(20, 1) == pytest.approx(graded_score(200, 10))

    def test_is_monotone_in_error_count(self) -> None:
        """Each additional non-decodable word strictly lowers the score."""
        scores = [graded_score(50, n) for n in range(5)]
        assert scores == sorted(scores, reverse=True)
        assert len(set(scores)) == len(scores)


class TestPerItemDecodability:
    """Per-item counts, rates and flags."""

    def test_fully_decodable_text(self) -> None:
        """A text the student can read scores 1.0 and passes strictly."""
        row = score_one("Mama anaona paka.")
        assert row["graded_score"] == 1.0
        assert row["strict_pass"]
        assert row["word_decodable_rate"] == 1.0
        assert row["unique_nondecodable_words"] == []

    def test_flags_nondecodable_words(self) -> None:
        """Words needing an untaught cluster are counted and named."""
        row = score_one("Mama anaona mwalimu.")
        assert row["n_words"] == 3
        assert row["n_nondecodable_words"] == 1
        assert row["unique_nondecodable_words"] == ["mwalimu"]
        assert not row["strict_pass"]
        assert row["graded_score"] == pytest.approx((2 / 3) ** REFERENCE_LENGTH)

    def test_counts_every_occurrence_of_a_repeated_bad_word(self) -> None:
        """The rate is per occurrence: a repeated bad word is a repeated failure."""
        row = score_one("Mama mwalimu mama mwalimu.")
        assert row["n_words"] == 4
        assert row["n_nondecodable_words"] == 2
        assert row["n_unique_nondecodable_words"] == 1
        assert row["word_decodable_rate"] == pytest.approx(0.5)

    def test_word_and_unique_rates_diverge_under_repetition(self) -> None:
        """Padding with a safe word dilutes the per-word rate, not the unique rate."""
        row = score_one("mwalimu " + "mama " * 9)
        assert row["word_decodable_rate"] == pytest.approx(0.9)
        assert row["unique_word_decodable_rate"] == pytest.approx(0.5)

    def test_empty_text_scores_zero_with_undefined_rates(self) -> None:
        """An empty text earns zero, but its rates are NaN not a flattering 1.0."""
        row = score_one("")
        assert row["graded_score"] == 0.0
        assert not row["strict_pass"]
        assert row["n_words"] == 0
        assert math.isnan(row["word_decodable_rate"])

    def test_text_with_no_extractable_words_is_treated_as_empty(self) -> None:
        """Punctuation and digits alone yield no words to score."""
        row = score_one("123 ... !!!")
        assert row["n_words"] == 0
        assert row["graded_score"] == 0.0

    def test_format_failure_scores_zero_and_leaves_rates_undefined(self) -> None:
        """A response without <text> tags is scored 0, not dropped.

        Dropping it would let a model dodge hard items by malformatting;
        grading the raw response would charge it for its own preamble.
        """
        row = score_one("Sure! Here is your story.", delimiter_found=False)
        assert row["format_failure"]
        assert row["graded_score"] == 0.0
        assert not row["strict_pass"]
        assert math.isnan(row["word_decodable_rate"])
        assert math.isnan(row["n_words"])

    def test_attributes_failures_to_the_scorer_that_rejected_them(self) -> None:
        """The cause split separates an untaught cluster from an untaught grapheme.

        With the whole inventory minus clusters taught, "mwalimu" fails only on
        its cluster — the diagnostic that matters because only the grade-2
        profiles in the dataset are taught clusters.
        """
        row = score_one("Mama anaona mwalimu.")
        assert row["n_fail_score_known_clusters_and_patterns_kiswahili"] == 1
        assert row["n_fail_score_known_graphemes_kiswahili"] == 0

    def test_flags_texts_shorter_than_the_reference_length(self) -> None:
        """Short texts are flagged because their score is high variance."""
        assert score_one("Mama anaona paka.")["under_reference_length"]
        assert not score_one("mama " * 30)["under_reference_length"]

    def test_unique_word_ratio_records_repetition(self) -> None:
        """Diversity is recorded as a diagnostic, never scored."""
        assert score_one("mama " * 4)["unique_word_ratio"] == pytest.approx(0.25)

    def test_requires_its_input_columns(self) -> None:
        """A frame missing the extracted text fails loudly rather than silently."""
        with pytest.raises(ValueError, match="missing required column"):
            add_per_item_decodability(pd.DataFrame({"model": ["m1"]}), "sw")


class TestSummarizeByModel:
    """Per-model rollups."""

    @pytest.fixture
    def scored(self) -> pd.DataFrame:
        rows = [
            build_row("Mama anaona paka.", benchmark_item_id="a"),
            build_row("Mama anaona mwalimu.", benchmark_item_id="b"),
            build_row("nope", benchmark_item_id="c", delimiter_found=False),
        ]
        return add_per_item_decodability(pd.DataFrame(rows), "sw")

    def test_one_row_per_model(self, scored: pd.DataFrame) -> None:
        """The summary is keyed by model name."""
        summary = summarize_by_model(scored)
        assert list(summary["model_name"]) == ["m1"]
        assert summary.loc[0, "n_items"] == 3

    def test_headline_metrics_count_format_failures_as_zero(
        self, scored: pd.DataFrame
    ) -> None:
        """A malformatted response drags the headline score down."""
        summary = summarize_by_model(scored).iloc[0]
        expected = (1.0 + (2 / 3) ** REFERENCE_LENGTH + 0.0) / 3
        assert summary["graded_score"] == pytest.approx(expected)
        assert summary["strict_pass_rate"] == pytest.approx(1 / 3)
        assert summary["format_failure_rate"] == pytest.approx(1 / 3)

    def test_pooled_rates_exclude_format_failures(self, scored: pd.DataFrame) -> None:
        """Word rates describe phonics, so an unparsed response is left out."""
        summary = summarize_by_model(scored).iloc[0]
        # 6 words across the two scorable texts, 1 of them non-decodable.
        assert summary["word_decodable_rate"] == pytest.approx(5 / 6)
        assert summary["nondecodable_per_100_words"] == pytest.approx(100 / 6)

    def test_undefined_metrics_are_nan_not_zero(self) -> None:
        """With nothing scorable, rates are NaN rather than a misleading 0 or 1."""
        rows = [build_row("x", benchmark_item_id="a", delimiter_found=False)]
        summary = summarize_by_model(
            add_per_item_decodability(pd.DataFrame(rows), "sw")
        ).iloc[0]
        assert math.isnan(summary["word_decodable_rate"])
        assert math.isnan(summary["nondecodable_per_100_words"])
        assert summary["graded_score"] == 0.0

    def test_pools_failure_causes(self, scored: pd.DataFrame) -> None:
        """Failure attribution is summed per model for the report."""
        summary = summarize_by_model(scored).iloc[0]
        key = "pooled_n_fail_score_known_clusters_and_patterns_kiswahili"
        assert summary[key] == 1


class TestBreakdowns:
    """Grouped views used for the report's plots."""

    @pytest.fixture
    def scored(self) -> pd.DataFrame:
        rows = [
            build_row(
                "Mama anaona paka.",
                benchmark_item_id="a",
                curriculum_phase="late",
                scenario_id="story",
            ),
            build_row(
                "Mama anaona mwalimu.",
                benchmark_item_id="b",
                curriculum_phase="very-early",
                scenario_id="dialogue",
            ),
        ]
        return add_per_item_decodability(pd.DataFrame(rows), "sw")

    def test_by_phase(self, scored: pd.DataFrame) -> None:
        """Each model/phase pair gets its own mean score and item count."""
        result = summarize_by_phase(scored).set_index("curriculum_phase")
        assert result.loc["late", "graded_score"] == 1.0
        assert result.loc["very-early", "graded_score"] < 1.0
        assert result.loc["late", "n_items"] == 1

    def test_by_scenario(self, scored: pd.DataFrame) -> None:
        """The same rollup, grouped by generation scenario instead."""
        result = summarize_by_scenario(scored).set_index("scenario_id")
        assert result.loc["story", "strict_pass_rate"] == 1.0
        assert result.loc["dialogue", "strict_pass_rate"] == 0.0

    def test_breakdowns_require_their_grouping_column(
        self, scored: pd.DataFrame
    ) -> None:
        """Missing the grouping column is an error, not an empty frame."""
        with pytest.raises(ValueError, match="missing required column"):
            summarize_by_phase(scored.drop(columns=["curriculum_phase"]))
