"""Tests for Kiswahili per-word decodability analysis."""

import pytest
from decodability.kiswahili import (
    KiswahiliStudentKnowledge,
    KiswahiliWordAnalysis,
    aggregate_scores_kenya_tusome,
    analyse_word_kiswahili,
    assess_decodability_kenya_tusome,
)
from decodability.kiswahili.analyse import is_decodable_kenya_tusome
from decodability.kiswahili.definitions import VALID_GRAPHEMES
from decodability.kiswahili.extract_words import extract_words_kiswahili
from decodability.kiswahili.scorers import (
    score_known_clusters_and_patterns_kiswahili,
    score_known_graphemes_kiswahili,
    score_whole_words_kiswahili,
)
from decodability.kiswahili.segment import Span


@pytest.fixture(scope="module")
def knowledge():
    return KiswahiliStudentKnowledge.model_validate(
        dict(
            graphemes=set("auioemktnsbyzgdjrw"),
            clusters={"kw", "st"},
            cluster_patterns={"prenasalised"},
            whole_words={"mama", "ama", "anaona", "na", "paka"},
        )
    )


@pytest.fixture(scope="module")
def no_knowledge():
    return KiswahiliStudentKnowledge()


class TestFindKiswahili:
    def test_a_fully_known_word_with_no_clusters_yields_basic_word_analysis(
        self, knowledge
    ):
        analysis = analyse_word_kiswahili("mayowe", knowledge)

        assert analysis == KiswahiliWordAnalysis(
            word="mayowe",
            graphemes=[
                Span(text="m", start=0, end=1),
                Span(text="a", start=1, end=2),
                Span(text="y", start=2, end=3),
                Span(text="o", start=3, end=4),
                Span(text="w", start=4, end=5),
                Span(text="e", start=5, end=6),
            ],
        )

    def test_a_fully_known_word_with_clusters_yields_basic_word_analysis(
        self, knowledge
    ):
        analysis = analyse_word_kiswahili("stima", knowledge)

        assert analysis == KiswahiliWordAnalysis(
            word="stima",
            graphemes=[
                Span(text="s", start=0, end=1),
                Span(text="t", start=1, end=2),
                Span(text="i", start=2, end=3),
                Span(text="m", start=3, end=4),
                Span(text="a", start=4, end=5),
            ],
            clusters=[
                Span(text="st", start=0, end=2),
            ],
        )

    def test_untaught_graphemes_are_located(self, knowledge):
        analysis = analyse_word_kiswahili("tulisikia", knowledge)

        assert analysis.unknown_graphemes == [Span(text="l", start=2, end=3)]
        assert analysis.invalid_graphemes == []

    def test_digraph_spans_cover_both_characters(self, knowledge):
        analysis = analyse_word_kiswahili("mchw", knowledge)

        assert analysis.unknown_graphemes == [Span(text="ch", start=1, end=3)]

    def test_trigraph_spans_cover_all_three_characters(self, knowledge):
        """``ng'`` is one grapheme, so its span must include the apostrophe."""
        analysis = analyse_word_kiswahili("ng'ombe", knowledge)

        assert analysis.unknown_graphemes == [Span(text="ng'", start=0, end=3)]

    def test_untaught_clusters_are_located(self, knowledge):
        analysis = analyse_word_kiswahili("mchw", knowledge)

        assert analysis.unknown_clusters == [Span(text="mchw", start=0, end=4)]

    def test_clusters_covered_by_a_taught_pattern_are_not_reported(self, knowledge):
        """``mb`` is prenasalised, which this student has been taught."""
        analysis = analyse_word_kiswahili("mbuzi", knowledge)

        assert analysis.unknown_clusters == []

    def test_explicitly_taught_clusters_are_not_reported(self, knowledge):
        analysis = analyse_word_kiswahili("stima", knowledge)

        assert analysis.unknown_clusters == []
        assert analysis.unknown_graphemes == []

    def test_a_word_with_no_cluster_reports_no_cluster_analysis(self, knowledge):
        assert analyse_word_kiswahili("baba", knowledge).unknown_clusters == []

    def test_whole_words_are_flagged_case_insensitively(self, knowledge):
        assert analyse_word_kiswahili("MAMA", knowledge).known_whole_word is True
        assert analyse_word_kiswahili("mama", knowledge).known_whole_word is True
        assert analyse_word_kiswahili("mayowe", knowledge).known_whole_word is False

    def test_spans_keep_the_original_casing_of_the_word(self, knowledge):
        analysis = analyse_word_kiswahili("Tulisikia", knowledge)

        assert analysis.unknown_graphemes == [Span(text="l", start=2, end=3)]
        assert analyse_word_kiswahili("Xylophone", knowledge).invalid_graphemes == [
            Span(text="X", start=0, end=1)
        ]

    def test_every_span_indexes_back_into_the_word(self, knowledge):
        word = "Mchanganyikox"
        # word with unknwon graphemes, clusters and an invalid grapheme
        analysis = analyse_word_kiswahili(word, knowledge)

        all_spans = (
            analysis.graphemes
            + analysis.clusters
            + analysis.invalid_graphemes
            + analysis.unknown_graphemes
            + analysis.unknown_clusters
        )
        assert all_spans, "expected this word to produce analysis"
        for span in all_spans:
            assert word[span.start : span.end] == span.text


class TestKenyaTusomeRule:
    @pytest.mark.parametrize(
        "word, expected",
        [
            ("mayowe", True),  # every grapheme taught, no clusters
            ("stima", True),  # cluster taught explicitly
            ("mbuzi", True),  # cluster covered by a taught pattern
            ("mama", True),  # sight word
            ("tulisikia", False),  # untaught grapheme "l"
            ("mchw", False),  # untaught grapheme and untaught cluster
            ("xa", False),  # grapheme outside the inventory
        ],
    )
    def test_verdicts(self, word, expected, knowledge):
        assert is_decodable_kenya_tusome(word, knowledge) is expected

    def test_a_known_whole_word_is_decodable_despite_untaught_parts(self):
        knowledge = KiswahiliStudentKnowledge.model_validate(
            dict(graphemes={"a"}, whole_words={"tulisikia"})
        )

        analysis = analyse_word_kiswahili("tulisikia", knowledge)

        assert assess_decodability_kenya_tusome(analysis) is True
        assert analysis.unknown_graphemes, "the untaught parts are still recorded"

    def test_invalid_graphemes_alone_make_a_word_non_decodable(self):
        """A rule that only checked untaught graphemes would miss this."""
        analysis = KiswahiliWordAnalysis(
            word="x", invalid_graphemes=[Span(text="x", start=0, end=1)]
        )

        assert assess_decodability_kenya_tusome(analysis) is False

    def test_a_student_taught_nothing_decodes_no_real_word(self, no_knowledge):
        words = extract_words_kiswahili("Tulisikia mayowe tukiwa sebuleni")

        decodability_verdicts = [
            is_decodable_kenya_tusome(word, no_knowledge) for word in words
        ]

        assert decodability_verdicts == [False] * len(words)

    def test_the_empty_word_has_no_obstacles_and_is_decodable(
        self, no_knowledge, knowledge
    ):
        """Deliberate: the empty word cannot be blocked by anything."""
        assert is_decodable_kenya_tusome("", no_knowledge) is True
        assert is_decodable_kenya_tusome("", knowledge) is True


class TestAnalysisModel:
    def test_the_analysis_carries_the_word_as_given(self, knowledge):
        assert analyse_word_kiswahili("Baba", knowledge).word == "Baba"

    def test_the_analysis_is_frozen(self, knowledge):
        from pydantic import ValidationError

        analysis = analyse_word_kiswahili("baba", knowledge)

        with pytest.raises(ValidationError):
            analysis.decodable = False

    def test_the_analysis_round_trips_through_serialisation(self, knowledge):
        analysis = analyse_word_kiswahili("Xylophone", knowledge)

        assert KiswahiliWordAnalysis.model_validate(analysis.model_dump()) == analysis


class TestParityWithTheScoreBasedPath:
    """The analysis path must reproduce the verdicts the scorers produced.

    The one intended difference is the empty word for a student taught nothing:
    the old path called it non-decodable, the analysis path calls it decodable.
    ``extract_words_kiswahili`` never yields an empty word, so no caller sees it.
    """

    TEXT = (
        "Tulisikia mayowe tukiwa sebuleni. Sote tukatoka. Kutazama tukaona moto "
        "ukiwaka sana. Waya wa stima uliwaka moto. Mchanganyiko wa mwembamba na "
        "transista. Xylophone quiz blanketi hospitali mbwea kunywea ng'ombe mchw "
        "Baba MAMA paka."
    )

    @staticmethod
    def old_verdict(word: str, knowledge: KiswahiliStudentKnowledge) -> bool:
        scores = {
            score.__name__: score(word, knowledge)
            for score in (
                score_known_graphemes_kiswahili,
                score_known_clusters_and_patterns_kiswahili,
                score_whole_words_kiswahili,
            )
        }
        return aggregate_scores_kenya_tusome(scores) == 1.0

    @pytest.mark.parametrize("taught_count", [1, 5, 12, len(VALID_GRAPHEMES)])
    @pytest.mark.parametrize(
        "patterns",
        [set(), {"prenasalised"}, {"labialised_glide", "palatalised_glide"}],
    )
    def test_verdicts_match_for_every_word_and_profile(self, taught_count, patterns):
        knowledge = KiswahiliStudentKnowledge.model_validate(
            dict(
                graphemes=set(sorted(VALID_GRAPHEMES)[:taught_count]),
                cluster_patterns=patterns,
                whole_words={"mama", "paka"},
            )
        )
        words = sorted(set(extract_words_kiswahili(self.TEXT)))

        nondecodable_words = [
            word for word in words if not is_decodable_kenya_tusome(word, knowledge)
        ]

        assert nondecodable_words == sorted(
            {word for word in words if not self.old_verdict(word, knowledge)}
        )
