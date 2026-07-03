import pandas as pd
import pytest
from decodability import rowwise
from decodability.kiswahili.aggregators import (
    aggregate_scores_kenya_tusome,
)
from decodability.kiswahili.extract_words import extract_words_kiswahili
from decodability.kiswahili.models import KiswahiliStudentKnowledge
from decodability.kiswahili.scorers import (
    score_known_graphemes_kiswahili,
    score_whole_words_kiswahili,
)


class TestKiswahiliStudentKnowledgeModel:
    @pytest.mark.parametrize(
        "known_graphemes, should_pass",
        [
            ({"q"}, False),
            ({"Q"}, False),
            (set(), True),
            ({"m", "a", "c"}, False),
            ({"m", "ng'"}, True),
        ],
    )
    def test_kiswahili_student_knowledge_grapheme_checks_for_valid_graphemes(
        self, known_graphemes: set[str], should_pass: bool
    ):
        """Test that the KiswahiliStudentKnowledge model raises a ValueError when
        invalid graphemes are provided
        """
        if should_pass:
            KiswahiliStudentKnowledge(graphemes=known_graphemes)
        else:
            with pytest.raises(ValueError):
                KiswahiliStudentKnowledge(graphemes=known_graphemes)

    def test_kiswahili_student_knowledge_lowercases_whole_words(self):
        """Test that the KiswahiliStudentKnowledge model lowercases whole words."""
        knowledge = KiswahiliStudentKnowledge(whole_words={"Baba", "mama", "ANAONA"})
        assert knowledge.whole_words == {"baba", "mama", "anaona"}

    @pytest.mark.parametrize(
        "known_graphemes, known_whole_words, expected_repr",
        [
            (
                {"m", "a", "n"},
                {"baba", "mama"},
                '- Known graphemes: "a", "m", "n"\n- Known whole words: "baba", "mama"',
            ),
            (set(), {"baba"}, '- Known graphemes: None\n- Known whole words: "baba"'),
            (
                {"m", "a"},
                set(),
                '- Known graphemes: "a", "m"\n- Known whole words: None',
            ),
            (set(), set(), "- Known graphemes: None\n- Known whole words: None"),
        ],
    )
    def test_kiswahili_student_knowledge_repr(
        self, known_graphemes: set[str], known_whole_words: set[str], expected_repr: str
    ):
        """Test the string representation of KiswahiliStudentKnowledge."""
        knowledge = KiswahiliStudentKnowledge(
            graphemes=known_graphemes, whole_words=known_whole_words
        )
        assert repr(knowledge) == expected_repr


class TestScorers:
    @pytest.mark.parametrize(
        "word, known_graphemes, expected",
        [
            ("baba", {"b", "a"}, 1.0),
            ("baba", {"b"}, 0.0),
            ("baba", set(), 0.0),
            ("Baba", {"b", "a"}, 1.0),
            ("kunywea", {"k", "u", "ny", "w", "e", "a"}, 1.0),
            ("kunywea", {"k", "u", "n", "w", "e", "a"}, 0.0),
        ],
    )
    def test_score_known_graphemes_kiswahili(
        self, word: str, known_graphemes: set[str], expected: float
    ):
        """Test the score_known_graphemes_kiswahili function."""
        student_knowledge = KiswahiliStudentKnowledge(graphemes=known_graphemes)
        score = score_known_graphemes_kiswahili(word, student_knowledge)
        assert score == expected

    def test_score_whole_words_kiswahili(self):
        """Test the score_whole_words_kiswahili function."""
        student_knowledge = KiswahiliStudentKnowledge(whole_words={"baba", "mama"})
        assert score_whole_words_kiswahili("baba", student_knowledge) == 1.0
        assert score_whole_words_kiswahili("mama", student_knowledge) == 1.0
        assert score_whole_words_kiswahili("babu", student_knowledge) == 0.0


class TestKiswahiliTokenizer:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("Juma ana baba na mama.", ["Juma", "ana", "baba", "na", "mama"]),
            ("Baba ana duka sokoni.", ["Baba", "ana", "duka", "sokoni"]),
            ("Ng'ombe ni mnyama.", ["Ng'ombe", "ni", "mnyama"]),
            ("Ng'ang'ania.", ["Ng'ang'ania"]),
        ],
    )
    def test_tokenize_kiswahili(self, text: str, expected: list[str]):
        """Test the tokenize function."""
        assert extract_words_kiswahili(text) == expected


class TestKiswahiliAggregators:
    def test_tusome_aggregator_returns_1_for_known_whole_words(self):
        """A known whole word scores 1.0 regardless of grapheme knowledge."""
        scores = {
            "score_whole_words_kiswahili": 1.0,
            "score_known_graphemes_kiswahili": 0.0,
        }
        assert aggregate_scores_kenya_tusome(scores) == 1.0

    def test_tusome_aggregator_falls_back_to_grapheme_score(self):
        """An unknown whole word falls back to the grapheme score."""
        scores = {
            "score_whole_words_kiswahili": 0.0,
            "score_known_graphemes_kiswahili": 1.0,
        }
        assert aggregate_scores_kenya_tusome(scores) == 1.0

    def test_tusome_aggregator_through_rowwise(self):
        """The adapter applies the row function across a frame."""
        df = pd.DataFrame(
            {
                "score_whole_words_kiswahili": [1.0, 0.0, 0.0],
                "score_known_graphemes_kiswahili": [0.0, 0.0, 1.0],
            }
        )
        score_columns = list(df.columns)
        final_scores = rowwise(aggregate_scores_kenya_tusome)(df, score_columns)

        pd.testing.assert_series_equal(final_scores, pd.Series([1.0, 0.0, 1.0]))
