import pandas as pd
import pytest
from decodability import rowwise
from decodability.kiswahili.aggregators import (
    aggregate_scores_kenya_tusome,
    aggregate_scores_kiswahili_complex,
)
from decodability.kiswahili.models import KiswahiliStudentKnowledge
from decodability.kiswahili.scorers import (
    primerpro_settings,
    score_known_graphemes_kiswahili,
    score_known_syllables_kiswahili,
    score_syllable_pattern_kiswahili,
    score_whole_words_kiswahili,
    score_word_length_kiswahili,
)
from decodability.kiswahili.tokenizer import tokenize_kiswahili
from primerpro import Word


class TestKiswahiliStudentKnowledgeValidator:
    @pytest.mark.parametrize(
        "known_syllable_patterns, expected",
        [
            ({"CV", "CVCV", "CCV"}, True),
            (set(), True),
            ({"CV", "ABC"}, False),
        ],
    )
    def test_kiswahili_student_knowledge_syllable_pattern_checks_for_valid_cv_patterns(
        self, known_syllable_patterns: set[str], expected: bool
    ):
        """Test that the KiswahiliKnowledge model raises a ValueError when invalid CV
        patterns are provided.
        """
        if expected:
            KiswahiliStudentKnowledge(syllable_patterns=known_syllable_patterns)
        else:
            with pytest.raises(ValueError):
                KiswahiliStudentKnowledge(syllable_patterns=known_syllable_patterns)


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

    @pytest.mark.parametrize(
        "word, known_syllables, expected",
        [
            ("baba", {"ba"}, 1.0),
            ("baba", {"be"}, 0.0),
            ("baba", set(), 0.0),
            ("Baba", {"ba"}, 1.0),  # case-insensitive
            # ("NG'OMBE", {"ng'o", "m", "be"}, 1.0),  # case-insensitive with apostrophe
        ],
    )
    def test_score_known_syllables_kiswahili(
        self, word: str, known_syllables: set[str], expected: float
    ):
        """Test the score_known_syllables_kiswahili function."""
        student_knowledge = KiswahiliStudentKnowledge(syllables=known_syllables)
        score = score_known_syllables_kiswahili(word, student_knowledge)
        primerpro_word = Word(word, primerpro_settings)
        syllable_symbols = [
            syllable.display_syllable.lower() for syllable in primerpro_word.syllables
        ]
        assert score == expected, (
            f"Expected {expected} but got {score} (syllables: {syllable_symbols})"
        )

    @pytest.mark.parametrize(
        "word, max_word_length, expected",
        [
            ("baba", 2, 0.5),
            ("baba", 4, 1.0),
            ("baba", 5, 1.0),
            ("mti", 3, 1.0),
            ("mti", 2, 0.6666666666666666),
            ("mti", 0, 0.0),
        ],
    )
    def test_score_word_length_kiswahili(
        self, word: str, max_word_length: int, expected: float
    ):
        """Test the score_word_length_kiswahili function."""
        student_knowledge = KiswahiliStudentKnowledge(max_word_length=max_word_length)
        score = score_word_length_kiswahili(word, student_knowledge)
        assert score == pytest.approx(expected), f"Expected {expected} but got {score}"

    @pytest.mark.parametrize(
        "word, known_syllable_patterns, expected",
        [
            ("baba", {"CV"}, 0.0),
            ("baba", {"CVCV"}, 1.0),
            ("baba", {"CCV"}, 0.0),
            ("baba", set(), 0.0),
            ("mti", {"CV"}, 0.0),
            ("mti", {"CV", "CCV", "CVCV"}, 1.0),
            ("Mti", {"CV", "CCV"}, 1.0),  # case-insensitive
        ],
    )
    def test_score_syllable_pattern_kiswahili(
        self, word: str, known_syllable_patterns: set[str], expected: float
    ):
        """Test the score_syllable_pattern_kiswahili function."""
        student_knowledge = KiswahiliStudentKnowledge(
            syllable_patterns=known_syllable_patterns
        )
        score = score_syllable_pattern_kiswahili(word, student_knowledge)
        assert score == expected, f"Expected {expected} but got {score}"


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
        assert tokenize_kiswahili(text) == expected


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

    @pytest.mark.parametrize(
        "scores, expected_final_score",
        [
            (
                {
                    "score_whole_words_kiswahili": 1.0,
                    "score_known_graphemes_kiswahili": 1.0,
                    "score_syllable_pattern_kiswahili": 1.0,
                    "score_word_length_kiswahili": 0.6,
                },
                1.0,
            ),
            (
                {
                    "score_whole_words_kiswahili": 0.0,
                    "score_known_graphemes_kiswahili": 0.0,
                    "score_syllable_pattern_kiswahili": 1.0,
                    "score_word_length_kiswahili": 0.6,
                },
                0.0,
            ),
            (
                {
                    "score_whole_words_kiswahili": 0.0,
                    "score_known_graphemes_kiswahili": 1.0,
                    "score_syllable_pattern_kiswahili": 0.0,
                    "score_word_length_kiswahili": 0.6,
                },
                0.0,
            ),
            (
                {
                    "score_whole_words_kiswahili": 0.0,
                    "score_known_graphemes_kiswahili": 1.0,
                    "score_syllable_pattern_kiswahili": 1.0,
                    "score_word_length_kiswahili": 0.6,
                },
                0.6,
            ),
        ],
    )
    def test_complex_aggregator_decision_tree(self, scores, expected_final_score):
        """Whole word -> 1.0; else gated on graphemes and syllable pattern, then
        falls through to the word-length score."""

        assert aggregate_scores_kiswahili_complex(scores) == expected_final_score
