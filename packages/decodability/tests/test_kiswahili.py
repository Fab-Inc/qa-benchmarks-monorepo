import pytest
from decodability.kiswahili import (
    score_known_syllables,
    score_syllable_pattern,
    score_word_length,
    tokenize,
)
from decodability.kiswahili.models import KiswahiliKnowledge


class TestKiswahiliKnowledgeValidator:
    @pytest.mark.parametrize(
        "known_syllable_patterns, expected",
        [
            ({"CV", "CVCV", "CCV"}, True),
            (set(), True),
            ({"CV", "ABC"}, False),
        ],
    )
    def test_kiswahili_knowledge_syllable_pattern_checks_for_valid_cv_patterns(
        self, known_syllable_patterns: set[str], expected: bool
    ):
        """Test that the KiswahiliKnowledge model raises a ValueError when invalid CV
        patterns are provided.
        """
        if expected:
            KiswahiliKnowledge(known_syllable_patterns=known_syllable_patterns)
        else:
            with pytest.raises(ValueError):
                KiswahiliKnowledge(known_syllable_patterns=known_syllable_patterns)


class TestScorers:
    @pytest.mark.parametrize(
        "word, known_syllables, expected",
        [
            ("baba", {"ba"}, 1.0),
            ("baba", {"be"}, 0.0),
            ("baba", set(), 0.0),
        ],
    )
    def test_score_known_syllables(
        self, word: str, known_syllables: set[str], expected: float
    ):
        """Test the sw_score_known_syllables function."""
        student_knowledge = KiswahiliKnowledge(known_syllables=known_syllables)
        score = score_known_syllables(word, student_knowledge)
        assert score == expected, f"Expected {expected} but got {score}"

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
    def test_score_word_length(self, word: str, max_word_length: int, expected: float):
        """Test the score_word_length function."""
        student_knowledge = KiswahiliKnowledge(max_word_length=max_word_length)
        score = score_word_length(word, student_knowledge)
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
        ],
    )
    def test_score_syllable_pattern(
        self, word: str, known_syllable_patterns: set[str], expected: float
    ):
        """Test the score_syllable_pattern function."""
        student_knowledge = KiswahiliKnowledge(
            known_syllable_patterns=known_syllable_patterns
        )
        score = score_syllable_pattern(word, student_knowledge)
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
    def test_tokenize_swahili(self, text: str, expected: list[str]):
        """Test the tokenize function."""
        assert tokenize(text) == expected
