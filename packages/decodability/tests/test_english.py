import pandas as pd
import pytest
from decodability import rowwise
from decodability.english.aggregators import aggregate_scores_upl_curriculum
from decodability.english.extract_words import extract_words_english
from decodability.english.models import EnglishStudentKnowledge
from decodability.english.scorers import (
    score_known_gpcs_english,
    score_whole_words_english,
)


class TestEnglishStudentKnowledgeModel:
    @pytest.mark.parametrize(
        "known_gpcs, should_pass",
        [
            ([], True),
            ([("s", ("s",)), ("a", ("æ",)), ("t", ("t",))], True),
            ([("x", ("k", "s"))], True),
            ([("a-e", ("eɪ",))], True),
            ([("z", ("xyz",))], False),
            ([("a", ("æ",)), ("bogus", ("q",))], False),
        ],
    )
    def test_english_student_knowledge_gpc_checks_for_valid_gpcs(
        self,
        known_gpcs: list[tuple[str, tuple[str, ...]]],
        should_pass: bool,
    ):
        """Reject GPCs that are not in UPL_GPC_INVENTORY."""
        if should_pass:
            EnglishStudentKnowledge(gpcs=known_gpcs)
        else:
            with pytest.raises(ValueError):
                EnglishStudentKnowledge(gpcs=known_gpcs)

    def test_english_student_knowledge_lowercases_whole_words(self):
        """Whole words are lowercased on ingest."""
        knowledge = EnglishStudentKnowledge(whole_words={"The", "cat", "DOG"})
        assert knowledge.whole_words == {"the", "cat", "dog"}


class TestScorers:
    @pytest.mark.parametrize(
        "word, known_gpcs, expected",
        [
            ("cat", [("c", ("k",)), ("a", ("æ",)), ("t", ("t",))], 1.0),
            ("cat", [("c", ("k",)), ("t", ("t",))], 0.0),
            ("cat", [], 0.0),
            ("Cat", [("c", ("k",)), ("a", ("æ",)), ("t", ("t",))], 1.0),
            (
                "smoke",
                [
                    ("s", ("s",)),
                    ("m", ("m",)),
                    ("o-e", ("oʊ",)),
                    ("k", ("k",)),
                ],
                1.0,
            ),
            (
                "smoke",
                [("s", ("s",)), ("m", ("m",)), ("k", ("k",))],
                0.0,
            ),
            ("fifth", [("f", ("f",)), ("i", ("ɪ",)), ("th", ("θ",))], 0.0),
        ],
    )
    def test_score_known_gpcs_english(
        self,
        word: str,
        known_gpcs: list[tuple[str, tuple[str, ...]]],
        expected: float,
    ):
        """Score is 1.0 iff every GPC the parser derives is in the student's set."""
        student_knowledge = EnglishStudentKnowledge(gpcs=known_gpcs)
        assert score_known_gpcs_english(word, student_knowledge) == expected

    def test_score_whole_words_english(self):
        """Sight-word check is case-insensitive."""
        student_knowledge = EnglishStudentKnowledge(whole_words={"the", "and"})
        assert score_whole_words_english("the", student_knowledge) == 1.0
        assert score_whole_words_english("The", student_knowledge) == 1.0
        assert score_whole_words_english("and", student_knowledge) == 1.0
        assert score_whole_words_english("cat", student_knowledge) == 0.0


class TestEnglishTokenizer:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("The cat sat on the mat.", ["The", "cat", "sat", "on", "the", "mat"]),
            (
                "It's a beautiful day, isn't it?",
                ["It's", "a", "beautiful", "day", "isn't", "it"],
            ),
            (
                "Don't stop. We've been waiting.",
                ["Don't", "stop", "We've", "been", "waiting"],
            ),
            ('She said, "Hello, world!"', ["She", "said", "Hello", "world"]),
            (
                "We’ve got a we’re situation.",
                ["We've", "got", "a", "we're", "situation"],
            ),
            (
                "Compound-word hyphens split; one-two",
                ["Compound", "word", "hyphens", "split", "one", "two"],
            ),
            ("Numbers 123 mixed 456 in.", ["Numbers", "mixed", "in"]),
            ("", []),
        ],
    )
    def test_tokenize_english(self, text: str, expected: list[str]):
        """Test the extract_words_english function."""
        assert extract_words_english(text) == expected


class TestEnglishAggregators:
    def test_upl_aggregator_returns_1_for_known_whole_words(self):
        """A known whole word scores 1.0 regardless of GPC knowledge."""
        scores = {
            "score_whole_words_english": 1.0,
            "score_known_gpcs_english": 0.0,
        }
        assert aggregate_scores_upl_curriculum(scores) == 1.0

    def test_upl_aggregator_falls_back_to_gpc_score(self):
        """An unknown whole word falls back to the GPC score."""
        scores = {
            "score_whole_words_english": 0.0,
            "score_known_gpcs_english": 1.0,
        }
        assert aggregate_scores_upl_curriculum(scores) == 1.0

    def test_upl_aggregator_returns_0_when_both_missing(self):
        """No sight-word match and no GPC coverage → 0.0."""
        scores = {
            "score_whole_words_english": 0.0,
            "score_known_gpcs_english": 0.0,
        }
        assert aggregate_scores_upl_curriculum(scores) == 0.0

    def test_upl_aggregator_through_rowwise(self):
        """The adapter applies the row function across a frame."""
        df = pd.DataFrame(
            {
                "score_whole_words_english": [1.0, 0.0, 0.0],
                "score_known_gpcs_english": [0.0, 0.0, 1.0],
            }
        )
        score_columns = list(df.columns)
        final_scores = rowwise(aggregate_scores_upl_curriculum)(df, score_columns)

        pd.testing.assert_series_equal(final_scores, pd.Series([1.0, 0.0, 1.0]))
