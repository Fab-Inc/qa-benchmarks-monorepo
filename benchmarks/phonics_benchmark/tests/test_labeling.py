"""Tests for the bridge to the deterministic decodability checker."""

import pytest
from phonics_benchmark.labeling import (
    DECODABILITY_CONFIGS,
    get_decodability_config,
    resolve_decodability_config,
    score_text_decodability,
)

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

CONFIG, SCORE_COLUMN = get_decodability_config("sw")


def score(text: str, knowledge: dict | None = None):
    return score_text_decodability(
        text,
        knowledge if knowledge is not None else KNOWLEDGE,
        CONFIG,
        "sw",
        score_column=SCORE_COLUMN,
    )


class TestGetDecodabilityConfig:
    """Looking up the config a language is scored with."""

    def test_returns_the_config_and_its_score_column(self) -> None:
        """The score column travels with the config so callers can't mismatch."""
        config, score_column = get_decodability_config("sw")
        assert config is DECODABILITY_CONFIGS["sw"]
        assert score_column == "kenya_tusome_score"

    def test_unregistered_language_is_an_explicit_gap(self) -> None:
        """An unconfigured language fails with a message naming what exists."""
        with pytest.raises(NotImplementedError, match="No decodability config"):
            get_decodability_config("xx")


class TestResolveDecodabilityConfig:
    """Turning config strings into callables."""

    def test_resolves_every_named_scorer_and_aggregation(self) -> None:
        """Each configured name becomes a callable from the language registry."""
        scoring_methods, aggregations = resolve_decodability_config(CONFIG, "sw")
        assert len(scoring_methods) == len(CONFIG["scoring_methods"])
        assert all(callable(method) for method in scoring_methods)
        assert set(aggregations) == {"kenya_tusome_score"}

    def test_binds_aggregation_params(self) -> None:
        """A params block is bound onto the aggregation via functools.partial."""
        config = {
            "scoring_methods": ["score_known_graphemes_kiswahili"],
            "aggregations": {
                "weighted": {
                    "name": "weighted_average",
                    "params": {"weights": {"score_known_graphemes_kiswahili": 1.0}},
                }
            },
        }
        _, aggregations = resolve_decodability_config(config, "sw")
        assert aggregations["weighted"].keywords == {
            "weights": {"score_known_graphemes_kiswahili": 1.0}
        }

    def test_unknown_scorer_name_raises(self) -> None:
        """A typo'd scorer name fails rather than silently scoring nothing."""
        config = {"scoring_methods": ["not_a_scorer"], "aggregations": {}}
        with pytest.raises(KeyError):
            resolve_decodability_config(config, "sw")


class TestScoreTextDecodability:
    """Scoring a whole generated text word by word."""

    def test_counts_words_not_unique_words(self) -> None:
        """Every occurrence counts, so repetition of a bad word is repeated harm."""
        result = score("Mama mwalimu mama mwalimu.")
        assert result.n_words == 4
        assert result.n_unique_words == 2
        assert result.n_nondecodable_words == 2
        assert result.n_unique_nondecodable_words == 1

    def test_names_the_nondecodable_words_lowercased_and_sorted(self) -> None:
        """The offending words are reported for inspection, in a stable order."""
        result = score("Mwalimu anaona mbuzi.")
        assert result.unique_nondecodable_words == ["mbuzi", "mwalimu"]

    def test_case_is_ignored_when_scoring(self) -> None:
        """A capitalised word is the same word: no double-counting."""
        assert score("Mama mama MAMA.").n_nondecodable_words == 0

    def test_fully_decodable_text_has_no_failures(self) -> None:
        """A text within the student's knowledge yields zero non-decodable words."""
        result = score("Mama anaona paka.")
        assert result.n_nondecodable_words == 0
        assert result.unique_nondecodable_words == []

    def test_attributes_failures_per_scorer(self) -> None:
        """Failure causes are keyed by scorer so grapheme vs cluster is separable."""
        result = score("Mama anaona mwalimu.")
        assert result.failure_causes["score_known_clusters_and_patterns_kiswahili"] == 1
        assert result.failure_causes["score_known_graphemes_kiswahili"] == 0

    def test_untaught_grapheme_is_attributed_to_the_grapheme_scorer(self) -> None:
        """A word using a letter-sound never taught fails on graphemes."""
        result = score("mama tai", {"graphemes": ["m", "a"], "whole_words": []})
        assert result.failure_causes["score_known_graphemes_kiswahili"] == 1

    def test_cluster_patterns_make_cluster_words_decodable(self) -> None:
        """Marking the native cluster patterns known unblocks words like mwalimu.

        This is how the grade-2 profiles in the canonical dataset express the
        cluster knowledge that lets a student read cluster words at all.
        """
        with_patterns = dict(KNOWLEDGE)
        with_patterns["cluster_patterns"] = ["labialised_glide"]
        assert score("mwalimu", with_patterns).n_nondecodable_words == 0
        assert score("mwalimu").n_nondecodable_words == 1

    def test_empty_text_returns_an_all_zero_result(self) -> None:
        """Nothing to score is not an error; the caller decides the score."""
        result = score("")
        assert result.n_words == 0
        assert result.n_unique_words == 0
        assert result.unique_nondecodable_words == []
        assert result.failure_causes == {}

    def test_text_of_only_punctuation_yields_no_words(self) -> None:
        """The word extractor drops punctuation and digits, leaving nothing to score."""
        assert score("123 --- !!!").n_words == 0

    def test_rejects_a_score_column_the_config_does_not_produce(self) -> None:
        """Asking for a column no aggregation emits is caught immediately."""
        with pytest.raises(ValueError, match="not one of the config's aggregation"):
            score_text_decodability(
                "mama", KNOWLEDGE, CONFIG, "sw", score_column="nope"
            )

    def test_invalid_student_knowledge_raises(self) -> None:
        """Knowledge that fails the language model's rules is rejected loudly."""
        with pytest.raises(Exception):
            score("mama", {"graphemes": ["NOT_A_GRAPHEME"]})
