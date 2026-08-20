import pandas as pd
import pytest
from decodability import rowwise
from decodability.kiswahili.aggregators import (
    aggregate_scores_kenya_tusome,
)
from decodability.kiswahili.definitions import (
    VALID_CLUSTERS,
    ClusterPattern,
    get_clusters,
    get_graphemes,
)
from decodability.kiswahili.extract_words import extract_words_kiswahili
from decodability.kiswahili.models import KiswahiliStudentKnowledge
from decodability.kiswahili.scorers import (
    score_known_clusters_and_patterns_kiswahili,
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

    @pytest.mark.parametrize(
        "known_whole_words, should_pass",
        [
            ({"baba", "mama"}, True),
            (set(), True),
            ({"Baba"}, False),
            ({"baba", "ANAONA"}, False),
        ],
    )
    def test_kiswahili_student_knowledge_rejects_uppercase_whole_words(
        self, known_whole_words: set[str], should_pass: bool
    ):
        """Whole words must already be lowercase; they are no longer normalised."""
        if should_pass:
            knowledge = KiswahiliStudentKnowledge(whole_words=known_whole_words)
            assert knowledge.whole_words == known_whole_words
        else:
            with pytest.raises(ValueError):
                KiswahiliStudentKnowledge(whole_words=known_whole_words)

    @pytest.mark.parametrize(
        "kwargs, expected_str",
        [
            (
                {"graphemes": {"m", "a", "n"}, "whole_words": {"baba", "mama"}},
                '- Known graphemes: "a", "m", "n"\n'
                "- Known clusters: None\n"
                "- Known cluster patterns: None\n"
                '- Known whole words: "baba", "mama"',
            ),
            (
                {"whole_words": {"baba"}},
                "- Known graphemes: None\n"
                "- Known clusters: None\n"
                "- Known cluster patterns: None\n"
                '- Known whole words: "baba"',
            ),
            (
                {"graphemes": {"m", "a"}},
                '- Known graphemes: "a", "m"\n'
                "- Known clusters: None\n"
                "- Known cluster patterns: None\n"
                "- Known whole words: None",
            ),
            (
                {},
                "- Known graphemes: None\n"
                "- Known clusters: None\n"
                "- Known cluster patterns: None\n"
                "- Known whole words: None",
            ),
            # Populated clusters and cluster patterns, so every line has a
            # non-empty form too.
            (
                {
                    "graphemes": {"m", "b", "d"},
                    "clusters": {"mb", "md"},
                    "cluster_patterns": {ClusterPattern.PALATALISED_GLIDE},
                },
                '- Known graphemes: "b", "d", "m"\n'
                '- Known clusters: "mb", "md"\n'
                '- Known cluster patterns: "consonant + "y""\n'
                "- Known whole words: None",
            ),
        ],
    )
    def test_kiswahili_student_knowledge_str(self, kwargs: dict, expected_str: str):
        """Test the string representation of KiswahiliStudentKnowledge."""
        assert str(KiswahiliStudentKnowledge(**kwargs)) == expected_str


class TestKiswahiliClusters:
    @pytest.mark.parametrize(
        "known_clusters, should_pass",
        [
            ({"mb"}, True),
            ({"mb", "nd", "kw", "chy"}, True),
            (set(), True),
            # Loan clusters follow no native pattern but are still legitimate:
            # brashi, stesheni, klabu.
            ({"br"}, True),
            ({"st", "kl"}, True),
            ({"MB"}, False),  # must be lowercase
            ({"br", "ST"}, False),  # one uppercase entry invalidates the set
        ],
    )
    def test_check_known_clusters(self, known_clusters: set[str], should_pass: bool):
        """Any lowercase cluster is accepted, including non-native loan clusters."""
        # Supply every component grapheme so check_graphemes_in_clusters_known
        # cannot be the thing that fails.
        graphemes = {"m", "b", "n", "d", "k", "w", "ch", "y", "r", "s", "t", "l"}
        if should_pass:
            KiswahiliStudentKnowledge(clusters=known_clusters, graphemes=graphemes)
        else:
            with pytest.raises(ValueError):
                KiswahiliStudentKnowledge(clusters=known_clusters, graphemes=graphemes)

    @pytest.mark.parametrize(
        "known_clusters, known_graphemes, should_pass",
        [
            ({"mb"}, {"m", "b"}, True),
            ({"mb"}, {"m"}, False),
            ({"mb"}, set(), False),
            # "mng'" segments as ["m", "ng'"] by longest match, so the student
            # needs the "ng'" grapheme -- "n" does not satisfy it.
            ({"mng'"}, {"m", "ng'"}, True),
            ({"mng'"}, {"m", "n"}, False),
            # Loan clusters are subject to the same grapheme requirement.
            ({"br"}, {"b", "r"}, True),
            ({"br"}, {"b"}, False),
        ],
    )
    def test_check_graphemes_in_clusters_known(
        self, known_clusters: set[str], known_graphemes: set[str], should_pass: bool
    ):
        """Every grapheme inside a known cluster must itself be a known grapheme."""
        if should_pass:
            KiswahiliStudentKnowledge(
                clusters=known_clusters, graphemes=known_graphemes
            )
        else:
            with pytest.raises(ValueError):
                KiswahiliStudentKnowledge(
                    clusters=known_clusters, graphemes=known_graphemes
                )

    @pytest.mark.parametrize(
        "known_patterns, should_pass",
        [
            ({ClusterPattern.PALATALISED_GLIDE}, True),
            (set(), True),
            ({ClusterPattern.PRENASALISED_M}, False),
            ({ClusterPattern.PRENASALISED_M, ClusterPattern.PALATALISED_GLIDE}, False),
            # A combined pattern covers "mb" too (union of the m- and n- forms).
            ({ClusterPattern.PRENASALISED}, False),
        ],
    )
    def test_check_clusters_not_covered_by_patterns(
        self, known_patterns: set[ClusterPattern], should_pass: bool
    ):
        """A cluster must not be listed when a known pattern already covers it."""
        # "mb" is generated by PRENASALISED_M but not by PALATALISED_GLIDE.
        kwargs = {
            "clusters": {"mb"},
            "graphemes": {"m", "b"},
            "cluster_patterns": known_patterns,
        }
        if should_pass:
            KiswahiliStudentKnowledge(**kwargs)
        else:
            with pytest.raises(ValueError):
                KiswahiliStudentKnowledge(**kwargs)

    @pytest.mark.parametrize(
        "word, expected",
        [
            ("mng'", ["m", "ng'"]),
            ("kunywea", ["k", "u", "ny", "w", "e", "a"]),
            ("ngw", ["ng", "w"]),
            ("Baba", ["b", "a", "b", "a"]),
        ],
    )
    def test_get_graphemes(self, word: str, expected: list[str]):
        """Graphemes are segmented by longest match and returned lowercase."""
        assert get_graphemes(word) == expected

    @pytest.mark.parametrize(
        "word, expected",
        [
            ("mbwea'", ["mbw"]),
            ("kunywea", ["nyw"]),
            ("ng'ombe", ["mb"]),
            ("Baba", []),
            ("mchw", ["mchw"]),
            ("hospitali", ["sp"]),
            ("blanketi", ["bl", "nk"]),
        ],
    )
    def test_get_clusters(self, word: str, expected: list[str]):
        """Check consonant clusters are captured as expected."""
        assert get_clusters(word) == set(expected)

    @pytest.mark.parametrize(
        "pattern, expected_grapheme_count",
        [
            (ClusterPattern.PRENASALISED_M, 2),
            (ClusterPattern.PRENASALISED_N, 2),
            (ClusterPattern.PRENASALISED, 2),
            (ClusterPattern.LABIALISED_GLIDE, 2),
            (ClusterPattern.PALATALISED_GLIDE, 2),
            (ClusterPattern.M_LABIALISED_GLIDE, 3),
            (ClusterPattern.N_LABIALISED_GLIDE, 3),
            (ClusterPattern.PRENASALISED_LABIALISED_GLIDE, 3),
            (ClusterPattern.M_PALATALISED_GLIDE, 3),
            (ClusterPattern.N_PALATALISED_GLIDE, 3),
            (ClusterPattern.PRENASALISED_PALATALISED_GLIDE, 3),
        ],
    )
    def test_valid_clusters_have_expected_grapheme_count(
        self, pattern: ClusterPattern, expected_grapheme_count: int
    ):
        """Each generated cluster spans exactly the graphemes its pattern implies.

        Catches illegal combinations that collide with an existing digraph: e.g.
        omitting "n" from the palatalised-glide rejections would generate "ny",
        which is a single grapheme rather than a cluster.
        """
        assert VALID_CLUSTERS[pattern], f"{pattern} generated no clusters"

        offenders = {
            cluster: get_graphemes(cluster)
            for cluster in VALID_CLUSTERS[pattern]
            if len(get_graphemes(cluster)) != expected_grapheme_count
        }
        assert not offenders

    def test_every_cluster_pattern_has_valid_clusters(self):
        """VALID_CLUSTERS is built in stages; no member may be left out."""
        assert set(VALID_CLUSTERS) == set(ClusterPattern)


class TestClusterPatternEnum:
    """The enum's ``value`` is the data-config contract; ``label`` is display."""

    # The exact set of codes a human expert may write in the jsonl. Pinning it
    # here means a member rename or a fat-fingered code fails loudly rather than
    # silently changing what the data files must contain.
    EXPECTED_CODES = {
        "prenasalised_m",
        "prenasalised_n",
        "prenasalised",
        "labialised_glide",
        "palatalised_glide",
        "m_labialised_glide",
        "n_labialised_glide",
        "prenasalised_labialised_glide",
        "m_palatalised_glide",
        "n_palatalised_glide",
        "prenasalised_palatalised_glide",
    }

    def test_codes_match_contract(self):
        """Every member's value is the quote-free slug used in the jsonl."""
        assert {p.value for p in ClusterPattern} == self.EXPECTED_CODES

    def test_labels_are_distinct_and_nonempty(self):
        """Each member has its own human-readable label for the prompt."""
        labels = [p.label for p in ClusterPattern]
        assert all(labels)
        assert len(set(labels)) == len(labels)

    def test_codes_are_quote_free(self):
        """Codes must be easy to type in Excel: no double quotes to escape."""
        assert all('"' not in p.value for p in ClusterPattern)

    @pytest.mark.parametrize(
        "code, expected_pattern",
        [
            ("palatalised_glide", ClusterPattern.PALATALISED_GLIDE),
            ("prenasalised", ClusterPattern.PRENASALISED),
        ],
    )
    def test_validates_from_config_code(
        self, code: str, expected_pattern: ClusterPattern
    ):
        """A jsonl string code is coerced into the matching enum member.

        model_validate (rather than the constructor) mirrors the real path --
        jsonl -> dict -> model -- and keeps the string input type-clean.
        """
        knowledge = KiswahiliStudentKnowledge.model_validate(
            {"cluster_patterns": [code]}
        )
        assert knowledge.cluster_patterns == {expected_pattern}

    def test_rejects_unknown_config_code(self):
        """An unrecognised code (e.g. a label leaking into the config) is rejected."""
        with pytest.raises(ValueError):
            KiswahiliStudentKnowledge.model_validate(
                {"cluster_patterns": ['"m" + consonant']}
            )

    def test_json_round_trip_uses_codes(self):
        """Serialising for jsonl emits the codes, so it round-trips cleanly."""
        knowledge = KiswahiliStudentKnowledge(
            cluster_patterns={ClusterPattern.PRENASALISED}
        )
        dumped = knowledge.model_dump(mode="json")
        assert dumped["cluster_patterns"] == ["prenasalised"]
        assert (
            KiswahiliStudentKnowledge.model_validate(dumped).cluster_patterns
            == knowledge.cluster_patterns
        )


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

    def test_score_whole_words_kiswahili_with_no_taught_whole_words(self):
        """A student who has been taught no whole words scores 0."""
        student_knowledge = KiswahiliStudentKnowledge(whole_words=set())
        assert score_whole_words_kiswahili("baba", student_knowledge) == 0.0

    @pytest.mark.parametrize(
        "word, known_graphemes, known_clusters, expected",
        [
            ("mbwea'", {"m", "b", "w", "e", "a"}, {"mbw"}, 1.0),
            ("mbwea'", {"m", "b", "w", "e", "a"}, set(), 0.0),
            ("kunywea", {"k", "u", "n", "y", "ny", "w", "e", "a"}, {"nyw"}, 1.0),
            ("ng'ombe", {"ng'", "o", "m", "b", "e"}, {"mb"}, 1.0),
            ("ng'ombe", {"ng'", "o", "m", "b", "e"}, set(), 0.0),
            ("Baba", {"b", "a"}, set(), 1.0),
            ("mchw", {"m", "ch", "w"}, {"mchw"}, 1.0),
            ("hospitali", {"h", "o", "s", "p", "i", "t", "a", "l"}, {"sp"}, 1.0),
            ("blanketi", {"b", "l", "a", "n", "k", "e", "t", "i"}, {"bl", "nk"}, 1.0),
        ],
    )
    def test_score_known_clusters_and_patterns_kiswahili(
        self, word, known_graphemes, known_clusters, expected
    ):
        """Test the score_known_clusters_and_patterns_kiswahili function."""
        student_knowledge = KiswahiliStudentKnowledge(
            graphemes=known_graphemes,
            clusters=known_clusters,
        )
        assert (
            score_known_clusters_and_patterns_kiswahili(word, student_knowledge)
            == expected
        )

    @pytest.mark.parametrize(
        "word, known_graphemes, known_cluster_patterns, expected",
        [
            (
                "mbwea'",
                {"m", "b", "w", "e", "a"},
                {ClusterPattern.M_LABIALISED_GLIDE},
                1.0,
            ),
            (
                "mbwea'",
                {"m", "b", "w", "e", "a"},
                {ClusterPattern.PRENASALISED_M, ClusterPattern.LABIALISED_GLIDE},
                0.0,
            ),
            (
                "kunywea",
                {"k", "u", "ny", "w", "e", "a"},
                {ClusterPattern.LABIALISED_GLIDE},
                1.0,
            ),
            (
                "ng'ombe",
                {"ng'", "o", "m", "b", "e"},
                {ClusterPattern.PRENASALISED_M},
                1.0,
            ),
            ("ng'ombe", {"ng'", "o", "m", "b", "e"}, set(), 0.0),
            ("Baba", {"b", "a"}, set(), 1.0),
            ("mchw", {"m", "ch", "w"}, {ClusterPattern.LABIALISED_GLIDE}, 0.0),
            ("mpya", {"m", "p", "y", "a"}, {ClusterPattern.M_PALATALISED_GLIDE}, 1.0),
            (
                "hospitali",
                {"h", "o", "s", "p", "i", "t", "a", "l"},
                set(),
                0.0,
            ),  # loan clusters must be known explicitly
            (
                "blanketi",
                {"b", "l", "a", "n", "k", "e", "t", "i"},
                set(),
                0.0,
            ),  # loan clusters must be known explicitly
        ],
    )
    def test_score_known_cluster_patterns_kiswahili(
        self, word, known_graphemes, known_cluster_patterns, expected
    ):
        """Test the score_known_cluster_patterns_kiswahili function."""
        student_knowledge = KiswahiliStudentKnowledge(
            graphemes=known_graphemes,
            cluster_patterns=known_cluster_patterns,
        )
        assert (
            score_known_clusters_and_patterns_kiswahili(word, student_knowledge)
            == expected
        )

    @pytest.mark.parametrize(
        "word, known_graphemes, known_clusters, known_cluster_patterns, expected",
        [
            (
                "mbwea'",
                {"m", "b", "w", "e", "a"},
                set(),
                {ClusterPattern.M_LABIALISED_GLIDE},
                1.0,
            ),
            (
                "mbwea'",
                {"m", "b", "w", "e", "a"},
                {"mb"},
                {ClusterPattern.LABIALISED_GLIDE},
                0.0,
            ),
            (
                "kunywea",
                {"k", "u", "ny", "w", "e", "a"},
                {"nyw"},
                set(),
                1.0,
            ),
            (
                "mpya",
                {"m", "p", "y", "a"},
                {"mp", "mpy"},
                {ClusterPattern.PALATALISED_GLIDE},
                1.0,
            ),
            (
                "mpya",
                {"m", "p", "y", "a"},
                {
                    "mp",
                },
                {ClusterPattern.PALATALISED_GLIDE},
                0.0,
            ),
            (
                "mpya",
                {"m", "b", "p", "y", "a"},
                {"mb"},
                {ClusterPattern.M_PALATALISED_GLIDE},
                1.0,
            ),
        ],
    )
    def test_score_known_clusters_and_patterns_kiswahili_with_patterns(
        self, word, known_graphemes, known_clusters, known_cluster_patterns, expected
    ):
        """Test the score_known_clusters_and_patterns_kiswahili function with cluster
        patterns."""
        student_knowledge = KiswahiliStudentKnowledge(
            graphemes=known_graphemes,
            clusters=known_clusters,
            cluster_patterns=known_cluster_patterns,
        )
        assert (
            score_known_clusters_and_patterns_kiswahili(word, student_knowledge)
            == expected
        )


class TestKiswahiliWordExtraction:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("Juma ana baba na mama.", ["Juma", "ana", "baba", "na", "mama"]),
            ("Baba ana duka sokoni.", ["Baba", "ana", "duka", "sokoni"]),
            ("Ng'ombe ni mnyama.", ["Ng'ombe", "ni", "mnyama"]),
            ("Ng'ang'ania.", ["Ng'ang'ania"]),
            (
                "Waliona ng’ombe na mbuzi shambani",
                ["Waliona", "ng'ombe", "na", "mbuzi", "shambani"],
            ),
        ],
    )
    def test_extract_words_kiswahili(self, text: str, expected: list[str]):
        """Test the extract_words_kiswahili function."""
        assert extract_words_kiswahili(text) == expected


class TestKiswahiliAggregators:
    def test_tusome_aggregator_returns_1_for_known_whole_words(self):
        """A known whole word scores 1.0 regardless of grapheme knowledge."""
        scores = {
            "score_whole_words_kiswahili": 1.0,
        }
        assert aggregate_scores_kenya_tusome(scores) == 1.0

    def test_tusome_aggregator_falls_back_to_grapheme_and_clusters_score(self):
        """An unknown whole word falls back to the grapheme and cluster scores."""
        scores = {
            "score_whole_words_kiswahili": 0.0,
            "score_known_graphemes_kiswahili": 1.0,
            "score_known_clusters_and_patterns_kiswahili": 1.0,
            "score_known_cluster_patterns_kiswahili": 1.0,
        }
        assert aggregate_scores_kenya_tusome(scores) == 1.0

    def test_tusome_aggregator_through_rowwise(self):
        """The adapter applies the row function across a frame."""
        df = pd.DataFrame(
            {
                "score_whole_words_kiswahili": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "score_known_graphemes_kiswahili": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
                "score_known_clusters_and_patterns_kiswahili": [
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                ],
            }
        )
        # 1st row: known as whole word, don't know any graphemes or clusters in the word
        #   -> decodable
        # 2nd row: don't know any graphemes in the word -> non-decodable
        # 3rd row: know all graphemes and clusters in the word -> decodable
        # 4th row: don't know any graphemes or clusters in the word -> non-decodable
        # 5th row: know all graphemes but don't know any clusters in the word
        #   -> non-decodable
        # 6th row: know all graphemes and cluster patterns in the word -> decodable
        score_columns = list(df.columns)
        final_scores = rowwise(aggregate_scores_kenya_tusome)(df, score_columns)

        pd.testing.assert_series_equal(
            final_scores, pd.Series([1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
        )

    def test_tusome_aggregator_scores_correctly_from_word_and_student_knowledge(self):
        """The aggregator produces the same result as the scorers + aggregator."""
        word = "mbwea"
        student_knowledge = KiswahiliStudentKnowledge(
            graphemes={"m", "b", "w", "e", "a"},
            clusters={"mbw"},
            cluster_patterns={ClusterPattern.PRENASALISED_M},
            whole_words=set(),
        )
        cluster_score = score_known_clusters_and_patterns_kiswahili(
            word, student_knowledge
        )
        scores = {
            "score_whole_words_kiswahili": score_whole_words_kiswahili(
                word, student_knowledge
            ),
            "score_known_graphemes_kiswahili": score_known_graphemes_kiswahili(
                word, student_knowledge
            ),
            "score_known_clusters_and_patterns_kiswahili": cluster_score,
        }
        final_score = aggregate_scores_kenya_tusome(scores)
        assert final_score == 1.0
