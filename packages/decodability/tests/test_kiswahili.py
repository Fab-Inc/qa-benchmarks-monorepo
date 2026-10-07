import pandas as pd
import pytest
from decodability import rowwise
from decodability.kiswahili.aggregators import (
    aggregate_scores_kenya_tusome,
)
from decodability.kiswahili.definitions import (
    VALID_CLUSTERS,
    ClusterPattern,
)
from decodability.kiswahili.extract_words import extract_words_kiswahili
from decodability.kiswahili.scorers import (
    score_known_clusters_and_patterns_kiswahili,
    score_known_clusters_and_patterns_lenient_kiswahili,
    score_known_graphemes_kiswahili,
    score_whole_words_kiswahili,
)
from decodability.kiswahili.segment import (
    decompose_word,
    find_ambiguous_boundaries,
    find_cluster_spans,
    find_clustering_variants,
    get_clusters,
    get_grapheme_symbols,
)
from decodability.kiswahili.student_knowledge import KiswahiliStudentKnowledge

# (word, [(cluster, start_index), ...]) for each expected consonant cluster.
# find_cluster_spans resolves ambiguous nasal boundaries (m + b/v can be the
# syllabic m- marker or the fused prenasalised onset) as fused, so it lists
# every cluster any viable reading could require. Nasals before consonants
# other than the taught onsets are syllabic and never part of a cluster.
cluster_outputs = [
    ("mbwea'", [("mb", 0)]),
    ("kunywea", [("nyw", 2)]),
    ("ng'ombe", [("mb", 4)]),
    ("Baba", []),
    ("mchw", [("chw", 1)]),
    ("hospitali", [("sp", 2)]),
    ("blanketi", [("bl", 0)]),
    ("mwenzangu", [("mw", 0), ("nz", 3)]),
    ("mchanganyiko", []),
    ("mwembamba", [("mw", 0), ("mb", 3), ("mb", 6)]),
    ("transista", [("tr", 0), ("st", 6)]),
    # Syllabic nasals: nasal + consonant is decodable from the graphemes alone.
    ("mtoto", []),
    ("mmoja", []),
    ("mnyama", []),
    ("mstatili", [("st", 1)]),
    ("mvua", [("mv", 0)]),
    ("nchi", []),
    ("wanne", []),
    ("ndizi", [("nd", 0)]),
    ("njia", [("nj", 0)]),
    ("pandwa", [("nd", 2)]),
    # Word-initial n + d/j/z on a one-vowel root is syllabic, so no cluster.
    ("nzi", []),
    ("nje", []),
    # On a longer root or word-medially the same onset stays fused.
    ("nziwa", [("nz", 0)]),
    ("anza", [("nz", 1)]),
    # Consonant + nasal (no following consonant) is still a cluster.
    ("rasmi", [("sm", 2)]),
]


class TestKiswahiliDefinition:
    @pytest.mark.parametrize(
        "word, expected_graphemes",
        [
            ("baba", ["b", "a", "b", "a"]),
            ("Baba", ["b", "a", "b", "a"]),
            ("kunywea", ["k", "u", "ny", "w", "e", "a"]),
            ("ng'ombe", ["ng'", "o", "m", "b", "e"]),
            ("mng'", ["m", "ng'"]),
            ("ngw", ["ng", "w"]),
        ],
    )
    def test_get_grapheme_symbols(self, word, expected_graphemes):
        """Graphemes are segmented by longest match and returned lowercase."""
        assert get_grapheme_symbols(word) == expected_graphemes

    @pytest.mark.parametrize("word, expected_outputs", cluster_outputs)
    def test_get_clusters(self, word, expected_outputs):
        """Clusters are returned lowercase, in the order they appear."""
        assert get_clusters(word) == [cluster for cluster, _ in expected_outputs]

    @pytest.mark.parametrize("word, expected_outputs", cluster_outputs)
    def test_find_cluster_spans_locates_each_cluster(self, word, expected_outputs):
        spans = find_cluster_spans(decompose_word(word))

        assert [(span.normalised, span.start) for span in spans] == expected_outputs

    @pytest.mark.parametrize(
        "word, expected_variants",
        [
            # Only m + b/v boundaries are ambiguous (provisional rule); every
            # other boundary has a single reading, so most words have exactly
            # one clustering.
            ("ndizi", [["nd"]]),
            ("mtoto", [[]]),
            ("mstatili", [["st"]]),
            # nzi's initial n is syllabic (monosyllabic root) — no cluster.
            ("nzi", [[]]),
            # nziwa's root has two vowels, so nz stays fused.
            ("nziwa", [["nz"]]),
            ("ng'ombe", [["mb"], []]),
            # m + b/v can be the syllabic m- marker or the fused onset.
            ("mbuzi", [["mb"], []]),
            ("mvua", [["mv"], []]),
            # Read syllabically, the m leaves a bw stretch cluster behind.
            ("mbwea'", [["mb"], ["bw"]]),
            # Variants are fused-first: index 0 resolves every ambiguous
            # boundary as a cluster.
            (
                "mwembamba",
                [["mw", "mb", "mb"], ["mw", "mb"], ["mw", "mb"], ["mw"]],
            ),
        ],
    )
    def test_find_clustering_variants(self, word, expected_variants):
        variants = find_clustering_variants(decompose_word(word))

        assert [
            [span.normalised for span in variant] for variant in variants
        ] == expected_variants

    @pytest.mark.parametrize(
        "word, expected_boundaries",
        [
            ("mbuzi", [("mb", 0)]),
            ("mvua", [("mv", 0)]),
            ("ng'ombe", [("mb", 4)]),
            ("mwembamba", [("mb", 3), ("mb", 6)]),
            # nd is unambiguously fused, mt unambiguously syllabic.
            ("ndizi", []),
            ("mtoto", []),
            # nzi's n is resolved syllabic by rule, not left ambiguous.
            ("nzi", []),
            ("baba", []),
        ],
    )
    def test_find_ambiguous_boundaries(self, word, expected_boundaries):
        boundaries = find_ambiguous_boundaries(decompose_word(word))

        assert [(s.normalised, s.start) for s in boundaries] == expected_boundaries

    @pytest.mark.parametrize("word", [word for word, _ in cluster_outputs])
    def test_every_span_indexes_back_into_the_word(self, word):
        """Spans must be usable to slice the original word, casing included."""
        graphemes = decompose_word(word)

        grapheme_spans = [grapheme.to_span() for grapheme in graphemes]
        for span in grapheme_spans + find_cluster_spans(graphemes):
            assert word[span.start : span.end] == span.text

    def test_decompose_word_covers_the_whole_word_without_gaps(self):
        word = "ng'ombe"

        graphemes = decompose_word(word)

        assert "".join(grapheme.text for grapheme in graphemes) == word
        assert graphemes[0].start == 0
        assert graphemes[-1].end == len(word)

    def test_decompose_word_flags_consonants(self):
        graphemes = decompose_word("kunywea")

        assert [grapheme.is_consonant for grapheme in graphemes] == [
            True,  # k
            False,  # u
            True,  # ny
            True,  # w
            False,  # e
            False,  # a
        ]

    def test_decompose_word_preserves_casing_but_lowercases_the_symbol(self):
        grapheme = decompose_word("Baba")[0]

        assert grapheme.text == "B"
        assert grapheme.symbol == "b"

    @pytest.mark.parametrize("word", ["", "a"])
    def test_words_too_short_to_hold_a_cluster(self, word):
        assert find_cluster_spans(decompose_word(word)) == []


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

    # Grapheme and cluster segmentation are covered by TestKiswahiliDefinition.

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
            cluster: get_grapheme_symbols(cluster)
            for cluster in VALID_CLUSTERS[pattern]
            if len(get_grapheme_symbols(cluster)) != expected_grapheme_count
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
            ("mbwea'", {"m", "b", "w", "e", "a"}, {"mb"}, 1.0),
            # Strictly, the ambiguous mb boundary counts as fused, so knowing
            # bw (the syllabic reading's leftover cluster) does not suffice.
            ("mbwea'", {"m", "b", "w", "e", "a"}, {"bw"}, 0.0),
            ("mbwea'", {"m", "b", "w", "e", "a"}, set(), 0.0),
            ("kunywea", {"k", "u", "n", "y", "ny", "w", "e", "a"}, {"nyw"}, 1.0),
            ("ng'ombe", {"ng'", "o", "m", "b", "e"}, {"mb"}, 1.0),
            # mb is ambiguous, and ambiguous counts as fused under this scorer.
            ("ng'ombe", {"ng'", "o", "m", "b", "e"}, set(), 0.0),
            ("Baba", {"b", "a"}, set(), 1.0),
            ("mchw", {"m", "ch", "w"}, {"chw"}, 1.0),
            # Syllabic nasals need no cluster knowledge at all.
            ("mtoto", {"m", "t", "o"}, set(), 1.0),
            ("nchi", {"n", "ch", "i"}, set(), 1.0),
            ("ndizi", {"n", "d", "i", "z"}, set(), 0.0),
            ("ndizi", {"n", "d", "i", "z"}, {"nd"}, 1.0),
            # nzi's initial n is syllabic (one-vowel root): no cluster to know.
            ("nzi", {"n", "z", "i"}, set(), 1.0),
            # nziwa's root has two vowels, so nz stays fused and must be known.
            ("nziwa", {"n", "z", "i", "w", "a"}, set(), 0.0),
            ("nziwa", {"n", "z", "i", "w", "a"}, {"nz"}, 1.0),
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
        "word, known_graphemes, known_clusters, expected",
        [
            # Under the lenient rule any cluster that is not a fused nasal
            # onset decomposes into its graphemes — so mb, mv, bw and the
            # non-nasal stretches all read from grapheme knowledge alone.
            ("ng'ombe", {"ng'", "o", "m", "b", "e"}, set(), 1.0),
            ("mvua", {"m", "v", "u", "a"}, set(), 1.0),
            ("mbwea'", {"m", "b", "w", "e", "a"}, set(), 1.0),
            ("blanketi", set("blanketi"), set(), 1.0),
            ("hospitali", set("hospital"), set(), 1.0),
            ("mpya", {"m", "p", "y", "a"}, set(), 1.0),
            # With m untaught the fused reading of mbwea' does not decompose,
            # but the syllabic reading's bw does.
            ("mbwea'", {"b", "w", "e", "a"}, set(), 1.0),
            # Taught clusters still satisfy the word outright.
            ("ng'ombe", {"ng'", "o", "m", "b", "e"}, {"mb"}, 1.0),
            ("mbwea'", {"m", "b", "w", "e", "a"}, {"bw"}, 1.0),
            ("mbwea'", {"m", "b", "w", "e", "a"}, {"mb"}, 1.0),
            # Only untaught fused onsets still block the lenient reading.
            ("ndizi", {"n", "d", "i", "z"}, set(), 0.0),
            ("ndizi", {"n", "d", "i", "z"}, {"nd"}, 1.0),
            ("mwembamba", set("mwembamba"), set(), 0.0),
            ("mwembamba", set("mwembamba"), {"mw"}, 1.0),
            ("mtoto", {"m", "t", "o"}, set(), 1.0),
            ("baba", {"b", "a"}, set(), 1.0),
        ],
    )
    def test_score_known_clusters_and_patterns_lenient_kiswahili(
        self, word, known_graphemes, known_clusters, expected
    ):
        """Leniently, only fused nasal onsets must be taught as clusters."""
        student_knowledge = KiswahiliStudentKnowledge(
            graphemes=known_graphemes,
            clusters=known_clusters,
        )
        assert (
            score_known_clusters_and_patterns_lenient_kiswahili(
                word, student_knowledge
            )
            == expected
        )

    @pytest.mark.parametrize(
        "word, known_graphemes, known_cluster_patterns, expected",
        [
            (
                "mbwea'",
                {"m", "b", "w", "e", "a"},
                {ClusterPattern.M_LABIALISED_GLIDE},
                0.0,
            ),  # "mbw" knowledge does not cover the required "mb" onset
            (
                "mbwea'",
                {"m", "b", "w", "e", "a"},
                {ClusterPattern.PRENASALISED_M, ClusterPattern.LABIALISED_GLIDE},
                1.0,
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
            # mb is ambiguous, and ambiguous counts as fused under this scorer.
            ("ng'ombe", {"ng'", "o", "m", "b", "e"}, set(), 0.0),
            ("Baba", {"b", "a"}, set(), 1.0),
            ("mchw", {"m", "ch", "w"}, {ClusterPattern.LABIALISED_GLIDE}, 1.0),
            ("mpya", {"m", "p", "y", "a"}, {ClusterPattern.M_PALATALISED_GLIDE}, 0.0),
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
                0.0,
            ),
            (
                "mbwea'",
                {"m", "b", "w", "e", "a"},
                {"mb"},
                {ClusterPattern.LABIALISED_GLIDE},
                1.0,
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
                1.0,
            ),
            (
                "mpya",
                {"m", "b", "p", "y", "a"},
                {"mb"},
                {ClusterPattern.M_PALATALISED_GLIDE},
                0.0,
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
