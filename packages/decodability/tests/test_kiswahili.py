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
    score_known_graphemes_kiswahili,
    score_whole_words_kiswahili,
)
from decodability.kiswahili.segment import (
    ClusterSpan,
    GraphemeSpan,
    Span,
    count_syllables,
    decompose_word,
    find_cluster_spans,
    get_clusters,
    get_grapheme_spans,
    get_grapheme_symbols,
)
from decodability.kiswahili.student_knowledge import KiswahiliStudentKnowledge
from decodability.kiswahili.ungate_syllabic_nasals import (
    is_syllabic_nasal,
    ok_to_ungate,
)
from pydantic import ValidationError

# (word, [(cluster, start_index), ...]) for each expected consonant cluster.
cluster_outputs = [
    ("mbwea'", [("mbw", 0)]),
    ("kunywea", [("nyw", 2)]),
    ("ng'ombe", [("mb", 4)]),
    ("Baba", []),
    ("mchw", [("mchw", 0)]),
    ("hospitali", [("sp", 2)]),
    ("blanketi", [("bl", 0), ("nk", 3)]),
    ("mwenzangu", [("mw", 0), ("nz", 3)]),
    ("mchanganyiko", [("mch", 0)]),
    ("mwembamba", [("mw", 0), ("mb", 3), ("mb", 6)]),
    ("transista", [("tr", 0), ("ns", 3), ("st", 6)]),
    ("chai", []),  # a digraph with no cluster anywhere in the word
    ("shamba", [("mb", 3)]),  # a digraph, then a cluster with a vowel still to come
    ("nchi", [("nch", 0)]),  # the cluster absorbs a digraph and a tail remains
    ("mtoto", [("mt", 0)]),  # the plain word-initial syllabic nasal, for contrast
]

# Words for the tiling invariant: the cluster table, plus shapes it has no reason
# to carry. "" is excluded because it has no spans to index.
tiling_words = [word for word, _ in cluster_outputs] + [
    "a",  # a lone vowel
    "b",  # a lone consonant, so no vowel nucleus
    "mcq",  # c and q are out of inventory, so the consonant run does not form
    "ngw",  # the whole word is one cluster, ending on a digraph
    "mng'",  # the whole word is one cluster, ending on a trigraph
    "Mtoto",
    "SHAMBA",  # all-caps, so the digraphs do not match and "S", "H" stay apart
    "habari",  # a long word with no clusters at all
]


class TestKiswahiliSegmentation:
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
        spans = find_cluster_spans(get_grapheme_spans(word))

        assert [(span.normalised, span.start) for span in spans] == expected_outputs

    @pytest.mark.parametrize(
        "word, expected_nasal",
        [
            ("mtoto", [True]),
            ("Mtoto", [True]),  # the symbol is normalised, so a capital "M" counts
            ("nchi", [True]),
            ("Nchi", [True]),
            ("ng'ombe", [True]),  # the nasal cluster "mb" sits mid-word
            ("kunywea", [False]),  # "nyw" opens with the "ny" digraph, not "n"
            ("ngw", [False]),  # "ngw" opens with the "ng" digraph, not "n"
            ("blanketi", [False, True]),  # "bl" then "nk", in that order
            ("transista", [False, True, False]),  # "ns" counts even mid-word
            ("Baba", []),
        ],
    )
    def test_nasal_flags_clusters_that_open_with_m_or_n(self, word, expected_nasal):
        """A cluster is nasal when its first grapheme normalises to "m" or "n"."""
        clusters = find_cluster_spans(get_grapheme_spans(word))

        assert [cluster.nasal for cluster in clusters] == expected_nasal

    @pytest.mark.parametrize("word", [word for word, _ in cluster_outputs])
    def test_every_span_indexes_back_into_the_word(self, word):
        """Spans must be usable to slice the original word, casing included."""
        graphemes = get_grapheme_spans(word)

        grapheme_spans = [grapheme.to_span() for grapheme in graphemes]
        for span in grapheme_spans + find_cluster_spans(graphemes):
            assert word[span.start : span.end] == span.text

    def test_get_grapheme_spans_covers_the_whole_word(self):
        word = "ng'ombe"

        graphemes = get_grapheme_spans(word)

        assert "".join(grapheme.text for grapheme in graphemes) == word
        assert graphemes[0].start == 0
        assert graphemes[-1].end == len(word)

    def test_get_grapheme_spans_flags_consonants(self):
        graphemes = get_grapheme_spans("kunywea")

        assert [grapheme.is_consonant for grapheme in graphemes] == [
            True,  # k
            False,  # u
            True,  # ny
            True,  # w
            False,  # e
            False,  # a
        ]

    def test_get_grapheme_spans_preserves_casing(self):
        grapheme = get_grapheme_spans("Baba")[0]

        assert grapheme.text == "B"
        assert grapheme.symbol == "b"

    @pytest.mark.parametrize("word", ["", "a"])
    def test_words_too_short_to_hold_a_cluster(self, word):
        assert find_cluster_spans(get_grapheme_spans(word)) == []


class TestSpanModel:
    @pytest.mark.parametrize(
        "word, index, expected_length",
        [
            ("baba", 0, 1),  # a single-letter grapheme
            ("chai", 0, 2),  # a digraph
            ("ng'ombe", 0, 3),  # a trigraph, apostrophe included
        ],
    )
    def test_length_is_the_width_of_the_span(self, word, index, expected_length):
        """``length`` is the half-open range's width, which is the text's length."""
        grapheme = get_grapheme_spans(word)[index]

        assert grapheme.length == expected_length == len(grapheme.text)

    def test_cluster_symbol_is_the_lowercased_text(self):
        """``symbol`` on a cluster is an alias of ``normalised``."""
        cluster = find_cluster_spans(get_grapheme_spans("Mchw"))[0]

        assert cluster.text == "Mchw"
        assert cluster.symbol == "mchw"

    def test_a_span_whose_range_contradicts_its_text_is_rejected(self):
        """The range and the text must agree, so a mismatch fails validation."""
        with pytest.raises(ValidationError, match="does not match text"):
            Span(text="ch", start=0, end=1)


class TestDecomposeWord:
    @pytest.mark.parametrize("word", tiling_words)
    def test_decomposition_tiles_the_whole_word(self, word):
        """Graphemes and the flat decomposition each cover the word exactly once."""
        result = decompose_word(word)

        for spans in (result.graphemes, result.decomposition):
            assert "".join(span.text for span in spans) == word
            assert spans[0].start == 0
            assert spans[-1].end == len(word)
            assert [span.start for span in spans[1:]] == [
                span.end for span in spans[:-1]
            ]

    @pytest.mark.parametrize(
        "word, expected_decomposition",
        [
            ("chai", ["ch", "a", "i"]),  # a digraph outside any cluster
            ("shamba", ["sh", "a", "mb", "a"]),
            ("nchi", ["nch", "i"]),
            ("mchezo", ["mch", "e", "z", "o"]),
            ("ng'ombe", ["ng'", "o", "mb", "e"]),  # a trigraph, then a later cluster
            ("mchw", ["mchw"]),  # the whole word is a single cluster
            ("kunywea", ["k", "u", "nyw", "e", "a"]),  # the cluster sits mid-word
            ("Baba", ["B", "a", "b", "a"]),  # entries keep the original casing
        ],
    )
    def test_decomposition_lists_each_cluster_once_and_every_other_grapheme(
        self, word, expected_decomposition
    ):
        """Every cluster collapses to one entry and the other graphemes stay put."""
        assert [
            span.text for span in decompose_word(word).decomposition
        ] == expected_decomposition

    @pytest.mark.parametrize("word", [word for word, _ in cluster_outputs])
    def test_decomposition_partitions_clusters_from_leftover_graphemes(self, word):
        """Cluster entries are the clusters; the rest are the unabsorbed graphemes."""
        result = decompose_word(word)

        absorbed = {
            offset
            for cluster in result.clusters
            for offset in range(cluster.start, cluster.end)
        }
        cluster_entries = [
            span for span in result.decomposition if isinstance(span, ClusterSpan)
        ]
        grapheme_entries = [
            span for span in result.decomposition if isinstance(span, GraphemeSpan)
        ]

        assert cluster_entries == result.clusters
        assert grapheme_entries == [
            grapheme for grapheme in result.graphemes if grapheme.start not in absorbed
        ]
        assert len(cluster_entries) + len(grapheme_entries) == len(result.decomposition)

    @pytest.mark.parametrize("word", [word for word, _ in cluster_outputs])
    def test_fields_agree_with_the_standalone_helpers(self, word):
        """The model's fields are exactly what the segmentation helpers return."""
        result = decompose_word(word)

        graphemes = get_grapheme_spans(word)
        assert result.word == word
        assert result.graphemes == graphemes
        assert result.clusters == find_cluster_spans(graphemes)
        assert result.syllable_count == count_syllables(graphemes)

    @pytest.mark.parametrize(
        "word, expected_syllable_count",
        [
            ("chai", 2),  # "ch" is one consonant grapheme, then two nuclei
            ("shamba", 2),
            ("nchi", 1),  # the syllabic nasal is not a nucleus
            ("mchanganyiko", 4),
            ("mchw", 0),  # no vowel at all
            ("", 0),
            # c and q are out of inventory, so they are not flagged as consonants
            # and are counted as nuclei. This records current behaviour.
            ("mcq", 2),
        ],
    )
    def test_syllable_count_counts_vowel_nuclei(self, word, expected_syllable_count):
        """Syllables are counted by vowel nuclei, so consonant runs add none."""
        assert decompose_word(word).syllable_count == expected_syllable_count

    @pytest.mark.parametrize(
        "word, expected_decomposition",
        [
            ("", []),  # no grapheme is produced, so no Span is ever constructed
            ("a", ["a"]),
            ("b", ["b"]),
            ("mcq", ["m", "c", "q"]),  # out-of-inventory letters break the run
        ],
    )
    def test_words_with_no_cluster_to_collapse(self, word, expected_decomposition):
        """With no cluster to collapse, the flat decomposition is just the graphemes."""
        result = decompose_word(word)

        assert [span.text for span in result.decomposition] == expected_decomposition
        assert [span.text for span in result.graphemes] == expected_decomposition
        assert result.clusters == []


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


class TestKiswahiliSyllabicNasalUngating:
    @pytest.mark.parametrize(
        "word,cluster_index,expected",
        [
            ("mpaka", 0, True),
            ("sentensi", 0, True),
            ("mwili", 0, False),
            ("kwa", 0, False),
            # word-initial, one syllable after the nasal
            ("nzi", 0, True),
            ("nje", 0, True),
            ("mchwa", 0, True),
            ("mbu", 0, True),
            ("mbwa", 0, True),
            # fused before j/z/d once more than one syllable follows
            ("njia", 0, False),
            ("kanzu", 0, False),
            # medial mb is fused
            ("nyumba", 0, False),
            ("mjomba", 0, True),
            ("mjomba", 1, False),
            ("chumvi", 0, False),
        ],
    )
    def test_is_syllabic_nasal(self, word, cluster_index, expected):
        word_decomp = decompose_word(word)
        cluster_span = word_decomp.clusters[cluster_index]
        assert is_syllabic_nasal(word_decomp, cluster_span) == expected

    @pytest.mark.parametrize(
        "word,profile,expected",
        [
            (
                "mbwea",
                KiswahiliStudentKnowledge(
                    graphemes={"m", "b", "w", "e", "a"},
                    clusters={"mbw"},
                    cluster_patterns={ClusterPattern.PRENASALISED_M},
                    whole_words=set(),
                ),
                False,
            ),
            (
                # mj is syllabic, but the medial mb is fused and unknown
                "mjomba",
                KiswahiliStudentKnowledge(
                    graphemes={"m", "b", "j", "o", "a"},
                    clusters=set(),
                    cluster_patterns=set(),
                    whole_words=set(),
                ),
                False,
            ),
            (
                # the "m" + consonant pattern covers mb, so the word can be ungated
                "mjomba",
                KiswahiliStudentKnowledge(
                    graphemes={"m", "b", "j", "o", "a"},
                    clusters=set(),
                    cluster_patterns={ClusterPattern.PRENASALISED_M},
                    whole_words=set(),
                ),
                True,
            ),
            (
                "nje",
                KiswahiliStudentKnowledge(
                    graphemes={"n", "j", "e"},
                    clusters=set(),
                    cluster_patterns=set(),
                    whole_words=set(),
                ),
                True,
            ),
        ],
    )
    def test_ok_to_ungate(self, word, profile, expected):
        word_decomp = decompose_word(word)
        assert ok_to_ungate(word_decomp, profile) == expected
