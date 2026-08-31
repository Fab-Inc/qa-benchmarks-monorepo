import pandas as pd
import pytest
from decodability import rowwise
from decodability.english.aggregators import aggregate_scores_ufli_curriculum
from decodability.english.extract_phonemes import (
    Phoneme,
    _normalize_phonemes,
    convert_arpabet_to_phoneme,
    get_word_phonemes,
)
from decodability.english.extract_words import extract_words_english
from decodability.english.gpc_parser import GPC, parse_gpcs
from decodability.english.models import EnglishStudentKnowledge
from decodability.english.phoneme_overrides import CMUDICT_OVERRIDES
from decodability.english.scorers import (
    GPC_INVENTORY,
    score_known_gpcs_english,
    score_sight_words_english,
)


class TestEnglishStudentKnowledgeModel:
    @pytest.mark.parametrize(
        "known_gpcs, should_pass",
        [
            ([], True),
            ([("s", "s"), ("a", "æ"), ("t", "t")], True),
            ([("x", "ks")], True),
            ([("a-e", "eɪ")], True),
            ([("z", "xyz")], False),
            ([("a", "æ"), ("bogus", "q")], False),
        ],
    )
    def test_english_student_knowledge_gpc_checks_for_valid_gpcs(
        self,
        known_gpcs: list[tuple[str, str]],
        should_pass: bool,
    ):
        """Reject GPCs that are not in UFLI_GPC_INVENTORY."""
        if should_pass:
            EnglishStudentKnowledge(gpcs=known_gpcs)
        else:
            with pytest.raises(ValueError):
                EnglishStudentKnowledge(gpcs=known_gpcs)

    def test_english_student_knowledge_lowercases_sight_words(self):
        """Whole words are lowercased on ingest."""
        knowledge = EnglishStudentKnowledge(sight_words={"The", "cat", "DOG"})
        assert knowledge.sight_words == {"the", "cat", "dog"}

    @pytest.mark.parametrize(
        "kwargs, expected_str",
        [
            (
                {},
                "- Known GPCs: None\n- Known sight words: None",
            ),
            (
                {"gpcs": [("c", "k"), ("a", "æ"), ("t", "t")]},
                '- Known GPCs: "c" → "k"; "a" → "æ"; "t" → "t"\n'
                "- Known sight words: None",
            ),
            (
                {"sight_words": {"The", "and"}},
                '- Known GPCs: None\n- Known sight words: "and", "the"',
            ),
            (
                {
                    "gpcs": [("th", "ð"), ("th", "θ")],
                    "sight_words": {"the"},
                },
                '- Known GPCs: "th" → "ð"; "th" → "θ"\n- Known sight words: "the"',
            ),
        ],
    )
    def test_english_student_knowledge_str(self, kwargs: dict, expected_str: str):
        """Test the string representation of EnglishStudentKnowledge."""
        assert str(EnglishStudentKnowledge(**kwargs)) == expected_str


class TestScorers:
    @pytest.mark.parametrize(
        "word, known_gpcs, expected",
        [
            ("cat", [("c", "k"), ("a", "æ"), ("t", "t")], 1.0),
            ("cat", [("c", "k"), ("t", "t")], 0.0),
            ("cat", [], 0.0),
            ("Cat", [("c", "k"), ("a", "æ"), ("t", "t")], 1.0),
            (
                "smoke",
                [
                    ("s", "s"),
                    ("m", "m"),
                    ("o-e", "oʊ"),
                    ("k", "k"),
                ],
                1.0,
            ),
            (
                "smoke",
                [("s", "s"), ("m", "m"), ("k", "k")],
                0.0,
            ),
            ("fifth", [("f", "f"), ("i", "ɪ"), ("th", "θ")], 1.0),
        ],
    )
    def test_score_known_gpcs_english(
        self,
        word: str,
        known_gpcs: list[tuple[str, str]],
        expected: float,
    ):
        """Score is 1.0 iff every GPC the parser derives is in the student's set."""
        student_knowledge = EnglishStudentKnowledge(gpcs=known_gpcs)
        assert score_known_gpcs_english(word, student_knowledge) == expected

    def test_score_sight_words_english(self):
        """Sight-word check is case-insensitive."""
        student_knowledge = EnglishStudentKnowledge(sight_words={"the", "and"})
        assert score_sight_words_english("the", student_knowledge) == 1.0
        assert score_sight_words_english("The", student_knowledge) == 1.0
        assert score_sight_words_english("and", student_knowledge) == 1.0
        assert score_sight_words_english("cat", student_knowledge) == 0.0


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
    def test_ufli_aggregator_returns_1_for_known_whole_words(self):
        """A known whole word scores 1.0 regardless of GPC knowledge."""
        scores = {
            "score_sight_words_english": 1.0,
            "score_known_gpcs_english": 0.0,
        }
        assert aggregate_scores_ufli_curriculum(scores) == 1.0

    def test_ufli_aggregator_falls_back_to_gpc_score(self):
        """An unknown whole word falls back to the GPC score."""
        scores = {
            "score_sight_words_english": 0.0,
            "score_known_gpcs_english": 1.0,
        }
        assert aggregate_scores_ufli_curriculum(scores) == 1.0

    def test_ufli_aggregator_returns_0_when_both_missing(self):
        """No sight-word match and no GPC coverage → 0.0."""
        scores = {
            "score_sight_words_english": 0.0,
            "score_known_gpcs_english": 0.0,
        }
        assert aggregate_scores_ufli_curriculum(scores) == 0.0

    def test_ufli_aggregator_through_rowwise(self):
        """The adapter applies the row function across a frame."""
        df = pd.DataFrame(
            {
                "score_sight_words_english": [1.0, 0.0, 0.0],
                "score_known_gpcs_english": [0.0, 0.0, 1.0],
            }
        )
        score_columns = list(df.columns)
        final_scores = rowwise(aggregate_scores_ufli_curriculum)(df, score_columns)

        pd.testing.assert_series_equal(final_scores, pd.Series([1.0, 0.0, 1.0]))


class TestPhonemeArpabetLookup:
    def test_unknown_arpabet_phoneme_raises(self):
        with pytest.raises(ValueError, match="Unknown ARPABET phoneme"):
            convert_arpabet_to_phoneme("QQ9")


def _tag(*phonemes: str) -> tuple[Phoneme, ...]:
    """Wrap plain IPA symbols as Phoneme objects (is_unstressed=False) for tests."""
    return tuple(Phoneme(p, False) for p in phonemes)


class TestNormalizePhonemes:
    @pytest.mark.parametrize(
        "phonemes, expected",
        [
            # /ɪ/+/r/ in coda position → /i/+/r/
            (_tag("b", "ɪ", "r"), _tag("b", "i", "r")),
            (_tag("b", "ɪ", "r", "d"), _tag("b", "i", "r", "d")),
            # /ɪ/+/r/ before a vowel is preserved (mirror)
            (_tag("m", "ɪ", "r", "ə", "r"), _tag("m", "ɪ", "r", "ə", "r")),
        ],
    )
    def test_pre_r_laxing_reversal(
        self,
        phonemes: tuple[Phoneme, ...],
        expected: tuple[Phoneme, ...],
    ):
        assert _normalize_phonemes(phonemes) == expected

    @pytest.mark.parametrize(
        "phonemes, expected",
        [
            # /ɚ/ splits to /ə/+/r/ (sister, taller, picture)
            (_tag("s", "ɪ", "s", "t", "ɚ"), _tag("s", "ɪ", "s", "t", "ə", "r")),
            (_tag("t", "ɔ", "l", "ɚ"), _tag("t", "ɔ", "l", "ə", "r")),
            (_tag("p", "ɪ", "k", "tʃ", "ɚ"), _tag("p", "ɪ", "k", "tʃ", "ə", "r")),
            # Mid-word /ɚ/ (mirror) splits too
            (_tag("m", "ɪ", "r", "ɚ"), _tag("m", "ɪ", "r", "ə", "r")),
        ],
    )
    def test_er_split(
        self,
        phonemes: tuple[Phoneme, ...],
        expected: tuple[Phoneme, ...],
    ):
        assert _normalize_phonemes(phonemes) == expected


class TestPhonemeOverrides:
    def test_override_wins_over_cmudict(self, monkeypatch):
        """When both the override table and CMUdict have the word, override wins."""
        monkeypatch.setitem(CMUDICT_OVERRIDES, "either", [["IY1", "DH", "ER0"]])
        variants = get_word_phonemes("either")
        # Only the override variant survives — CMU's /aɪ/ alternate is dropped.
        assert len(variants) == 1
        assert variants[0][0].ipa == "i"

    def test_override_supplies_oov_pronunciation(self, monkeypatch):
        """An override rescues a word that CMUdict doesn't know."""
        monkeypatch.setitem(CMUDICT_OVERRIDES, "zzznotaword", [["Z", "IY1"]])
        variants = get_word_phonemes("zzznotaword")
        assert variants == ((Phoneme("z", False), Phoneme("i", False)),)


class TestGPCValidation:
    @pytest.mark.parametrize(
        "grapheme, phonemes",
        [
            ("", ("k",)),
            ("a", ()),
        ],
    )
    def test_gpc_requires_grapheme_and_phonemes(
        self, grapheme: str, phonemes: tuple[str, ...]
    ):
        with pytest.raises(ValueError, match="grapheme and at least one phoneme"):
            GPC(id="x", grapheme=grapheme, phonemes=phonemes)

    def test_gpc_rejects_more_than_one_split(self):
        with pytest.raises(ValueError, match="two-part split graphemes only"):
            GPC(id="x", grapheme="a-b-c", phonemes=("k",))

    @pytest.mark.parametrize("grapheme", ["-a", "a-", "-"])
    def test_gpc_rejects_split_missing_a_side(self, grapheme: str):
        with pytest.raises(ValueError, match="letters on both sides"):
            GPC(id="x", grapheme=grapheme, phonemes=("k",))


class TestParseGpcsValidation:
    def test_max_split_gap_must_be_positive(self):
        with pytest.raises(ValueError, match="max_split_gap must be at least one"):
            parse_gpcs("cat", _tag("k", "æ", "t"), (), max_split_gap=0)


class TestReducedVowelFreePass:
    """The parser accepts a reduced (unstressed) vowel phoneme paired with a
    vowel letter even when no inventory GPC covers that pair — vowel reduction
    is a universal spoken-English process, not a taught correspondence.
    """

    def _minimal_planet_inventory(self) -> tuple[GPC, ...]:
        # Deliberately omits any ``e → …`` GPC so the only way ``planet`` parses
        # is via the reduced-vowel free pass on the trailing ``e``.
        return (
            GPC("P", "p", ("p",)),
            GPC("L", "l", ("l",)),
            GPC("A", "a", ("æ",)),
            GPC("N", "n", ("n",)),
            GPC("T", "t", ("t",)),
        )

    def test_parser_uses_free_pass_when_no_inventory_gpc_matches(self):
        """/ə/ marked unstressed on letter ``e`` parses without an ``e → ə`` GPC."""
        phonemes = (
            Phoneme("p", False),
            Phoneme("l", False),
            Phoneme("æ", False),
            Phoneme("n", False),
            Phoneme("ə", True),  # unstressed schwa — needs the free pass
            Phoneme("t", False),
        )
        parses = parse_gpcs("planet", phonemes, self._minimal_planet_inventory())

        assert len(parses) == 1
        (parse,) = parses
        e_match = parse[4]
        assert e_match.gpc.id == "REDUCED_E"
        assert e_match.gpc.grapheme == "e"
        assert e_match.gpc.phonemes == ("ə",)

    def test_parser_does_not_free_pass_when_phoneme_is_stressed(self):
        """The same ``e``/``ə`` pair without the unstressed flag doesn't fire."""
        phonemes = (
            Phoneme("p", False),
            Phoneme("l", False),
            Phoneme("æ", False),
            Phoneme("n", False),
            Phoneme("ə", False),  # NOT flagged reduced — no free pass
            Phoneme("t", False),
        )
        parses = parse_gpcs("planet", phonemes, self._minimal_planet_inventory())
        assert parses == ()


class TestScoreReducedVowelCreditsAnyGraphemeGpc:
    """The scorer credits a reduced-vowel free-pass match as taught whenever the
    student knows *any* GPC for the same grapheme letter.
    """

    def test_any_e_gpc_credits_reduced_e_in_planet(self):
        """Knowing ``e → ɛ`` is enough to score ``planet`` even though the
        parser uses ``e → ə`` (reduced) for the trailing e."""
        student = EnglishStudentKnowledge(
            gpcs=[
                ("p", "p"),
                ("l", "l"),
                ("a", "æ"),
                ("n", "n"),
                ("e", "ɛ"),  # short e — parser needs ə but this counts
                ("t", "t"),
            ]
        )
        assert score_known_gpcs_english("planet", student) == 1.0

    def test_missing_all_e_gpcs_fails_reduced_e_in_planet(self):
        """No ``e`` GPC at all → the reduced-vowel free pass is not credited."""
        student = EnglishStudentKnowledge(
            gpcs=[
                ("p", "p"),
                ("l", "l"),
                ("a", "æ"),
                ("n", "n"),
                ("t", "t"),
            ]
        )
        assert score_known_gpcs_english("planet", student) == 0.0


class TestApostropheIsSilent:
    """Possessive/contracted ``'s`` and its siblings are an assumed correspondence
    in the curriculum — the apostrophe carries no phoneme, and the letters
    around it decode via their existing GPCs. The parser skips ``'`` positions
    without emitting a match or consuming a phoneme.
    """

    def _minimal_hes_inventory(self) -> tuple[GPC, ...]:
        # Enough to decode ``he's`` letter-by-letter modulo the apostrophe.
        return (
            GPC("H", "h", ("h",)),
            GPC("E_OPEN", "e", ("i",)),  # open-syllable e → /i/
            GPC("S_VOICED", "s", ("z",)),  # G018 s → /z/
        )

    def test_parser_skips_apostrophe_in_hes(self):
        """``he's`` parses via h + e + s (apostrophe silently skipped)."""
        phonemes = (Phoneme("h", False), Phoneme("i", False), Phoneme("z", False))
        parses = parse_gpcs("he's", phonemes, self._minimal_hes_inventory())

        assert len(parses) == 1
        (parse,) = parses
        # Three matches only — no synthetic apostrophe entry
        assert len(parse) == 3
        assert [m.gpc.id for m in parse] == ["H", "E_OPEN", "S_VOICED"]
        # Letter positions cover 0, 1, 3 — position 2 (the ``'``) is uncovered
        # by design.
        assert parse[0].letter_positions == (0,)
        assert parse[1].letter_positions == (1,)
        assert parse[2].letter_positions == (3,)

    def test_parser_still_requires_phoneme_match_around_apostrophe(self):
        """Skipping ``'`` doesn't absolve the surrounding letters from matching
        phonemes — if the ``s`` phoneme is wrong, the whole parse still fails.
        """
        # phonemes claim final /s/ (voiceless) but the inventory only teaches
        # s → /z/, so nothing covers the trailing phoneme.
        phonemes = (Phoneme("h", False), Phoneme("i", False), Phoneme("s", False))
        parses = parse_gpcs("he's", phonemes, self._minimal_hes_inventory())
        assert parses == ()


class TestScoreContractionsViaAssumedApostrophe:
    """End-to-end: a student who knows the underlying letter GPCs can decode
    possessives and contractions without any dedicated ``'s → …`` GPC.
    """

    def test_student_with_s_z_scores_hes(self):
        """Knowing ``s → z`` (plus h and e) is enough to score ``he's``."""
        student = EnglishStudentKnowledge(
            gpcs=[
                ("h", "h"),
                ("e", "i"),  # open-syllable e
                ("s", "z"),
            ]
        )
        assert score_known_gpcs_english("he's", student) == 1.0

    def test_student_missing_s_z_fails_hes(self):
        """Without ``s → z``, ``he's`` is undecodable."""
        student = EnglishStudentKnowledge(
            gpcs=[
                ("h", "h"),
                ("e", "i"),
                ("s", "s"),  # only voiceless s
            ]
        )
        assert score_known_gpcs_english("he's", student) == 0.0


class TestScorersEmptyKnowledge:
    def test_score_sight_words_english_returns_0_when_no_taught_sight_words(self):
        """Empty sight_words set short-circuits to 0.0."""
        student = EnglishStudentKnowledge()
        assert score_sight_words_english("cat", student) == 0.0


def _parse_corpus_word(word: str):
    """Parse a real corpus word under the full UFLI inventory across every CMU
    variant. Returns (variants, [(variant_index, parse), ...]).
    """
    variants = get_word_phonemes(word)
    parses = [
        (i, parse)
        for i, variant in enumerate(variants)
        for parse in parse_gpcs(word, variant, GPC_INVENTORY)
    ]
    return variants, parses


class TestRegressionReducedVowelFreePassOnCorpusWords:
    """Corpus words whose UFLI parse is only reachable via the parser's
    reduced-vowel free pass: each pairs a bare vowel letter with an unstressed
    schwa that no taught GPC covers. Regression guard for the free pass in
    ``parse_gpcs`` — remove it and every one of these becomes undecodable.
    """

    @pytest.mark.parametrize(
        "word, reduced_letter",
        [
            ("about", "a"),  # leading unstressed a — ultimate_phonics
            ("sofa", "a"),  # trailing unstressed a — ultimate_phonics
            ("lion", "o"),  # medial unstressed o — ultimate_phonics
            ("apron", "o"),  # trailing unstressed o — ultimate_phonics
            ("planet", "e"),  # trailing unstressed e — ultimate_phonics
        ],
    )
    def test_corpus_word_parses_via_reduced_vowel_free_pass(
        self, word: str, reduced_letter: str
    ):
        _, parses = _parse_corpus_word(word)
        assert parses, f"{word} should parse under the full UFLI inventory"
        assert any(
            any(
                match.gpc.is_reduced_vowel and match.gpc.grapheme == reduced_letter
                for match in parse
            )
            for _, parse in parses
        ), f"{word} should use the reduced-vowel free pass on '{reduced_letter}'"


class TestRegressionSilentApostropheOnCorpusWords:
    """Contractions and possessives from the corpora parse only because the
    parser silently skips ``'`` positions. Regression guard for the silent-
    apostrophe handling in ``parse_gpcs``.
    """

    @pytest.mark.parametrize(
        "word",
        [
            "he's",  # ufli
            "it's",  # ufli
            "don't",  # ufli
            "let's",  # ufli
            "that's",  # ufli
        ],
    )
    def test_corpus_contraction_parses_with_silent_apostrophe(self, word: str):
        _, parses = _parse_corpus_word(word)
        assert parses, f"{word} should parse under the full UFLI inventory"
        apostrophe_index = word.index("'")
        for _, parse in parses:
            covered = {
                position for match in parse for position in match.letter_positions
            }
            assert apostrophe_index not in covered, (
                f"apostrophe position should be uncovered in {word} parse"
            )
            assert all("'" not in match.gpc.grapheme for match in parse), (
                f"no GPC should carry an apostrophe grapheme in {word} parse"
            )


class TestRegressionCotCaughtMergerOnCorpusWords:
    """Words from the corpora that only parse under UFLI once cot-caught merger GPCs
    were added. Each parametrization pins the expected new-GPC id — if the entry is
    removed from ``UFLI_GPC_INVENTORY`` the parse either disappears or falls
    back to a different (unattested) decomposition.
    """

    @pytest.mark.parametrize(
        "word, gpc_id",
        [
            # G168 o → /ɔ/ (cot-caught un-merged; short-o words in the corpora
            # whose CMU variant carries /ɔ/ rather than /ɑ/).
            ("dog", "G168"),  # ufli
            ("long", "G168"),  # ultimate_phonics
            ("off", "G168"),  # ufli
            # G169 au → /ɑ/
            ("august", "G169"),  # ultimate_phonics
            ("caution", "G169"),  # ultimate_phonics
            # G170 augh → /ɑ/
            ("caught", "G170"),  # ultimate_phonics
            # G171 ough → /ɑ/
            ("bought", "G171"),  # ultimate_phonics
        ],
    )
    def test_corpus_word_parses_using_new_gpc(self, word: str, gpc_id: str):
        _, parses = _parse_corpus_word(word)
        assert parses, f"{word} should parse under the full UFLI inventory"
        assert any(
            any(match.gpc.id == gpc_id for match in parse) for _, parse in parses
        ), f"{word} should parse via {gpc_id}"


class TestRegressionNormalizePhonemesOnCorpusWords:
    """Corpus words whose UFLI parse depends on the two CMUdict fixups in
    ``_normalize_phonemes``: splitting /ɚ/ into /ə/+/r/ so ``er/ir/ur``
    endings match the ``er → (ə, r)`` family, and reversing pre-r laxing so
    coda ``/ɪ r/`` becomes ``/i r/`` and matches ``ee/ea → i``. Regression
    guard for both fixups.
    """

    @pytest.mark.parametrize(
        "word, expected_er_gpc",
        [
            ("sister", "G079"),  # ultimate_phonics: er → (ə, r)
            ("after", "G079"),  # ultimate_phonics: er → (ə, r)
            ("better", "G079"),  # ultimate_phonics: er → (ə, r)
        ],
    )
    def test_r_colored_schwa_split_lets_er_ending_parse(
        self, word: str, expected_er_gpc: str
    ):
        variants, parses = _parse_corpus_word(word)
        # Fixup: no variant should still carry the raw /ɚ/ that CMU emits.
        for variant in variants:
            assert all(p.ipa != "ɚ" for p in variant), (
                f"{word}: /ɚ/ should have been split into /ə/+/r/"
            )
        assert parses, f"{word} should parse under the full UFLI inventory"
        assert any(
            any(match.gpc.id == expected_er_gpc for match in parse)
            for _, parse in parses
        ), f"{word} should parse via {expected_er_gpc} (er → ə, r)"

    @pytest.mark.parametrize(
        "word, expected_vowel_gpc",
        [
            ("deer", "G085"),  # ultimate_phonics: ee → i
            ("beer", "G085"),  # ultimate_phonics: ee → i
            ("cheer", "G085"),  # ultimate_phonics: ee → i
            ("year", "G086"),  # ultimate_phonics: ea → i
        ],
    )
    def test_pre_r_laxing_reversal_lets_ee_ea_parse(
        self, word: str, expected_vowel_gpc: str
    ):
        variants, parses = _parse_corpus_word(word)
        # Fixup: coda /ɪ r/ should have been rewritten to /i r/.
        for variant in variants:
            ipas = tuple(p.ipa for p in variant)
            assert "ɪ" not in ipas, (
                f"{word}: coda /ɪ/+/r/ should have been re-tensed to /i/+/r/"
            )
        assert parses, f"{word} should parse under the full UFLI inventory"
        assert any(
            any(match.gpc.id == expected_vowel_gpc for match in parse)
            for _, parse in parses
        ), f"{word} should parse via {expected_vowel_gpc}"

    def test_pre_r_laxing_reversal_preserves_prevocalic_ir(self):
        """A pre-vocalic /ɪ r/ (``lyric`` /lɪrɪk/) is left alone — the fixup
        only fires in coda position. Regression guard against overreach.
        """
        variants = get_word_phonemes("lyric")
        assert variants
        # Every variant should still carry the internal /ɪ/+/r/ pair.
        assert all(
            any(
                v[i].ipa == "ɪ" and i + 1 < len(v) and v[i + 1].ipa == "r"
                for i in range(len(v) - 1)
            )
            for v in variants
        )
