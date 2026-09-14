import pandas as pd
import pytest
from decodability import rowwise
from decodability.english.aggregators import aggregate_scores_ufli_curriculum
from decodability.english.extract_phonemes import (
    Phoneme,
    Pronunciation,
    _add_cot_caught_variants,
    _normalize_phonemes,
    _suffixed_s_variants,
    convert_arpabet_to_phoneme,
    count_syllables,
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


def _say(*phonemes: str, is_monosyllabic: bool = True) -> Pronunciation:
    """Wrap plain IPA symbols as a Pronunciation for parser tests."""
    return Pronunciation(phonemes=_tag(*phonemes), is_monosyllabic=is_monosyllabic)


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
        assert variants[0].phonemes[0].ipa == "i"

    def test_override_supplies_oov_pronunciation(self, monkeypatch):
        """An override rescues a word that CMUdict doesn't know."""
        monkeypatch.setitem(CMUDICT_OVERRIDES, "zzznotaword", [["Z", "IY1"]])
        variants = get_word_phonemes("zzznotaword")
        assert variants[0].phonemes == (Phoneme("z", False), Phoneme("i", False))

    def test_fine_has_only_one_pronunciation_variant(self):
        variants = get_word_phonemes("fine")

        assert len(variants) == 1
        assert variants[0].phonemes == (
            Phoneme("f", False),
            Phoneme("aɪ", False),
            Phoneme("n", False),
        )


class TestSuffixedSFallback:
    """OOV words ending in ``s``/``'s`` inherit the base word's CMUdict
    pronunciation with /s/ and /z/ appended (``haps`` → ``hap`` + /s/ or /z/)."""

    @pytest.mark.parametrize("word", ["haps", "hap's"])
    def test_oov_plural_and_possessive_inherit_base_plus_s(self, word: str):
        """``hap`` is in CMUdict; ``haps``/``hap's`` are not."""
        variants = get_word_phonemes(word)
        assert variants == (
            (
                Phoneme("h", False),
                Phoneme("æ", False),
                Phoneme("p", False),
                Phoneme("s", False),
            ),
            (
                Phoneme("h", False),
                Phoneme("æ", False),
                Phoneme("p", False),
                Phoneme("z", False),
            ),
        )

    def test_fallback_does_not_fire_for_cmudict_entries(self):
        """``hiss`` is in CMUdict as /h ɪ s/ — the fallback must not rewrite it
        as ``his`` + /s/ (which would be /h ɪ z s/)."""
        variants = get_word_phonemes("hiss")
        assert variants == (
            (Phoneme("h", False), Phoneme("ɪ", False), Phoneme("s", False)),
        )

    def test_fallback_uses_override_for_base(self, monkeypatch):
        """An override for the base also serves its ``s``-suffixed form."""
        monkeypatch.setitem(CMUDICT_OVERRIDES, "zzzbase", [["Z", "IY1"]])
        variants = get_word_phonemes("zzzbases")
        assert variants == (
            (Phoneme("z", False), Phoneme("i", False), Phoneme("s", False)),
            (Phoneme("z", False), Phoneme("i", False), Phoneme("z", False)),
        )

    @pytest.mark.parametrize("word", ["zzznotaword", "zzznotawords"])
    def test_still_raises_when_no_base_available(self, word: str):
        """No suffix or unknown base → KeyError as before."""
        with pytest.raises(KeyError):
            get_word_phonemes(word)

    @pytest.mark.parametrize("word", ["s", "'s"])
    def test_empty_base_returns_none(self, word: str):
        """A bare ``s``/``'s`` has no base to inherit from — no fallback.
        (Both happen to be real CMUdict entries anyway, so the guard is only
        reachable via direct calls or non-CMU keys.)"""
        assert _suffixed_s_variants(word) is None

    def test_fallback_preserves_all_base_variants(self, monkeypatch):
        """Every variant of the base gets /s/ and /z/ appended, not just the
        first."""
        monkeypatch.setitem(CMUDICT_OVERRIDES, "zzzbase", [["Z", "IY1"], ["Z", "EH1"]])
        variants = get_word_phonemes("zzzbase's")
        assert variants == (
            (Phoneme("z", False), Phoneme("i", False), Phoneme("s", False)),
            (Phoneme("z", False), Phoneme("i", False), Phoneme("z", False)),
            (Phoneme("z", False), Phoneme("ɛ", False), Phoneme("s", False)),
            (Phoneme("z", False), Phoneme("ɛ", False), Phoneme("z", False)),
        )

    def test_haps_scores_end_to_end(self):
        """A student who knows h/æ/p/s decodes ``haps`` via the fallback."""
        student = EnglishStudentKnowledge(
            gpcs=[("h", "h"), ("a", "æ"), ("p", "p"), ("s", "s")]
        )
        assert score_known_gpcs_english("haps", student) == 1.0
        assert score_known_gpcs_english("hap's", student) == 1.0

    def test_voiced_variant_scores_end_to_end(self):
        """The /z/ variant lets a student with ``s → z`` (but not ``s → s``)
        decode ``haps``."""
        student = EnglishStudentKnowledge(
            gpcs=[("h", "h"), ("a", "æ"), ("p", "p"), ("s", "z")]
        )
        assert score_known_gpcs_english("haps", student) == 1.0
        assert score_known_gpcs_english("hap's", student) == 1.0


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
            parse_gpcs("cat", _say("k", "æ", "t"), (), max_split_gap=0)


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
        parses = parse_gpcs(
            "planet",
            Pronunciation(phonemes=phonemes, is_monosyllabic=False),
            self._minimal_planet_inventory(),
        )

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
        parses = parse_gpcs(
            "planet",
            Pronunciation(phonemes=phonemes, is_monosyllabic=False),
            self._minimal_planet_inventory(),
        )
        assert parses == ()


class TestMonosyllabicBlocksReducedVowelFreePass:
    """A one-syllable pronunciation has no unstressed syllable to reduce into,
    so its vowel is a correspondence the student must have been taught. The
    free pass is withheld even when CMU marks the vowel stress-0 — which it
    does for reduced function words like ``was``, ``a`` and ``the``.
    """

    def _minimal_was_inventory(self) -> tuple[GPC, ...]:
        # Deliberately omits any ``a → …`` GPC, so the only route to a parse
        # is the reduced-vowel free pass on the ``a``.
        return (
            GPC("W", "w", ("w",)),
            GPC("S", "s", ("z",)),
        )

    def _was_phonemes(self) -> tuple[Phoneme, ...]:
        return (
            Phoneme("w", False),
            Phoneme("ə", True),  # CMU gives "was" as W AH0 Z
            Phoneme("z", False),
        )

    def test_monosyllabic_variant_gets_no_free_pass(self):
        """``was`` /wəz/ is one syllable, so the unstressed /ə/ earns nothing."""
        parses = parse_gpcs(
            "was",
            Pronunciation(phonemes=self._was_phonemes(), is_monosyllabic=True),
            self._minimal_was_inventory(),
        )
        assert parses == ()

    def test_same_phonemes_parse_when_not_monosyllabic(self):
        """Only the flag differs — this isolates the gate from every other rule."""
        parses = parse_gpcs(
            "was",
            Pronunciation(phonemes=self._was_phonemes(), is_monosyllabic=False),
            self._minimal_was_inventory(),
        )
        assert len(parses) == 1
        (parse,) = parses
        assert parse[1].gpc.id == "REDUCED_A"

    @pytest.mark.parametrize("word", ["was", "a", "the"])
    def test_reduced_function_words_lose_the_free_pass_end_to_end(self, word: str):
        """Through the real pipeline: every monosyllabic variant whose vowel is
        unstressed now fails to parse on the free pass alone.
        """
        reduced_variants = [
            variant
            for variant in get_word_phonemes(word)
            if variant.is_monosyllabic
            and any(phoneme.is_unstressed for phoneme in variant.phonemes)
        ]
        assert reduced_variants, f"expected a reduced variant for {word!r}"
        for variant in reduced_variants:
            assert not any(
                match.gpc.is_reduced_vowel
                for parse in parse_gpcs(word, variant, GPC_INVENTORY)
                for match in parse
            ), f"{word}: monosyllabic variant should not parse via a free pass"

    def test_multisyllabic_free_pass_still_fires(self):
        """Regression guard: the gate must not touch multisyllabic reduction."""
        variants = get_word_phonemes("planet")
        assert any(
            match.gpc.is_reduced_vowel
            for variant in variants
            for parse in parse_gpcs("planet", variant, GPC_INVENTORY)
            for match in parse
        )


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
        parses = parse_gpcs(
            "he's",
            Pronunciation(phonemes=phonemes, is_monosyllabic=True),
            self._minimal_hes_inventory(),
        )

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
        parses = parse_gpcs(
            "he's",
            Pronunciation(phonemes=phonemes, is_monosyllabic=True),
            self._minimal_hes_inventory(),
        )
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


class TestSyllableCounting:
    """``count_syllables`` counts stress-bearing tokens. ARPABET marks every
    vowel — and only vowels — with a stress digit, so that count is the number
    of syllable nuclei. All stress levels count, not just primary.
    """

    @pytest.mark.parametrize(
        "arpabet, expected",
        [
            (["K", "AE1", "T"], 1),
            (["HH", "EH0", "L", "OW1"], 2),
            # Only AE1 carries primary stress, but all three vowels are nuclei.
            (["B", "AH0", "N", "AE1", "N", "AH0"], 3),
            # "strengths": eight tokens, one vowel, so one syllable.
            (["S", "T", "R", "EH1", "NG", "K", "TH", "S"], 1),
            ([], 0),
        ],
    )
    def test_count_syllables(self, arpabet: list[str], expected: int):
        assert count_syllables(arpabet) == expected

    @pytest.mark.parametrize(
        "word, expected",
        [
            ("cat", True),
            ("banana", False),
            ("planet", False),
        ],
    )
    def test_is_monosyllabic_on_unambiguous_words(self, word: str, expected: bool):
        """Words whose variants agree — every variant carries the same flag."""
        variants = get_word_phonemes(word)
        assert variants
        assert all(v.is_monosyllabic is expected for v in variants)

    @pytest.mark.parametrize(
        "word",
        [
            "fire",  # faɪər / faɪr
            "fires",  # the alternation survives inflection
            "flour",  # flaʊər / flaʊr
            "dial",  # daɪəl / daɪl
            "cruel",  # kruəl / krul
            "aisle",  # aɪl / aɪəl
        ],
    )
    def test_flag_is_per_variant_not_per_word(self, word: str):
        """CMU records both a one- and a two-syllable reading for these, which
        is why the flag lives on ``Pronunciation`` rather than on the word.

        The provided sample matches against upstream CMUdict
        data, so a future release could in principle change alter these pronunciations.
        """
        flags = {v.is_monosyllabic for v in get_word_phonemes(word)}
        assert flags == {True, False}, f"{word} should have variants of both shapes"


class TestCotCaughtVariantExpansion:
    """Every CMU variant transcribed with AO (``caught``, /ɔ/) gains an AA
    (``cot``, /ɑ/) counterpart, so a student taught only the merged reading
    still matches the word. Runs on ARPABET before conversion.
    """

    def test_ao_variant_gains_an_aa_counterpart(self):
        assert _add_cot_caught_variants([["K", "AO1", "T"]]) == [
            ["K", "AO1", "T"],
            ["K", "AA1", "T"],
        ]

    def test_variant_without_ao_is_untouched(self):
        assert _add_cot_caught_variants([["K", "AA1", "T"]]) == [["K", "AA1", "T"]]

    def test_stress_digit_is_preserved(self):
        """``AO1`` -> ``AA1``, not a bare ``AA`` — downstream reads stress-0."""
        assert _add_cot_caught_variants([["AO2", "G", "AH0"]]) == [
            ["AO2", "G", "AH0"],
            ["AA2", "G", "AH0"],
        ]

    def test_every_ao_in_a_variant_is_swapped(self):
        assert _add_cot_caught_variants([["AO1", "T", "AO0"]]) == [
            ["AO1", "T", "AO0"],
            ["AA1", "T", "AA0"],
        ]

    def test_existing_aa_counterpart_is_not_duplicated(self):
        """CMU already lists both readings for some words; don't emit a repeat."""
        assert _add_cot_caught_variants([["D", "AO1", "G"], ["D", "AA1", "G"]]) == [
            ["D", "AO1", "G"],
            ["D", "AA1", "G"],
        ]

    def test_original_variant_is_kept_first(self):
        """The unmerged reading stays primary; the merged one is the alternate."""
        first, second = _add_cot_caught_variants([["AO1"]])
        assert first == ["AO1"]
        assert second == ["AA1"]

    @pytest.mark.parametrize("word", ["dog", "off", "long", "caught"])
    def test_word_yields_both_readings_end_to_end(self, word: str):
        """Through ``get_word_phonemes``: both /ɔ/ and /ɑ/ reach the caller."""
        vowels = {
            phoneme.ipa
            for variant in get_word_phonemes(word)
            for phoneme in variant.phonemes
        }
        assert {"ɔ", "ɑ"} <= vowels, f"{word} should offer both merger readings"

    def test_added_variant_keeps_the_syllable_count(self):
        """Swapping the vowel must not change how many syllables the word has."""
        for variant in get_word_phonemes("dog"):
            assert variant.is_monosyllabic


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
            assert all(p.ipa != "ɚ" for p in variant.phonemes), (
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
            ipas = tuple(p.ipa for p in variant.phonemes)
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
                v.phonemes[i].ipa == "ɪ"
                and i + 1 < len(v.phonemes)
                and v.phonemes[i + 1].ipa == "r"
                for i in range(len(v.phonemes) - 1)
            )
            for v in variants
        )
