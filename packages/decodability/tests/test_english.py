import json
from pathlib import Path

import pandas as pd
import pytest
from decodability import rowwise
from decodability.english.aggregators import aggregate_scores_ufli_curriculum
from decodability.english.definitions import (
    UFLI_GPC_INVENTORY,
    UFLI_GPC_PAIRS_FLAT,
    ufli_gpc_inventory,
    ufli_gpc_pairs,
)
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
    _compound_segments,
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
        assert EnglishStudentKnowledge(**kwargs).render("quoted") == expected_str

    @pytest.mark.parametrize(
        "kwargs, expected_plain",
        [
            (
                {},
                "- Known GPCs: None\n- Known sight words: None",
            ),
            (
                {"gpcs": [("c", "k"), ("a", "æ"), ("t", "t")]},
                "- Known GPCs: c -> /k/; a -> /æ/; t -> /t/\n- Known sight words: None",
            ),
            (
                {"sight_words": {"The", "and"}},
                "- Known GPCs: None\n- Known sight words: and, the",
            ),
            (
                {
                    "gpcs": [("th", "ð"), ("th", "θ")],
                    "sight_words": {"the"},
                },
                "- Known GPCs: th -> /ð/; th -> /θ/\n- Known sight words: the",
            ),
        ],
    )
    def test_english_student_knowledge_render_plain(
        self, kwargs: dict, expected_plain: str
    ):
        """The "plain" notation used by decodable text generation."""
        assert EnglishStudentKnowledge(**kwargs).render("plain") == expected_plain

    def test_english_student_knowledge_render_omits_affixes(self):
        """Affixes stay out of the rendered block; generation adds its own line."""
        knowledge = EnglishStudentKnowledge(gpcs=[("-ed", "t")], affixes={"-ed"})
        for fmt in ("quoted", "plain"):
            assert "affix" not in knowledge.render(fmt).lower()

    def test_english_student_knowledge_render_rejects_unknown_formats(self):
        with pytest.raises(ValueError, match="Unknown knowledge format 'v2'"):
            EnglishStudentKnowledge().render("v2")  # type: ignore[arg-type]


PINNED_PAIRS_PATH = Path(__file__).parent / "fixtures/ufli_gpc_pairs_2026-09-10.json"


class TestUFLIInventoryVersions:
    def test_latest_version_is_the_inventory(self):
        assert ufli_gpc_inventory() == UFLI_GPC_INVENTORY
        assert ufli_gpc_pairs() == UFLI_GPC_PAIRS_FLAT

    def test_earlier_version_only_undoes_respellings(self):
        """Ids, graphemes and lessons stay; G159 reads /juni/ as it did before."""
        pinned = {gpc.id: gpc for gpc in ufli_gpc_inventory("2026-09-10")}
        changed = {
            gpc.id: pinned[gpc.id].phonemes
            for gpc in UFLI_GPC_INVENTORY
            if pinned[gpc.id] != gpc
        }
        assert changed == {"G159": ("j", "u", "n", "i")}

    def test_earlier_version_keeps_every_pair_it_was_frozen_with(self):
        """Fails when an entry is respelled without registering a version."""
        frozen = {
            tuple(pair) for pair in json.loads(PINNED_PAIRS_PATH.read_text())["pairs"]
        }
        assert frozen <= ufli_gpc_pairs("2026-09-10")

    def test_unknown_version_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown UFLI inventory version"):
            ufli_gpc_inventory("2025-01-01")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "inventory, gpc, should_pass",
        [
            ("2026-10-27", ("uni", "junɪ"), True),
            ("2026-10-27", ("uni", "juni"), False),
            ("2026-09-10", ("uni", "juni"), True),
            ("2026-09-10", ("uni", "junɪ"), False),
        ],
    )
    def test_gpcs_are_checked_against_the_profile_inventory(
        self, inventory: str, gpc: tuple[str, str], should_pass: bool
    ):
        if should_pass:
            EnglishStudentKnowledge(inventory=inventory, gpcs=[gpc])
        else:
            with pytest.raises(ValueError, match=f"inventory {inventory}"):
                EnglishStudentKnowledge(inventory=inventory, gpcs=[gpc])

    def test_inventory_is_not_part_of_the_profile(self):
        knowledge = EnglishStudentKnowledge(inventory="2026-09-10", gpcs=[("t", "t")])
        assert "inventory" not in knowledge.model_dump()

    @pytest.mark.parametrize(
        "inventory, uni",
        [("2026-10-27", ("uni", "junɪ")), ("2026-09-10", ("uni", "juni"))],
    )
    def test_scorer_parses_with_the_profile_inventory(
        self, inventory: str, uni: tuple[str, str]
    ):
        """A profile's spelling matches the parser only at its own version."""
        knowledge = EnglishStudentKnowledge(inventory=inventory, gpcs=[uni, ("t", "t")])
        assert score_known_gpcs_english("unit", knowledge) == 1.0


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

    def test_score_sight_words_english_inflections(self):
        """An inflected sight word passes when its affix was taught."""
        knowledge = EnglishStudentKnowledge(
            affixes={"-s", "-ed", "-ing", "-ies"},
            sight_words={"friend", "walk", "cry", "make", "want"},
        )
        assert score_sight_words_english("friends", knowledge) == 1.0
        assert score_sight_words_english("walking", knowledge) == 1.0
        # the taught affix unit covers all of -ed's allomorphs
        assert score_sight_words_english("wanted", knowledge) == 1.0
        assert score_sight_words_english("walked", knowledge) == 1.0
        # a correspondence pair licenses the same ending before the affix
        # is introduced — but -ed→əd only reads it after t/d stems; the
        # silent-e /d/,/t/ readings need their own pair
        ed_schwa = EnglishStudentKnowledge(
            gpcs=[("ed", "əd")], sight_words={"want", "walk"}
        )
        assert score_sight_words_english("wanted", ed_schwa) == 1.0
        assert score_sight_words_english("walked", ed_schwa) == 0.0
        # cry -> cries via the -ies unit (the ie letters spell the stem's y)
        assert score_sight_words_english("cries", knowledge) == 1.0
        # silent-e restoration: make + s
        assert score_sight_words_english("makes", knowledge) == 1.0

        # a taught correspondence pair licenses the matching ending before
        # its affix is officially introduced: s→z reads friend + s and
        # ed→t reads walk + ed
        no_affixes = EnglishStudentKnowledge(
            gpcs=[("s", "z"), ("ed", "t")], sight_words={"friend", "walk"}
        )
        assert score_sight_words_english("friends", no_affixes) == 1.0
        assert score_sight_words_english("walked", no_affixes) == 1.0

        # and the stem must be a known sight word
        unknown_stem = EnglishStudentKnowledge(affixes={"-s"}, sight_words={"pal"})
        assert score_sight_words_english("friends", unknown_stem) == 0.0

    def test_score_sight_words_english_prefixes(self):
        """A prefixed sight word passes when its affix was taught."""
        knowledge = EnglishStudentKnowledge(
            affixes={"pre-", "un-", "re-"},
            sight_words={"view", "happy", "do", "friend"},
        )
        assert score_sight_words_english("preview", knowledge) == 1.0
        assert score_sight_words_english("unhappy", knowledge) == 1.0
        assert score_sight_words_english("redo", knowledge) == 1.0

        # the affix route requires the taught unit — but ``pre``/``un`` are
        # real words and decodable chunks here, so the compound rule picks
        # these up (pre + view, un + happy)
        untaught_affix = EnglishStudentKnowledge(
            gpcs=[("pre", "pri"), ("un", "ʌn")],
            sight_words={"view", "happy"},
        )
        assert score_sight_words_english("preview", untaught_affix) == 1.0
        assert score_sight_words_english("unhappy", untaught_affix) == 1.0

        # a decodable prefix chunk alone isn't enough — the rest must be a
        # sight word or decodable too (``view`` is neither here)
        no_stem = EnglishStudentKnowledge(gpcs=[("pre", "pri")], sight_words={"happy"})
        assert score_sight_words_english("preview", no_stem) == 0.0

    def test_score_sight_words_english_stacked_affixes(self):
        """A sight word wearing a taught prefix and suffix resolves."""
        knowledge = EnglishStudentKnowledge(
            affixes={"un-", "-ly", "pre-", "-s", "-ing", "re-"},
            sight_words={"friend", "view", "do"},
        )
        assert score_sight_words_english("unfriendly", knowledge) == 1.0
        assert score_sight_words_english("previews", knowledge) == 1.0
        assert score_sight_words_english("redoing", knowledge) == 1.0

        # one missing affix sinks the chain
        no_suffix = EnglishStudentKnowledge(affixes={"un-"}, sight_words={"friend"})
        assert score_sight_words_english("unfriendly", no_suffix) == 0.0


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

    def test_sloth_short_o_override(self):
        """The override adds the short-o reading to ``sloth``; OOV ``sloths``
        inherits both readings through the s-suffix fallback."""
        sloth_ipas = {
            tuple(p.ipa for p in v.phonemes) for v in get_word_phonemes("sloth")
        }
        assert ("s", "l", "ɑ", "θ") in sloth_ipas
        assert ("s", "l", "oʊ", "θ") in sloth_ipas

        sloths_ipas = {
            tuple(p.ipa for p in v.phonemes) for v in get_word_phonemes("sloths")
        }
        assert ("s", "l", "ɑ", "θ", "s") in sloths_ipas
        assert ("s", "l", "oʊ", "θ", "z") in sloths_ipas

        student = EnglishStudentKnowledge(
            gpcs=[("s", "s"), ("l", "l"), ("o", "ɑ"), ("th", "θ")]
        )
        assert score_known_gpcs_english("sloth", student) == 1.0
        assert score_known_gpcs_english("sloths", student) == 1.0


class TestSuffixedSFallback:
    """OOV words ending in ``s``/``'s`` inherit the base word's CMUdict
    pronunciation with /s/ and /z/ appended (``haps`` → ``hap`` + /s/ or /z/)."""

    @pytest.mark.parametrize("word", ["haps", "hap's"])
    def test_oov_plural_and_possessive_inherit_base_plus_s(self, word: str):
        """``hap`` is in CMUdict; ``haps``/``hap's`` are not."""
        variants = get_word_phonemes(word)
        assert variants == (
            _say("h", "æ", "p", "s"),
            _say("h", "æ", "p", "z"),
        )

    def test_fallback_does_not_fire_for_cmudict_entries(self):
        """``hiss`` is in CMUdict as /h ɪ s/ — the fallback must not rewrite it
        as ``his`` + /s/ (which would be /h ɪ z s/)."""
        variants = get_word_phonemes("hiss")
        assert variants == (_say("h", "ɪ", "s"),)

    def test_fallback_uses_override_for_base(self, monkeypatch):
        """An override for the base also serves its ``s``-suffixed form."""
        monkeypatch.setitem(CMUDICT_OVERRIDES, "zzzbase", [["Z", "IY1"]])
        variants = get_word_phonemes("zzzbases")
        assert variants == (
            _say("z", "i", "s"),
            _say("z", "i", "z"),
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
            _say("z", "i", "s"),
            _say("z", "i", "z"),
            _say("z", "ɛ", "s"),
            _say("z", "ɛ", "z"),
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


class TestPronunciationDedup:
    """Distinct ARPABET variants can normalize to the same ``Pronunciation`` —
    stress-level swaps (``AE1/AY2`` vs ``AE2/AY1``), ``ER0`` vs a spelled-out
    ``AH0 R``, and ``IH R`` undone to ``IY R`` — so the dedup happens after
    normalization, not just on the raw ARPABET tuples."""

    @pytest.mark.parametrize(
        "word",
        [
            "advertised",  # AE1/AY2 vs AE2/AY1 stress swap
            "disarray",  # ER0 vs AH0 R
            "disappear",  # IH1 R vs IY1 R
        ],
    )
    def test_no_duplicate_pronunciations(self, word: str):
        variants = get_word_phonemes(word)
        assert len(variants) == len(set(variants))


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

    def test_gpc_rejects_bare_marker(self):
        with pytest.raises(ValueError, match="letters on both sides"):
            GPC(id="x", grapheme="-", phonemes=("k",))

    @pytest.mark.parametrize(
        "grapheme, symbol, is_prefix, is_suffix",
        [
            ("-ed", "ed", False, True),  # word-final affix
            ("-ve", "ve", False, True),  # positional grapheme
            ("pre-", "pre", True, False),  # word-initial affix
            ("un-", "un", True, False),
            ("a-e", "a-e", False, False),  # split digraph, no edge marker
            ("s", "s", False, False),  # plain letter
        ],
    )
    def test_gpc_edge_markers_set_position(
        self,
        grapheme: str,
        symbol: str,
        is_prefix: bool,
        is_suffix: bool,
    ):
        """A leading ``-`` marks a word-final unit, a trailing ``-`` a
        word-initial one; either way the marker comes off ``symbol``."""
        gpc = GPC(id="x", grapheme=grapheme, phonemes=("k",))
        assert gpc.symbol == symbol
        assert gpc.is_prefix is is_prefix
        assert gpc.is_suffix is is_suffix
        assert gpc.is_split is ("-" in symbol)

    def test_suffix_gpc_matches_medially(self):
        """``-ing`` is stripped to ``ing`` for matching — the same chunk
        appears mid-word in inflected forms, so ``sings`` parses as
        s + ing + s, not only at the word's end."""
        inventory = (GPC("S", "s", ("s",)), GPC("ING", "-ing", ("ɪ", "ŋ")))
        parses = parse_gpcs("sing", _say("s", "ɪ", "ŋ"), inventory)
        assert parses and parses[0][1].gpc.id == "ING"
        medial = parse_gpcs("sings", _say("s", "ɪ", "ŋ", "s"), inventory)
        assert medial and medial[0][1].gpc.id == "ING"

    def test_prefix_gpc_matches_at_word_start_only(self):
        """``pre-`` anchors to the word start — the ``re`` inside ``pre`` is
        not a prefix, unlike suffixes which match medially."""
        inventory = (GPC("S", "s", ("s",)), GPC("PRE", "pre-", ("p", "r", "i")))
        parses = parse_gpcs("pre", _say("p", "r", "i"), inventory)
        assert parses and parses[0][0].gpc.id == "PRE"
        medial = parse_gpcs("spre", _say("s", "p", "r", "i"), inventory)
        assert not medial or all(match.gpc.id != "PRE" for match in medial[0])

    @pytest.mark.parametrize(
        "word, gpcs, expected",
        [
            # -ing as a taught unit decodes words containing the chunk.
            ("sing", [("s", "s"), ("ing", "ɪŋ")], 1.0),
            ("king", [("k", "k"), ("ing", "ɪŋ")], 1.0),
            # medial -ve in an inflection: give + s.
            ("gives", [("g", "g"), ("i", "ɪ"), ("ve", "v"), ("s", "z")], 1.0),
            # medial -le: table + s.
            (
                "tables",
                [("t", "t"), ("a", "eɪ"), ("b", "b"), ("le", "əl"), ("s", "z")],
                1.0,
            ),
            # the unit also covers non-morphemic occurrences (sing + er).
            ("singer", [("s", "s"), ("ing", "ɪŋ"), ("er", "ər")], 1.0),
            ("singer", [("s", "s"), ("i", "ɪ"), ("ng", "ŋ"), ("er", "ər")], 1.0),
        ],
    )
    def test_affix_gpcs_match_at_any_position(
        self,
        word: str,
        gpcs: list[tuple[str, str]],
        expected: float,
    ):
        student = EnglishStudentKnowledge(gpcs=gpcs)
        assert score_known_gpcs_english(word, student) == expected

    @pytest.mark.parametrize(
        "gpcs",
        [
            [("-ed", "t")],  # marked form, as written in the workbook
            [("ed", "t")],  # stripped form, as stored in profiles
            [("pre-", "pri")],
            [("_ce", "s")],  # underscore marker, as written in the workbook
            [("a_e", "eɪ")],  # underscore split digraph
        ],
    )
    def test_marked_and_stripped_gpc_forms_validate(self, gpcs):
        """Affix markers on taught GPCs are presentation, not identity."""
        EnglishStudentKnowledge(gpcs=gpcs)

    def test_marked_affix_gpc_scores_like_stripped(self):
        """A profile listing ``("-ed", "d")`` gets the same credit as
        ``("ed", "d")``: ``played`` parses via ``play`` + ``-ed``."""
        for gpcs in ([("-ed", "d")], [("ed", "d")]):
            student = EnglishStudentKnowledge(
                gpcs=[("p", "p"), ("l", "l"), ("ay", "eɪ"), *gpcs]
            )
            assert score_known_gpcs_english("played", student) == 1.0


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


class TestReducedYodFreePass:
    """When the inventory teaches ``u → (j, u)``, a bare ``u`` paired with
    /j/ plus an unstressed vowel parses via ``REDUCED_YOD_U`` — the glide is
    part of the taught correspondence; only its vowel reduces. This is what
    lets ``partic-u-larly`` /j ə/ decode for a student who knows ``u → ju``.
    """

    # ``tuler`` is a stand-in word: t + u + l + er, /t j ə l ə r/. The only
    # way ``u`` covers /j ə/ is the reduced-yod pass.
    _TULER_PHONEMES = (
        Phoneme("t", False),
        Phoneme("j", False),
        Phoneme("ə", True),  # unstressed — the reducible vowel of the yod pair
        Phoneme("l", False),
        Phoneme("ə", True),
        Phoneme("r", False),
    )
    _TULER_INVENTORY = (
        GPC("T", "t", ("t",)),
        GPC("U_YOD", "u", ("j", "u")),
        GPC("L", "l", ("l",)),
        GPC("ER", "er", ("ə", "r")),
    )

    def test_u_covers_j_plus_unstressed_vowel(self):
        parses = parse_gpcs(
            "tuler",
            Pronunciation(phonemes=self._TULER_PHONEMES, is_monosyllabic=False),
            self._TULER_INVENTORY,
        )
        assert len(parses) == 1
        (parse,) = parses
        u_match = parse[1]
        assert u_match.gpc.id == "REDUCED_YOD_U"
        assert u_match.gpc.grapheme == "u"
        assert u_match.gpc.phonemes == ("j", "ə")
        assert u_match.letter_positions == (1,)
        assert u_match.phoneme_span == (1, 3)
        assert u_match.gpc.is_reduced_vowel

    def test_yod_pass_requires_u_to_ju_in_inventory(self):
        """Without a ``u → (j, u)`` GPC the pass is withheld — the glide is
        part of that correspondence, not a universal reduction."""
        inventory = tuple(gpc for gpc in self._TULER_INVENTORY if gpc.id != "U_YOD")
        parses = parse_gpcs(
            "tuler",
            Pronunciation(phonemes=self._TULER_PHONEMES, is_monosyllabic=False),
            inventory,
        )
        assert parses == ()

    def test_yod_pass_requires_unstressed_vowel(self):
        """A stressed vowel after /j/ doesn't reduce — ``u`` → /j ə/ only
        fires on the unstressed form."""
        phonemes = (
            Phoneme("t", False),
            Phoneme("j", False),
            Phoneme("ə", False),  # stressed schwa — no reduction
            Phoneme("l", False),
            Phoneme("ə", True),
            Phoneme("r", False),
        )
        parses = parse_gpcs(
            "tuler",
            Pronunciation(phonemes=phonemes, is_monosyllabic=False),
            self._TULER_INVENTORY,
        )
        assert parses == ()

    def test_yod_pass_withheld_when_monosyllabic(self):
        parses = parse_gpcs(
            "tuler",
            Pronunciation(phonemes=self._TULER_PHONEMES, is_monosyllabic=True),
            self._TULER_INVENTORY,
        )
        assert parses == ()

    def test_yod_pass_fires_only_on_u(self):
        """The glide belongs to the ``u`` correspondence — another vowel
        letter can't claim /j ə/."""
        parses = parse_gpcs(
            "toler",
            Pronunciation(phonemes=self._TULER_PHONEMES, is_monosyllabic=False),
            self._TULER_INVENTORY,
        )
        assert parses == ()

    def test_particularly_parses_via_reduced_yod_end_to_end(self):
        """Through the real pipeline: ``particularly`` parses under the full
        UFLI inventory only because ``u`` covers /j ə/."""
        variants, parses = _parse_corpus_word("particularly")
        assert parses, "particularly should parse under the full UFLI inventory"
        assert any(
            any(match.gpc.id == "REDUCED_YOD_U" for match in parse)
            for _, parse in parses
        )


class TestScoreReducedYodRequiresYodPair:
    """The scorer credits ``REDUCED_YOD_U`` only when the unreduced pair
    ``u → ju`` was taught — knowing just any ``u`` GPC (``u → ʌ``) is not
    enough, since the /j/ glide is part of the correspondence.
    """

    _BASE_GPCS = [
        ("p", "p"),
        ("ar", "ɑr"),
        ("t", "t"),
        ("i", "ɪ"),
        ("c", "k"),
        ("l", "l"),
        ("ar", "ər"),
        ("-ly", "li"),
    ]

    def test_u_ju_taught_decodes_particularly(self):
        student = EnglishStudentKnowledge(gpcs=[*self._BASE_GPCS, ("u", "ju")])
        assert score_known_gpcs_english("particularly", student) == 1.0

    def test_other_u_gpcs_do_not_unlock_the_yod(self):
        """``u → ʌ`` (or ``u → u``) teaches the letter but not the glide."""
        for u_pair in ("ʌ", "u", "ʊ"):
            student = EnglishStudentKnowledge(gpcs=[*self._BASE_GPCS, ("u", u_pair)])
            assert score_known_gpcs_english("particularly", student) == 0.0

    def test_u_ju_also_credits_plain_reduced_u(self):
        """``u → ju`` is a ``u`` GPC, so it satisfies the ordinary reduced-
        vowel rule too — ``album``'s ``u → ə`` parses for free."""
        student = EnglishStudentKnowledge(
            gpcs=[("a", "æ"), ("l", "l"), ("b", "b"), ("m", "m"), ("u", "ju")]
        )
        assert score_known_gpcs_english("album", student) == 1.0


class TestCompoundSightWordDecoding:
    """A word built from known pieces is decodable: every contiguous chunk
    must be a taught sight word or independently GPC-decodable, and at least
    one chunk must be a sight word.
    """

    def test_two_sight_words(self):
        """``noone`` = ``no`` + ``one`` — undecodable as a whole (the middle
        ``o`` has no phoneme) but both halves are memorized."""
        student = EnglishStudentKnowledge(sight_words={"no", "one"})
        assert score_sight_words_english("noone", student) == 1.0
        assert _compound_segments("noone", student) == ["no", "one"]

    def test_sight_word_plus_decodable_rest(self):
        """``someone`` = ``some`` (s, o-e → ʌ, m) + sight word ``one``."""
        student = EnglishStudentKnowledge(
            gpcs=[("s", "s"), ("o-e", "ʌ"), ("m", "m")],
            sight_words={"one"},
        )
        assert score_sight_words_english("someone", student) == 1.0
        assert _compound_segments("someone", student) == ["some", "one"]

    def test_decodable_chunk_can_follow_the_sight_word(self):
        """The sight word needn't come last: ``onesome``-style splits work."""
        student = EnglishStudentKnowledge(
            gpcs=[("s", "s"), ("o-e", "ʌ"), ("m", "m")],
            sight_words={"one"},
        )
        assert _compound_segments("onesome", student) == ["one", "some"]

    def test_requires_a_sight_word_chunk(self):
        """A compound of purely decodable chunks is out of scope — ``catdog``
        stays undecodable even though ``cat`` and ``dog`` both decode."""
        student = EnglishStudentKnowledge(
            gpcs=[
                ("c", "k"),
                ("a", "æ"),
                ("t", "t"),
                ("d", "d"),
                ("o", "ɑ"),
                ("g", "g"),
            ],
            sight_words={"the"},
        )
        assert score_sight_words_english("catdog", student) == 0.0

    def test_missing_sight_word_chunk_fails(self):
        """``someone`` without ``one`` in sight words stays undecodable."""
        student = EnglishStudentKnowledge(
            gpcs=[("s", "s"), ("o-e", "ʌ"), ("m", "m")],
            sight_words={"two"},
        )
        assert score_sight_words_english("someone", student) == 0.0

    def test_undecodable_rest_fails(self):
        """The non-sight chunks must actually decode — without ``o-e → ʌ``
        the ``some`` half can't be sounded out."""
        student = EnglishStudentKnowledge(
            gpcs=[("s", "s"), ("m", "m")],
            sight_words={"one"},
        )
        assert score_sight_words_english("someone", student) == 0.0

    def test_no_sight_words_at_all_short_circuits(self):
        """No taught sight words → no compound can qualify."""
        student = EnglishStudentKnowledge(gpcs=[("s", "s"), ("o-e", "ʌ"), ("m", "m")])
        assert score_sight_words_english("someone", student) == 0.0


class TestSharedLetters:
    """A grapheme may reuse one boundary letter claimed by a neighbor — the
    ``g`` in ``jungle`` serves ``ng → ŋ`` and ``g → g``; the ``e`` in
    ``placed`` is the ``a-e`` tail, part of ``ce → s``, and part of
    ``-ed → t``. Only boundary letters share; interiors stay exclusive.
    """

    def test_jungle_g_shared_between_ng_and_g(self):
        variants, parses = _parse_corpus_word("jungle")
        assert parses, "jungle should parse under the full UFLI inventory"
        # some parse puts ng on (2,3) and g on (3,) — the shared g
        assert any(
            ("ng", (2, 3)) in matches and ("g", (3,)) in matches
            for _, parse in parses
            for matches in [[(m.gpc.grapheme, m.letter_positions) for m in parse]]
        )

    def test_placed_e_shared_three_ways(self):
        variants, parses = _parse_corpus_word("placed")
        assert parses, "placed should parse under the full UFLI inventory"
        assert any(
            [(m.gpc.grapheme, m.letter_positions) for m in parse]
            == [
                ("p", (0,)),
                ("l", (1,)),
                ("a-e", (2, 4)),
                ("ce", (3, 4)),
                ("-ed", (4, 5)),
            ]
            for _, parse in parses
        )

    def test_shared_letter_only_one_position_back(self):
        """A grapheme can borrow the previous match's last letter — not two
        letters back. ``bc`` after ``abc`` would share two letters."""
        inventory = (
            GPC("ABC", "abc", ("x",)),
            GPC("BC", "bc", ("y", "z")),
        )
        phonemes = _say("x", "y", "z")
        assert parse_gpcs("abc", phonemes, inventory) == ()
        # but bc starting on abc's last letter (one back) is allowed
        inventory2 = (
            GPC("AB", "ab", ("x",)),
            GPC("BC", "bc", ("y", "z")),
        )
        parses = parse_gpcs("abc", phonemes, inventory2)
        assert len(parses) == 1
        assert [m.letter_positions for m in parses[0]] == [(0, 1), (1, 2)]

    def test_reserved_tail_shareable_only_as_last_letter(self):
        """A split tail can be shared as a match's last letter (``be`` below)
        but not buried in a match's interior (``bec``)."""
        # a-e[0,2] reserves position 2; bec[1,3] needs it mid-span → blocked
        blocked = parse_gpcs(
            "abec",
            _say("eɪ", "x"),
            (GPC("A_E", "a-e", ("eɪ",)), GPC("BEC", "bec", ("x",))),
        )
        assert blocked == ()
        # be[1,2] ends on the reserved tail — the shared silent-e case
        allowed = parse_gpcs(
            "abe",
            _say("eɪ", "j", "ə"),
            (GPC("A_E", "a-e", ("eɪ",)), GPC("BE", "be", ("j", "ə"))),
        )
        assert len(allowed) == 1
        assert [m.letter_positions for m in allowed[0]] == [(0, 2), (1, 2)]

    def test_jungle_scores_end_to_end(self):
        student = EnglishStudentKnowledge(
            gpcs=[("j", "dʒ"), ("u", "ʌ"), ("ng", "ŋ"), ("g", "g"), ("-le", "əl")]
        )
        assert score_known_gpcs_english("jungle", student) == 1.0

    def test_placed_scores_end_to_end(self):
        student = EnglishStudentKnowledge(
            gpcs=[("p", "p"), ("l", "l"), ("a_e", "eɪ"), ("_ce", "s"), ("-ed", "t")]
        )
        assert score_known_gpcs_english("placed", student) == 1.0

    def test_placed_needs_the_shared_letter_pieces(self):
        """Without ``ce → s`` the shared-``e`` parse collapses — ``placed``
        stays undecodable."""
        student = EnglishStudentKnowledge(
            gpcs=[("p", "p"), ("l", "l"), ("a_e", "eɪ"), ("-ed", "t")]
        )
        assert score_known_gpcs_english("placed", student) == 0.0

    def test_judged_ed_shares_the_dge_e(self):
        """``judged`` parses as ``j`` + ``u`` + ``dge`` + ``-ed`` — the
        suffix's first letter is the trigraph's last."""
        _, parses = _parse_corpus_word("judged")
        assert any(
            [(m.gpc.grapheme, m.letter_positions) for m in parse]
            == [
                ("j", (0,)),
                ("u", (1,)),
                ("dge", (2, 3, 4)),
                ("-ed", (4, 5)),
            ]
            for _, parse in parses
        )

    def test_judged_needs_the_dge_trigraph_not_ge(self):
        """The ``d`` before ``g`` has no other /dʒ/ correspondence, so every
        parse goes through ``dge`` — ``ge`` → /dʒ/ is unreachable here."""
        _, parses = _parse_corpus_word("judged")
        assert parses, "judged should parse under the full UFLI inventory"
        for _, parse in parses:
            graphemes = [m.gpc.grapheme for m in parse]
            assert "dge" in graphemes
            assert "ge" not in graphemes

    def test_judged_scores_end_to_end(self):
        student = EnglishStudentKnowledge(
            gpcs=[("j", "dʒ"), ("u", "ʌ"), ("dge", "dʒ"), ("-ed", "d")]
        )
        assert score_known_gpcs_english("judged", student) == 1.0

    def test_ge_alone_does_not_unlock_judged(self):
        """Taught ``ge`` → /dʒ/ (and even ``g`` → /dʒ/) can't substitute for
        the trigraph — ``dge`` is atomic, not ``d`` + ``ge``."""
        student = EnglishStudentKnowledge(
            gpcs=[
                ("j", "dʒ"),
                ("u", "ʌ"),
                ("d", "d"),
                ("ge", "dʒ"),
                ("g", "dʒ"),
                ("-ed", "d"),
            ]
        )
        assert score_known_gpcs_english("judged", student) == 0.0


class TestProductiveSpellingAndPronunciationRules:
    def test_positional_final_e_is_silent_after_a_consonant(self):
        student = EnglishStudentKnowledge(gpcs=[("h", "h"), ("ou", "aʊ"), ("s", "s")])
        assert score_known_gpcs_english("house", student) == 0.0
        # Once the silent-e family is fully taught (UFLI lesson 62), the
        # trailing-e convention unlocks and the same parse decodes.
        student = EnglishStudentKnowledge(
            gpcs=[
                ("h", "h"),
                ("ou", "aʊ"),
                ("s", "s"),
                ("a-e", "eɪ"),
                ("i-e", "aɪ"),
                ("o-e", "oʊ"),
                ("e-e", "i"),
                ("u-e", "u"),
                ("u-e", "ju"),
                ("ce", "s"),
                ("ge", "dʒ"),
                ("ve", "v"),
                ("o-e", "ʌ"),
            ]
        )
        assert score_known_gpcs_english("house", student) == 1.0

    def test_final_e_does_not_turn_bare_u_into_the_ue_vowel_team(self):
        student = EnglishStudentKnowledge(gpcs=[("b", "b"), ("l", "l"), ("u", "u")])
        assert score_known_gpcs_english("blue", student) == 0.0

    def test_dropped_e_soft_c_is_licensed_by_taught_ce(self):
        student = EnglishStudentKnowledge(
            gpcs=[("d", "d"), ("a", "æ"), ("n", "n"), ("ce", "s"), ("ing", "ɪŋ")]
        )
        assert score_known_gpcs_english("dancing", student) == 1.0

    def test_dropped_e_rule_does_not_replace_ordinary_hard_c(self):
        student = EnglishStudentKnowledge(
            gpcs=[("a", "æ"), ("n", "n"), ("ce", "s"), ("ing", "ɪŋ")]
        )
        assert score_known_gpcs_english("canning", student) == 0.0

    @pytest.mark.parametrize(
        "spelling, phonemes, derived_id",
        [
            ("x", ("g", "z"), "DERIVED_X_VOICED"),
            ("xc", ("k", "s"), "DERIVED_XC"),
            ("xh", ("g", "z"), "DERIVED_XH"),
        ],
    )
    def test_contextual_x_readings_are_available(
        self, spelling: str, phonemes: tuple[str, ...], derived_id: str
    ):
        parses = parse_gpcs(spelling, _say(*phonemes), ())
        assert any(parse[0].gpc.id == derived_id for parse in parses)

    def test_contextual_x_reading_still_requires_taught_x(self):
        student = EnglishStudentKnowledge(
            gpcs=[("e", "ɛ"), ("a", "æ"), ("c", "k"), ("t", "t")]
        )
        assert score_known_gpcs_english("exact", student) == 0.0
        student.gpcs.append(("x", "ks"))
        assert score_known_gpcs_english("exact", student) == 1.0

    def test_es_allows_the_reduced_iz_pronunciation_and_shared_e(self):
        pronunciation = Pronunciation(
            phonemes=(
                *_tag("p", "l", "ɛ", "dʒ"),
                Phoneme("ɪ", True),
                Phoneme("z", False),
            ),
            is_monosyllabic=False,
        )
        inventory = (
            GPC("P", "p", ("p",)),
            GPC("L", "l", ("l",)),
            GPC("E", "e", ("ɛ",)),
            GPC("DGE", "dge", ("dʒ",)),
            GPC("ES", "-es", ("ə", "z")),
        )
        parses = parse_gpcs("pledges", pronunciation, inventory)
        assert any(
            [(m.gpc.symbol, m.letter_positions) for m in parse]
            == [
                ("p", (0,)),
                ("l", (1,)),
                ("e", (2,)),
                ("dge", (3, 4, 5)),
                ("es", (5, 6)),
            ]
            for parse in parses
        )

    def test_letter_name_initialisms_use_known_constituent_letters(self):
        assert (
            score_known_gpcs_english(
                "TV", EnglishStudentKnowledge(gpcs=[("t", "t"), ("v", "v")])
            )
            == 1.0
        )
        assert (
            score_known_gpcs_english(
                "OK", EnglishStudentKnowledge(gpcs=[("o", "oʊ"), ("k", "k")])
            )
            == 1.0
        )

    def test_letter_name_readings_require_all_caps_spelling(self):
        assert (
            score_known_gpcs_english(
                "tv", EnglishStudentKnowledge(gpcs=[("t", "t"), ("v", "v")])
            )
            == 0.0
        )
        assert (
            score_known_gpcs_english(
                "am",
                EnglishStudentKnowledge(gpcs=[("a", "eɪ"), ("m", "m")]),
            )
            == 0.0
        )

    def test_taught_digraph_is_not_reinterpreted_as_an_initialism(self):
        student = EnglishStudentKnowledge(gpcs=[("t", "t"), ("h", "h")])
        assert score_known_gpcs_english("th", student) == 0.0

    def test_gu_glide_is_licensed_by_taught_g_and_u(self):
        student = EnglishStudentKnowledge(
            gpcs=[
                ("bi", "baɪ"),
                ("l", "l"),
                ("i", "ɪ"),
                ("ng", "ŋ"),
                ("g", "g"),
                ("u", "ʌ"),
                ("a", "æ"),
            ]
        )
        assert score_known_gpcs_english("bilingual", student) == 1.0
        without_u = EnglishStudentKnowledge(gpcs=student.gpcs[:-2] + [("a", "æ")])
        assert score_known_gpcs_english("bilingual", without_u) == 0.0

    @pytest.mark.parametrize("word", ["isn't", "wasn't"])
    def test_nt_contraction_can_use_syllabic_n(self, word: str):
        variants = get_word_phonemes(word)
        assert any(
            not any(phoneme.ipa == "ə" for phoneme in variant.phonemes)
            for variant in variants
        )


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

    def test_ao_before_r_is_not_swapped(self):
        """The merger does not apply before /r/: ``for`` stays /fɔr/ — it
        never homophonizes with ``far`` /fɑr/ even for merged speakers."""
        assert _add_cot_caught_variants([["F", "AO1", "R"]]) == [["F", "AO1", "R"]]

    def test_only_non_pre_r_aos_are_swapped(self):
        """In a mixed variant the pre-/r/ AO stays; the other still merges."""
        assert _add_cot_caught_variants([["AO1", "T", "AO0", "R"]]) == [
            ["AO1", "T", "AO0", "R"],
            ["AA1", "T", "AO0", "R"],
        ]

    def test_for_yields_no_merged_reading_end_to_end(self):
        vowels = {
            phoneme.ipa
            for variant in get_word_phonemes("for")
            for phoneme in variant.phonemes
        }
        assert "ɑ" not in vowels, "for should not offer the merged /ɑr/ reading"

    def test_for_parses_only_via_ɔr_correspondences(self):
        """``o → ɑ`` must not match the /ɔ/ in /fɔr/; the word still parses
        via ``or → ɔr`` and the unmerged ``o → ɔ``."""
        _, parses = _parse_corpus_word("for")
        assert parses, "for should still parse under the full UFLI inventory"
        used = {match.gpc.id for _, parse in parses for match in parse}
        assert "G010" not in used, "o -> ɑ should not match /ɔ/ before r"
        assert {"G077", "G168"} <= used, "or -> ɔr and o -> ɔ should still parse"


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
