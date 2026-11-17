"""Tests for English per-word decodability analysis."""

import pytest
from decodability.english import (
    EnglishStudentKnowledge,
    EnglishWordAnalysis,
    aggregate_scores_ufli_curriculum,
    analyse_word_english,
    assess_decodability_ufli,
)
from decodability.english.analyse import is_decodable_ufli
from decodability.english.definitions import UFLI_GPC_PAIRS_FLAT
from decodability.english.extract_words import extract_words_english
from decodability.english.scorers import (
    score_known_gpcs_english,
    score_sight_words_english,
)


@pytest.fixture(scope="module")
def knowledge():
    return EnglishStudentKnowledge(
        gpcs=[
            ("c", "k"),
            ("a", "æ"),
            ("t", "t"),
            ("s", "s"),
            ("m", "m"),
            ("n", "n"),
            ("h", "h"),
            ("o", "ɑ"),
            ("i", "ɪ"),
            ("th", "θ"),
            ("th", "ð"),
            ("sh", "ʃ"),
        ],
        sight_words={"the", "he", "said", "to"},
    )


@pytest.fixture(scope="module")
def no_knowledge():
    return EnglishStudentKnowledge()


class TestFindEnglish:
    def test_a_fully_decodable_word_has_a_parse_with_no_untaught(self, knowledge):
        analysis = analyse_word_english("cat", knowledge)

        assert analysis.decodable_parses
        assert all(parse.untaught == [] for parse in analysis.decodable_parses)

    def test_untaught_gpcs_are_located(self):
        """``cat`` for a student missing ``a`` parses, blocked only on ``a``."""
        student = EnglishStudentKnowledge(gpcs=[("c", "k"), ("t", "t")])
        analysis = analyse_word_english("cat", student)

        (parse,) = analysis.candidate_parses
        assert [
            (match.grapheme, match.phonemes, match.letter_positions)
            for match in parse.untaught
        ] == [("a", ("æ",), (1,))]

    def test_a_word_missing_from_the_dictionary_is_flagged(self, knowledge):
        """OOV words surface as ``in_dictionary=False``, not as empty parses."""
        analysis = analyse_word_english("zzznotaword", knowledge)

        assert analysis.in_dictionary is False
        assert analysis.pronunciations == []
        assert analysis.candidate_parses == []

    def test_an_unparseable_word_is_flagged(self, knowledge):
        """``queue`` is in CMUdict but can't be aligned under the UFLI inventory."""
        analysis = analyse_word_english("queue", knowledge)

        assert analysis.in_dictionary is True
        assert analysis.pronunciations, "the pronunciation is still recorded"
        assert analysis.candidate_parses == []

    def test_sight_words_are_flagged_case_insensitively(self, knowledge):
        assert analyse_word_english("The", knowledge).known_sight_word is True
        assert analyse_word_english("THE", knowledge).known_sight_word is True
        assert analyse_word_english("cat", knowledge).known_sight_word is False

    def test_affixed_and_clitic_sight_words_are_flagged(self, knowledge):
        """``he's`` is a sight word via the ``he`` + ``'s`` clitic route."""
        assert analyse_word_english("he's", knowledge).known_sight_word is True

    def test_every_match_indexes_back_into_the_word(self, knowledge):
        word = "Placed"
        analysis = analyse_word_english(word, knowledge)

        assert analysis.candidate_parses, "expected this word to parse"
        for parse in analysis.candidate_parses:
            for match in parse.matches:
                covered = "".join(
                    word.lower()[position] for position in match.letter_positions
                )
                assert covered == match.symbol.replace("-", "")

    def test_ge_alone_leaves_judged_blocked_on_dge(self):
        """``judged`` needs the ``dge`` trigraph: the ``d`` before ``g`` has
        no other /dʒ/ correspondence, so taught ``ge`` can't substitute and
        every parse flags ``dge`` across letters 2-4 as the blocker."""
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
        analysis = analyse_word_english("judged", student)

        assert analysis.candidate_parses, "expected this word to parse"
        assert analysis.decodable_parses == []
        for parse in analysis.candidate_parses:
            assert [
                (match.grapheme, match.letter_positions) for match in parse.untaught
            ] == [("dge", (2, 3, 4))]

    def test_judged_display_parse_prefers_the_shared_ed(self):
        """Between spelling the final /d/ as ``d`` and as ``-ed`` sharing the
        trigraph's ``e``, the affix-unit parse wins the display ranking.
        (The profile has no ``ge`` — ``judged`` never needs it.)"""
        student = EnglishStudentKnowledge(
            gpcs=[("j", "dʒ"), ("u", "ʌ"), ("dge", "dʒ"), ("-ed", "d")]
        )
        analysis = analyse_word_english("judged", student)

        display = analysis.display_parse
        assert display is not None
        assert display.untaught == []
        assert [
            (match.grapheme, match.letter_positions) for match in display.matches
        ] == [
            ("j", (0,)),
            ("u", (1,)),
            ("dge", (2, 3, 4)),
            ("-ed", (4, 5)),
        ]

    def test_judged_without_d_still_decodes_via_shared_ed(self):
        """Missing ``d → d`` doesn't block ``judged``: the ``-ed`` match reuses
        the trigraph's ``e``, so the affix parse alone is decodable."""
        student = EnglishStudentKnowledge(
            gpcs=[("j", "dʒ"), ("u", "ʌ"), ("dge", "dʒ"), ("-ed", "d")]
        )
        analysis = analyse_word_english("judged", student)

        (decodable,) = analysis.decodable_parses
        assert [match.grapheme for match in decodable.matches] == [
            "j",
            "u",
            "dge",
            "-ed",
        ]
        assert analysis.display_parse is decodable

    def test_judged_without_ed_still_decodes_via_bare_d(self):
        """Missing ``-ed → d`` leaves the letter-by-letter parse, and a parse
        with no blockers outranks the affix-unit parse on ``display_parse``'s
        fewest-untaught key — the affix preference only breaks ties."""
        student = EnglishStudentKnowledge(
            gpcs=[("j", "dʒ"), ("u", "ʌ"), ("dge", "dʒ"), ("d", "d")]
        )
        analysis = analyse_word_english("judged", student)

        (decodable,) = analysis.decodable_parses
        assert [match.grapheme for match in decodable.matches] == [
            "j",
            "u",
            "dge",
            "d",
        ]
        assert analysis.display_parse is decodable

    def test_judged_missing_both_endings_reports_each_parse_blocker(self):
        """With neither way to spell the final /d/, each parse is blocked by
        its own ending — ``d`` at (5,) or ``-ed`` at (4, 5) — and at one
        blocker each the affix-unit parse wins the display tiebreak."""
        student = EnglishStudentKnowledge(gpcs=[("j", "dʒ"), ("u", "ʌ"), ("dge", "dʒ")])
        analysis = analyse_word_english("judged", student)

        assert analysis.decodable_parses == []
        assert {
            tuple((m.grapheme, m.letter_positions) for m in parse.untaught)
            for parse in analysis.candidate_parses
        } == {(("d", (5,)),), (("-ed", (4, 5)),)}
        assert [
            (m.grapheme, m.letter_positions) for m in analysis.display_parse.untaught
        ] == [("-ed", (4, 5))]


class TestUFLIRule:
    @pytest.mark.parametrize(
        "word, expected",
        [
            ("cat", True),  # every GPC taught
            ("the", True),  # sight word
            ("he's", True),  # sight word via the 's clitic
            ("smoke", False),  # needs o -> oʊ (or o-e -> oʊ) and k -> k
            ("planet", False),  # reduced e needs some taught e GPC
            ("queue", False),  # no parse under the UFLI inventory
            ("zzznotaword", False),  # not in CMUdict
        ],
    )
    def test_verdicts(self, word, expected, knowledge):
        assert is_decodable_ufli(word, knowledge) is expected

    def test_a_known_sight_word_is_decodable_despite_no_parse(self):
        """``queue`` can't be parsed at all, yet a taught sight word decodes."""
        knowledge = EnglishStudentKnowledge(gpcs=[("c", "k")], sight_words={"queue"})

        analysis = analyse_word_english("queue", knowledge)

        assert assess_decodability_ufli(analysis) is True
        assert analysis.candidate_parses == [], "the failure is still recorded"

    def test_a_student_taught_nothing_decodes_no_real_word(self, no_knowledge):
        words = extract_words_english("The cat sat on the mat")

        decodability_verdicts = [
            is_decodable_ufli(word, no_knowledge) for word in words
        ]

        assert decodability_verdicts == [False] * len(words)

    def test_compound_word_decodability_follows_its_chunks(self):
        """``someone`` decodes when every chunk is a sight word or GPC-
        decodable and at least one is a sight word. ``one`` has no GPC parse
        (``o → w`` isn't a correspondence), so it must be the sight-word
        anchor — ``some`` sight alone can't carry it."""
        some_gpcs = [("s", "s"), ("o-e", "ʌ"), ("m", "m")]
        cases = [
            ({"some", "one"}, [], True),  # both sight words
            ({"one"}, some_gpcs, True),  # one sight, some decodable
            ({"some"}, some_gpcs, False),  # one untaught — undecodable
            (set(), some_gpcs, False),  # both untaught
        ]
        for sight_words, gpcs, expected in cases:
            knowledge = EnglishStudentKnowledge(gpcs=gpcs, sight_words=sight_words)
            analysis = analyse_word_english("someone", knowledge)
            assert analysis.known_sight_word is expected
            assert assess_decodability_ufli(analysis) is expected


class TestEarliestDecodableLesson:
    def test_each_match_reports_the_lesson_its_gpc_is_taught(self, no_knowledge):
        # c -> k is L14, a -> æ is L1, t -> t is L4: the word decodes at L14.
        analysis = analyse_word_english("cat", no_knowledge)

        (parse,) = analysis.candidate_parses
        assert [(m.grapheme, m.first_taught_lesson) for m in parse.matches] == [
            ("c", 14),
            ("a", 1),
            ("t", 4),
        ]
        assert parse.decodable_at_lesson == 14
        assert analysis.earliest_decodable_lesson == 14

    def test_earliest_decodable_parses_finish_teaching_first(self, no_knowledge):
        # ``smoke`` parses via the o-e digraph (L56) or via the open-syllable
        # o -> oʊ (L68) plus a silent e; the o-e route decodes earlier.
        analysis = analyse_word_english("smoke", no_knowledge)

        assert analysis.earliest_decodable_lesson == 56
        assert all(
            parse.decodable_at_lesson == 56
            for parse in analysis.earliest_decodable_parses
        )
        assert any(
            match.grapheme == "o-e"
            for parse in analysis.earliest_decodable_parses
            for match in parse.matches
        )

    def test_curriculum_sight_words_report_introduction_lesson(self, no_knowledge):
        # ``the`` is introduced as a sight word at L1, long before th -> ð
        # (L46) finishes its only GPC parse (L68).
        analysis = analyse_word_english("the", no_knowledge)

        assert analysis.earliest_decodable_lesson == 1
        # …but this particular student hasn't been taught it yet.
        assert assess_decodability_ufli(analysis) is False

    def test_derived_matches_take_the_lesson_of_their_licensing_gpcs(
        self, no_knowledge
    ):
        # ``running`` needs the doubled-consonant convention, assumed from L67.
        analysis = analyse_word_english("running", no_knowledge)

        assert analysis.earliest_decodable_lesson == 67
        nn_uses = [
            match
            for parse in analysis.candidate_parses
            for match in parse.matches
            if match.grapheme == "nn" or match.gpc_id.startswith("DERIVED_DOUBLE")
        ]
        assert nn_uses
        assert all(match.first_taught_lesson == 67 for match in nn_uses)

    def test_a_word_with_no_decodable_route_reports_none(self, no_knowledge):
        analysis = analyse_word_english("queue", no_knowledge)

        assert analysis.earliest_decodable_lesson is None

    def test_judged_decodes_once_the_dge_trigraph_is_taught(self, no_knowledge):
        # dge -> dʒ is L70 — later than ge -> dʒ (L61), which can't serve here:
        # the parse's ceiling is the trigraph, whichever ending follows.
        analysis = analyse_word_english("judged", no_knowledge)

        assert analysis.earliest_decodable_lesson == 70

    @pytest.mark.parametrize(
        "word, expected",
        [
            # Compound: ``one`` is a sight word at L58 and ``some`` first
            # GPC-decodes at L62 (o-e -> ʌ) — the route unlocks at the max.
            ("someone", 62),
            # Both chunks are sight words: max(no L43, one L58).
            ("noone", 58),
            # Affixed stem: friend L66 + -s L20; walk L48 + -ed L64.
            ("friends", 66),
            ("walked", 64),
            # Clitic: he's decodes once ``he`` is a sight word (L23) — the
            # assumed 's correspondence needs no taught GPC.
            ("he's", 23),
        ],
    )
    def test_derived_sight_word_forms_report_their_own_lesson(
        self, word, expected, no_knowledge
    ):
        """Clitic, affixed, and compound sight-word forms report the lesson
        their last-taught constituent arrives — not just direct members."""
        analysis = analyse_word_english(word, no_knowledge)

        assert analysis.earliest_decodable_lesson == expected


class TestAnalysisModel:
    def test_the_analysis_carries_the_word_as_given(self, knowledge):
        assert analyse_word_english("Cat", knowledge).word == "Cat"

    def test_the_analysis_is_frozen(self, knowledge):
        from pydantic import ValidationError

        analysis = analyse_word_english("cat", knowledge)

        with pytest.raises(ValidationError):
            analysis.known_sight_word = not analysis.known_sight_word

    def test_the_analysis_round_trips_through_serialisation(self, knowledge):
        analysis = analyse_word_english("placed", knowledge)

        assert EnglishWordAnalysis.model_validate(analysis.model_dump()) == analysis
        assert (
            EnglishWordAnalysis.model_validate_json(analysis.model_dump_json())
            == analysis
        )


class TestParityWithTheScoreBasedPath:
    """The analysis path must reproduce the verdicts the scorers produced."""

    TEXT = (
        "The cat sat on the mat and he's running to a friend. "
        "Someone beautiful walked through the jungle quickly, isn't it lovely. "
        "Xylophone queue isn't planet smoke either."
    )

    FLAT = sorted(UFLI_GPC_PAIRS_FLAT)

    @staticmethod
    def old_verdict(word: str, knowledge: EnglishStudentKnowledge) -> bool:
        scores = {
            score.__name__: score(word, knowledge)
            for score in (
                score_known_gpcs_english,
                score_sight_words_english,
            )
        }
        return aggregate_scores_ufli_curriculum(scores) == 1.0

    @pytest.mark.parametrize("taught_count", [0, 1, 25, 80, len(FLAT)])
    @pytest.mark.parametrize("sight_words", [set(), {"the", "friend", "someone"}])
    def test_verdicts_match_for_every_word_and_profile(self, taught_count, sight_words):
        knowledge = EnglishStudentKnowledge(
            gpcs=self.FLAT[:taught_count], sight_words=sight_words
        )
        words = sorted(set(extract_words_english(self.TEXT)))

        nondecodable_words = [
            word for word in words if not is_decodable_ufli(word, knowledge)
        ]

        assert nondecodable_words == sorted(
            {word for word in words if not self.old_verdict(word, knowledge)}
        )
