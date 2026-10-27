"""Per-word decodability analysis for English.

- :func:`analyse_word_english` analyses a word based on a student knowledge
    profile, returning an English-specific analysis object.
- :func:`assess_decodability_ufli` takes the analysis object and returns a
    boolean indicating decodability.
- :func:`is_decodable_ufli` composes the two, taking a word and a student
    knowledge profile and returning a boolean indicating decodability.
"""

from __future__ import annotations

from functools import cache

from pydantic import BaseModel, ConfigDict

from decodability.english.definitions import (
    UFLI_SIGHT_WORD_LESSONS,
    normalize_gpc_grapheme,
)
from decodability.english.extract_phonemes import get_word_phonemes
from decodability.english.gpc_parser import GPCMatch, parse_gpcs
from decodability.english.models import EnglishStudentKnowledge
from decodability.english.scorers import (
    GPC_INVENTORY,
    _is_taught_match,
    normalize_taught_gpcs,
    score_sight_words_english,
)


class GPCUse(BaseModel):
    """One GPC match inside a parse, marked with whether it was taught.

    ``grapheme`` is the inventory form, so it may carry a positional marker
    (``-ed``, ``pre-``) or a split-digraph hyphen (``a-e``); ``symbol`` is the
    matched-letters form with markers stripped. ``letter_positions`` indexes
    into the word and may be non-contiguous (split digraphs) or overlap a
    neighbouring match (shared boundary letters). ``phoneme_span`` is the
    half-open range of the pronunciation variant's phonemes the match covers.

    ``taught`` is relative to the analysed student's knowledge.
    ``first_taught_lesson`` is curriculum-relative: the earliest UFLI lesson
    at which the correspondence is taught — for derived, silent and reduced
    matches, the lesson at which the GPCs licensing them are all taught.
    ``None`` means the UFLI sequence never teaches the correspondence.
    """

    model_config = ConfigDict(frozen=True)

    gpc_id: str
    grapheme: str
    symbol: str
    phonemes: tuple[str, ...]
    letter_positions: tuple[int, ...]
    phoneme_span: tuple[int, int]
    taught: bool
    first_taught_lesson: int | None


class ParseAssessment(BaseModel):
    """One valid parse of one pronunciation variant, with per-match flags.

    ``decodable_at_lesson`` is the earliest UFLI lesson at which every GPC in
    the parse is taught — i.e. when this parse decodes for a student following
    the curriculum — or ``None`` when some match's correspondence is never
    taught.
    """

    model_config = ConfigDict(frozen=True)

    variant_index: int
    matches: list[GPCUse]
    decodable_at_lesson: int | None

    @property
    def untaught(self) -> list[GPCUse]:
        """The matches the student has not been taught — this parse's blockers."""
        return [match for match in self.matches if not match.taught]


class PronunciationVariant(BaseModel):
    """One pronunciation variant the parser was tried against."""

    model_config = ConfigDict(frozen=True)

    phonemes: list[str]
    is_monosyllabic: bool


class EnglishWordAnalysis(BaseModel):
    """Analysis of an English word based on a student knowledge profile.

    A word is non-decodable when every route fails, in order:
    a known sight word is decodable regardless of anything else; otherwise
    ``in_dictionary`` being False means no pronunciation was found in CMUdict
    or the overrides; ``candidate_parses`` being empty means no pronunciation
    variant could be aligned to the spelling under the UFLI inventory; and any
    remaining parse is blocked by the matches in ``ParseAssessment.untaught``.
    Consumers surface ``display_parse`` — the package's ranked pick — rather
    than choosing among ``candidate_parses`` themselves.

    ``earliest_decodable_lesson`` is curriculum-relative, independent of the
    analysed student: the earliest UFLI lesson at which the word decodes for
    a student following the sequence — the best parse's
    ``decodable_at_lesson``, or the first lesson whose cumulative sight words,
    affixes, and GPCs unlock the word via the sight-word route (direct, clitic,
    affixed-stem, or compound), whichever is earlier. ``None`` when no route
    ever decodes it (e.g. an unparseable word that is never a sight word or
    a compound of known pieces).
    """

    model_config = ConfigDict(frozen=True)

    word: str
    known_sight_word: bool = False
    in_dictionary: bool = True
    pronunciations: list[PronunciationVariant] = []
    candidate_parses: list[ParseAssessment] = []
    earliest_decodable_lesson: int | None = None

    @property
    def decodable_parses(self) -> list[ParseAssessment]:
        """The parses using only taught GPCs — any one makes the word decodable."""
        return [parse for parse in self.candidate_parses if not parse.untaught]

    @property
    def earliest_decodable_parses(self) -> list[ParseAssessment]:
        """The parses that decode at ``earliest_decodable_lesson``.

        Empty when the earliest route is not a GPC parse — e.g. when the
        sight-word route decodes the word before any parse does.
        """
        return [
            parse
            for parse in self.candidate_parses
            if parse.decodable_at_lesson == self.earliest_decodable_lesson
        ]

    @property
    def display_parse(self) -> ParseAssessment | None:
        """The single parse a consumer should surface.

        Ranking, in order:
        1. a parse attaining ``earliest_decodable_lesson`` — the
           curriculum-canonical reading — when one exists; otherwise all
           candidate parses are considered (e.g. when the earliest route is
           the sight-word route and produces no parse);
        2. fewest untaught matches for this student — the least-blocked parse;
        3. parses using taught-as-unit affixes (edge-marked graphemes like
           ``-ing``, ``un-``) preferred over letter-by-letter decompositions —
           so "wishing" displays ``-ing`` as one morpheme rather than shredding
           it into ``i`` + ``ng``;
        4. earliest UFLI lesson at which the parse fully decodes.
        """
        if not self.candidate_parses:
            return None
        candidates = self.earliest_decodable_parses or self.candidate_parses

        def key(p: ParseAssessment) -> tuple[int, bool, float | int]:
            has_affix = any(
                m.grapheme.startswith("-") or m.grapheme.endswith("-")
                for m in p.matches
            )
            return (
                len(p.untaught),
                not has_affix,
                p.decodable_at_lesson
                if p.decodable_at_lesson is not None
                else float("inf"),
            )

        return min(candidates, key=key)


@cache
def _curriculum_steps() -> tuple[
    tuple[int, frozenset[tuple[str, str]], frozenset[str]], ...
]:
    """Cumulative taught sets after each UFLI lesson, in lesson order.

    Each step is ``(lesson, taught_pairs, taught_graphemes)`` for a student
    who has been taught every GPC up to and including that lesson — the same
    canonical form ``normalize_taught_gpcs`` produces for a profile.
    """
    by_lesson: dict[int, list[tuple[str, str]]] = {}
    for gpc in GPC_INVENTORY:
        by_lesson.setdefault(gpc.decode_lesson or gpc.lesson, []).append(
            (normalize_gpc_grapheme(gpc.grapheme), "".join(gpc.phonemes))
        )

    steps = []
    taught_pairs: set[tuple[str, str]] = set()
    taught_graphemes: set[str] = set()
    for lesson in sorted(by_lesson):
        taught_pairs.update(by_lesson[lesson])
        taught_graphemes.update(grapheme for grapheme, _ in by_lesson[lesson])
        steps.append((lesson, frozenset(taught_pairs), frozenset(taught_graphemes)))
    return tuple(steps)


def _first_taught_lesson(match: GPCMatch) -> int | None:
    """Earliest UFLI lesson at which ``_is_taught_match`` credits ``match``.

    ``_is_taught_match`` is monotonic in the taught sets, so the first lesson
    whose cumulative set accepts the match is the earliest it can be taught.
    Returns ``None`` when no lesson's set accepts it.
    """
    for lesson, taught_pairs, taught_graphemes in _curriculum_steps():
        if _is_taught_match(match, taught_pairs, taught_graphemes):
            return lesson
    return None


@cache
def _curriculum_profiles() -> tuple[tuple[int, EnglishStudentKnowledge], ...]:
    """Cumulative student profiles after each UFLI lesson, in lesson order.

    Each step is ``(lesson, profile)`` for a student taught every GPC, affix
    unit, and sight word introduced up to and including that lesson — the
    inputs ``score_sight_words_english`` reads. Lessons are the union of the
    GPC and sight-word schedules: some lessons introduce sight words only
    (e.g. ``and`` at lesson 5) and still count.
    """
    gpcs_by_lesson: dict[int, list[tuple[str, str]]] = {}
    affixes_by_lesson: dict[int, list[str]] = {}
    for gpc in GPC_INVENTORY:
        gpcs_by_lesson.setdefault(gpc.decode_lesson or gpc.lesson, []).append(
            (normalize_gpc_grapheme(gpc.grapheme), "".join(gpc.phonemes))
        )
        if gpc.is_prefix or gpc.is_suffix:
            affixes_by_lesson.setdefault(gpc.lesson, []).append(gpc.grapheme)
    sight_words_by_lesson: dict[int, list[str]] = {}
    for sight_word, lesson in UFLI_SIGHT_WORD_LESSONS.items():
        sight_words_by_lesson.setdefault(lesson, []).append(sight_word)

    gpcs: list[tuple[str, str]] = []
    affixes: set[str] = set()
    sight_words: set[str] = set()
    profiles = []
    for lesson in sorted(gpcs_by_lesson.keys() | sight_words_by_lesson.keys()):
        gpcs += gpcs_by_lesson.get(lesson, [])
        affixes.update(affixes_by_lesson.get(lesson, ()))
        sight_words.update(sight_words_by_lesson.get(lesson, ()))
        profiles.append(
            (
                lesson,
                EnglishStudentKnowledge(
                    gpcs=list(gpcs),
                    sight_words=set(sight_words),
                    affixes=set(affixes),
                ),
            )
        )
    return tuple(profiles)


@cache
def _first_sight_word_lesson(word: str) -> int | None:
    """Earliest UFLI lesson at which the sight-word route credits ``word``.

    Runs ``score_sight_words_english`` against each cumulative curriculum
    profile, so clitic, affixed-stem, and compound forms count at the lesson
    their last-taught constituent arrives — ``someone`` decodes when ``one``
    is a known sight word and ``some`` decodes via GPCs — not just direct
    sight-word membership.
    """
    for lesson, profile in _curriculum_profiles():
        if score_sight_words_english(word, profile) == 1.0:
            return lesson
    return None


def analyse_word_english(
    word: str, student_knowledge: EnglishStudentKnowledge
) -> EnglishWordAnalysis:
    """Analyze an English word based on a student knowledge profile.

    Sight-word status is computed by the same rule the scorer uses (exact,
    clitic, affixed-stem, and compound forms all count). Every pronunciation
    variant is then parsed once and each parse's matches are checked against
    the student's taught set — the same check the GPC scorer applies, but
    kept per-match so untaught GPCs can be located in the word — and against
    the UFLI teaching order so the earliest decodable lesson is known.
    """
    lowered = word.lower()
    known_sight_word = score_sight_words_english(word, student_knowledge) == 1.0

    pronunciations: list[PronunciationVariant] = []
    parses: list[ParseAssessment] = []
    try:
        variants = get_word_phonemes(lowered)
    except KeyError:
        in_dictionary = False
    else:
        in_dictionary = True
        taught_set, taught_graphemes = normalize_taught_gpcs(student_knowledge.gpcs)
        for variant_index, variant in enumerate(variants):
            pronunciations.append(
                PronunciationVariant(
                    phonemes=[phoneme.ipa for phoneme in variant.phonemes],
                    is_monosyllabic=variant.is_monosyllabic,
                )
            )
            for parse in parse_gpcs(word, variant, GPC_INVENTORY):
                matches = [
                    GPCUse(
                        gpc_id=match.gpc.id,
                        grapheme=match.gpc.grapheme,
                        symbol=match.gpc.symbol,
                        phonemes=match.gpc.phonemes,
                        letter_positions=match.letter_positions,
                        phoneme_span=match.phoneme_span,
                        taught=_is_taught_match(match, taught_set, taught_graphemes),
                        first_taught_lesson=_first_taught_lesson(match),
                    )
                    for match in parse
                ]
                match_lessons = [m.first_taught_lesson for m in matches]
                decodable_at = (
                    max(match_lessons)
                    if matches and None not in match_lessons
                    else None
                )
                parses.append(
                    ParseAssessment(
                        variant_index=variant_index,
                        matches=matches,
                        decodable_at_lesson=decodable_at,
                    )
                )

    lesson_candidates = [
        parse.decodable_at_lesson
        for parse in parses
        if parse.decodable_at_lesson is not None
    ]
    sight_word_lesson = _first_sight_word_lesson(lowered)
    if sight_word_lesson is not None:
        lesson_candidates.append(sight_word_lesson)

    return EnglishWordAnalysis(
        word=word,
        known_sight_word=known_sight_word,
        in_dictionary=in_dictionary,
        pronunciations=pronunciations,
        candidate_parses=parses,
        earliest_decodable_lesson=min(lesson_candidates, default=None),
    )


def assess_decodability_ufli(word_analysis: EnglishWordAnalysis) -> bool:
    """Decide decodability of a word based on the analysis results and the
    UFLI curriculum.

    The rule: a known sight word is decodable regardless of anything else;
    otherwise the word is decodable iff some pronunciation variant has a
    parse that uses only taught GPCs.
    """
    if word_analysis.known_sight_word:
        return True

    return bool(word_analysis.decodable_parses)


def is_decodable_ufli(word: str, student_knowledge: EnglishStudentKnowledge) -> bool:
    """Decide decodability of a word based on the UFLI curriculum.

    The rule: a known sight word is decodable regardless of anything else;
    otherwise the word is decodable iff some pronunciation variant has a
    parse that uses only taught GPCs.
    """
    findings = analyse_word_english(word, student_knowledge)
    return assess_decodability_ufli(findings)
