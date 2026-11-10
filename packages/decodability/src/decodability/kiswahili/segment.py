"""Segmentation and span definitions for Kiswahili words."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Annotated

from primerpro import Word
from pydantic import BaseModel, ConfigDict, Field, model_validator

from decodability.kiswahili.definitions import primerpro_settings


class Span(BaseModel):
    """A half-open ``[start, end)`` character range within a Kiswahili word.

    ``text`` is the matched substring exactly as it appears in the word, so callers
    can render a finding without re-slicing. Use :attr:`normalised` to compare a span
    against the grapheme inventory or a student's knowledge, both of which are
    lowercase.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    start: Annotated[int, Field(ge=0)]
    end: Annotated[int, Field(ge=1)]

    @model_validator(mode="after")
    def check_span_matches_text(self) -> Span:
        """Ensure the range is consistent with the matched text."""
        if self.end - self.start != len(self.text):
            raise ValueError(
                f"Span [{self.start}, {self.end}) does not match text {self.text!r} "
                f"of length {len(self.text)}."
            )
        return self

    @property
    def normalised(self) -> str:
        """Lowercased text, for matching against inventories and student knowledge."""
        return self.text.lower()


class GraphemeSpan(Span):
    """A span that is known to be a single grapheme, with consonant flag."""

    is_consonant: bool

    def to_span(self) -> Span:
        """Return a plain :class:`Span` with the same text and range."""
        return Span(text=self.text, start=self.start, end=self.end)

    @property
    def symbol(self) -> str:
        """The lowercased grapheme symbol: an alias of :attr:`normalised`."""
        return self.normalised


def decompose_word(word: str) -> list[GraphemeSpan]:
    """Decompose a word into its located graphemes using PrimerPro settings.

    NOTE: PrimerPro's grapheme decomposition doesn't flag out-of-inventory graphemes,
    so they come back as-is as single characters. It is the caller's responsibility to
    check validity against ``VALID_GRAPHEMES``.
    """
    primerpro_word = Word(text=word, settings=primerpro_settings)

    graphemes: list[GraphemeSpan] = []
    word_index = 0
    for grapheme in primerpro_word.graphemes:
        end_index = word_index + len(grapheme.symbol)
        graphemes.append(
            GraphemeSpan(
                text=word[word_index:end_index],
                start=word_index,
                end=end_index,
                is_consonant=grapheme.is_consonant,
            )
        )
        word_index = end_index

    return graphemes


# Words reviewed by Kiswahili experts as having syllabic nasalsm=,
# mapped to the clusters the rules should leave behind in the strict
# view (ambiguous boundaries count as fused, cf. ``segment.py``).
SYLLABIC_NASAL_EXPECTED_CLUSTERS: dict[str, list[str]] = {
    "akamsalimia": [],
    "akamtuma": [],
    "alimnunulia": [],
    "alimpeleka": [],
    "alimpenda": ["nd"],  # medial nd is fused
    "hakumpenda": ["nd"],  # medial nd is fused
    "humpeleka": [],
    "inamlinda": ["nd"],  # medial nd is fused
    "kumchapa": [],
    "kumfukuza": [],
    "kumkaribisha": [],
    "mbunge": [],
    "mbwa": ["bw"],  # syllabic m on a monosyllabic root leaves bw
    "mchana": [],
    "mchanganyiko": [],
    "mchele": [],
    "mchezo": [],
    "mchi": [],
    "mchwa": ["chw"],  # syllabic m leaves chw
    "mdogo": [],
    "mdomo": [],
    "mdudu": [],
    "mfalme": ["lm"],  # lm is an ordinary cluster
    "mfano": [],
    "mfupi": [],
    "mgeni": [],
    "mgonjwa": ["njw"],  # medial n+j is fused and njw is treated as a single cluster
    "mheshimiwa": [],
    "mjini": [],
    "mjomba": ["mb"],  # medial mb is ambiguous; strict view keeps it
    "mkali": [],
    "mkate": [],
    "mkono": [],
    "mkononi": [],
    "mkubwa": ["bw"],  # bw is an ordinary cluster
    "mkuu": [],
    "mlango": [],
    "mlingoti": [],
    "mmoja": [],
    "mpeketoni": [],
    "mpunga": [],
    "mraba": [],
    "mrefu": [],
    "msafiri": [],
    "msalani": [],
    "msichana": [],
    "msingi": [],
    "msituni": [],
    "mstatili": ["st"],  # syllabic m leaves st
    "mti": [],
    "mtihani": [],
    "mtini": [],
    "mtoto": [],
    "mtu": [],
    "mvua": [],
    "mzazi": [],
    "mzee": [],
    "nchi": [],
    "nne": [],
    "wamsalimie": [],
    "wanamjua": [],
    "zilizompatia": [],
    "nzi": [],
    "kuamka": [],
    "kiamshakinywa": ["nyw"],  # nyw is an ordinary cluster
    "kuzungumza": [],
    "namna": [],
    "umri": [],
    "huteremsha": [],
    "nje": [],
}

SYLLABIC_NASAL_REVIEWED_WORDS: frozenset[str] = frozenset(
    SYLLABIC_NASAL_EXPECTED_CLUSTERS.keys()
)


class NasalReading(Enum):
    """How a nasal + consonant boundary is read.

    FUSED means the two graphemes form a cluster onset that must be taught as a
    unit (mb, nd, ...). SYLLABIC means the nasal is a syllabic consonant that
    reads as the bare grapheme, so no cluster exists. AMBIGUOUS means the rules
    cannot decide and both readings are viable; scorers choose how to treat it.
    """

    FUSED = "fused"
    SYLLABIC = "syllabic"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class NasalBoundary:
    """Context handed to nasal rules when deciding a nasal + consonant reading.

    ``graphemes`` is the whole word's decomposition and ``index`` locates the
    nasal within it, so rules can use any surrounding context they need (word
    position, neighbouring vowels, later consonants).
    """

    graphemes: tuple[GraphemeSpan, ...]
    index: int

    @property
    def nasal(self) -> GraphemeSpan:
        """The nasal grapheme at the boundary."""
        return self.graphemes[self.index]

    @property
    def follower(self) -> GraphemeSpan:
        """The consonant immediately after the nasal."""
        return self.graphemes[self.index + 1]

    @property
    def word_initial(self) -> bool:
        """Whether the nasal is the first grapheme of the word."""
        return self.nasal.start == 0

    @property
    def vowels_after(self) -> int:
        """Number of vowel graphemes after the nasal.

        For a word-initial nasal this is the syllable count of the root it
        attaches to — a one-vowel root makes the nasal carry its own syllable.
        """
        return sum(
            not grapheme.is_consonant for grapheme in self.graphemes[self.index + 1 :]
        )


# A rule inspects one nasal + consonant boundary and returns a reading, or None
# to abstain and defer to the next rule. Rules run in order and the first
# non-None verdict wins, so more specific rules should come first. Decision
# trees from linguist notes slot in here as small functions — no other code
# needs to change when rules are added, reordered, or refined.
NasalRule = Callable[[NasalBoundary], NasalReading | None]


def _exception_words(boundary: NasalBoundary) -> NasalReading | None:
    """Handle exception words that do not follow the usual nasal rules."""
    word = "".join(grapheme.symbol for grapheme in boundary.graphemes)

    if word == "mvua":
        return NasalReading.SYLLABIC

    if word == "jumanne":
        return NasalReading.SYLLABIC

    return None


def _unambiguous_due_to_expert_review(boundary: NasalBoundary) -> NasalReading | None:
    word = "".join(grapheme.symbol for grapheme in boundary.graphemes)
    disambiguated = [
        # reviewed from benchmark item texts
        ("mbunge", ("mb", 0)),
        # other examples reviewed
        ("mbaya", ("mb", 0)),
        ("mvulana", ("mv", 0)),
        ("mbovu", ("mb", 0)),
        ("akambeba", ("mb", 3)),
        ("mbebe", ("mb", 0)),
        ("hambebi", ("mb", 2)),
        ("mwako", ("mw", 0)),
        ("mwaridi", ("mw", 0)),
        ("myombo", ("my", 0)),
        ("myugwa", ("my", 0)),
    ]
    for w, (_, pos) in disambiguated:
        if word == w and boundary.index == pos:
            return NasalReading.SYLLABIC
    return None


def _nw_always_fused(boundary: NasalBoundary) -> NasalReading | None:
    """The sequence ``nw`` is always fused."""
    if boundary.nasal.symbol == "n" and boundary.follower.symbol == "w":
        return NasalReading.FUSED
    return None


def _medial_m_glide_always_fused(boundary: NasalBoundary) -> NasalReading | None:
    """Medial ``my`` and ``mw`` are always fused."""
    if (
        not boundary.word_initial
        and boundary.nasal.symbol == "m"
        and boundary.follower.symbol in {"y", "w"}
    ):
        return NasalReading.FUSED
    return None


def _initial_nasal_before_monosyllabic_root_is_syllabic(
    boundary: NasalBoundary,
) -> NasalReading | None:
    """Word-initial nasal consonant running on a one-vowel root reads as syllabic.

    This is true even for consonants that are typically fused after the nasals, like
    mb, mv, nd, nj, nz (except nw). On longer roots (ndizi, njia) and word-medially
    those stay fused.
    """
    if boundary.word_initial and boundary.nasal and boundary.vowels_after == 1:
        return NasalReading.SYLLABIC
    return None


def _taught_n_onset_clusters(boundary: NasalBoundary) -> NasalReading | None:
    """Default rule: the taught n- onsets are gated, regardless of the following
    consonant.

    n before d, j, z, w is usually fused hence gated.
    n before any other consonant is either a syllable coda or part of a loan
    cluster or an unusual sound early grade students aren't expected to blend
    well, hence gated.
    """
    if boundary.nasal.symbol == "n":
        return NasalReading.FUSED
    return None


def _m_ambiguity(boundary: NasalBoundary) -> NasalReading | None:
    """Nasal verdict on ``m`` before b/v/w/y depends on word analysis which
    the current tool cannot perform.

    More specifically, for words starting with m-glides, we need to know their
    plural form to determine if the ``m`` is syllabic or fused. Word-medial
    m-glides are always fused.

    For ``m`` before b/v, it depends on whether
    it is an object or second-person plural marker for a verb, the intended
    meaning of the word, or its noun class or agreement features.
    """
    if not boundary.nasal.symbol == "m":
        return None

    if boundary.follower.symbol in {"b", "v"}:
        return NasalReading.AMBIGUOUS

    if boundary.word_initial and boundary.follower.symbol in {"w", "y"}:
        return NasalReading.AMBIGUOUS

    return None


def _taught_m_onset_clusters(boundary: NasalBoundary) -> NasalReading | None:
    """Default rule: the taught m- onsets fuse; any other m-cluster is syllabic.

    Before ``b, v, w, y`` an m is a prenasalised/glide onset taught as a cluster.
    Before any other consonant the m is syllabic and reads as the bare grapheme.
    """
    followers = _NASAL_CLUSTER_FOLLOWERS.get(boundary.nasal.symbol)
    if followers is None:
        return None
    if boundary.nasal.symbol == "m" and boundary.follower.symbol in followers:
        return NasalReading.FUSED
    return NasalReading.SYLLABIC


NASAL_RULES: tuple[NasalRule, ...] = (
    _exception_words,
    _unambiguous_due_to_expert_review,
    # TODO: check with justin whether to include it here or not
    _nw_always_fused,
    _medial_m_glide_always_fused,
    _initial_nasal_before_monosyllabic_root_is_syllabic,
    _taught_n_onset_clusters,
    _m_ambiguity,
    _taught_m_onset_clusters,
)


# Before these consonants a nasal *could be* a prenasalised/glide onset that is
# taught as a cluster (mb, mv, mw, my; nd, nj, nz, nw). The keys are the
# graphemes that can head a nasal + consonant boundary at all.
_NASAL_CLUSTER_FOLLOWERS = {
    "m": frozenset({"b", "v", "w", "y"}),
    "n": frozenset({"d", "j", "z", "w"}),
}


def _nasal_reading(boundary: NasalBoundary, rules: Sequence[NasalRule]) -> NasalReading:
    """Run the rule list over one boundary; first non-abstaining rule wins.

    If every rule abstains the boundary is treated as FUSED, so a gap in the
    rules makes a word harder to decode rather than silently easier.
    """
    for rule in rules:
        verdict = rule(boundary)
        if verdict is not None:
            return verdict
    return NasalReading.FUSED


def _consonant_runs(graphemes: Sequence[GraphemeSpan]) -> Iterator[list[int]]:
    """Yield the grapheme indices of each maximal consonant run."""
    run: list[int] = []
    for index, grapheme in enumerate(graphemes):
        if grapheme.is_consonant:
            run.append(index)
        elif run:
            yield run
            run = []
    if run:
        yield run


def _nasal_verdict(
    graphemes: Sequence[GraphemeSpan], run: Sequence[int]
) -> NasalReading | None:
    """The reading of the run's opening nasal boundary, if it has one.

    Only the first consonant of a run can head a nasal onset: a nasal later
    in the run sits inside a loan cluster (e.g. the "nd" of "blnd"), which is
    gated as a unit rather than split off as a taught onset.
    """
    if len(run) < 2 or graphemes[run[0]].symbol not in _NASAL_CLUSTER_FOLLOWERS:
        return None
    return _nasal_reading(
        NasalBoundary(graphemes=tuple(graphemes), index=run[0]), NASAL_RULES
    )


def _run_span(
    graphemes: Sequence[GraphemeSpan], run: Sequence[int], pos: int
) -> Span | None:
    """The cluster covering ``run[pos:]``, if two or more consonants remain.

    A consonant run is one cluster — a fused onset cannot decompose into its
    parts, so "ndw", "ndr" and "mbw" are emitted whole rather than "nd" plus
    a loose consonant. A lone consonant is not a cluster.
    """
    if len(run) - pos < 2:
        return None
    return Span(
        text="".join(graphemes[i].text for i in run[pos:]),
        start=graphemes[run[pos]].start,
        end=graphemes[run[-1]].end,
    )


def _clusterings_of_run(
    graphemes: Sequence[GraphemeSpan],
    run: Sequence[int],
    verdict: NasalReading | None,
) -> list[list[Span]]:
    """Enumerate every viable clustering of one consonant run.

    A run has at most one nasal boundary — its opening position — so each
    verdict maps to a fixed shape: the whole run is one cluster, and a
    syllabic leading nasal simply drops out of it. An ambiguous boundary
    yields both readings, fused first.
    """

    def clustering(start: int) -> list[Span]:
        span = _run_span(graphemes, run, start)
        return [] if span is None else [span]

    if verdict is NasalReading.AMBIGUOUS:
        return [clustering(0), clustering(1)]
    if verdict is NasalReading.SYLLABIC:
        return [clustering(1)]
    return [clustering(0)]


def find_clustering_variants(
    graphemes: Sequence[GraphemeSpan],
) -> list[list[Span]]:
    """Enumerate every viable consonant clustering of an already-decomposed word.

    Each variant is the list of cluster spans under one resolution of the
    ambiguous nasal boundaries. Variants are ordered fused-first: index 0 is
    the strict resolution in which every ambiguous boundary is a cluster, and
    the last variant is the fully syllabic resolution. A word with no
    ambiguous boundaries yields a single variant.
    """
    variants: list[list[Span]] = [[]]
    for run in _consonant_runs(graphemes):
        run_variants = _clusterings_of_run(
            graphemes, run, _nasal_verdict(graphemes, run)
        )
        variants = [prefix + suffix for prefix in variants for suffix in run_variants]
    return variants


def find_ambiguous_boundaries(graphemes: Sequence[GraphemeSpan]) -> list[Span]:
    """Locate nasal + consonant boundaries the rules could not resolve.

    Each returned span covers the whole run the boundary heads — the cluster
    it becomes under a fused reading.
    """
    return [
        span
        for run in _consonant_runs(graphemes)
        if _nasal_verdict(graphemes, run) is NasalReading.AMBIGUOUS
        if (span := _run_span(graphemes, run, 0)) is not None
    ]


def find_cluster_spans(graphemes: Sequence[GraphemeSpan]) -> list[Span]:
    """Locate the consonant clusters in an already-decomposed word.

    A cluster is a run of two or more consecutive consonant graphemes, except
    that a syllabic nasal (m or n + consonant other than the taught onsets mb,
    mv, mw, my and nd, nj, nz, nw) is not part of a cluster. A run is emitted
    as one span — a fused onset cannot decompose into its parts, so "ndw" is
    one cluster rather than "nd" followed by a loose "w".

    Ambiguous nasal boundaries are resolved as fused, so the result is the
    maximal set of clusters — every cluster that any viable reading of the
    word could require. Use :func:`find_clustering_variants` for the full set
    of readings and :func:`find_ambiguous_boundaries` for the boundaries the
    rules left unresolved.
    """
    return find_clustering_variants(graphemes)[0]


def get_grapheme_symbols(word: str) -> list[str]:
    """Return the lowercased graphemes of a word, in order."""
    return [grapheme.symbol for grapheme in decompose_word(word)]


def get_clusters(word: str) -> list[str]:
    """Return the lowercased consonant clusters of a word, in order.

    Ambiguous nasal boundaries are resolved as fused, so this lists every
    cluster the word could require under any viable reading.
    """
    return [span.normalised for span in find_cluster_spans(decompose_word(word))]


__all__ = [
    "SYLLABIC_NASAL_EXPECTED_CLUSTERS",
    "SYLLABIC_NASAL_REVIEWED_WORDS",
    "Span",
    "GraphemeSpan",
    "NasalReading",
    "NasalBoundary",
    "NasalRule",
    "NASAL_RULES",
    "decompose_word",
    "find_clustering_variants",
    "find_ambiguous_boundaries",
    "find_cluster_spans",
    "get_grapheme_symbols",
    "get_clusters",
]
