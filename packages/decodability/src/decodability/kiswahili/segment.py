"""Segmentation and span definitions for Kiswahili words."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Annotated

from primerpro import Word
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from decodability.kiswahili.definitions import VALID_GRAPHEMES, primerpro_settings


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
    def length(self) -> int:
        """Return the length of the span."""
        return self.end - self.start

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

    @property
    def is_valid(self) -> bool:
        """Check if the grapheme is valid against the known inventory."""
        return self.normalised in VALID_GRAPHEMES


class ClusterSpan(Span):
    """A span that is known to be a consonant cluster."""

    nasal: bool

    @field_validator("text")
    def check_cluster_length(cls, value: str) -> str:  # noqa: N805
        graphemes = get_grapheme_spans(value)
        if len(graphemes) < 2:
            raise ValueError("A cluster must consist of at least two characters.")
        return value

    @field_validator("text")
    def check_all_consonants(cls, value: str) -> str:  # noqa: N805
        assert all(grapheme.is_consonant for grapheme in get_grapheme_spans(value))
        return value

    @property
    def grapheme_spans(self) -> list[GraphemeSpan]:
        """Return the individual grapheme spans that make up this cluster."""
        return get_grapheme_spans(self.text)

    def to_span(self) -> Span:
        """Return a plain :class:`Span` with the same text and range."""
        return Span(text=self.text, start=self.start, end=self.end)

    @property
    def symbol(self) -> str:
        """The lowercased cluster symbol: an alias of :attr:`normalised`."""
        return self.normalised


class WordDecomposition(BaseModel):
    """A decomposition of a word into grapheme spans and cluster spans."""

    word: str
    graphemes: list[GraphemeSpan]
    clusters: list[ClusterSpan]
    decomposition: list[GraphemeSpan | ClusterSpan]
    syllable_count: int


def count_syllables(graphemes: list[GraphemeSpan]) -> int:
    """Syllable count by vowel nuclei."""
    return sum(1 for grapheme in graphemes if not grapheme.is_consonant)


def get_grapheme_spans(word: str) -> list[GraphemeSpan]:
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


def _initial_n_before_monosyllabic_root_is_syllabic(
    boundary: NasalBoundary,
) -> NasalReading | None:
    """Word-initial ``n`` + d/j/z on a one-vowel root reads as syllabic n.

    When the n- class prefix attaches to a monosyllabic root (nzi, nje) the
    nasal carries its own syllable rather than fusing into the prenasalised
    onset; on longer roots (ndizi, njia) and word-medially it stays fused.
    """
    if (
        boundary.word_initial
        and boundary.nasal.symbol == "n"
        and boundary.vowels_after == 1
    ):
        return NasalReading.SYLLABIC
    return None


def _provisional_mb_mv_ambiguity(boundary: NasalBoundary) -> NasalReading | None:
    """PROVISIONAL pending linguist notes: ``m`` + b/v may be either reading.

    "m" before b or v can be the syllabic m- marker (e.g. a verb marker or
    class prefix) or the fused prenasalised onset, at any position in the
    word, so both readings are kept open.
    """
    if boundary.nasal.symbol == "m" and boundary.follower.symbol in {"b", "v"}:
        return NasalReading.AMBIGUOUS
    return None


def _taught_onset_clusters(boundary: NasalBoundary) -> NasalReading | None:
    """Default rule: the taught nasal onsets fuse; any other nasal is syllabic.

    Before ``b, v, w, y`` (after m) or ``d, j, z, w`` (after n) a nasal is a
    prenasalised/glide onset taught as a cluster. Before any other consonant
    the nasal is syllabic and reads as the bare grapheme.
    """
    followers = _NASAL_CLUSTER_FOLLOWERS.get(boundary.nasal.symbol)
    if followers is None:
        return None
    if boundary.follower.symbol in followers:
        return NasalReading.FUSED
    return NasalReading.SYLLABIC


NASAL_RULES: tuple[NasalRule, ...] = (
    _initial_n_before_monosyllabic_root_is_syllabic,
    _provisional_mb_mv_ambiguity,
    _taught_onset_clusters,
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
) -> ClusterSpan | None:
    """The cluster covering ``run[pos:]``, if two or more consonants remain.

    A consonant run is one cluster — a fused onset cannot decompose into its
    parts, so "ndw", "ndr" and "mbw" are emitted whole rather than "nd" plus
    a loose consonant. A lone consonant is not a cluster.

    ``nasal`` is set when the cluster itself starts with a nasal (m or n). When
    the rules read a leading nasal as syllabic it is dropped from the cluster,
    so the remaining cluster is not flagged as nasal.
    """
    if len(run) - pos < 2:
        return None
    return ClusterSpan(
        text="".join(graphemes[i].text for i in run[pos:]),
        start=graphemes[run[pos]].start,
        end=graphemes[run[-1]].end,
        nasal=graphemes[run[pos]].symbol in _NASAL_CLUSTER_FOLLOWERS,
    )


def _clusterings_of_run(
    graphemes: Sequence[GraphemeSpan],
    run: Sequence[int],
    verdict: NasalReading | None,
) -> list[list[ClusterSpan]]:
    """Enumerate every viable clustering of one consonant run.

    A run has at most one nasal boundary — its opening position — so each
    verdict maps to a fixed shape: the whole run is one cluster, and a
    syllabic leading nasal simply drops out of it. An ambiguous boundary
    yields both readings, fused first.
    """

    def clustering(start: int) -> list[ClusterSpan]:
        span = _run_span(graphemes, run, start)
        return [] if span is None else [span]

    if verdict is NasalReading.AMBIGUOUS:
        return [clustering(0), clustering(1)]
    if verdict is NasalReading.SYLLABIC:
        return [clustering(1)]
    return [clustering(0)]


def find_clustering_variants(
    graphemes: Sequence[GraphemeSpan],
) -> list[list[ClusterSpan]]:
    """Enumerate every viable consonant clustering of an already-decomposed word.

    Each variant is the list of cluster spans under one resolution of the
    ambiguous nasal boundaries. Variants are ordered fused-first: index 0 is
    the strict resolution in which every ambiguous boundary is a cluster, and
    the last variant is the fully syllabic resolution. A word with no
    ambiguous boundaries yields a single variant.
    """
    variants: list[list[ClusterSpan]] = [[]]
    for run in _consonant_runs(graphemes):
        run_variants = _clusterings_of_run(
            graphemes, run, _nasal_verdict(graphemes, run)
        )
        variants = [prefix + suffix for prefix in variants for suffix in run_variants]
    return variants


def find_ambiguous_boundaries(
    graphemes: Sequence[GraphemeSpan],
) -> list[ClusterSpan]:
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


def find_cluster_spans(graphemes: Sequence[GraphemeSpan]) -> list[ClusterSpan]:
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


def decompose_word(word: str) -> WordDecomposition:
    """Decompose a word into its graphemes, consonant clusters, and full
    decomposition."""
    graphemes = get_grapheme_spans(word)
    clusters = find_cluster_spans(graphemes)
    cluster_by_start = {cluster.start: cluster for cluster in clusters}

    decomposition: list[GraphemeSpan | ClusterSpan] = []
    i = 0
    while i < len(graphemes):
        cluster = cluster_by_start.get(graphemes[i].start)
        if cluster is None:
            decomposition.append(graphemes[i])
            i += 1
            continue
        decomposition.append(cluster)
        # Skip past every grapheme the cluster absorbed.
        while i < len(graphemes) and graphemes[i].end <= cluster.end:
            i += 1

    return WordDecomposition(
        word=word,
        graphemes=graphemes,
        clusters=clusters,
        decomposition=decomposition,
        syllable_count=count_syllables(graphemes),
    )


def get_grapheme_symbols(word: str) -> list[str]:
    """Return the lowercased graphemes of a word, in order."""
    return [grapheme.symbol for grapheme in get_grapheme_spans(word)]


def get_clusters(word: str) -> list[str]:
    """Return the lowercased consonant clusters of a word, in order."""
    return [span.normalised for span in find_cluster_spans(get_grapheme_spans(word))]


__all__ = [
    "Span",
    "GraphemeSpan",
    "ClusterSpan",
    "WordDecomposition",
    "count_syllables",
    "decompose_word",
    "get_grapheme_spans",
    "NasalReading",
    "NasalBoundary",
    "NasalRule",
    "NASAL_RULES",
    "find_clustering_variants",
    "find_ambiguous_boundaries",
    "find_cluster_spans",
    "get_grapheme_symbols",
    "get_clusters",
]
