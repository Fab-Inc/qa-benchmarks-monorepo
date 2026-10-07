"""Segmentation and span definitions for Kiswahili words."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
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
    def preceded_by_vowel(self) -> bool:
        """Whether the grapheme before the nasal is a vowel."""
        return self.index > 0 and not self.graphemes[self.index - 1].is_consonant

    @property
    def followed_by_vowel(self) -> bool:
        """Whether the grapheme after the follower is a vowel."""
        after = self.index + 2
        return after < len(self.graphemes) and not self.graphemes[after].is_consonant

    @property
    def vowels_after(self) -> int:
        """Number of vowel graphemes after the nasal.

        For a word-initial nasal this is the syllable count of the root it
        attaches to — a one-vowel root makes the nasal carry its own syllable.
        """
        return sum(
            not grapheme.is_consonant
            for grapheme in self.graphemes[self.index + 1 :]
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
        and boundary.follower.symbol in {"d", "j", "z"}
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


def _nasal_reading(
    boundary: NasalBoundary, rules: Sequence[NasalRule]
) -> NasalReading:
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


def _nasal_verdicts(
    graphemes: Sequence[GraphemeSpan],
    run: Sequence[int],
    rules: Sequence[NasalRule],
) -> dict[int, NasalReading]:
    """Map run-local positions that head a nasal boundary to their reading."""
    graphemes_tuple = tuple(graphemes)
    verdicts: dict[int, NasalReading] = {}
    for pos, grapheme_index in enumerate(run):
        if pos + 1 >= len(run):
            continue
        if graphemes[grapheme_index].symbol in _NASAL_CLUSTER_FOLLOWERS:
            verdicts[pos] = _nasal_reading(
                NasalBoundary(graphemes=graphemes_tuple, index=grapheme_index),
                rules,
            )
    return verdicts


def _pair_span(nasal: GraphemeSpan, follower: GraphemeSpan) -> Span:
    """The cluster span covering a nasal and its following consonant."""
    return Span(
        text=nasal.text + follower.text, start=nasal.start, end=follower.end
    )


def _clusterings_of_run(
    graphemes: Sequence[GraphemeSpan],
    run: Sequence[int],
    verdicts: Mapping[int, NasalReading],
) -> list[list[Span]]:
    """Enumerate every viable clustering of one consonant run.

    Variants are ordered fused-first: each ambiguous boundary branches into a
    fused reading (nasal + consonant emitted as a cluster span) and a syllabic
    reading (nasal skipped, read as the bare grapheme), with the fused branch
    explored first.
    """

    def rec(pos: int, acc: list[Span]) -> list[list[Span]]:
        while pos < len(run):
            verdict = verdicts.get(pos)
            if verdict is NasalReading.AMBIGUOUS:
                fused = rec(
                    pos + 2,
                    acc + [_pair_span(graphemes[run[pos]], graphemes[run[pos + 1]])],
                )
                syllabic = rec(pos + 1, acc)
                return fused + syllabic
            if verdict is NasalReading.FUSED:
                acc = acc + [_pair_span(graphemes[run[pos]], graphemes[run[pos + 1]])]
                pos += 2
                continue
            if verdict is NasalReading.SYLLABIC:
                pos += 1
                continue
            # Non-nasal stretch. A trailing nasal (not followed by a consonant)
            # belongs to the stretch: e.g. the "sm" of rasmi is a consonant + m
            # cluster. The stretch breaks before any nasal that heads a
            # boundary, whatever its reading.
            end = pos
            while end < len(run) and end not in verdicts:
                end += 1
            if end - pos >= 2:
                acc = acc + [
                    Span(
                        text="".join(graphemes[i].text for i in run[pos:end]),
                        start=graphemes[run[pos]].start,
                        end=graphemes[run[end - 1]].end,
                    )
                ]
            pos = end
        return [acc]

    return rec(0, [])


def find_clustering_variants(
    graphemes: Sequence[GraphemeSpan],
    *,
    rules: Sequence[NasalRule] = NASAL_RULES,
) -> list[list[Span]]:
    """Enumerate every viable clustering of an already-decomposed word.

    Each variant is the list of cluster spans under one resolution of the
    ambiguous nasal boundaries. Variants are ordered fused-first: index 0 is
    the strict resolution in which every ambiguous boundary is a cluster, and
    the last variant is the fully syllabic resolution. A word with no
    ambiguous boundaries yields a single variant.
    """
    variants: list[list[Span]] = [[]]
    for run in _consonant_runs(graphemes):
        verdicts = _nasal_verdicts(graphemes, run, rules)
        run_variants = _clusterings_of_run(graphemes, run, verdicts)
        variants = [
            prefix + suffix for prefix in variants for suffix in run_variants
        ]
    return variants


def find_ambiguous_boundaries(
    graphemes: Sequence[GraphemeSpan],
    *,
    rules: Sequence[NasalRule] = NASAL_RULES,
) -> list[Span]:
    """Locate nasal + consonant boundaries the rules could not resolve.

    Each returned span covers the nasal and its following consonant — the fused
    cluster the boundary becomes under a fused reading.
    """
    boundaries: list[Span] = []
    for run in _consonant_runs(graphemes):
        for pos, reading in _nasal_verdicts(graphemes, run, rules).items():
            if reading is NasalReading.AMBIGUOUS:
                boundaries.append(
                    _pair_span(graphemes[run[pos]], graphemes[run[pos + 1]])
                )
    return boundaries


def find_fused_cluster_spans(
    graphemes: Sequence[GraphemeSpan],
    *,
    rules: Sequence[NasalRule] = NASAL_RULES,
) -> list[Span]:
    """Locate the clusters the rules resolved as fused nasal onsets.

    These are the clusters that must be taught as units under any reading —
    they appear in every viable clustering. Every other cluster a variant can
    produce (an ambiguous boundary read as fused, or a non-nasal stretch such
    as "bl" or "st") is decomposable: it may be read as its component
    graphemes.
    """
    fused: list[Span] = []
    for run in _consonant_runs(graphemes):
        for pos, reading in sorted(_nasal_verdicts(graphemes, run, rules).items()):
            if reading is NasalReading.FUSED:
                fused.append(
                    _pair_span(graphemes[run[pos]], graphemes[run[pos + 1]])
                )
    return fused


def find_cluster_spans(
    graphemes: Sequence[GraphemeSpan],
    *,
    rules: Sequence[NasalRule] = NASAL_RULES,
) -> list[Span]:
    """Locate the consonant clusters in an already-decomposed word.

    A cluster is a run of two or more consecutive consonant graphemes, except
    that a syllabic nasal (m or n + consonant other than the taught onsets mb,
    mv, mw, my and nd, nj, nz, nw) is not part of a cluster.

    Ambiguous nasal boundaries are resolved as fused, so the result is the
    maximal set of clusters — every cluster that any viable reading of the
    word could require. Use :func:`find_clustering_variants` for the full set
    of readings and :func:`find_ambiguous_boundaries` for the boundaries the
    rules left unresolved.
    """
    return find_clustering_variants(graphemes, rules=rules)[0]


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
    "Span",
    "GraphemeSpan",
    "NasalReading",
    "NasalBoundary",
    "NasalRule",
    "NASAL_RULES",
    "decompose_word",
    "find_clustering_variants",
    "find_ambiguous_boundaries",
    "find_fused_cluster_spans",
    "find_cluster_spans",
    "get_grapheme_symbols",
    "get_clusters",
]
