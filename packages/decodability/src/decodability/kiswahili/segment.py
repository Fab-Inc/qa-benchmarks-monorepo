"""Segmentation and span definitions for Kiswahili words."""

from __future__ import annotations

from collections.abc import Sequence
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
        if len(value) < 2:
            raise ValueError("A cluster must consist of at least two characters.")
        return value

    @field_validator("text")
    def check_all_consonants(cls, value: str) -> str:  # noqa: N805
        assert all(grapheme.is_consonant for grapheme in get_grapheme_spans(value))
        return value

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


def find_cluster_spans(graphemes: Sequence[GraphemeSpan]) -> list[ClusterSpan]:
    """Locate the consonant clusters in an already-decomposed word.

    A cluster is any run of two or more consecutive consonant graphemes.
    """
    clusters: list[ClusterSpan] = []
    consonant_run: list[GraphemeSpan] = []

    def close_run() -> None:
        """Record the consonants seen so far as a cluster, if there are enough."""
        if len(consonant_run) < 2:
            return
        clusters.append(
            ClusterSpan(
                text="".join(grapheme.text for grapheme in consonant_run),
                start=consonant_run[0].start,
                end=consonant_run[-1].end,
                nasal=consonant_run[0].normalised in ["m", "n"],
            )
        )

    for grapheme in graphemes:
        if grapheme.is_consonant:
            consonant_run.append(grapheme)
        else:
            close_run()
            consonant_run.clear()

    close_run()
    return clusters


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
    "find_cluster_spans",
    "get_grapheme_symbols",
    "get_clusters",
]
