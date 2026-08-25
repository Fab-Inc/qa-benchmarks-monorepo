"""Constrained aligner for English spelling and phoneme sequences.

Given a word, its phoneme sequence, and a GPC inventory, ``parse_gpcs``
returns every valid parse — a tuple of ``GPCMatch`` objects covering
every letter and every phoneme. The module is deliberately narrow: it
does not pronounce, translate, or score; it only answers the alignment
question. See ``parse_gpcs`` for the match kinds it recognizes and the
taught-cluster preference applied to the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

from decodability.english.definitions import VOWEL_LETTERS
from decodability.english.extract_phonemes import Phoneme


@dataclass(frozen=True)
class GPC:
    """One licensed correspondence, including multi-phoneme and split units."""

    id: str
    grapheme: str
    phonemes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.grapheme or not self.phonemes:
            raise ValueError("a GPC needs a grapheme and at least one phoneme")
        if self.grapheme.count("-") > 1:
            raise ValueError("the prototype supports two-part split graphemes only")


@dataclass(frozen=True)
class GPCMatch:
    """A GPC plus the spelling positions and phoneme slice it covers."""

    gpc: GPC
    letter_positions: tuple[int, ...]
    phoneme_span: tuple[int, int]


type Parse = tuple[GPCMatch, ...]


def _positions_are_free(start: int, end: int, reserved: int) -> bool:
    """Check whether every letter position in ``range(start, end)`` is unclaimed.

    ``reserved`` is a bitmask where bit ``i`` is set if letter position ``i``
    has already been claimed by a split-grapheme tail earlier in the walk.
    Returns True iff none of the positions in the half-open range have their
    bit set.
    """
    return all(not reserved & (1 << position) for position in range(start, end))


def parse_gpcs(
    word: str,
    phonemes: tuple[Phoneme, ...],
    inventory: tuple[GPC, ...],
    *,
    max_split_gap: int = 2,
) -> tuple[Parse, ...]:
    """Return every valid parse of ``word`` under the given inventory.

    A parse is a tuple of ``GPCMatch`` objects that together cover every
    letter of the spelling and every phoneme in the sequence. Multiple
    parses arise when the inventory offers overlapping ways to cover the
    same span — ``fish`` decomposes as ``f + i + sh`` (via ``sh → /ʃ/``)
    and as ``f + ish`` (via ``ish → (ɪ, ʃ)``), both yielding /f, ɪ, ʃ/.

    Beyond ordinary contiguous matches, the parser handles:

    - Split graphemes (``a-e``, ``i-e``, …): the two parts cover
      non-adjacent letter positions, with the gap bounded by
      ``max_split_gap``.
    - Free pass on unstressed vowels: a bare vowel letter paired with
      an unstressed phoneme parses via a synthesized ``REDUCED_*`` GPC —
      vowel reduction is a universal spoken-English process, not a taught
      correspondence.
    - Silent apostrophe: ``'`` positions are skipped without consuming
      a phoneme (``he's`` decodes as ``h + e + s``) — the curriculum
      treats possessive/contracted ``'s`` as an assumed correspondence.

    A post-walk filter prefers parses that avoid the free pass whenever an
    inventory aligned parse exists, so taught clusters (``ful`` for
    ``careful``) win over letter-by-letter decomposition.
    """

    if max_split_gap < 1:
        raise ValueError("max_split_gap must be at least one letter")

    spelling = word.casefold()
    phoneme_strings = tuple(p.ipa for p in phonemes)
    contiguous = tuple(
        (gpc, gpc.grapheme.casefold()) for gpc in inventory if "-" not in gpc.grapheme
    )
    split = tuple(
        (gpc, tuple(part.casefold() for part in gpc.grapheme.split("-")))
        for gpc in inventory
        if "-" in gpc.grapheme
    )

    @cache
    def walk(letter_index: int, phoneme_index: int, reserved: int) -> tuple[Parse, ...]:
        """Enumerate every completion of a parse from ``(letter_index, phoneme_index)``.

        Each returned ``Parse`` covers the remaining letters and phonemes using
        inventory GPCs (contiguous, split, or free-pass). Returns ``((),)``
        when both sequences are exhausted (one completion — the empty parse)
        and ``()`` when only one is exhausted (no valid completion).

        ``reserved`` is a bitmask of letter positions already claimed by an
        earlier split-grapheme tail; see ``_positions_are_free``.
        """
        while letter_index < len(spelling) and (
            reserved & (1 << letter_index) or spelling[letter_index] == "'"
        ):
            letter_index += 1

        if letter_index == len(spelling):
            return ((),) if phoneme_index == len(phonemes) else ()

        parses: list[Parse] = []

        for gpc, letters in contiguous:
            letter_end = letter_index + len(letters)
            phoneme_end = phoneme_index + len(gpc.phonemes)
            if (
                spelling.startswith(letters, letter_index)
                and _positions_are_free(letter_index, letter_end, reserved)
                and phoneme_strings[phoneme_index:phoneme_end] == gpc.phonemes
            ):
                match = GPCMatch(
                    gpc,
                    tuple(range(letter_index, letter_end)),
                    (phoneme_index, phoneme_end),
                )
                parses.extend(
                    (match, *remainder)
                    for remainder in walk(letter_end, phoneme_end, reserved)
                )

        if phoneme_index < len(phonemes):
            reduced_ipa = phonemes[phoneme_index].ipa
            is_unstressed = phonemes[phoneme_index].is_unstressed
            letter = spelling[letter_index]
            if is_unstressed and letter in VOWEL_LETTERS:
                free_pass = GPCMatch(
                    GPC(
                        id=f"REDUCED_{letter.upper()}",
                        grapheme=letter,
                        phonemes=(reduced_ipa,),
                    ),
                    (letter_index,),
                    (phoneme_index, phoneme_index + 1),
                )
                parses.extend(
                    (free_pass, *remainder)
                    for remainder in walk(letter_index + 1, phoneme_index + 1, reserved)
                )

        for gpc, (head, tail) in split:
            head_end = letter_index + len(head)
            phoneme_end = phoneme_index + len(gpc.phonemes)
            if not (
                spelling.startswith(head, letter_index)
                and _positions_are_free(letter_index, head_end, reserved)
                and phoneme_strings[phoneme_index:phoneme_end] == gpc.phonemes
            ):
                continue

            for gap in range(1, max_split_gap + 1):
                tail_start = head_end + gap
                tail_end = tail_start + len(tail)
                if spelling.startswith(tail, tail_start) and _positions_are_free(
                    letter_index, tail_end, reserved
                ):
                    tail_positions = tuple(range(tail_start, tail_end))
                    tail_mask = sum(1 << position for position in tail_positions)
                    match = GPCMatch(
                        gpc,
                        (*range(letter_index, head_end), *tail_positions),
                        (phoneme_index, phoneme_end),
                    )
                    remainders = walk(head_end, phoneme_end, reserved | tail_mask)
                    parses.extend((match, *remainder) for remainder in remainders)

        return tuple(parses)

    all_parses = walk(0, 0, 0)
    strip_free_parses = tuple(
        parse
        for parse in all_parses
        if not any(match.gpc.id.startswith("REDUCED_") for match in parse)
    )
    return strip_free_parses if strip_free_parses else all_parses
