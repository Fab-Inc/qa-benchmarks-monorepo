"""Small constrained parser for spelling-to-phoneme correspondences.
The parser only answers the alignment question: which inventory GPCs cover
every letter and every phoneme?
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache


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


def parse_gpcs(
    word: str,
    phonemes: tuple[str, ...],
    inventory: tuple[GPC, ...],
    *,
    max_split_gap: int = 2,
) -> tuple[Parse, ...]:
    """Return every complete inventory-constrained parse of ``word``.

    A hyphen marks a split grapheme: ``a-e`` can cover the ``a`` and ``e`` in
    ``make`` while the intervening ``k`` is matched normally.  Results are ordered
    by pronunciation, so that split ``a-e`` precedes ``k`` in that parse.

    """

    if max_split_gap < 1:
        raise ValueError("max_split_gap must be at least one letter")

    spelling = word.casefold()
    contiguous = tuple(
        (gpc, gpc.grapheme.casefold()) for gpc in inventory if "-" not in gpc.grapheme
    )
    split = tuple(
        (gpc, tuple(part.casefold() for part in gpc.grapheme.split("-")))
        for gpc in inventory
        if "-" in gpc.grapheme
    )

    def positions_are_free(start: int, end: int, reserved: int) -> bool:
        return all(not reserved & (1 << position) for position in range(start, end))

    @cache
    def walk(letter_index: int, phoneme_index: int, reserved: int) -> tuple[Parse, ...]:
        while letter_index < len(spelling) and reserved & (1 << letter_index):
            letter_index += 1

        if letter_index == len(spelling):
            return ((),) if phoneme_index == len(phonemes) else ()

        parses: list[Parse] = []

        for gpc, letters in contiguous:
            letter_end = letter_index + len(letters)
            phoneme_end = phoneme_index + len(gpc.phonemes)
            if (
                spelling.startswith(letters, letter_index)
                and positions_are_free(letter_index, letter_end, reserved)
                and phonemes[phoneme_index:phoneme_end] == gpc.phonemes
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

        for gpc, (head, tail) in split:
            head_end = letter_index + len(head)
            phoneme_end = phoneme_index + len(gpc.phonemes)
            if not (
                spelling.startswith(head, letter_index)
                and positions_are_free(letter_index, head_end, reserved)
                and phonemes[phoneme_index:phoneme_end] == gpc.phonemes
            ):
                continue

            for gap in range(1, max_split_gap + 1):
                tail_start = head_end + gap
                tail_end = tail_start + len(tail)
                if spelling.startswith(tail, tail_start) and positions_are_free(
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

    return walk(0, 0, 0)
