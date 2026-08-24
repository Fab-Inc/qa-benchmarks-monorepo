"""Small constrained parser for spelling-to-phoneme correspondences.
The parser only answers the alignment question: which inventory GPCs cover
every letter and every phoneme?
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Literal

from decodability.english.definitions import UFLI_GPC_PAIRS, VOWEL_LETTERS
from decodability.english.extract_phonemes import Phoneme

FreePassMode = Literal["none", "schwa", "stress0"]


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
    phonemes: tuple[Phoneme, ...],
    inventory: tuple[GPC, ...],
    *,
    max_split_gap: int = 2,
    free_pass_mode: FreePassMode = "stress0",
    curriculum_guarded: bool = True,
) -> tuple[Parse, ...]:
    """Return every complete inventory-constrained parse of ``word``.

    A hyphen marks a split grapheme: ``a-e`` can cover the ``a`` and ``e`` in
    ``make`` while the intervening ``k`` is matched normally.  Results are ordered
    by pronunciation, so that split ``a-e`` precedes ``k`` in that parse.

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

    def positions_are_free(start: int, end: int, reserved: int) -> bool:
        return all(not reserved & (1 << position) for position in range(start, end))

    @cache
    def walk(letter_index: int, phoneme_index: int, reserved: int) -> tuple[Parse, ...]:
        # Skip positions already covered by a split-grapheme tail, and treat
        # apostrophes as silent (the expert curriculum treats possessive/
        # contracted ``'s`` and its siblings as an assumed correspondence —
        # the ``'`` itself carries no phoneme, and the surrounding letters
        # decode via their existing GPCs).
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
                and positions_are_free(letter_index, letter_end, reserved)
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

        # Free pass: an unstressed (reduced) phoneme mapped to a single vowel
        # letter — vowel reduction is a universal spoken-English process, not a
        # taught GPC. The expert's principle: fire the free pass only where no
        # lesson already covers that sound in that position. Three guards:
        #   1. letter must be a bare vowel (VOWEL_LETTERS excludes y — see
        #      ``happy``, where the taught y→i lesson would be erased);
        #   2. next phoneme must not be /r/ — prevents the ɚ-split loophole
        #      that would decode ``doctor`` as o+r instead of the taught
        #      ``or → (ə, r)`` cluster;
        #   3. no inventory GPC already teaches this exact (letter, phoneme)
        #      pair — otherwise we'd double the parse count and dilute the
        #      taught lesson (``unicorn``'s middle i, ``dishes``' final s).
        if free_pass_mode != "none" and phoneme_index < len(phonemes):
            reduced_ipa = phonemes[phoneme_index].ipa
            is_unstressed = phonemes[phoneme_index].is_unstressed
            letter = spelling[letter_index]
            next_phoneme = (
                phoneme_strings[phoneme_index + 1]
                if phoneme_index + 1 < len(phonemes)
                else None
            )
            # Experiment axis: "schwa" restricts the reduction to /ə/ (ARPABET
            # AH0), "stress0" allows any unstressed vowel phoneme (IH0, IY0,
            # OW0, ...). Guards 1-3 below are held constant across modes.
            reducible = is_unstressed and (
                free_pass_mode == "stress0" or reduced_ipa == "ə"
            )
            # Guards 2 and 3 (next != /r/, pair not already taught) encode
            # curriculum precedence at walk time. The post-walk clean/all-parses
            # filter enforces the same precedence at result time, so toggling
            # this off keeps taught parses winning whenever they exist and only
            # rescues words where no taught alternative applies.
            passes_curriculum_guards = (
                next_phoneme != "r" and (letter, (reduced_ipa,)) not in UFLI_GPC_PAIRS
            )
            if (
                reducible
                and letter in VOWEL_LETTERS
                and (passes_curriculum_guards or not curriculum_guarded)
            ):
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
                and positions_are_free(letter_index, head_end, reserved)
                and phoneme_strings[phoneme_index:phoneme_end] == gpc.phonemes
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

    all_parses = walk(0, 0, 0)
    # Prefer parses that use only taught GPCs. If any parse avoided the
    # reduced-vowel free pass entirely, drop the free-pass alternatives — a
    # taught cluster (e.g. ``ful`` for ``careful``) should take precedence over
    # decomposing the same span letter-by-letter through the free pass.
    clean = tuple(
        parse
        for parse in all_parses
        if not any(match.gpc.id.startswith("REDUCED_") for match in parse)
    )
    return clean if clean else all_parses
