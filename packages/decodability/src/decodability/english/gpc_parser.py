"""Constrained aligner for English spelling and phoneme sequences.

Given a word, its phoneme sequence, and a GPC inventory, ``parse_gpcs``
returns every valid parse — a tuple of ``GPCMatch`` objects covering
every letter and every phoneme. The module is deliberately narrow: it
does not pronounce, translate, or score; it only answers the alignment
question. See ``parse_gpcs`` for the match kinds it recognizes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache

from decodability.english.definitions import VOWEL_LETTERS
from decodability.english.extract_phonemes import Phoneme, Pronunciation

# Productive spelling/pronunciation conventions that are licensed by taught
# correspondences rather than introduced as additional curriculum GPCs.
_DERIVED_GPC_SPECS = (
    ("DERIVED_X_VOICED", "x", ("g", "z")),
    ("DERIVED_XC", "xc", ("k", "s")),
    ("DERIVED_XH", "xh", ("g", "z")),
    ("DERIVED_DROPPED_E_CE", "c", ("s",)),
    ("DERIVED_DROPPED_E_GE", "g", ("dʒ",)),
    ("DERIVED_VELAR_N", "n", ("ŋ",)),
    ("DERIVED_GU", "gu", ("g", "w")),
    ("DERIVED_AL", "al", ("ɔ", "l")),
    ("DERIVED_IE_FINAL", "ie", ("i",)),
    ("DERIVED_IEW", "iew", ("j", "u")),
    ("DERIVED_LK", "lk", ("k",)),
    ("DERIVED_AWE", "awe", ("ɔ",)),
    ("DERIVED_OE_U", "oe", ("u",)),
    ("DERIVED_WH_H", "wh", ("h",)),
    ("DERIVED_AR_AIR", "ar", ("ɛ", "r")),
    ("DERIVED_IRE", "ire", ("aɪ", "ə", "r")),
    ("DERIVED_SSION", "ssion", ("ʃ", "ə", "n")),
    ("DERIVED_TION_CH", "tion", ("tʃ", "ə", "n")),
    ("DERIVED_GE_ZH", "ge", ("ʒ",)),
    ("DERIVED_O_SHORT_U", "o", ("ʌ",)),
)
_DROPPED_E_SUFFIXES = ("able", "ible", "ing", "ed", "er", "est", "y")
_LETTER_NAME_IPAS = {
    "a": ("eɪ",),
    "b": ("b", "i"),
    "c": ("s", "i"),
    "d": ("d", "i"),
    "e": ("i",),
    "f": ("ɛ", "f"),
    "g": ("dʒ", "i"),
    "h": ("eɪ", "tʃ"),
    "i": ("aɪ",),
    "j": ("dʒ", "eɪ"),
    "k": ("k", "eɪ"),
    "l": ("ɛ", "l"),
    "m": ("ɛ", "m"),
    "n": ("ɛ", "n"),
    "o": ("oʊ",),
    "p": ("p", "i"),
    "q": ("k", "j", "u"),
    "r": ("ɑ", "r"),
    "s": ("ɛ", "s"),
    "t": ("t", "i"),
    "u": ("j", "u"),
    "v": ("v", "i"),
    "w": ("d", "ʌ", "b", "ə", "l", "j", "u"),
    "x": ("ɛ", "k", "s"),
    "y": ("w", "aɪ"),
    "z": ("z", "i"),
}


@dataclass(frozen=True)
class GPC:
    """One licensed correspondence, including multi-phoneme and split units.

    A hyphen at the *edge* of the grapheme is a positional marker, not a
    split: a leading ``-`` marks a word-final unit (``-ed``, ``-ing``,
    ``-ve``), a trailing ``-`` a word-initial one (``pre-``, ``un-``). The
    marker is stripped from ``symbol`` and recorded as ``is_suffix`` /
    ``is_prefix``; matching itself is position-agnostic — a medial
    occurrence (the ``ve`` in ``gives``) is the same correspondence. A
    medial hyphen remains a split digraph (``a-e``).
    """

    id: str
    grapheme: str
    phonemes: tuple[str, ...]
    # Derived at construction.
    symbol: str = field(init=False, repr=False, compare=False)
    is_split: bool = field(init=False, repr=False, compare=False)
    split_head: str = field(init=False, repr=False, compare=False)
    split_tail: str = field(init=False, repr=False, compare=False)
    is_prefix: bool = field(init=False, repr=False, compare=False)
    is_suffix: bool = field(init=False, repr=False, compare=False)
    is_reduced_vowel: bool = field(init=False, repr=False, compare=False)
    is_silent: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.grapheme or (
            not self.phonemes and not self.id.startswith(("SILENT_", "UNKNOWN_"))
        ):
            raise ValueError("a GPC needs a grapheme and at least one phoneme")
        folded = self.grapheme.casefold()
        is_suffix = len(folded) > 1 and folded.startswith("-")
        is_prefix = len(folded) > 1 and not is_suffix and folded.endswith("-")
        symbol = folded[1:] if is_suffix else folded[:-1] if is_prefix else folded
        if symbol.count("-") > 1:
            raise ValueError("the prototype supports two-part split graphemes only")
        head, hyphen, tail = symbol.partition("-")
        if hyphen and not (head and tail):
            raise ValueError("a split grapheme needs letters on both sides of the '-'")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "is_split", bool(hyphen))
        object.__setattr__(self, "split_head", head)
        object.__setattr__(self, "split_tail", tail)
        object.__setattr__(self, "is_prefix", is_prefix)
        object.__setattr__(self, "is_suffix", is_suffix)
        object.__setattr__(self, "is_reduced_vowel", self.id.startswith("REDUCED_"))
        object.__setattr__(self, "is_silent", self.id.startswith("SILENT_"))
        object.__setattr__(
            self,
            "is_unknown",
            self.id.startswith(("UNKNOWN_", "UNSPELLED_")),
        )


@cache
def _derived_gpcs(inventory: tuple[GPC, ...]) -> tuple[GPC, ...]:
    doubles = (
        GPC(f"DERIVED_DOUBLE_{gpc.id}", gpc.symbol * 2, gpc.phonemes)
        for gpc in inventory
        if len(gpc.symbol) == 1
        and gpc.symbol not in VOWEL_LETTERS
        and not gpc.is_split
    )
    repeated_finals = (
        GPC(
            f"DERIVED_REPEAT_FINAL_{gpc.id}",
            gpc.symbol + gpc.symbol[-1],
            gpc.phonemes,
        )
        for gpc in inventory
        if len(gpc.symbol) > 1
        and gpc.symbol[-1] not in VOWEL_LETTERS
        and not gpc.is_split
    )
    specs = (
        spec
        for spec in _DERIVED_GPC_SPECS
        if spec[0] != "DERIVED_AL"
        or any(gpc.symbol == "all" and gpc.phonemes == ("ɔ", "l") for gpc in inventory)
    )
    return (*tuple(GPC(*spec) for spec in specs), *doubles, *repeated_finals)


def _phonemes_match(
    gpc: GPC, actual: tuple[Phoneme, ...], following: Phoneme | None
) -> bool:
    """Match a GPC against the heard phonemes.

    Two relaxations of exact matching:

    - ``o``/``au``/``augh``/``ough`` accept the cot-caught counterpart of
      their vowel (/ɑ/ for /ɔ/ and vice versa) — except before /r/, where
      the merger does not apply and ``for`` /fɔr/ never becomes /fɑr/.
    - ``es`` accepts unstressed /ɪ/ for /ə/ — the ordinary /əz/~/ɪz/
      ``-es`` variation.

    ``following`` is the phoneme immediately after ``actual`` in the full
    pronunciation (``None`` at word end); the pre-/r/ check needs it when
    the vowel under test is the last phoneme the GPC covers.
    """
    if len(gpc.phonemes) != len(actual):
        return False
    for index, (wanted, heard) in enumerate(zip(gpc.phonemes, actual)):
        if wanted == heard.ipa:
            continue
        after = actual[index + 1] if index + 1 < len(actual) else following
        if (
            gpc.symbol in {"o", "au", "augh", "ough"}
            and {wanted, heard.ipa} == {"ɑ", "ɔ"}
            and (after is None or after.ipa != "r")
        ):
            continue
        if (
            gpc.symbol == "es"
            and wanted == "ə"
            and heard.is_unstressed
            and heard.ipa == "ɪ"
        ):
            continue
        return False
    return True


def _derived_match_is_licensed(gpc: GPC, spelling: str, start: int) -> bool:
    if gpc.id.startswith("DERIVED_DROPPED_E_"):
        return spelling[start + 1 :].startswith(_DROPPED_E_SUFFIXES)
    if gpc.id == "DERIVED_VELAR_N":
        return spelling[start + 1 :].startswith("c")
    if gpc.id == "DERIVED_IE_FINAL":
        return spelling[start + 2 :] in {"", "'s"}
    if gpc.id == "DERIVED_LK":
        return start > 0 and spelling[start - 1] == "o"
    if gpc.id == "DERIVED_AR_AIR":
        return spelling[start + 2 :] == "y"
    if gpc.id == "DERIVED_O_SHORT_U":
        return start + 1 < len(spelling) and spelling[start + 1] in "vlm"
    return True


def _interjection_parse(
    spelling: str, phoneme_strings: tuple[str, ...]
) -> Parse | None:
    if (
        len(phoneme_strings) == 1
        and len(spelling) >= 2
        and spelling[0] in VOWEL_LETTERS
        and set(spelling[1:]) == {"h"}
    ):
        return (
            GPCMatch(
                GPC("DERIVED_INTERJECTION", spelling, phoneme_strings),
                tuple(range(len(spelling))),
                (0, 1),
            ),
        )
    return None


def _is_letter_name_pronunciation(
    spelling: str, phoneme_strings: tuple[str, ...], inventory: tuple[GPC, ...]
) -> bool:
    if (
        len(spelling) < 2
        or not spelling.isalpha()
        or spelling in {gpc.symbol for gpc in inventory}
        or not all(letter in _LETTER_NAME_IPAS for letter in spelling)
    ):
        return False
    return (
        tuple(ipa for letter in spelling for ipa in _LETTER_NAME_IPAS[letter])
        == phoneme_strings
    )


@dataclass(frozen=True)
class GPCMatch:
    """A GPC plus the spelling positions and phoneme slice it covers."""

    gpc: GPC
    letter_positions: tuple[int, ...]
    phoneme_span: tuple[int, int]


type Parse = tuple[GPCMatch, ...]


def _can_reuse_previous_letter(gpc: GPC) -> bool:
    return gpc.is_suffix or gpc.symbol[0] not in VOWEL_LETTERS


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
    pronunciation: Pronunciation,
    inventory: tuple[GPC, ...],
    *,
    max_split_gap: int = 2,
) -> tuple[Parse, ...]:
    """Return every valid parse of ``word`` under the given inventory.

    Takes one ``Pronunciation`` — a phoneme sequence together with the
    word-shape facts derived from its ARPABET source, such as
    ``is_monosyllabic``. Callers hold several per word and should try each,
    accepting a match on any one.

    A parse is a tuple of ``GPCMatch`` objects that together cover every
    letter of the spelling and every phoneme in the sequence. Multiple
    parses arise when the inventory offers overlapping ways to cover the
    same span — ``fish`` decomposes as ``f + i + sh`` (via ``sh → /ʃ/``)
    and as ``f + ish`` (via ``ish → (ɪ, ʃ)``), both yielding /f, ɪ, ʃ/.

    Beyond ordinary contiguous matches, the parser handles:

    - Split graphemes (``a-e``, ``i-e``, …): the two parts cover
      non-adjacent letter positions, with the gap bounded by
      ``max_split_gap``.
    - Edge-marked units (``-ed``, ``-ing``, ``pre-``, ``un-``, ``-ve``,
      ``-le``): affixes and positional graphemes taught as chunks. The
      marker is stripped from ``symbol`` and recorded on the ``GPC``
      (``is_prefix``/``is_suffix``) but does not constrain matching —
      the same chunk appears medially in inflected words (``gives``,
      ``tables``) and the unit means the same correspondence there.
    - Free pass on unstressed vowels: a bare vowel letter paired with
      an unstressed phoneme parses via a synthesized ``REDUCED_*`` GPC —
      vowel reduction is a universal spoken-English process, not a taught
      correspondence. Withheld when the pronunciation is monosyllabic: a
      one-syllable word has no unstressed syllable to reduce into, so its
      vowel is a correspondence the student has to have been taught.
      The same reduction applies inside the yod correspondence: where the
      inventory licenses ``u → (j, u)``, a bare ``u`` paired with /j/ plus
      an unstressed vowel parses via ``REDUCED_YOD_U`` — the /j/ glide is
      part of the taught correspondence and only its vowel reduces
      (``partic-u-larly`` /j ə/).
    - Silent apostrophe: ``'`` positions are skipped without consuming
      a phoneme (``he's`` decodes as ``h + e + s``) — the curriculum
      treats possessive/contracted ``'s`` as an assumed correspondence.
    - Shared letters: a grapheme may reuse one boundary letter already
      claimed by a neighboring grapheme — either its first letter reusing
      the previous match's last letter (``jungle``: ``ng → ŋ`` *and*
      ``g → g`` share the ``g``), or its last letter sitting on a split
      digraph's reserved tail (``placed``: the ``e`` is the ``a-e`` tail,
      part of ``ce → s``, and part of ``-ed → t``). Interior positions
      must still be unclaimed; phonemes are never shared.
    """

    if max_split_gap < 1:
        raise ValueError("max_split_gap must be at least one letter")

    spelling = word.casefold()
    phonemes = pronunciation.phonemes
    phoneme_strings = tuple(p.ipa for p in phonemes)
    interjection = _interjection_parse(spelling, phoneme_strings)
    if interjection is not None:
        return (interjection,)
    contiguous = tuple(
        gpc for gpc in inventory if not gpc.is_split
    ) + _derived_gpcs(inventory)
    split = tuple(gpc for gpc in inventory if gpc.is_split)
    spells_letter_names = _is_letter_name_pronunciation(
        spelling, phoneme_strings, inventory
    )
    # The reduced-yod free pass is licensed only when the inventory teaches
    # u → (j, u): the glide belongs to that correspondence; only its vowel
    # reduces.
    has_yod_u = any(
        gpc.symbol == "u" and gpc.phonemes == ("j", "u") for gpc in contiguous
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

        for gpc in contiguous:
            phoneme_end = phoneme_index + len(gpc.phonemes)
            # Each GPC is tried at ``letter_index`` and one letter back —
            # the latter reuses the previous match's last letter.
            for start in (letter_index, letter_index - 1):
                if start < 0 or (
                    start < letter_index and not _can_reuse_previous_letter(gpc)
                ):
                    continue
                letter_end = start + len(gpc.symbol)
                if (
                    spelling.startswith(gpc.symbol, start)
                    and _derived_match_is_licensed(gpc, spelling, start)
                    # Only interior positions must be unclaimed: the first
                    # letter may be the previous match's last (start <
                    # letter_index), and the last may sit on a split tail.
                    and _positions_are_free(
                        max(start, letter_index), letter_end - 1, reserved
                    )
                    and _phonemes_match(
                        gpc,
                        phonemes[phoneme_index:phoneme_end],
                        phonemes[phoneme_end]
                        if phoneme_end < len(phonemes)
                        else None,
                    )
                ):
                    match = GPCMatch(
                        gpc,
                        tuple(range(start, letter_end)),
                        (phoneme_index, phoneme_end),
                    )
                    parses.extend(
                        (match, *remainder)
                        for remainder in walk(letter_end, phoneme_end, reserved)
                    )

        if phoneme_index < len(phonemes) and not pronunciation.is_monosyllabic:
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
            if (
                has_yod_u
                and letter == "u"
                and phonemes[phoneme_index].ipa == "j"
                and phoneme_index + 1 < len(phonemes)
                and phonemes[phoneme_index + 1].is_unstressed
            ):
                yod_pass = GPCMatch(
                    GPC(
                        id="REDUCED_YOD_U",
                        grapheme="u",
                        phonemes=("j", phonemes[phoneme_index + 1].ipa),
                    ),
                    (letter_index,),
                    (phoneme_index, phoneme_index + 2),
                )
                parses.extend(
                    (yod_pass, *remainder)
                    for remainder in walk(letter_index + 1, phoneme_index + 2, reserved)
                )

        letter = spelling[letter_index]
        if spells_letter_names:
            name = _LETTER_NAME_IPAS[letter]
            name_end = phoneme_index + len(name)
            if phoneme_strings[phoneme_index:name_end] == name:
                letter_name = GPCMatch(
                    GPC(f"LETTER_NAME_{letter.upper()}", letter, name),
                    (letter_index,),
                    (phoneme_index, name_end),
                )
                parses.extend(
                    (letter_name, *remainder)
                    for remainder in walk(letter_index + 1, name_end, reserved)
                )

        is_silent_final_e = (
            letter == "e"
            and letter_index == len(spelling) - 1
            and spelling[letter_index - 1] not in VOWEL_LETTERS
            and phoneme_index == len(phonemes)
        )
        is_interjection_h = (
            letter == "h"
            and phoneme_index == len(phonemes)
            and set(spelling[letter_index:]) == {"h"}
            and spelling[0] in VOWEL_LETTERS
            and pronunciation.is_monosyllabic
        )
        if is_silent_final_e or is_interjection_h:
            reason = "FINAL_E" if is_silent_final_e else "INTERJECTION_H"
            silent = GPCMatch(
                GPC(id=f"SILENT_{reason}", grapheme=letter, phonemes=()),
                (letter_index,),
                (phoneme_index, phoneme_index),
            )
            parses.extend(
                (silent, *remainder)
                for remainder in walk(letter_index + 1, phoneme_index, reserved)
            )

        for gpc in split:
            head_end = letter_index + len(gpc.split_head)
            phoneme_end = phoneme_index + len(gpc.phonemes)
            if not (
                spelling.startswith(gpc.split_head, letter_index)
                and _positions_are_free(letter_index, head_end, reserved)
                and phoneme_strings[phoneme_index:phoneme_end] == gpc.phonemes
            ):
                continue

            for gap in range(1, max_split_gap + 1):
                tail_start = head_end + gap
                tail_end = tail_start + len(gpc.split_tail)
                if spelling.startswith(
                    gpc.split_tail, tail_start
                ) and _positions_are_free(letter_index, tail_end, reserved):
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
    return all_parses


def align_word(
    word: str,
    pronunciation: Pronunciation,
    inventory: tuple[GPC, ...],
    *,
    max_split_gap: int = 2,
) -> Parse:
    """Best-effort alignment of ``word`` to ``pronunciation``.

    Unlike ``parse_gpcs`` — which returns only parses covering every letter
    and phoneme with inventory GPCs — this always returns exactly one
    ``Parse``: the alignment needing the fewest non-inventory
    correspondences. When a real parse exists it wins (cost 0), so this
    doubles as "the best parse or the closest thing to one".

    Non-inventory matches are synthesized ``GPC`` objects (``is_unknown``)
    whose ids mark the kind of gap:

    - ``UNKNOWN_<letter>_<ipa>`` — a letter↔phoneme pair with no inventory
      GPC (``n → ŋ`` in ``hungry``).
    - ``UNKNOWN_SILENT_<letter>`` — a letter covering no phoneme.
    - ``UNSPELLED_<ipa>`` — a phoneme no letter spells (the schwa in
      ``isn't``); its ``letter_positions`` are empty.

    Intended for diagnosis, not scoring: it answers "what would it take to
    read this word?" rather than "can this student read it?".
    """
    spelling = word.casefold()
    phonemes = pronunciation.phonemes
    phoneme_strings = tuple(p.ipa for p in phonemes)
    interjection = _interjection_parse(spelling, phoneme_strings)
    if interjection is not None:
        return interjection
    contiguous = tuple(
        gpc for gpc in inventory if not gpc.is_split
    ) + _derived_gpcs(inventory)
    split = tuple(gpc for gpc in inventory if gpc.is_split)
    spells_letter_names = _is_letter_name_pronunciation(
        spelling, phoneme_strings, inventory
    )
    has_yod_u = any(
        gpc.symbol == "u" and gpc.phonemes == ("j", "u") for gpc in contiguous
    )

    @cache
    def walk(letter_index: int, phoneme_index: int, reserved: int) -> tuple[int, Parse]:
        """Min-cost completion from ``(letter_index, phoneme_index)``.

        Cost is the number of non-inventory correspondences used. Always
        returns a parse — worst case every letter is ``UNKNOWN_SILENT``
        and every phoneme ``UNSPELLED``.
        """
        while letter_index < len(spelling) and (
            reserved & (1 << letter_index) or spelling[letter_index] == "'"
        ):
            letter_index += 1

        if letter_index == len(spelling):
            unspelled = tuple(
                GPCMatch(
                    GPC(f"UNSPELLED_{p.ipa}", "∅", (p.ipa,)),
                    (),
                    (k, k + 1),
                )
                for k, p in enumerate(phonemes[phoneme_index:], phoneme_index)
            )
            return len(unspelled), unspelled

        letter = spelling[letter_index]
        options: list[tuple[int, Parse]] = []

        for gpc in contiguous:
            phoneme_end = phoneme_index + len(gpc.phonemes)
            # Same shared-letter rules as ``parse_gpcs``: a match may start
            # one letter back or end on a reserved split tail.
            for start in (letter_index, letter_index - 1):
                if start < 0 or (
                    start < letter_index and not _can_reuse_previous_letter(gpc)
                ):
                    continue
                letter_end = start + len(gpc.symbol)
                if (
                    spelling.startswith(gpc.symbol, start)
                    and _derived_match_is_licensed(gpc, spelling, start)
                    and _positions_are_free(
                        max(start, letter_index), letter_end - 1, reserved
                    )
                    and _phonemes_match(
                        gpc,
                        phonemes[phoneme_index:phoneme_end],
                        phonemes[phoneme_end]
                        if phoneme_end < len(phonemes)
                        else None,
                    )
                ):
                    match = GPCMatch(
                        gpc,
                        tuple(range(start, letter_end)),
                        (phoneme_index, phoneme_end),
                    )
                    cost, rest = walk(letter_end, phoneme_end, reserved)
                    options.append((cost, (match, *rest)))

        if phoneme_index < len(phonemes) and not pronunciation.is_monosyllabic:
            if phonemes[phoneme_index].is_unstressed and letter in VOWEL_LETTERS:
                match = GPCMatch(
                    GPC(
                        id=f"REDUCED_{letter.upper()}",
                        grapheme=letter,
                        phonemes=(phonemes[phoneme_index].ipa,),
                    ),
                    (letter_index,),
                    (phoneme_index, phoneme_index + 1),
                )
                cost, rest = walk(letter_index + 1, phoneme_index + 1, reserved)
                options.append((cost, (match, *rest)))
            if (
                has_yod_u
                and letter == "u"
                and phonemes[phoneme_index].ipa == "j"
                and phoneme_index + 1 < len(phonemes)
                and phonemes[phoneme_index + 1].is_unstressed
            ):
                match = GPCMatch(
                    GPC(
                        id="REDUCED_YOD_U",
                        grapheme="u",
                        phonemes=("j", phonemes[phoneme_index + 1].ipa),
                    ),
                    (letter_index,),
                    (phoneme_index, phoneme_index + 2),
                )
                cost, rest = walk(letter_index + 1, phoneme_index + 2, reserved)
                options.append((cost, (match, *rest)))

        if spells_letter_names:
            name = _LETTER_NAME_IPAS[letter]
            name_end = phoneme_index + len(name)
            if phoneme_strings[phoneme_index:name_end] == name:
                match = GPCMatch(
                    GPC(f"LETTER_NAME_{letter.upper()}", letter, name),
                    (letter_index,),
                    (phoneme_index, name_end),
                )
                cost, rest = walk(letter_index + 1, name_end, reserved)
                options.append((cost, (match, *rest)))

        is_silent_final_e = (
            letter == "e"
            and letter_index == len(spelling) - 1
            and spelling[letter_index - 1] not in VOWEL_LETTERS
            and phoneme_index == len(phonemes)
        )
        is_interjection_h = (
            letter == "h"
            and phoneme_index == len(phonemes)
            and set(spelling[letter_index:]) == {"h"}
            and spelling[0] in VOWEL_LETTERS
            and pronunciation.is_monosyllabic
        )
        if is_silent_final_e or is_interjection_h:
            reason = "FINAL_E" if is_silent_final_e else "INTERJECTION_H"
            match = GPCMatch(
                GPC(f"SILENT_{reason}", letter, ()),
                (letter_index,),
                (phoneme_index, phoneme_index),
            )
            cost, rest = walk(letter_index + 1, phoneme_index, reserved)
            options.append((cost, (match, *rest)))

        if phoneme_index < len(phonemes):
            ipa = phonemes[phoneme_index].ipa
            match = GPCMatch(
                GPC(f"UNKNOWN_{letter}_{ipa}", letter, (ipa,)),
                (letter_index,),
                (phoneme_index, phoneme_index + 1),
            )
            cost, rest = walk(letter_index + 1, phoneme_index + 1, reserved)
            options.append((cost + 1, (match, *rest)))

        match = GPCMatch(
            GPC(f"UNKNOWN_SILENT_{letter.upper()}", letter, ()),
            (letter_index,),
            (phoneme_index, phoneme_index),
        )
        cost, rest = walk(letter_index + 1, phoneme_index, reserved)
        options.append((cost + 1, (match, *rest)))

        if phoneme_index < len(phonemes):
            ipa = phonemes[phoneme_index].ipa
            match = GPCMatch(
                GPC(f"UNSPELLED_{ipa}", "∅", (ipa,)),
                (),
                (phoneme_index, phoneme_index + 1),
            )
            cost, rest = walk(letter_index, phoneme_index + 1, reserved)
            options.append((cost + 1, (match, *rest)))

        for gpc in split:
            head_end = letter_index + len(gpc.split_head)
            phoneme_end = phoneme_index + len(gpc.phonemes)
            if not (
                spelling.startswith(gpc.split_head, letter_index)
                and _positions_are_free(letter_index, head_end, reserved)
                and phoneme_strings[phoneme_index:phoneme_end] == gpc.phonemes
            ):
                continue
            for gap in range(1, max_split_gap + 1):
                tail_start = head_end + gap
                tail_end = tail_start + len(gpc.split_tail)
                if spelling.startswith(
                    gpc.split_tail, tail_start
                ) and _positions_are_free(letter_index, tail_end, reserved):
                    tail_positions = tuple(range(tail_start, tail_end))
                    tail_mask = sum(1 << position for position in tail_positions)
                    match = GPCMatch(
                        gpc,
                        (*range(letter_index, head_end), *tail_positions),
                        (phoneme_index, phoneme_end),
                    )
                    cost, rest = walk(head_end, phoneme_end, reserved | tail_mask)
                    options.append((cost, (match, *rest)))

        return min(options, key=lambda option: (option[0], len(option[1])))

    return walk(0, 0, 0)[1]
