"""Phonology and GPC definitions for US-English decoding.

- ``ARPABET_TO_IPA``: CMU ARPABET phoneme codes → IPA. The parser consumes IPA.
- ``VOWEL_INITIALS`` / ``VOWEL_LETTERS``: character sets used by the parser's
  free-pass guards.
- ``UFLI_GPC_INVENTORY``: the ordered grapheme-phoneme correspondences taught
  in UFLI Foundations. Inline comments annotate entries whose mapping needs
  contextual disambiguation (split digraphs, open syllables, positional
  variants, etc.).
- ``UFLI_GPC_PAIRS_FLAT``: derived lookup sets for the student-knowledge validator
  (joined-IPA form).
- ``ufli_gpc_inventory(version)`` / ``ufli_gpc_pairs(version)``: the inventory
  and its pairs as spelled at an earlier ``UFLIInventoryVersion``, so profiles
  written against that spelling keep validating and matching parses.
"""

from dataclasses import dataclass, field, replace
from functools import cache
from typing import Literal, get_args

ARPABET_TO_IPA = {
    "AA": "ɑ",
    "AE": "æ",
    "AH0": "ə",
    "AH1": "ʌ",
    "AH2": "ʌ",
    "AO": "ɔ",
    "AW": "aʊ",
    "AY": "aɪ",
    "B": "b",
    "CH": "tʃ",
    "D": "d",
    "DH": "ð",
    "EH": "ɛ",
    "ER": "ɚ",
    "EY": "eɪ",
    "F": "f",
    "G": "g",
    "HH": "h",
    "IH": "ɪ",
    "IY": "i",
    "JH": "dʒ",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ŋ",
    "OW": "oʊ",
    "OY": "ɔɪ",
    "P": "p",
    "R": "r",
    "S": "s",
    "SH": "ʃ",
    "T": "t",
    "TH": "θ",
    "UH": "ʊ",
    "UW": "u",
    "V": "v",
    "W": "w",
    "Y": "j",
    "Z": "z",
    "ZH": "ʒ",
}


VOWEL_INITIALS = set("aeiouæɑɔɛɪʊʌəɚɝ")
VOWEL_LETTERS = set("aeiou")


@dataclass(frozen=True)
class GPC:
    """One licensed correspondence, including multi-phoneme and split units.

    A hyphen at the *edge* of the grapheme is a positional marker, not a
    split: a leading ``-`` marks a word-final unit (``-ed``, ``-ing``,
    ``-ve``), a trailing ``-`` a word-initial one (``pre-``, ``un-``). The
    marker is stripped from ``symbol`` and recorded as ``is_suffix`` /
    ``is_prefix``; suffix matching is position-agnostic — a medial
    occurrence (the ``ve`` in ``gives``) is the same correspondence — but
    a prefix only matches at the word start. A medial hyphen remains a
    split digraph (``a-e``).

    ``lesson`` is the UFLI lesson at which the correspondence is officially
    introduced; ``decode_lesson``, when set, is the earlier lesson at which
    it becomes decodable (``-ed→əd`` decodes at 18 though ``-ed`` is
    introduced at 64). Both are set only on inventory entries — synthesized
    matches (``REDUCED_*``, ``SILENT_*``, ``DERIVED_*``) leave them ``None``.
    """

    id: str
    grapheme: str
    phonemes: tuple[str, ...]
    lesson: int | None = field(default=None, repr=False)
    decode_lesson: int | None = field(default=None, repr=False)
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
            not self.phonemes and not self.id.startswith("SILENT_")
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


UFLI_GPC_INVENTORY: tuple[GPC, ...] = (
    GPC("G001", "a", ("æ",), lesson=1),
    GPC("G002", "m", ("m",), lesson=2),
    GPC("G003", "s", ("s",), lesson=3),
    GPC("G004", "t", ("t",), lesson=4),
    GPC("G005", "p", ("p",), lesson=6),
    GPC("G006", "f", ("f",), lesson=7),
    GPC("G007", "i", ("ɪ",), lesson=8),
    GPC("G008", "n", ("n",), lesson=9),
    GPC("G010", "o", ("ɑ",), lesson=12),
    GPC("G011", "d", ("d",), lesson=13),
    GPC("G012", "c", ("k",), lesson=14),
    GPC("G013", "u", ("ʌ",), lesson=15),
    GPC("G014", "g", ("g",), lesson=16),
    GPC("G015", "b", ("b",), lesson=17),
    GPC("G016", "e", ("ɛ",), lesson=18),
    GPC("G017", "-s", ("s",), lesson=20),
    GPC("G018", "s", ("z",), lesson=21),
    GPC("G019", "k", ("k",), lesson=22),
    GPC("G020", "h", ("h",), lesson=23),
    GPC("G021", "r", ("r",), lesson=24),
    GPC("G022", "l", ("l",), lesson=26),
    GPC("G023", "w", ("w",), lesson=28),
    GPC("G024", "j", ("dʒ",), lesson=29),
    GPC("G025", "y", ("j",), lesson=30),
    GPC("G026", "x", ("k", "s"), lesson=31),
    GPC("G027", "qu", ("k", "w"), lesson=32),
    GPC("G028", "v", ("v",), lesson=33),
    GPC("G029", "z", ("z",), lesson=34),
    GPC("G030", "ff", ("f",), lesson=42),
    GPC("G031", "ll", ("l",), lesson=42),
    GPC("G032", "ss", ("s",), lesson=42),
    GPC("G033", "zz", ("z",), lesson=42),
    GPC("G034", "all", ("ɔ", "l"), lesson=43),
    GPC("G035", "oll", ("oʊ", "l"), lesson=43),
    GPC("G036", "ull", ("ʊ", "l"), lesson=43),
    GPC("G037", "ck", ("k",), lesson=44),
    GPC("G038", "sh", ("ʃ",), lesson=45),
    GPC("G039", "th", ("ð",), lesson=46),
    GPC("G040", "th", ("θ",), lesson=47),
    GPC("G041", "ch", ("tʃ",), lesson=48),
    GPC("G042", "wh", ("w",), lesson=50),
    GPC("G043", "ph", ("f",), lesson=50),
    GPC("G044", "ng", ("ŋ",), lesson=51),
    GPC("G045", "nk", ("ŋ", "k"), lesson=52),
    GPC("G046", "a-e", ("eɪ",), lesson=54),  # split digraph
    GPC("G047", "i-e", ("aɪ",), lesson=55),  # split digraph
    GPC("G048", "o-e", ("oʊ",), lesson=56),  # split digraph
    GPC("G049", "e-e", ("i",), lesson=57),  # split digraph
    GPC("G050", "u-e", ("u",), lesson=58),  # split digraph
    GPC("G051", "u-e", ("j", "u"), lesson=58),  # split digraph (yod)
    GPC("G052", "ce", ("s",), lesson=60),  # soft c + silent e (_ce)
    GPC("G053", "ge", ("dʒ",), lesson=61),  # soft g + silent e (_ge)
    GPC("G054", "-ve", ("v",), lesson=62),  # words don't end in v (-ve)
    GPC("G055", "-es", ("ə", "z"), lesson=63, decode_lesson=21),
    GPC("G056", "-ed", ("t",), lesson=64),
    GPC("G057", "-ed", ("d",), lesson=64),
    GPC("G058", "-ed", ("ə", "d"), lesson=64, decode_lesson=18),
    GPC("G059", "-ing", ("ɪ", "ŋ"), lesson=65, decode_lesson=51),
    GPC("G060", "a", ("eɪ",), lesson=68),  # open syllable
    GPC("G061", "e", ("i",), lesson=68),  # open syllable
    GPC("G062", "i", ("aɪ",), lesson=68),  # open syllable
    GPC("G063", "o", ("oʊ",), lesson=68),  # open syllable
    GPC("G064", "u", ("j", "u"), lesson=68),  # open syllable
    GPC("G065", "u", ("u",), lesson=68),  # open syllable
    GPC("G066", "tch", ("tʃ",), lesson=69),
    GPC("G067", "dge", ("dʒ",), lesson=70),
    GPC("G068", "ild", ("aɪ", "l", "d"), lesson=72),
    GPC("G069", "old", ("oʊ", "l", "d"), lesson=72),
    GPC("G070", "ind", ("aɪ", "n", "d"), lesson=72),
    GPC("G071", "olt", ("oʊ", "l", "t"), lesson=72),
    GPC("G072", "ost", ("oʊ", "s", "t"), lesson=72),
    GPC("G073", "y", ("aɪ",), lesson=73),
    GPC("G074", "y", ("i",), lesson=74),
    GPC("G075", "-le", ("ə", "l"), lesson=75),  # -le syllable ending
    GPC("G076", "ar", ("ɑ", "r"), lesson=77),
    GPC("G077", "or", ("ɔ", "r"), lesson=78),
    GPC("G078", "ore", ("ɔ", "r"), lesson=78),
    GPC("G079", "er", ("ə", "r"), lesson=80),
    GPC("G080", "ir", ("ə", "r"), lesson=81),
    GPC("G081", "ur", ("ə", "r"), lesson=81),
    GPC("G082", "or", ("ə", "r"), lesson=82),  # or after w (unstressed)
    GPC("G083", "ai", ("eɪ",), lesson=84),
    GPC("G084", "ay", ("eɪ",), lesson=84),
    GPC("G085", "ee", ("i",), lesson=85),
    GPC("G086", "ea", ("i",), lesson=85),
    GPC("G087", "ey", ("i",), lesson=85),
    GPC("G088", "oa", ("oʊ",), lesson=86),
    GPC("G089", "ow", ("oʊ",), lesson=86),
    GPC("G090", "oe", ("oʊ",), lesson=86),
    GPC("G091", "ie", ("aɪ",), lesson=87),
    GPC("G092", "igh", ("aɪ",), lesson=87),
    GPC("G093", "u", ("ʊ",), lesson=89),
    GPC("G094", "oo", ("ʊ",), lesson=89),
    GPC("G095", "oo", ("u",), lesson=90),
    GPC("G096", "ew", ("u",), lesson=91),
    GPC("G097", "ui", ("u",), lesson=91),
    GPC("G098", "ue", ("u",), lesson=91),
    GPC("G099", "au", ("ɔ",), lesson=93),
    GPC("G100", "aw", ("ɔ",), lesson=93),
    GPC("G101", "augh", ("ɔ",), lesson=93),
    GPC("G102", "ea", ("ɛ",), lesson=94),
    GPC("G103", "a", ("ɑ",), lesson=94),  # a after w/qu
    GPC("G104", "oi", ("ɔɪ",), lesson=95),
    GPC("G105", "oy", ("ɔɪ",), lesson=95),
    GPC("G106", "ou", ("aʊ",), lesson=96),
    GPC("G107", "ow", ("aʊ",), lesson=96),
    GPC("G108", "kn", ("n",), lesson=98),
    GPC("G109", "wr", ("r",), lesson=98),
    GPC("G110", "mb", ("m",), lesson=98),
    GPC("G111", "-er", ("ə", "r"), lesson=100, decode_lesson=80),
    GPC("G112", "-est", ("ə", "s", "t"), lesson=100, decode_lesson=18),
    GPC("G113", "-ly", ("l", "i"), lesson=101, decode_lesson=74),
    GPC("G114", "-less", ("l", "ə", "s"), lesson=102, decode_lesson=42),
    GPC("G115", "-ful", ("f", "ə", "l"), lesson=102, decode_lesson=26),
    GPC("G116", "un-", ("ʌ", "n"), lesson=103, decode_lesson=15),
    GPC("G117", "pre-", ("p", "r", "i"), lesson=104, decode_lesson=66),
    GPC("G118", "re-", ("r", "i"), lesson=104, decode_lesson=66),
    GPC("G119", "dis-", ("d", "ɪ", "s"), lesson=105, decode_lesson=13),
    GPC("G120", "-ied", ("aɪ", "d"), lesson=110, decode_lesson=87),
    GPC("G121", "-ies", ("aɪ", "z"), lesson=110, decode_lesson=87),
    GPC("G122", "-ies", ("i", "z"), lesson=110),
    GPC("G123", "ar", ("ə", "r"), lesson=111),  # unstressed ar: beggar, dollar
    GPC("G125", "air", ("ɛ", "r"), lesson=112),
    GPC("G126", "are", ("ɛ", "r"), lesson=112),
    GPC("G127", "ear", ("ɛ", "r"), lesson=112),
    GPC("G128", "ear", ("ɪ", "r"), lesson=113),
    GPC("G129", "ei", ("eɪ",), lesson=114),
    GPC("G130", "eigh", ("eɪ",), lesson=114),
    GPC("G131", "ey", ("eɪ",), lesson=114),
    GPC("G132", "ea", ("eɪ",), lesson=114),
    GPC("G133", "aigh", ("eɪ",), lesson=114),
    GPC("G134", "ew", ("j", "u"), lesson=115),
    GPC("G135", "eu", ("j", "u"), lesson=115),
    GPC("G136", "ue", ("j", "u"), lesson=115),
    GPC("G137", "ou", ("u",), lesson=115),
    GPC("G138", "ough", ("ɔ",), lesson=116),
    GPC("G139", "ough", ("oʊ",), lesson=116),
    GPC("G140", "c", ("s",), lesson=117),  # c before e, i, y
    GPC("G141", "g", ("dʒ",), lesson=117),  # g before e, i, y
    GPC("G142", "ch", ("ʃ",), lesson=118),
    GPC("G143", "ch", ("k",), lesson=118),
    GPC("G144", "gn", ("n",), lesson=118),
    GPC("G145", "gh", ("g",), lesson=118),
    GPC("G146", "st", ("s",), lesson=118),  # silent t
    GPC("G147", "-tion", ("ʃ", "ə", "n"), lesson=119),
    GPC("G148", "-sion", ("ʒ", "ə", "n"), lesson=119),
    GPC("G149", "-sion", ("ʃ", "ə", "n"), lesson=119),
    GPC("G150", "-ture", ("tʃ", "ə", "r"), lesson=120),
    GPC("G151", "-ist", ("ɪ", "s", "t"), lesson=121),
    GPC("G152", "-ish", ("ɪ", "ʃ"), lesson=122),
    GPC("G153", "-ness", ("n", "ə", "s"), lesson=124),
    GPC("G154", "-ment", ("m", "ə", "n", "t"), lesson=125),
    GPC("G155", "-able", ("ə", "b", "ə", "l"), lesson=126),
    GPC("G156", "-ible", ("ə", "b", "ə", "l"), lesson=126),
    GPC("G157", "bi-", ("b", "aɪ"), lesson=127, decode_lesson=66),
    GPC("G158", "tri-", ("t", "r", "aɪ"), lesson=127, decode_lesson=66),
    GPC("G159", "uni-", ("j", "u", "n", "ɪ"), lesson=127, decode_lesson=66),
    GPC("G160", "bb", ("b",), lesson=67),
    GPC("G161", "cc", ("k",), lesson=67),
    GPC("G162", "dd", ("d",), lesson=67),
    GPC("G163", "gg", ("g",), lesson=67),
    GPC("G164", "mm", ("m",), lesson=67),
    GPC("G165", "nn", ("n",), lesson=67),
    GPC("G166", "pp", ("p",), lesson=67),
    GPC("G167", "tt", ("t",), lesson=67),
    GPC("G168", "o", ("ɔ",), lesson=12),
    GPC("G169", "au", ("ɑ",), lesson=93),
    GPC("G170", "augh", ("ɑ",), lesson=93),
    GPC("G171", "ough", ("ɑ",), lesson=116),  # resolve cot-caught merger
    GPC("G172", "ssion", ("ʃ", "ə", "n"), lesson=119),
    GPC("G173", "o-e", ("ʌ",), lesson=62),
    GPC("G174", "rr", ("r",), lesson=67),
    GPC("G175", "-ier", ("i", "ə", "r"), lesson=110),  # comparative suffix: happier
    GPC(
        "G176", "-iest", ("i", "ə", "s", "t"), lesson=110
    ),  # superlative suffix: happiest
    GPC("G177", "-ied", ("i", "d"), lesson=110),  # y-to-i past tense: carried, studied
)


def normalize_gpc_grapheme(grapheme: str) -> str:
    """Map a grapheme to its canonical taught-pair form.

    Edge positional markers are removed — a leading ``-``/``_`` marks a
    word-final unit (``-ed``, ``_ce``), a trailing one a word-initial unit
    (``pre-``) — while a medial ``_`` is rewritten to the split-digraph
    hyphen the inventory uses (``a_e`` → ``a-e``). Matches the normalization
    workbook ingestion applies, so marked, stripped, and underscore-written
    forms of the same unit all compare equal.
    """
    return grapheme.strip("-_").replace("_", "-")


# Inventory versions, oldest first, named by the date their spelling took
# effect. Only respelling an existing entry needs a new version: entries added
# later don't change what an earlier profile means, so every version keeps them.
UFLIInventoryVersion = Literal["2026-09-10", "2026-10-27"]
LATEST_UFLI_INVENTORY: UFLIInventoryVersion = "2026-10-27"

# The phonemes each version replaced, by GPC id. A respelling keeps the id and
# grapheme, so a profile written with the old spelling still means the same
# correspondence.
_UFLI_RESPELLINGS: dict[UFLIInventoryVersion, dict[str, tuple[str, ...]]] = {
    "2026-10-27": {"G159": ("j", "u", "n", "i")},  # uni-: /juni/ → /junɪ/
}


@cache
def ufli_gpc_inventory(
    version: UFLIInventoryVersion = LATEST_UFLI_INVENTORY,
) -> tuple[GPC, ...]:
    """Return ``UFLI_GPC_INVENTORY`` as spelled at ``version``.

    Every respelling made after ``version`` is undone; all other entries,
    including ones added since, are the latest. ``"2026-09-10"``, the spelling
    the phonics benchmark items were built with, reads ``uni-`` as /juni/.
    """
    versions = get_args(UFLIInventoryVersion)
    if version not in versions:
        raise ValueError(
            f"Unknown UFLI inventory version {version!r}; known: {versions}"
        )
    earlier: dict[str, tuple[str, ...]] = {}
    # Newest first, so an entry respelled twice ends at its spelling at `version`.
    for later in reversed(versions[versions.index(version) + 1 :]):
        earlier.update(_UFLI_RESPELLINGS.get(later, {}))
    return tuple(
        replace(gpc, phonemes=earlier[gpc.id]) if gpc.id in earlier else gpc
        for gpc in UFLI_GPC_INVENTORY
    )


# Student-facing form: the phoneme side is a single joined IPA string, so a
# multi-phoneme GPC like ``("x", ("k", "s"))`` reads as ``("x", "ks")``. The
# grapheme side is normalized to its canonical form (affix markers stripped:
# ``-ed`` → ``ed``, ``pre-`` → ``pre``), matching how taught GPCs are stored
# in student profiles. Used by ``EnglishStudentKnowledge`` validation and by
# the scorer to check parser matches against the student's taught set.
@cache
def ufli_gpc_pairs(
    version: UFLIInventoryVersion = LATEST_UFLI_INVENTORY,
) -> frozenset[tuple[str, str]]:
    """Return the student-facing pairs of ``ufli_gpc_inventory(version)``."""
    return frozenset(
        (normalize_gpc_grapheme(gpc.grapheme), "".join(gpc.phonemes))
        for gpc in ufli_gpc_inventory(version)
    )


UFLI_GPC_PAIRS_FLAT: frozenset[tuple[str, str]] = ufli_gpc_pairs()


# UFLI lesson (1-128) at which each sight (heart) word is introduced, from the
# "Sight words" sheet of the same workbook.
UFLI_SIGHT_WORD_LESSONS: dict[str, int] = {
    "the": 1,
    "i": 3,
    "and": 5,
    "a": 7,
    "is": 9,
    "as": 11,
    "said": 13,
    "to": 15,
    "do": 15,
    "of": 17,
    "see": 19,
    "he": 23,
    "be": 23,
    "me": 23,
    "from": 25,
    "look": 26,
    "book": 26,
    "are": 27,
    "was": 29,
    "you": 31,
    "what": 33,
    "have": 33,
    "your": 42,
    "want": 42,
    "go": 43,
    "no": 43,
    "so": 43,
    "goes": 44,
    "says": 44,
    "she": 45,
    "we": 45,
    "they": 46,
    "their": 46,
    "were": 47,
    "talk": 48,
    "walk": 48,
    "could": 50,
    "would": 50,
    "should": 50,
    "or": 51,
    "for": 51,
    "there": 52,
    "where": 52,
    "who": 54,
    "by": 55,
    "my": 55,
    "one": 58,
    "once": 58,
    "two": 63,
    "does": 63,
    "any": 64,
    "many": 64,
    "been": 65,
    "into": 65,
    "friend": 66,
    "because": 67,
    "woman": 69,
    "women": 69,
    "move": 70,
    "both": 72,
    "four": 73,
    "fourth": 73,
    "forty": 74,
    "people": 75,
    "pretty": 77,
    "nothing": 78,
    "other": 80,
    "another": 80,
    "mother": 81,
    "brother": 81,
    "father": 83,
    "water": 83,
    "today": 84,
    "very": 84,
    "above": 85,
    "among": 85,
    "again": 86,
    "against": 86,
    "always": 87,
    "almost": 87,
    "poor": 89,
    "door": 89,
    "floor": 89,
    "won": 90,
    "son": 90,
    "month": 91,
    "hour": 93,
    "minute": 93,
    "monday": 94,
    "wednesday": 94,
    "february": 95,
    "eye": 96,
    "heart": 96,
    "about": 98,
    "answer": 99,
    "honest": 101,
    "honor": 101,
    "truth": 103,
    "truly": 103,
    "busy": 107,
    "build": 108,
    "built": 108,
    "sure": 109,
    "laugh": 110,
    "whom": 111,
    "whose": 112,
    "toward": 113,
    "through": 114,
    "learn": 115,
    "earth": 115,
    "early": 115,
    "buy": 116,
    "guy": 116,
    "guess": 117,
    "guest": 117,
    "guide": 117,
    "young": 118,
    "touch": 118,
    "enough": 119,
    "tough": 119,
    "rough": 119,
}
