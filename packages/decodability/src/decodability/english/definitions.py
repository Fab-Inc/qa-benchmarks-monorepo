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
"""

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

UFLI_GPC_INVENTORY: list[tuple[str, str, tuple[str, ...]]] = [
    ("G001", "a", ("æ",)),
    ("G002", "m", ("m",)),
    ("G003", "s", ("s",)),
    ("G004", "t", ("t",)),
    ("G005", "p", ("p",)),
    ("G006", "f", ("f",)),
    ("G007", "i", ("ɪ",)),
    ("G008", "n", ("n",)),
    ("G010", "o", ("ɑ",)),
    ("G011", "d", ("d",)),
    ("G012", "c", ("k",)),
    ("G013", "u", ("ʌ",)),
    ("G014", "g", ("g",)),
    ("G015", "b", ("b",)),
    ("G016", "e", ("ɛ",)),
    ("G018", "s", ("z",)),
    ("G019", "k", ("k",)),
    ("G020", "h", ("h",)),
    ("G021", "r", ("r",)),
    ("G022", "l", ("l",)),
    ("G023", "w", ("w",)),
    ("G024", "j", ("dʒ",)),
    ("G025", "y", ("j",)),
    ("G026", "x", ("k", "s")),
    ("G027", "qu", ("k", "w")),
    ("G028", "v", ("v",)),
    ("G029", "z", ("z",)),
    ("G030", "ff", ("f",)),
    ("G031", "ll", ("l",)),
    ("G032", "ss", ("s",)),
    ("G033", "zz", ("z",)),
    ("G034", "all", ("ɔ", "l")),
    ("G035", "oll", ("oʊ", "l")),
    ("G036", "ull", ("ʊ", "l")),
    ("G037", "ck", ("k",)),
    ("G038", "sh", ("ʃ",)),
    ("G039", "th", ("ð",)),
    ("G040", "th", ("θ",)),
    ("G041", "ch", ("tʃ",)),
    ("G042", "wh", ("w",)),
    ("G043", "ph", ("f",)),
    ("G044", "ng", ("ŋ",)),
    ("G045", "nk", ("ŋ", "k")),
    ("G046", "a-e", ("eɪ",)),  # split digraph
    ("G047", "i-e", ("aɪ",)),  # split digraph
    ("G048", "o-e", ("oʊ",)),  # split digraph
    ("G049", "e-e", ("i",)),  # split digraph
    ("G050", "u-e", ("u",)),  # split digraph
    ("G051", "u-e", ("j", "u")),  # split digraph (yod)
    ("G052", "ce", ("s",)),  # soft c + silent e (_ce)
    ("G053", "ge", ("dʒ",)),  # soft g + silent e (_ge)
    ("G054", "ve", ("v",)),  # words don't end in v (-ve)
    ("G055", "es", ("ə", "z")),
    ("G056", "ed", ("t",)),
    ("G057", "ed", ("d",)),
    ("G058", "ed", ("ə", "d")),
    ("G059", "ing", ("ɪ", "ŋ")),
    ("G060", "a", ("eɪ",)),  # open syllable
    ("G061", "e", ("i",)),  # open syllable
    ("G062", "i", ("aɪ",)),  # open syllable
    ("G063", "o", ("oʊ",)),  # open syllable
    ("G064", "u", ("j", "u")),  # open syllable
    ("G065", "u", ("u",)),  # open syllable
    ("G066", "tch", ("tʃ",)),
    ("G067", "dge", ("dʒ",)),
    ("G068", "ild", ("aɪ", "l", "d")),
    ("G069", "old", ("oʊ", "l", "d")),
    ("G070", "ind", ("aɪ", "n", "d")),
    ("G071", "olt", ("oʊ", "l", "t")),
    ("G072", "ost", ("oʊ", "s", "t")),
    ("G073", "y", ("aɪ",)),
    ("G074", "y", ("i",)),
    ("G075", "le", ("ə", "l")),  # -le syllable ending
    ("G076", "ar", ("ɑ", "r")),
    ("G077", "or", ("ɔ", "r")),
    ("G078", "ore", ("ɔ", "r")),
    ("G079", "er", ("ə", "r")),
    ("G080", "ir", ("ə", "r")),
    ("G081", "ur", ("ə", "r")),
    ("G082", "or", ("ə", "r")),  # or after w (unstressed)
    ("G083", "ai", ("eɪ",)),
    ("G084", "ay", ("eɪ",)),
    ("G085", "ee", ("i",)),
    ("G086", "ea", ("i",)),
    ("G087", "ey", ("i",)),
    ("G088", "oa", ("oʊ",)),
    ("G089", "ow", ("oʊ",)),
    ("G090", "oe", ("oʊ",)),
    ("G091", "ie", ("aɪ",)),
    ("G092", "igh", ("aɪ",)),
    ("G093", "u", ("ʊ",)),
    ("G094", "oo", ("ʊ",)),
    ("G095", "oo", ("u",)),
    ("G096", "ew", ("u",)),
    ("G097", "ui", ("u",)),
    ("G098", "ue", ("u",)),
    ("G099", "au", ("ɔ",)),
    ("G100", "aw", ("ɔ",)),
    ("G101", "augh", ("ɔ",)),
    ("G102", "ea", ("ɛ",)),
    ("G103", "a", ("ɑ",)),  # a after w/qu
    ("G104", "oi", ("ɔɪ",)),
    ("G105", "oy", ("ɔɪ",)),
    ("G106", "ou", ("aʊ",)),
    ("G107", "ow", ("aʊ",)),
    ("G108", "kn", ("n",)),
    ("G109", "wr", ("r",)),
    ("G110", "mb", ("m",)),
    ("G112", "est", ("ə", "s", "t")),
    ("G113", "ly", ("l", "i")),
    ("G114", "less", ("l", "ə", "s")),
    ("G115", "ful", ("f", "ə", "l")),
    ("G116", "un", ("ʌ", "n")),
    ("G117", "pre", ("p", "r", "i")),
    ("G118", "re", ("r", "i")),
    ("G119", "dis", ("d", "ɪ", "s")),
    ("G120", "ied", ("aɪ", "d")),
    ("G121", "ies", ("aɪ", "z")),
    ("G122", "ies", ("i", "z")),
    ("G123", "ar", ("ə", "r")),  # unstressed ar: beggar, dollar
    ("G125", "air", ("ɛ", "r")),
    ("G126", "are", ("ɛ", "r")),
    ("G127", "ear", ("ɛ", "r")),
    ("G128", "ear", ("ɪ", "r")),
    ("G129", "ei", ("eɪ",)),
    ("G130", "eigh", ("eɪ",)),
    ("G131", "ey", ("eɪ",)),
    ("G132", "ea", ("eɪ",)),
    ("G133", "aigh", ("eɪ",)),
    ("G134", "ew", ("j", "u")),
    ("G135", "eu", ("j", "u")),
    ("G136", "ue", ("j", "u")),
    ("G137", "ou", ("u",)),
    ("G138", "ough", ("ɔ",)),
    ("G139", "ough", ("oʊ",)),
    ("G140", "c", ("s",)),  # c before e, i, y
    ("G141", "g", ("dʒ",)),  # g before e, i, y
    ("G142", "ch", ("ʃ",)),
    ("G143", "ch", ("k",)),
    ("G144", "gn", ("n",)),
    ("G145", "gh", ("g",)),
    ("G146", "st", ("s",)),  # silent t
    ("G147", "tion", ("ʃ", "ə", "n")),
    ("G148", "sion", ("ʒ", "ə", "n")),
    ("G149", "sion", ("ʃ", "ə", "n")),
    ("G150", "ture", ("tʃ", "ə", "r")),
    ("G151", "ist", ("ɪ", "s", "t")),
    ("G152", "ish", ("ɪ", "ʃ")),
    ("G153", "ness", ("n", "ə", "s")),
    ("G154", "ment", ("m", "ə", "n", "t")),
    ("G155", "able", ("ə", "b", "ə", "l")),
    ("G156", "ible", ("ə", "b", "ə", "l")),
    ("G157", "bi", ("b", "aɪ")),
    ("G158", "tri", ("t", "r", "aɪ")),
    ("G159", "uni", ("j", "u", "n", "i")),
    ("G160", "bb", ("b",)),
    ("G161", "cc", ("k",)),
    ("G162", "dd", ("d",)),
    ("G163", "gg", ("g",)),
    ("G164", "mm", ("m",)),
    ("G165", "nn", ("n",)),
    ("G166", "pp", ("p",)),
    ("G167", "tt", ("t",)),
    ("G168", "o", ("ɔ",)),
    ("G169", "au", ("ɑ",)),
    ("G170", "augh", ("ɑ",)),
    ("G171", "ough", ("ɑ",)),  # resolve cot-caught merger
    ("G172", "ssion", ("ʃ", "ə", "n")),
    ("G173", "o-e", ("ʌ",)),
]

# Student-facing form: the phoneme side is a single joined IPA string, so a
# multi-phoneme GPC like ``("x", ("k", "s"))`` reads as ``("x", "ks")``. Used
# by ``EnglishStudentKnowledge`` validation and by the scorer to check parser
# matches against the student's taught set.
UFLI_GPC_PAIRS_FLAT: frozenset[tuple[str, str]] = frozenset(
    (grapheme, "".join(phonemes)) for _, grapheme, phonemes in UFLI_GPC_INVENTORY
)
