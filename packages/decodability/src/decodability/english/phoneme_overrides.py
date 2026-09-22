"""Manual pronunciation overrides for CMUdict.

Keys are lowercased words. Values are lists of ARPABET pronunciation variants,
matching CMUdict's own return shape (``list[list[str]]``). Overrides pass
through the same normalization pipeline as CMU output — including stress-digit
handling and dialect fixups — so entries stay in ARPABET rather than
post-normalization IPA.

Use to:

- Force a specific variant when CMU offers multiple (e.g. drop /aɪðɚ/ for
  ``either`` and keep only /iːðɚ/).
- Supply a pronunciation for a word that is out-of-vocabulary in CMUdict.
- Correct a CMU pronunciation that doesn't match the curriculum's target.
"""

CMUDICT_OVERRIDES: dict[str, list[list[str]]] = {
    "fine": [["F", "AY1", "N"]],
    # CMUdict lists no /j/ (yod) reading of "beyond"; these variants spell the
    # yod with y so the word parses as b + e + y->j + o + n + d. CMUdict's own
    # variants (which parse via ey->i) are kept alongside.
    "beyond": [
        ["B", "IH0", "Y", "AA1", "N", "D"],
        ["B", "IY0", "Y", "AA1", "N", "D"],
        ["B", "IY0", "Y", "AO1", "N", "D"],
        ["B", "IH0", "AA1", "N", "D"],
        ["B", "IY2", "AO1", "N", "D"],
        ["B", "IH0", "AO1", "N", "D"],
    ],
    # Out-of-vocabulary words from the UFLI decodables corpus — mostly
    # character names plus a few real words CMUdict lacks. Possessive/'s
    # forms are derived automatically by the s-suffix fallback.
    "akeem": [["AH0", "K", "IY1", "M"]],
    "backflips": [["B", "AE1", "K", "F", "L", "IH2", "P", "S"]],
    "bim": [["B", "IH1", "M"]],
    "cassie": [["K", "AE1", "S", "IY0"]],
    "cath": [["K", "AE1", "TH"]],
    "corndog": [["K", "AO1", "R", "N", "D", "AO2", "G"]],
    "duffle": [["D", "AH1", "F", "AH0", "L"]],
    "flicked": [["F", "L", "IH1", "K", "T"]],
    "greyson": [["G", "R", "EY1", "S", "AH0", "N"]],
    "jace": [["JH", "EY1", "S"]],
    "jax": [["JH", "AE1", "K", "S"]],
    "jayvon": [["JH", "EY0", "V", "AA1", "N"]],
    "jaz": [["JH", "AE1", "Z"]],
    "kaylen": [["K", "EY1", "L", "AH0", "N"]],
    "kenzer": [["K", "EH1", "N", "Z", "ER0"]],
    "lunchbox": [["L", "AH1", "N", "CH", "B", "AA2", "K", "S"]],
    "palish": [["P", "EY1", "L", "IH0", "SH"]],
    "pim": [["P", "IH1", "M"]],
    "preheat": [["P", "R", "IY2", "HH", "IY1", "T"]],
    "premade": [["P", "R", "IY2", "M", "EY1", "D"]],
    "powerfully": [["P", "AW1", "ER0", "F", "AH0", "L", "IY0"]],
    "rashawn": [["R", "AH0", "SH", "AA1", "N"], ["R", "AH0", "SH", "AO1", "N"]],
    "retie": [["R", "IY2", "T", "AY1"]],
    "retold": [["R", "IY2", "T", "OW1", "L", "D"]],
    "salma": [["S", "AA1", "L", "M", "AH0"]],
    "saturday": [["S", "AE1", "T", "ER0", "D", "EY2"]],
    "shirly": [["SH", "ER1", "L", "IY0"]],
    # CMUdict gives "sloth" only /sloʊθ/; the corpus target is the short-o
    # reading. "sloths" is OOV and inherits both readings (+/s/, +/z/)
    # through the s-suffix fallback.
    "sloth": [["S", "L", "OW1", "TH"], ["S", "L", "AA1", "TH"]],
    "smirked": [["S", "M", "ER1", "K", "T"]],
    "squealed": [["S", "K", "W", "IY1", "L", "D"]],
    "squirmy": [["S", "K", "W", "ER1", "M", "IY0"]],
    "trilingual": [["T", "R", "AY0", "L", "IH1", "NG", "G", "W", "AH0", "L"]],
    "unscrewed": [["AH0", "N", "S", "K", "R", "UW1", "D"]],
    "useable": [["Y", "UW1", "Z", "AH0", "B", "AH0", "L"]],
    "yummiest": [["Y", "AH1", "M", "IY0", "AH0", "S", "T"]],
}
