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

CMUDICT_OVERRIDES: dict[str, list[list[str]]] = {}
