"""Report per-lesson parser coverage over the UFLI texts corpus.

Reads ``ufli_texts_words.json`` (a mapping from GPC-lesson ID to the words
that introduce it) and, for each word, prints either the ``(grapheme,
phonemes)`` pairs of every valid parse, an ``OOV in CMUdict`` line, or a
``no parse`` line. A lesson whose example word can't be parsed signals a
gap in the inventory or a parser bug.
"""

import json
from pathlib import Path

from decodability.english.definitions import UFLI_GPC_INVENTORY
from decodability.english.extract_phonemes import get_word_phonemes
from decodability.english.gpc_parser import parse_gpcs

INVENTORY = UFLI_GPC_INVENTORY

_WORD_LISTS_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "decodability"
    / "english"
    / "data"
    / "ufli_texts_words.json"
)


def main() -> None:
    with open(_WORD_LISTS_PATH, encoding="utf-8") as f:
        gpc_ids: dict[str, list[str]] = json.load(f)

    for gpc in gpc_ids:
        for word in gpc_ids[gpc]:
            try:
                variants = get_word_phonemes(word.lower())
            except KeyError:
                print(f"GPC {gpc}: {word} — OOV in CMUdict")
                continue

            any_parse = False
            for variant_index, variant in enumerate(variants):
                for parse_index, parse in enumerate(
                    parse_gpcs(word.lower(), variant, INVENTORY)
                ):
                    any_parse = True
                    # Mirror scorers.py: the ``(grapheme, phonemes)`` pair is
                    # what the scorer checks against the student's taught set.
                    pairs = [
                        (match.gpc.grapheme, match.gpc.phonemes) for match in parse
                    ]
                    print(
                        f"GPC {gpc}: {word} "
                        f"[variant {variant_index}, parse {parse_index}] {pairs}"
                    )

            if not any_parse:
                print(f"GPC {gpc}: {word} — no parse")


if __name__ == "__main__":
    main()
