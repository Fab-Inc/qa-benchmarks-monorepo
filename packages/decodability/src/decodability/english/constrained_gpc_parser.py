"""
The script identifies words from the Ultimate Phonics lessons that the parser
cannot cover. This is useful for surfacing gaps in the GPC inventory or issues
with the parser's implementation.
"""

import json
from pathlib import Path

from decodability.english.definitions import UFLI_GPC_INVENTORY
from decodability.english.extract_phonemes import get_word_phonemes
from decodability.english.gpc_parser import GPC, parse_gpcs

INVENTORY = tuple(
    GPC(gpc_id, grapheme, phonemes) for gpc_id, grapheme, phonemes in UFLI_GPC_INVENTORY
)

_WORD_LISTS_PATH = (
    Path(__file__).resolve().parent / "data" / "ultimate_phonics_word_lists.json"
)


def main() -> None:
    with open(_WORD_LISTS_PATH) as f:
        gpc_ids: dict[str, list[str]] = json.load(f)

    for gpc in gpc_ids:
        for word in gpc_ids[gpc]:
            try:
                variants = get_word_phonemes(word.lower())
            except KeyError:
                print(f"GPC {gpc}: {word} — OOV in CMUdict")
                continue
            if not any(
                parse_gpcs(word.lower(), variant, INVENTORY) for variant in variants
            ):
                print(f"GPC {gpc}: {word} — no parse")


if __name__ == "__main__":
    main()
