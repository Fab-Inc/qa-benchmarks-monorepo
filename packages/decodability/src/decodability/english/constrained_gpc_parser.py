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
    Path(__file__).resolve().parent / "data" / "ufli_model_word_list.json"
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
