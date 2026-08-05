"""Report words from the Ultimate Phonics lessons the parser cannot cover."""

import json

from decodability.english.definitions import UPL_GPC_INVENTORY
from decodability.english.extract_phonemes import get_word_phonemes
from decodability.english.gpc_parser import GPC, parse_gpcs
from decodability.english.paths import DATA_DIR

INVENTORY = tuple(
    GPC(gpc_id, grapheme, phonemes) for gpc_id, grapheme, phonemes in UPL_GPC_INVENTORY
)

_WORD_LISTS_PATH = DATA_DIR / "ultimate_phonics_word_lists.json"


def main() -> None:
    with open(_WORD_LISTS_PATH) as f:
        lessons: dict[str, list[str]] = json.load(f)

    missing_alignment = 0
    for lesson_str in sorted(lessons, key=int):
        lesson = int(lesson_str)
        for word in lessons[lesson_str]:
            try:
                phonemes = get_word_phonemes(word.lower())
            except ValueError:
                missing_alignment += 1
                continue
            parses = parse_gpcs(word.lower(), phonemes, INVENTORY)
            if not parses:
                print(f"Lesson {lesson}: {word}")
                continue
    if missing_alignment:
        print(f"\n({missing_alignment} words skipped: not in alignments)")


if __name__ == "__main__":
    main()
