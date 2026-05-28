"""Short PrimerPro decodability tutorial.

Run from the repository root with:

    uv run --package primerpro python \
        benchmarks/phonics_benchmark/scripts/decodable_text_demo.py

PrimerPro decodability depends on two project-specific inputs:

1. A grapheme inventory: the written units PrimerPro should segment words into.
2. A taught-grapheme list: the graphemes the learner is assumed to know.
"""

from __future__ import annotations

from primerpro import (
    BuildableWordSearchTD,
    Consonant,
    GraphemeInventory,
    Paragraph,
    ResidueSearch,
    Settings,
    SightWords,
    TextData,
    Vowel,
)

ENGLISH_CONSONANTS = [
    "b",
    "c",
    "ch",
    "ck",
    "d",
    "f",
    "g",
    "h",
    "j",
    "k",
    "l",
    "m",
    "n",
    "ng",
    "p",
    "qu",
    "r",
    "s",
    "sh",
    "t",
    "th",
    "v",
    "w",
    "wh",
    "x",
    "y",
    "z",
]

ENGLISH_VOWELS = [
    "a",
    "e",
    "i",
    "o",
    "u",
    "ai",
    "ay",
    "ea",
    "ee",
    "igh",
    "oa",
    "oo",
    "ar",
    "er",
    "ir",
    "or",
    "ur",
]

EARLY_TAUGHT_GRAPHEMES = [
    "s",
    "a",
    "t",
    "m",
    "sh",
    "i",
    "p",
    "n",
    "b",
]

SIGHT_WORDS = ["the"]

SAMPLE_TEXT = "the sat ship queen night beach. mat thin sing."


def create_english_settings() -> Settings:
    settings = Settings()
    settings.option_settings.max_size_grapheme = 4
    settings.option_settings.general_punct = " ,:;-\"'"
    settings.option_settings.ending_punct = ".?!"

    inventory = GraphemeInventory(settings)
    for symbol in ENGLISH_CONSONANTS:
        consonant = Consonant(symbol)
        consonant.uppercase = symbol.upper()
        inventory.add_consonant(consonant)
    for symbol in ENGLISH_VOWELS:
        vowel = Vowel(symbol)
        vowel.uppercase = symbol.upper()
        inventory.add_vowel(vowel)

    settings.grapheme_inventory = inventory
    settings.sight_words = SightWords(settings)
    for word in SIGHT_WORDS:
        settings.sight_words.add_word(word)
    return settings


def parse_text(settings: Settings, text: str) -> TextData:
    text_data = TextData(settings)
    text_data.add_paragraph(Paragraph(text, settings))
    return text_data


def grapheme_symbols(word) -> list[str]:
    return [word.get_grapheme(index).symbol for index in range(word.grapheme_count())]


def missing_graphemes(word, taught_graphemes: list[str]) -> list[str]:
    taught = set(taught_graphemes)
    return [symbol for symbol in grapheme_symbols(word) if symbol not in taught]


def print_word_by_word_report(text_data: TextData, taught_graphemes: list[str]) -> None:
    print("Word-by-word decodability")
    print("------------------------")
    for word in text_data.iter_words():
        symbols = grapheme_symbols(word)
        buildable = word.is_buildable_word(taught_graphemes)
        if buildable:
            status = "decodable"
        elif word.is_sight_word():
            status = "sight word exception"
        else:
            status = "not decodable"
        missing = missing_graphemes(word, taught_graphemes)
        missing_note = f" missing={','.join(missing)}" if missing else ""
        print(f"{word.display_word:>6}  {symbols!s:<24} {status}{missing_note}")


def print_search_reports(
    settings: Settings,
    text_data: TextData,
    taught_graphemes: list[str],
) -> None:
    buildable = BuildableWordSearchTD(1, settings)
    buildable.graphemes = taught_graphemes
    buildable.execute_buildable_word_search(text_data)

    residue = ResidueSearch(1, settings)
    residue.graphemes = taught_graphemes
    residue.ignore_sight_words = True
    residue.execute_residue_search(text_data)

    print()
    print("PrimerPro search objects")
    print("------------------------")
    print(f"Buildable words ({buildable.search_count}):")
    print(buildable.search_results)
    print()
    print(
        "Residue / not-yet-decodable words, "
        f"ignoring sight words ({residue.search_count}):"
    )
    print(residue.search_results)


def main() -> None:
    settings = create_english_settings()
    text_data = parse_text(settings, SAMPLE_TEXT)

    print(f"Text: {SAMPLE_TEXT}")
    print(f"Taught graphemes: {', '.join(EARLY_TAUGHT_GRAPHEMES)}")
    print(f"Sight words: {', '.join(SIGHT_WORDS)}")
    print()

    print_word_by_word_report(text_data, EARLY_TAUGHT_GRAPHEMES)
    print_search_reports(settings, text_data, EARLY_TAUGHT_GRAPHEMES)


if __name__ == "__main__":
    main()
