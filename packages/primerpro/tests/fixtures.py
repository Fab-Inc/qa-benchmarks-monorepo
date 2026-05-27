from primerpro import (
    Consonant,
    GraphemeInventory,
    GraphemeTaughtOrder,
    Paragraph,
    Settings,
    SightWords,
    TextData,
    Tone,
    Vowel,
    Word,
    WordList,
)


def create_bantu_like_settings() -> Settings:
    return create_settings(
        ["b", "d", "g", "k", "m", "mb", "n", "ng", "p", "s", "t", "z"],
        ["a", "aa", "e", "i", "o", "u"],
    )


def create_english_settings() -> Settings:
    return create_settings(
        [
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
        ],
        [
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
        ],
    )


def create_spanish_like_settings() -> Settings:
    return create_settings(
        [
            "b",
            "c",
            "ch",
            "d",
            "f",
            "g",
            "gu",
            "h",
            "j",
            "l",
            "ll",
            "m",
            "n",
            "ñ",
            "p",
            "q",
            "qu",
            "r",
            "rr",
            "s",
            "t",
            "v",
            "y",
            "z",
        ],
        ["a", "e", "i", "o", "u", "á", "é", "í", "ó", "ú"],
    )


def create_overlapping_grapheme_settings() -> Settings:
    return create_settings(["n", "ng", "nga", "g"], ["a", "aa"])


def create_tone_settings() -> Settings:
    settings = create_settings(["m", "n", "t", "s"], ["a", "i", "u"])
    inventory = settings.grapheme_inventory
    a = inventory.get_vowel(inventory.find_vowel_index("a"))

    high = Tone("á")
    high.uppercase = "Á"
    high.level = "H"
    high.tone_bearing_unit = a
    high.teaching_order = 1
    inventory.add_tone(high)

    low = Tone("à")
    low.uppercase = "À"
    low.level = "L"
    low.tone_bearing_unit = a
    low.teaching_order = 2
    inventory.add_tone(low)

    return settings


def taught(*graphemes: str) -> list[str]:
    return list(graphemes)


def grapheme_symbols(word: Word) -> list[str]:
    return [word.get_grapheme(i).symbol for i in range(word.grapheme_count())]


def syllable_symbols(word: Word) -> list[str]:
    return [
        word.get_syllable(i).get_syllable_in_lower_case()
        for i in range(word.syllable_count())
    ]


def text_data_from_paragraphs(settings: Settings, *paragraphs: str) -> TextData:
    text_data = TextData(settings)
    text_data.paragraphs = []
    for paragraph in paragraphs:
        text_data.add_paragraph(Paragraph(paragraph, settings))
    return text_data


def word_list_from_words(settings: Settings, *words: str) -> WordList:
    word_list = WordList(settings)
    for word in words:
        word_list.add_word(Word(word, settings))
    return word_list


def assert_contains_highlighted(symbol: str, haystack: str) -> None:
    from primerpro import HIGHLIGHT_OFF, HIGHLIGHT_ON

    assert f"{HIGHLIGHT_ON}{symbol}{HIGHLIGHT_OFF}" in haystack


def create_settings(consonants: list[str], vowels: list[str]) -> Settings:
    settings = Settings()
    settings.option_settings.max_size_grapheme = 4
    settings.option_settings.general_punct = " ,:;-\"'"
    settings.option_settings.ending_punct = ".?!"

    inventory = GraphemeInventory(settings)
    for symbol in consonants:
        consonant = Consonant(symbol)
        consonant.uppercase = symbol.upper()
        inventory.add_consonant(consonant)
    for symbol in vowels:
        vowel = Vowel(symbol)
        vowel.uppercase = symbol.upper()
        inventory.add_vowel(vowel)

    settings.grapheme_inventory = inventory
    settings.graphemes_taught = GraphemeTaughtOrder(settings)
    settings.sight_words = SightWords(settings)
    settings.text_data = TextData(settings)
    settings.word_list = WordList(settings)
    return settings
