"""Word extractor for Kiswahili."""

import re

APOSTROPHE_RE = r"['’′´❜]"
VELAR_NASAL_APOSTROPHE_RE = re.compile(rf"(?<=[Nn][Gg]){APOSTROPHE_RE}")
WORD_RE = re.compile(rf"(?:[Nn][Gg]{APOSTROPHE_RE}|[A-Za-z])+")


def extract_words_kiswahili(text: str) -> list[str]:
    """Lenient Kiswahili word extractor. We intentionally don't limit to only valid
    Kiswahili graphemes, as we want to be able to extract words from any text.
    However, we explicitly include "ng'" as a valid character sequence so that the
    apostrophe in this case won't split up a word.
    """
    return [VELAR_NASAL_APOSTROPHE_RE.sub("'", w) for w in WORD_RE.findall(text)]
