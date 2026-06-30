"""Word extractor for Kiswahili."""

import re

WORD_RE = re.compile(r"(?:[Nn][Gg]'|[A-Za-z])+")


def extract_kiswahili_words(text: str) -> list[str]:
    """Lenient Kiswahili word extractor. We intentionally don't limit to only valid
    Kiswahili graphemes, as we want to be able to extract words from any text.
    However, we explicitly include "ng'" as a valid character sequence so that the
    apostrophe in this case won't split up a word.
    """
    return WORD_RE.findall(text)
