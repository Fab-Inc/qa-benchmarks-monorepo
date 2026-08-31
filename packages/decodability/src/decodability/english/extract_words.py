"""Word extractor for English."""

import re

APOSTROPHE_RE = r"['’′´❜]"
INTERIOR_APOSTROPHE_RE = re.compile(rf"(?<=[A-Za-z]){APOSTROPHE_RE}(?=[A-Za-z])")
WORD_RE = re.compile(rf"[A-Za-z]+(?:{APOSTROPHE_RE}[A-Za-z]+)*")


def extract_words_english(text: str) -> list[str]:
    """Lenient English word extractor.

    Preserves apostrophes inside contractions (``don't``, ``it's``, ``we're``)
    and normalizes fancy apostrophe forms (``’′´❜``) to the standard ``'``.
    Standalone quote marks and other punctuation are dropped.
    """
    return [INTERIOR_APOSTROPHE_RE.sub("'", w) for w in WORD_RE.findall(text)]
