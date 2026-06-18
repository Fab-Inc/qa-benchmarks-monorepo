"""Tokenizer for Kiswahili."""

import re

WORD_RE = re.compile(r"(?:[Nn][Gg]'|[A-Za-z])+")


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text)
