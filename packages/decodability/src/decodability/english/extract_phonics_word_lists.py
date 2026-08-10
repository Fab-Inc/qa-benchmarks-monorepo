"""Extract per-lesson word lists from the Ultimate Phonics PDF."""

import json
import re
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

from pypdf import PdfReader

_PDF_URL = (
    "https://s3.amazonaws.com/sl-us-standard/resources/ultimate-phonics-word-lists.pdf"
)
_OUT_PATH = (
    Path(__file__).resolve().parent / "data" / "ultimate_phonics_word_lists.json"
)
_HEADER_RE = re.compile(r"LESSON\s+(\d+)\s*-\s*Word List", re.IGNORECASE)


def extract_word_lists() -> dict[int, list[str]]:
    """Return ``{lesson_number: [words, ...]}`` from the Ultimate Phonics PDF.

    Fetches the PDF from ``_PDF_URL`` and scans every page for the header
    ``LESSON <n>  -  Word List``. Pulls whitespace-separated tokens between
    that header and the phonics-category footer line (the last non-empty
    line of each page). Multi-page lessons are merged in page order.
    """
    with urlopen(_PDF_URL) as response:
        pdf_bytes = response.read()
    reader = PdfReader(BytesIO(pdf_bytes))
    lessons: dict[int, list[str]] = defaultdict(list)
    for page in reader.pages:
        text = page.extract_text() or ""
        match = _HEADER_RE.search(text)
        if not match:
            continue
        lesson = int(match.group(1))
        body = text[match.end() :]
        # The last non-empty line is always a phonics-category breadcrumb
        # (e.g. "Closed Syllables »  Short Vowels »  a = / ă/" or a shorter
        # top-level label like "Silent e at end") — drop it.
        lines = [line for line in body.splitlines() if line.strip()]
        for line in lines[:-1]:
            for token in line.split():
                if any(c.isalpha() for c in token):
                    lessons[lesson].append(token)
    return dict(sorted(lessons.items()))


def write_word_lists() -> None:
    """Write the extracted word lists to a JSON file in the data directory."""
    lessons = extract_word_lists()
    serializable = {str(lesson): words for lesson, words in lessons.items()}
    with open(_OUT_PATH, "w") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    write_word_lists()
