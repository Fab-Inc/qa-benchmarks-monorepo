"""Extract the generated text from an LLM response to a benchmark item."""

import re
from typing import NamedTuple


class ExtractResult(NamedTuple):
    """The extracted text plus a flag recording whether the tags were found."""

    text: str
    delimiter_found: bool


# Captures the content of the first <text>...</text> pair. The inner group is
#   non-greedy so a response containing several pairs yields the first complete
#   one rather than spanning from the first opening tag to the last closing tag.
# DOTALL lets the text span newlines; IGNORECASE tolerates <TEXT> / <Text>.
_TEXT_TAG_RE = re.compile(
    r"<\s*text\s*>(.*?)<\s*/\s*text\s*>", re.DOTALL | re.IGNORECASE
)


def extract_generated_text(response: str) -> ExtractResult:
    """Pull the generated text out of a model response.

    The prompt asks the model to wrap its text in ``<text>...</text>`` so the
    text can be separated from any commentary deterministically.

    Returns an ``ExtractResult``: the extracted text, plus ``delimiter_found``
    which is False when the tags were absent and the whole response was
    returned instead.

    A missing delimiter is a scoring event, not just a warning. Scoring the
    whole response would charge the model for preamble such as "Sure! Here is a
    story" — foreign words that are non-decodable in the target language — so
    the caller treats a format failure as a zero rather than silently grading
    the preamble. See ``scoring.add_per_item_decodability``.
    """
    match = _TEXT_TAG_RE.search(response)
    if match:
        return ExtractResult(match.group(1).strip(), True)
    return ExtractResult(response.strip(), False)
