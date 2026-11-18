"""Parse a model response into question and answer pairs."""

import json
import re

from reading_comprehension_question_generation.models import QAPair
from reading_comprehension_question_generation.prompts import NUM_PAIRS

# Parse statuses. A response that was never parsed because the call failed gets
# CALL_ERROR in pairs.py, not here.
OK = "ok"
COUNT_MISMATCH = "count_mismatch"
PARSE_FAILED = "parse_failed"
NO_PAIRS = "no_pairs"

# Reasoning text some models put in the output before the answer. Removed first so a
# bracket inside it cannot be mistaken for the answer array.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

_DECODER = json.JSONDecoder()


def _json_arrays(text: str) -> list[list]:
    """Return every top-level JSON array found anywhere in ``text``, in order.

    Every ``[`` is tried as the start of an array. Code fences, preamble and trailing
    notes are skipped over, because only the array itself has to decode. A ``[`` that
    sits inside an array already decoded is skipped, so a nested list is never
    returned on its own.
    """
    arrays: list[list] = []
    position = 0
    while (start := text.find("[", position)) != -1:
        try:
            value, end = _DECODER.raw_decode(text, start)
        except json.JSONDecodeError:
            position = start + 1  # not an array here; try the next bracket
            continue
        if isinstance(value, list):
            arrays.append(value)
        position = end  # skip past it, nested brackets included
    return arrays


def _valid_pair(element: object) -> QAPair | None:
    """Return a pair if ``element`` is an object with non-empty text question/answer."""
    if not isinstance(element, dict):
        return None
    question, answer = element.get("question"), element.get("answer")
    if not isinstance(question, str) or not isinstance(answer, str):
        return None
    question, answer = question.strip(), answer.strip()
    if not question or not answer:
        return None
    return QAPair(question=question, answer=answer)


def parse_qa_pairs(response: str | None) -> tuple[list[QAPair], str]:
    """Extract the Q&A pairs from one raw model response.

    Returns the pairs and a status:

    - ``parse_failed``: no JSON array could be decoded (including an empty response);
    - ``no_pairs``: an array was found but none of its elements is a valid pair;
    - ``count_mismatch``: the number of valid pairs differs from ``NUM_PAIRS``;
    - ``ok``: exactly ``NUM_PAIRS`` valid pairs.

    Elements that are not objects with non-empty string ``question`` and ``answer``
    are dropped, extra keys are ignored, and nothing is guessed from free text. Pairs
    are kept on a count mismatch so the mismatch can be inspected.
    """
    if not response or not response.strip():
        return [], PARSE_FAILED

    arrays = _json_arrays(_THINK_BLOCK.sub("", response))
    if not arrays:
        return [], PARSE_FAILED

    # Models put the final answer last, so take the last array that holds a valid pair.
    # A trailing "[1]" footnote is valid JSON but must not hide the real answer.
    # If no array holds a pair, the response is no_pairs.
    for array in reversed(arrays):
        pairs = [p for element in array if (p := _valid_pair(element)) is not None]
        if pairs:
            return pairs, OK if len(pairs) == NUM_PAIRS else COUNT_MISMATCH
    return [], NO_PAIRS
