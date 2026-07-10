"""Functions for parsing LLM responses to benchmark items into a list of words."""

import re
from typing import NamedTuple


class ParseResult(NamedTuple):
    """The parsed word list plus a flag recording whether parsing succeeded."""

    words: list[str]
    parse_ok: bool


def parse_word_list(response: str) -> ParseResult:
    """Parse a model response to a benchmark item into a pure list of words (strings).

    A typical model response is expected to include a JSON array of words, e.g.:
    ["apple", "banana", "pear"]

    The model may also include extra text before or after the array,
    or may not format the array correctly, so this function attempts to extract
    the list of words from the response in a forgiving way.

    Returns a ``ParseResult``: the list of words, plus ``parse_ok`` which is
    False when the response did not match the expected pattern and the words
    were instead scraped from the full response text (so the caller may want
    to flag the result or retry the LLM call).
    """

    if re.search(r"\ball\b", response, re.IGNORECASE):
        print(
            "WARNING: model response contains the word 'all' so may be "
            "erroneously trying to say 'all words decodable' or similar"
        )

    # Stage 1 of parsing: locate the array region of the response and capture
    #   its contents, discarding any surrounding prose the model added.
    array_contents_pattern = r"^.*?\[([^\]\[]*)\][^\[]*?$"
    ## logic of regex:
    #   start of string - ^
    #   followed by zero or more of anything (non-greedy, specified by the ?
    #     after *) - .*?
    #   followed by a left bracket - \[
    #   followed by zero or more not-right-or-left-brackets (in a capturing group)
    #     where zero allows an empty list "[]" to match - ([^\]\[]*)
    #   followed by a right bracket - \]
    #   followed by zero or more non-left-brackets (non-greedy), which ensures
    #     the captured array is the last one in the string - [^\[]*?
    #   followed by end of string - $
    # The brackets are mandatory so that failure to match means "no array found",
    #   which triggers the fallback (and parse_ok=False) below.

    # flag DOTALL ensures that newlines are included in .
    array_contents_re = re.compile(array_contents_pattern, flags=re.DOTALL)

    match = array_contents_re.match(response)
    if match:
        inner_text = match.group(1)
        parse_ok = True
    else:
        # Fall back to scraping words from the full response text; parse_ok
        #   lets the caller flag the result or retry the LLM call.
        inner_text = response
        parse_ok = False

        # TODO: maybe use a secondary fallback regex to match a list of words that are
        #   not in brackets, e.g. "apple, banana, pear" but where precisely one word
        #   allowed between each comma or else marked as parse failure.

    # Stage 2 of parsing: extract the individual words from the bracket
    #   contents in one pass.
    # A word is a run of letters, optionally joined by internal apostrophes or
    #   hyphens (e.g. "don't", "two-word"). Quotes, commas and whitespace
    #   around each word are simply never matched, so '"a", 'b',c' all parse
    #   the same. Because the apostrophe/hyphen must be *followed by a letter*,
    #   a single-quote used as a delimiter ('word') is not swallowed into the
    #   word, while a genuine internal apostrophe (don't) is kept.
    # [^\W\d_] means "a word character that is not a digit or underscore",
    #   i.e. a letter — Unicode-aware, so this works for any alphabet/language.
    # (?:['’-][letters]+)* is optional tail that allows internal apostrophes and hyphens
    word_pattern = r"[^\W\d_]+(?:['’-][^\W\d_]+)*"
    words = re.findall(word_pattern, inner_text)

    # WARNING: ^this will fail if a word is legitimately supposed to begin
    #   or end with punctuation, e.g. a leading/trailing apostrophe will be dropped.

    # TODO: maybe check whether all words were actually present in the original text
    #  and if not then log a response validity flag to save with the result.

    return ParseResult(words, parse_ok)
