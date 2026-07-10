"""Functions for parsing LLM responses to benchmark items into a list of words."""

import re


def parse_word_list_from_response(response: str) -> list[str]:
    """Parse a model response to a benchmark item into a pure list of words (strings).

    A typical model response is expected to include a JSON array of words, e.g.:
    ["apple", "banana", "pear"]

    The model may also include extra text before or after the array,
    or may not format the array correctly, so this function attempts to extract
    the list of words from the response in a forgiving way.
    """

    if re.search(r"\ball\b", response, re.IGNORECASE):
        print(
            "WARNING: model response contains the word 'all' so may be "
            "erroneously trying to say 'all words decodable' or similar"
        )

    # Stage 1 of parsing: locate the array region of the response and capture
    #   its contents, discarding any surrounding prose the model added.
    array_contents_pattern = r"^.*?\[{0,1}([^\]\[]*)\]{0,1}[^\[]*?$"
    ## logic of regex:
    #   start of string - ^
    #   followed by zero or more of anything (non-greedy, specified by the ?
    #     after *) - .*?
    #   followed by zero or one left brackets - \[{0,1}
    #   followed by zero or more not-right-or-left-brackets (in a capturing group)
    #     where zero allows an empty list "[]" to match - ([^\]\[]*)
    #   followed by zero or one right brackets which ensures it is the last array
    #     in the string - \]{0,1}
    #   followed by zero or more non-left-brackets (non-greedy) - [^\[]*?
    #   followed by end of string - $

    # TODO: maybe use a secondary fallback regex to match a list of words that are not
    #   in brackets, e.g. "apple, banana, pear" but where precisely one word between
    #   each comma is required or else marked as parse failure.

    # flag DOTALL ensures that newlines are included in .
    array_contents_re = re.compile(array_contents_pattern, flags=re.DOTALL)

    match = array_contents_re.match(response)
    if match:
        inner_text = match.group(1)
    else:
        print("WARNING: model response did not match expected pattern:")
        print(response)
        # TODO: remove this print and instead set a datapoint that tracks parse failure
        #   and maybe return early to retry LLM again. For now just utilise full
        #   response text and hope for the best, but this is likely to fail
        inner_text = response

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

    return words
