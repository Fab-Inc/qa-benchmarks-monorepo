"""Functions for parsing LLM responses to benchmark items into a list of words."""

import re


def _strip_non_alpha(word: str) -> str:
    """Strip non-alphabetic characters from the start and end of a word.

    Alphabetic characters within the word are preserved, and ``str.isalpha``
    is Unicode-aware so this works for any alphabet/language.
    """
    start, end = 0, len(word)
    while start < end and not word[start].isalpha():
        start += 1
    while end > start and not word[end - 1].isalpha():
        end -= 1
    return word[start:end]


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

    answer_pattern = r"^.*?\[{0,1}([^\]\[]*)\]{0,1}[^\[]*?$"
    ## logic of regex:
    #   start of string - ^
    #   followed by zero or more of anything (non-greedy, specified by the ?
    #     after *) - .*?
    #   followed by zero or one left brackets - \[{0,1}
    #   followed by zero or more not-right-or-left-brackets (in a capturing group)
    #     where zero allows an empty list "[]" to match - ([^\]\[]*)
    #   followed by zero or one right brackets - \]{0,1}
    #   followed by zero or more non-left-brackets (non-greedy) - [^\[]*?
    #   followed by end of string - $

    # TODO: maybe use a secondary fallback regex to match a list of words that are not
    #   in brackets, e.g. "apple, banana, pear" but where precisely one word between
    #   each comma is required or else marked as parse failure.

    # flag DOTALL ensures that newlines are included in .
    answer_re = re.compile(answer_pattern, flags=re.DOTALL)

    match = answer_re.match(response)
    if match:
        inner_text = match.group(1)
    else:
        print("WARNING: model response did not match expected pattern:")
        print(response)
        # TODO: remove this print and instead set a datapoint that tracks parse failure
        #   and maybe return early to retry LLM again. For now just utilise full
        #   response text and hope for the best, but this is likely to fail
        inner_text = response

    words_raw = [word.strip() for word in inner_text.split(",")]

    # Now deal with any delimiters that may be surrounding each word.
    # WARNING: this will fail if the word is legitimately supposed to begin
    #   or end with punctuation, e.g. apostrophe could be problematic
    words_stripped = [_strip_non_alpha(word) for word in words_raw]

    # remove any empty strings that may have occurred,
    #   e.g. if the model returned an empty list or extra commas
    words_filtered = [w for w in words_stripped if w]

    # TODO: maybe check whether all words were actually present in the original text
    #  and if not then log a response validity flag to save with the result.

    return words_filtered
