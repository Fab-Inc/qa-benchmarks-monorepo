import re


def strip_non_alpha(word: str) -> str:
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
    """Parse a model response to a benchmark item into a pure list of words."""
    # TODO: we probably want to handle this more scientifically in longer term
    # by evaluating and logging whether the response met the expected schema,
    # so that invalid responses get flagged and not just ignored / marked wrong.

    response = response.strip()

    if response == "[]" or response == "":
        return []

    if re.search(r"\ball\b", response, re.IGNORECASE):
        print(
            "WARNING: model response contains the word 'all' so may be "
            "erroneously trying to say 'all words decodable' or similar"
        )

    # Parse forgivingly to try remove superfluous material the model may have added.
    # Find the final closing square bracket of the response, then the opening bracket
    #   that matches it, then extract all text in between them.
    # If no matching set of square brackets is found, then just keep response in full.
    response_trimmed = response
    close_idx = response.rfind("]")
    if close_idx != -1:
        # walk backwards from the final "]" to find its matching "[",
        # accounting for any nested brackets in between.
        depth = 0
        for i in range(close_idx, -1, -1):
            if response[i] == "]":
                depth += 1
            elif response[i] == "[":
                depth -= 1
                if depth == 0:
                    response_trimmed = response[i + 1 : close_idx]
                    break

    words_raw = [word.strip() for word in response_trimmed.split(",")]

    # Now deal with any delimiters that may be surrounding each word

    # WARNING: this will fail if the word is legitimately supposed to begin
    # or end with punctuation, e.g. apostrophe could be problematic
    words_stripped = [strip_non_alpha(word) for word in words_raw]

    # remove any empty strings that may have occurred (would be unusual)
    words_filtered = [w for w in words_stripped if w]

    # TODO: maybe check whether all words were actually present in the original text
    #  and if not then log a response validity flag to save with the result.

    return words_filtered
