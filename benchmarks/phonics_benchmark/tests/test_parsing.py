"""Test cases for parsing LLM responses (extracting list of non-decodable words)."""

import json

from phonics_benchmark.parsing import parse_word_list


def test_parse_word_list():
    expected_output_single = ["wordone"]
    expected_output_double = ["wordone", "wordtwo"]
    expected_output_triple = ["wordone", "wordtwo", "wordthree"]

    # Test with a valid JSON array of several words
    response = '["wordone", "wordtwo", "wordthree"]'
    assert parse_word_list(response).words == expected_output_triple

    # Test with extra text before and after the array
    response = 'Some extra text ["wordone", "wordtwo", "wordthree"] more text'
    assert parse_word_list(response).words == expected_output_triple

    # Test with newlines before and after the array
    response = 'Some extra \n text ["wordone", "wordtwo", "wordthree"] more \n text'
    assert parse_word_list(response).words == expected_output_triple

    # Test with newlines inside the array
    response = 'Some extra \n text ["wordone",\n"wordtwo",\n "wordthree"] more \n text'
    assert parse_word_list(response).words == expected_output_triple

    # Test with no spaces after commas in the array
    response = '["wordone","wordtwo","wordthree"]'
    assert parse_word_list(response).words == expected_output_triple

    # Test with single quotes around words and extra text
    response = "Some extra text ['wordone', 'wordtwo', 'wordthree'] more text"
    assert parse_word_list(response).words == expected_output_triple

    # Test with no quotes around words and extra text
    response = "Some extra text [wordone, wordtwo, wordthree] more text"
    assert parse_word_list(response).words == expected_output_triple

    # Test with a valid JSON array of one word
    response = '["wordone"]'
    assert parse_word_list(response).words == expected_output_single

    # Test with a valid JSON array of two words
    response = '["wordone", "wordtwo"]'
    assert parse_word_list(response).words == expected_output_double

    # Test with apostrophes in words
    expected_output_apostrophe = ["don't", "dog's", "ng'alang'o"]
    response = """Some extra text ["don't", "dog's", "ng'alang'o"] more text"""
    assert parse_word_list(response).words == expected_output_apostrophe

    # Test with hyphens in words
    expected_output_hyphen = ["two-word", "three-word-join"]
    response = """Some extra text ["two-word", "three-word-join"] more text"""
    assert parse_word_list(response).words == expected_output_hyphen

    # TODO: consider testing with pesky curly quotes and curly apostrophes,
    #   e.g. “wordone”, “word’two”, “wordthree”
    # TODO: test dodgy cases like ["word.one", "word,two", "word three"]
    #   where punctuation is included in the words; unclear how to handle these.

    # Test accented characters
    expected_output_accented = ["naïve", "résumé", "façade", "niño", "smörgåsbord"]
    response = """Pre ["naïve", "résumé", "façade", "niño", "smörgåsbord"] post"""
    assert parse_word_list(response).words == expected_output_accented

    # Test other scripts (e.g. Cyrillic, Greek, Chinese)
    expected_output_cyrillic = ["слово", "пример", "Кыргызстан"]
    response = """Some extra text ["слово", "пример", "Кыргызстан"] more text"""
    assert parse_word_list(response).words == expected_output_cyrillic

    # Test with an empty array
    response = "[]"
    assert parse_word_list(response).words == []

    # Test with an empty array with extra text before and after
    response = "Some extra text [] more text"
    assert parse_word_list(response).words == []

    # Test with an empty string
    response = ""
    assert parse_word_list(response).words == []

    # Test with the word "all" in the response
    response = "All words are decodable"
    # TODO: decide how to handle this case
    # assert parse_word_list(response).words == []
    # assert "WARNING" in capsys.readouterr().out

    # Test with the word "all" in the response, but also a valid array
    response = 'All words are decodable, but here is a list: ["wordone", "wordtwo"]'
    assert parse_word_list(response).words == expected_output_double

    # Test multiple arrays in the response; should only parse the last one
    #  since sometimes the model will change its mind or revise its answer mid-response.
    response = 'Bla ["wordone", "wordtwo"] actually ["wordthree", "wordfour"] blagh'
    assert parse_word_list(response).words == ["wordthree", "wordfour"]

    # Test full JSON style
    response = '{"words": ["wordone", "wordtwo"]}'
    assert parse_word_list(response).words == expected_output_double

    # Test alternate JSON style
    response = '```json\n[\n "wordone",\n "wordtwo",\n "wordthree"\n]\n```'
    assert parse_word_list(response).words == expected_output_triple


def test_parse_word_list_parse_ok_flag():
    # A well-formed bracketed array parses ok
    result = parse_word_list('["wordone", "wordtwo"]')
    assert result.parse_ok

    # Text before and after the array does not affect parse_ok
    result = parse_word_list('Some text ["wordone", "wordtwo"] more text')
    assert result.parse_ok

    # An empty array is still a successful parse
    result = parse_word_list("[]")
    assert result.parse_ok

    # No array at all: words are scraped from the full text but flagged
    result = parse_word_list("Here are the words: wordone, wordtwo")
    assert not result.parse_ok

    # Missing closing bracket: flagged, and surrounding prose leaks into the words
    #   (which is why the caller should check parse_ok before trusting the words)
    result = parse_word_list('Here you go ["wordone", "wordtwo"')
    assert not result.parse_ok

    # An empty response: no words, flagged as a parse failure
    result = parse_word_list("")
    assert result.words == []
    assert not result.parse_ok


def test_parse_word_list_preserves_apostrophe():
    """The ng' grapheme must survive parsing: Kiswahili words such as ng'ombe
    carry an apostrophe, and stripping it changes the word.
    """
    expected_words = ["ng'ombe", "shambani"]

    result = parse_word_list(json.dumps(expected_words, ensure_ascii=False))

    assert result.words == expected_words
    assert result.parse_ok
