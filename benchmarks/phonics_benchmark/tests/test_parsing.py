"""Test cases for parsing LLM responses (extracting list of non-decodable words)."""

from phonics_benchmark.parsing import parse_word_list_from_response


def test_parse_word_list_from_response():
    expected_output_single = ["wordone"]
    expected_output_double = ["wordone", "wordtwo"]
    expected_output_triple = ["wordone", "wordtwo", "wordthree"]

    # Test with a valid JSON array of several words
    response = '["wordone", "wordtwo", "wordthree"]'
    assert parse_word_list_from_response(response) == expected_output_triple

    # Test with extra text before and after the array
    response = 'Some extra text ["wordone", "wordtwo", "wordthree"] more text'
    assert parse_word_list_from_response(response) == expected_output_triple

    # Test with newlines before and after the array
    response = 'Some extra \n text ["wordone", "wordtwo", "wordthree"] more \n text'
    assert parse_word_list_from_response(response) == expected_output_triple

    # Test with newlines inside the array
    response = 'Some extra \n text ["wordone",\n"wordtwo",\n "wordthree"] more \n text'
    assert parse_word_list_from_response(response) == expected_output_triple

    # Test with no spaces after commas in the array
    response = '["wordone","wordtwo","wordthree"]'
    assert parse_word_list_from_response(response) == expected_output_triple

    # Test with single quotes around words and extra text
    response = "Some extra text ['wordone', 'wordtwo', 'wordthree'] more text"
    assert parse_word_list_from_response(response) == expected_output_triple

    # Test with no quotes around words and extra text
    response = "Some extra text [wordone, wordtwo, wordthree] more text"
    assert parse_word_list_from_response(response) == expected_output_triple

    # Test with a valid JSON array of one word
    response = '["wordone"]'
    assert parse_word_list_from_response(response) == expected_output_single

    # Test with a valid JSON array of two words
    response = '["wordone", "wordtwo"]'
    assert parse_word_list_from_response(response) == expected_output_double

    # Test with apostrophes in words
    expected_output_apostrophe = ["wordone's", "wordtwo's", "wordthree's"]
    response = """Some extra text ["wordone's", "wordtwo's", "wordthree's"] more text"""
    assert parse_word_list_from_response(response) == expected_output_apostrophe

    # TODO: consider testing with curly quotes and apostrophes,
    #   e.g. “wordone”, “wordtwo”, “wordthree”
    # TODO: consider testing with a valid JSON array of words that contain punctuation,
    #   e.g. "word-one", "word.two", "word,three"

    # Test with an empty array
    response = "[]"
    assert parse_word_list_from_response(response) == []

    # Test with an empty array with extra text before and after
    response = "Some extra text [] more text"
    assert parse_word_list_from_response(response) == []

    # Test with an empty string
    response = ""
    # TODO: decide how we actually want to handle this case; probably a retry to LLM

    # Test with the word "all" in the response
    response = "All words are decodable"
    # TODO: decide how to handle this case
    # assert parse_word_list_from_response(response) == []
    # assert "WARNING" in capsys.readouterr().out
