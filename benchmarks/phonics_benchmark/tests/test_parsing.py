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
    expected_output_apostrophe = ["don't", "dog's", "ng'alang'o"]
    response = """Some extra text ["don't", "dog's", "ng'alang'o"] more text"""
    assert parse_word_list_from_response(response) == expected_output_apostrophe

    # Test with hyphens in words
    expected_output_hyphen = ["two-word", "three-word-join"]
    response = """Some extra text ["two-word", "three-word-join"] more text"""
    assert parse_word_list_from_response(response) == expected_output_hyphen

    # TODO: consider testing with pesky curly quotes and curly apostrophes,
    #   e.g. “wordone”, “word’two”, “wordthree”
    # TODO: test with some parse failures, e.g. missing closing bracket.
    # TODO: test dodgy cases like ["word.one", "word,two", "word three"]
    #   where punctuation is included in the words; unclear how to handle these.

    # Test accented characters
    expected_output_accented = ["naïve", "résumé", "façade", "niño", "smörgåsbord"]
    response = """Pre ["naïve", "résumé", "façade", "niño", "smörgåsbord"] post"""
    assert parse_word_list_from_response(response) == expected_output_accented

    # Test other scripts (e.g. Cyrillic, Greek, Chinese)
    expected_output_cyrillic = ["слово", "пример", "Кыргызстан"]
    response = """Some extra text ["слово", "пример", "Кыргызстан"] more text"""
    assert parse_word_list_from_response(response) == expected_output_cyrillic

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

    # Test with the word "all" in the response, but also a valid array
    response = 'All words are decodable, but here is a list: ["wordone", "wordtwo"]'
    assert parse_word_list_from_response(response) == expected_output_double

    # Test multiple arrays in the response; should only parse the last one
    #  since sometimes the model will change its mind or revise its answer mid-response.
    response = 'Bla ["wordone", "wordtwo"] actually ["wordthree", "wordfour"] blagh'
    assert parse_word_list_from_response(response) == ["wordthree", "wordfour"]
