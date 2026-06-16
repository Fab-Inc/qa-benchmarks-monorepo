from functools import partial

import pytest
from decodability import annotate_words, mean_score, weighted_average


def known_letters_score(word, student_knowledge):
    letters = [character for character in word.casefold() if character.isalpha()]
    known_letters = student_knowledge["known_letters"]
    return sum(letter in known_letters for letter in letters) / len(letters)


def short_word_score(word, student_knowledge):
    if len(word) <= student_knowledge["max_easy_length"]:
        return 1.0
    return 0.5


def test_annotate_words_adds_scoring_and_default_mean_columns():
    df = annotate_words(
        ["cat", "thin"],
        scoring_methods=[known_letters_score, short_word_score],
        student_knowledge={
            "known_letters": set("acnt"),
            "max_easy_length": 3,
        },
    )

    assert list(df.columns) == [
        "word",
        "known_letters_score",
        "short_word_score",
        "mean_score",
    ]
    assert df.loc[0, "known_letters_score"] == 1.0
    assert df.loc[0, "short_word_score"] == 1.0
    assert df.loc[1, "known_letters_score"] == 0.5
    assert df.loc[1, "short_word_score"] == 0.5
    assert df.loc[1, "mean_score"] == 0.5


def test_annotate_words_accepts_explicit_aggregations():
    df = annotate_words(
        ["thin"],
        scoring_methods=[known_letters_score, short_word_score],
        student_knowledge={
            "known_letters": set("nt"),
            "max_easy_length": 3,
        },
        aggregations={
            "mean_score": mean_score,
            "weighted_score": partial(
                weighted_average,
                weights={
                    "known_letters_score": 0.75,
                    "short_word_score": 0.25,
                },
            ),
        },
    )

    assert df.loc[0, "mean_score"] == 0.5
    assert df.loc[0, "weighted_score"] == pytest.approx(0.5)
