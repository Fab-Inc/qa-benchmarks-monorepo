from pathlib import Path
from runpy import run_path

EXAMPLES_DIR = Path(__file__).parents[1] / "examples"


def test_english_example_builds_dataframe():
    example_path = EXAMPLES_DIR / "english_example.py"
    namespace = run_path(str(example_path))

    df = namespace["build_dataframe"]()

    assert list(df.columns) == [
        "word",
        "known_letters_score",
        "short_word_score",
        "mean_score",
        "weighted_score",
    ]
    assert df.loc[0, "word"] == "The"


def test_kiswahili_example_builds_dataframe():
    example_path = EXAMPLES_DIR / "kiswahili_example.py"
    namespace = run_path(str(example_path))

    df = namespace["build_dataframe"]()

    assert list(df.columns) == [
        "word",
        "score_known_graphemes_kiswahili",
        "score_known_clusters_and_patterns_kiswahili",
        "score_whole_words_kiswahili",
        "kenya_tusome_score",
        "mean_score",
        "weighted_score",
    ]
    assert df.loc[0, "word"] == "Tulisikia"
