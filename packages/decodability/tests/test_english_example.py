from pathlib import Path
from runpy import run_path


def test_english_example_builds_dataframe():
    example_path = Path(__file__).parents[1] / "examples" / "english_example.py"
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
