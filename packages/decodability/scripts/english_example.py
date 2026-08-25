"""Small runnable English example for the decodability scaffold.

Run from the repository root with:

    uv run --frozen --package decodability python \
        packages/decodability/examples/english_example.py
"""

from __future__ import annotations

from functools import partial

import pandas as pd
from decodability import annotate_words
from decodability.annotate import Aggregation
from decodability.english import (
    AGGREGATIONS,
    SCORING_METHODS,
    EnglishStudentKnowledge,
    extract_words_english,
)

TEXT = """The cat sat on the mat. He said hi to the cat.
The thin cat sat in a hat."""

# A student who has learned CVC consonants and short vowels, plus the ``th``
# and ``sh`` digraphs, and holds a small sight-word set. Mixed coverage across
# the passage exercises the GPC-score / sight-word split and the reduced-vowel
# free pass (on the unstressed ``a``).
STUDENT_KNOWLEDGE = EnglishStudentKnowledge(
    gpcs=[
        ("c", "k"),
        ("a", "æ"),
        ("t", "t"),
        ("s", "s"),
        ("m", "m"),
        ("n", "n"),
        ("h", "h"),
        ("o", "ɑ"),
        ("i", "ɪ"),
        ("th", "θ"),
        ("th", "ð"),
        ("sh", "ʃ"),
    ],
    sight_words={"the", "he", "said", "to"},
)


JSON_DATA = {
    "language": "english",
    "scoring_methods": [
        "score_known_gpcs_english",
        "score_sight_words_english",
    ],
    "aggregations": {
        "ufli_score": {"name": "ufli_curriculum_aggregator"},
        # Generic aggregators — not English-specific, included to show how a
        # config can request several aggregation strategies side by side.
        "mean_score": {"name": "mean_score"},
        "weighted_score": {
            "name": "weighted_average",
            "params": {
                "weights": {
                    "score_known_gpcs_english": 0.5,
                    "score_sight_words_english": 0.5,
                }
            },
        },
    },
}


def build_dataframe() -> pd.DataFrame:
    scoring_method_names = JSON_DATA["scoring_methods"]
    aggregations: dict[str, Aggregation] = {}
    for name, agg in JSON_DATA["aggregations"].items():
        params = agg.get("params")
        agg_func = AGGREGATIONS[agg["name"]]
        aggregations[name] = partial(agg_func, **params) if params else agg_func

    return annotate_words(
        extract_words_english(TEXT),
        scoring_methods=[SCORING_METHODS[name] for name in scoring_method_names],
        student_knowledge=STUDENT_KNOWLEDGE,
        aggregations=aggregations,
    )


def main() -> None:
    print(build_dataframe().to_string(index=False))


if __name__ == "__main__":
    main()
