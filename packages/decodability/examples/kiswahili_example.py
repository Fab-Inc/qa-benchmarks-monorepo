"""Small runnable Kiswahili example for the decodability scaffold.

Run from the repository root with:

    uv run --frozen --package decodability python \
        packages/decodability/examples/kiswahili_example.py
"""

from __future__ import annotations

from functools import partial

import pandas as pd
from decodability import annotate_words
from decodability.annotate import Aggregation
from decodability.kiswahili import (
    AGGREGATIONS,
    SCORING_METHODS,
    KiswahiliStudentKnowledge,
    extract_words_kiswahili,
)

TEXT = """Tulisikia mayowe tukiwa sebuleni. Sote tukatoka.
Kutazama tukaona moto ukiwaka sana. Waya wa stima uliwaka moto.
Sote tukawa na wasiwasi sana. Wazima moto wakatokea.
Mioyo yetu ikatulia, tukaelekea sebuleni tena."""

STUDENT_KNOWLEDGE = KiswahiliStudentKnowledge.model_validate(
    dict(
        graphemes={
            "a",
            "u",
            "i",
            "o",
            "e",
            "m",
            "k",
            "t",
            "n",
            "s",
            "b",
            "y",
            "z",
            "g",
            "d",
            "j",
            "r",
            "w",
        },
        clusters={"kw", "st"},
        cluster_patterns={"prenasalised"},
        whole_words={"mama", "ama", "anaona", "na", "paka"},
    )
)


JSON_DATA = {
    "language": "kiswahili",
    "scoring_methods": [
        "score_known_graphemes_kiswahili",
        "score_known_clusters_and_patterns_kiswahili",
        "score_whole_words_kiswahili",
    ],
    "aggregations": {
        "kenya_tusome_score": {"name": "kiswahili_kenya_tusome_aggregator"},
        # NOTE: The following are generic aggregations that are not
        # Kiswahili-specific, but we include them to demonstrate potential usage with
        # multiple aggregation methods.
        "mean_score": {"name": "mean_score"},
        "weighted_score": {
            "name": "weighted_average",
            "params": {
                "weights": {
                    "score_known_graphemes_kiswahili": 0.5,
                    "score_known_clusters_and_patterns_kiswahili": 0.2,
                    "score_whole_words_kiswahili": 0.3,
                }
            },
        },
    },
}


def build_dataframe() -> pd.DataFrame:
    scoring_method_names = JSON_DATA["scoring_methods"]
    aggregations: dict[str, Aggregation] = dict()
    for name, agg in JSON_DATA["aggregations"].items():
        params = agg.get("params")
        agg_func = AGGREGATIONS[agg["name"]]
        if params:
            aggregations[name] = partial(agg_func, **params)
        else:
            aggregations[name] = agg_func

    return annotate_words(
        extract_words_kiswahili(TEXT),
        scoring_methods=[SCORING_METHODS[name] for name in scoring_method_names],
        student_knowledge=STUDENT_KNOWLEDGE,
        aggregations=aggregations,
    )


def main() -> None:
    print(build_dataframe().to_string(index=False))


if __name__ == "__main__":
    main()
