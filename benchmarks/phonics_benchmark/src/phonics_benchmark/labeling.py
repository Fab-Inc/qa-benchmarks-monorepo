"""Label benchmark items with their non-decodable words (the correct answers).

This is the only module that talks to the `decodability` package directly. It
resolves a JSON-style decodability config into callables and applies them per
benchmark item, so swapping the language or the scoring/aggregation logic is a
contained change (config + decodability registry) rather than a code edit here.
"""

from __future__ import annotations

from functools import partial

import pandas as pd
from decodability import Aggregation, Measure, annotate_words, get_decodability_objects


def resolve_decodability_config(
    config: dict, language: str
) -> tuple[list[Measure], dict[str, Aggregation]]:
    """Translate a decodability config into scoring/aggregation callables.

    The config names scoring methods and aggregations as strings; they are
    looked up in the language's registries. Aggregations with a ``params`` block
    are bound with functools.partial (e.g. weights for a weighted average).
    """
    objs = get_decodability_objects(language)

    scoring_methods = [objs.scoring_methods[name] for name in config["scoring_methods"]]

    aggregations: dict[str, Aggregation] = {}
    for output_column, agg in config["aggregations"].items():
        agg_func = objs.aggregations[agg["name"]]
        params = agg.get("params")
        aggregations[output_column] = (
            partial(agg_func, **params) if params else agg_func
        )

    return scoring_methods, aggregations


def label_nondecodable_words(
    benchmark_items: pd.DataFrame,
    config: dict,
    language: str,
    *,
    score_column: str = "kenya_tusome_score",
) -> pd.Series:
    """Return, per benchmark item, the list of words scored as non-decodable.

    A word is non-decodable when its aggregated decodability score is 0.0 in the
    ``score_column`` (which the config must produce). Expects the DataFrame to
    have ``content`` and ``student_knowledge`` columns.
    """
    missing = {"content", "student_knowledge"} - set(benchmark_items.columns)
    if missing:
        raise ValueError(
            f"benchmark_items is missing required column(s): {sorted(missing)}"
        )

    objs = get_decodability_objects(language)
    scoring_methods, aggregations = resolve_decodability_config(config, language)

    def get_nondecodable_words(benchmark_item: pd.Series) -> list[str]:
        words = objs.tokenizer(benchmark_item.content)
        student_knowledge = objs.student_knowledge_model.model_validate(
            benchmark_item.student_knowledge
        )
        label_df = annotate_words(
            words,
            scoring_methods=scoring_methods,
            student_knowledge=student_knowledge,
            aggregations=aggregations,
        )
        label_df = label_df.drop_duplicates()
        return label_df["word"][label_df[score_column] == 0.0].tolist()

    return benchmark_items.apply(get_nondecodable_words, axis=1)
