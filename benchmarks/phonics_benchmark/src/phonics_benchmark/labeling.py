"""Label benchmark items with their non-decodable words (the correct answers).

This is the only module that talks to the `decodability` package directly. It
resolves a JSON-style decodability config into callables and applies them per
benchmark item, so swapping the language or the scoring/aggregation logic is a
contained change (config + decodability registry) rather than a code edit here.
"""

from __future__ import annotations

from functools import partial

import pandas as pd
from decodability import DECODABILITY_REGISTRY, Aggregation, Measure, annotate_words
from decodability.kiswahili.definitions import ClusterPattern, primerpro_settings
from primerpro import Word

from phonics_benchmark.utils import require_columns


def resolve_decodability_config(
    config: dict, language: str
) -> tuple[list[Measure], dict[str, Aggregation]]:
    """Translate a decodability config into scoring/aggregation callables.

    The config names scoring methods and aggregations as strings; they are
    looked up in the language's registries. Aggregations with a ``params`` block
    are bound with functools.partial (e.g. weights for a weighted average).
    """
    objs = DECODABILITY_REGISTRY[language]
    scoring_methods = [
        objs["scoring_methods"][name] for name in config["scoring_methods"]
    ]

    aggregations: dict[str, Aggregation] = {}
    for output_column, agg in config["aggregations"].items():
        agg_func = objs["aggregations"][agg["name"]]
        params = agg.get("params")
        aggregations[output_column] = (
            partial(agg_func, **params) if params else agg_func
        )

    return scoring_methods, aggregations


@require_columns(["content"])
def get_vocabulary_set(benchmark_items: pd.DataFrame, language: str) -> pd.Series:
    """Get unique set of vocabulary per benchmark item from its content."""
    objs = DECODABILITY_REGISTRY[language]

    # TODO: push out any normalization such as lowercasing or deduplication
    # to the decodability package in the future
    return benchmark_items["content"].apply(
        lambda content: {word.lower() for word in objs["extract_words"](content)}
    )


@require_columns(["content", "student_knowledge"])
def label_nondecodable_words(
    benchmark_items: pd.DataFrame,
    config: dict,
    language: str,
    *,
    score_column: str,
) -> pd.Series:
    """Return, per benchmark item, the list of words scored as non-decodable.

    Expects the DataFrame to have ``content`` and ``student_knowledge`` columns.
    ``score_column`` names which of the config's aggregation outputs decides
    decodability; it is required because it differs by language.
    """
    if "vocab_set" not in benchmark_items.columns:
        benchmark_items = benchmark_items.copy()
        benchmark_items["vocab_set"] = get_vocabulary_set(benchmark_items, language)

    scoring_methods, aggregations = resolve_decodability_config(config, language)
    if score_column not in aggregations:
        raise ValueError(
            f"score_column '{score_column}' is not one of the config's aggregation "
            f"outputs: {sorted(aggregations)}"
        )
    objs = DECODABILITY_REGISTRY[language]

    def get_nondecodable_words(benchmark_item: pd.Series) -> list[str]:
        words = benchmark_item.vocab_set
        student_knowledge = objs["student_knowledge_model"].model_validate(
            benchmark_item.student_knowledge
        )
        label_df = annotate_words(
            words,
            scoring_methods=scoring_methods,
            student_knowledge=student_knowledge,
            aggregations=aggregations,
        )

        # lowercase text
        nondecodable_words = sorted(
            label_df["word"][label_df[score_column] == 0.0]
            .str.lower()
            .unique()
            .tolist()
        )
        return nondecodable_words

    return benchmark_items.apply(get_nondecodable_words, axis=1)


def get_grapheme_breakdown(text: str, language: str) -> str:
    """Return a string showing the grapheme breakdown of a benchmark item."""
    objs = DECODABILITY_REGISTRY[language]
    words = {word.lower() for word in objs["extract_words"](text)}

    if language != "sw":
        raise NotImplementedError(
            f"Grapheme breakdown is only implemented for Kiswahili, not {language}."
        )

    word_to_graphemes = {
        word: "/".join(
            [
                grapheme.symbol
                for grapheme in Word(text=word, settings=primerpro_settings).graphemes
            ]
        )
        for word in words
    }

    longest_to_shortest_words = sorted(words, key=len, reverse=True)

    text_lower = text.lower()
    for word in longest_to_shortest_words:
        graphemes = word_to_graphemes[word]
        text_lower = text_lower.replace(word, graphemes)
    return text_lower


def get_valid_knowledge_str(language: str) -> str:
    """Get valid knowledge string representation for the specified language."""
    if language != "sw":
        raise NotImplementedError(
            f"Valid knowledge string is only implemented for Kiswahili, not {language}."
        )

    inventory = primerpro_settings.grapheme_inventory

    if inventory is None:
        raise ValueError("Grapheme inventory is not defined in primerpro_settings.")

    vowels_string = ", ".join([grapheme.symbol for grapheme in inventory.vowels])
    consonants_string = ", ".join(
        [grapheme.symbol for grapheme in inventory.consonants]
    )
    return f"- Vowels: {vowels_string}\n- Consonants: {consonants_string}"


def get_missing_knowledge_str(language: str, student_knowledge: dict) -> str:
    """Get string for missing knowledge for the specified language and student."""
    if language != "sw":
        raise NotImplementedError(
            "Missing knowledge string is only implemented for Kiswahili, not "
            f"{language}."
        )

    grapheme_inventory = primerpro_settings.grapheme_inventory

    if grapheme_inventory is None:
        raise ValueError("Grapheme inventory is not defined in primerpro_settings.")

    knowledge_model = DECODABILITY_REGISTRY[language]["student_knowledge_model"]
    profile = knowledge_model.model_validate(student_knowledge)

    unknown_graphemes = [
        grapheme.symbol
        for grapheme in grapheme_inventory._get_all_graphemes()
        if grapheme.symbol not in profile.graphemes
    ]
    prenasalised_patterns = (
        # patterns that are a superset of the m- and n- prenasalised patterns,
        # which makes the full list of unknown cluster patterns not MECE.
        # So, we want to exclude these from the full list of valid cluster patterns.
        ClusterPattern.PRENASALISED,
        ClusterPattern.PRENASALISED_LABIALISED_GLIDE,
        ClusterPattern.PRENASALISED_PALATALISED_GLIDE,
    )
    unknown_cluster_patterns = [
        pattern.label
        for pattern in ClusterPattern
        if (
            pattern.label not in profile.cluster_patterns
            and pattern not in prenasalised_patterns
        )
    ]

    unknown_graphemes_string = (
        ", ".join(unknown_graphemes) if unknown_graphemes else "None"
    )
    unknown_cluster_patterns_string = (
        ", ".join(unknown_cluster_patterns) if unknown_cluster_patterns else "None"
    )
    return (
        f"- Unknown graphemes: {unknown_graphemes_string}\n"
        f"- Unknown cluster patterns: {unknown_cluster_patterns_string}\n"
        f"- Unknown clusters: any other consonant clusters not covered by known "
        f"cluster patterns\n"
    )
