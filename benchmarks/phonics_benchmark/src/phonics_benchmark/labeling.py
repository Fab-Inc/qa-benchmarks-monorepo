"""Run the deterministic decodability checker over model-generated text.

This is the only module that talks to the `decodability` package directly. It
resolves a JSON-style decodability config into callables and applies them to a
generated text, so swapping the language or the scoring/aggregation logic is a
contained change (config + decodability registry) rather than a code edit here.
"""

from __future__ import annotations

from collections import Counter
from functools import partial
from typing import Any, NamedTuple

from decodability import DECODABILITY_REGISTRY, Aggregation, Measure, annotate_words

# The decodability configuration each language is scored with. Shared by the
# item-build script and the report script so the two can never drift apart.
#
# ``score_column`` names the aggregation output that decides decodability; it
# differs by language.
DECODABILITY_CONFIGS: dict[str, dict[str, Any]] = {
    "sw": {
        "scoring_methods": [
            "score_known_graphemes_kiswahili",
            "score_known_clusters_and_patterns_kiswahili",
            "score_whole_words_kiswahili",
        ],
        "aggregations": {
            "kenya_tusome_score": {"name": "kiswahili_kenya_tusome_aggregator"}
        },
        "score_column": "kenya_tusome_score",
    },
}


def get_decodability_config(language: str) -> tuple[dict[str, Any], str]:
    """Return the decodability config and score column for a language."""
    try:
        config = DECODABILITY_CONFIGS[language]
    except KeyError:
        raise NotImplementedError(
            f"No decodability config for language '{language}'; available: "
            f"{sorted(DECODABILITY_CONFIGS)}"
        ) from None
    return config, config["score_column"]


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


class TextDecodabilityResult(NamedTuple):
    """Per-text decodability counts produced by ``score_text_decodability``.

    Counts are **per occurrence**: a word repeated three times counts three
    times. The ``*_unique_*`` fields are the distinct-word equivalents, counted
    over lowercased words, so ``n_unique_nondecodable_words`` always equals
    ``len(unique_nondecodable_words)``.

    Note that "word" here means an actual word of the text, never an LLM token.

    ``failure_causes`` maps each scorer's column name to the number of
    non-decodable word occurrences that scorer rejected, which is what separates
    e.g. an untaught grapheme from an untaught consonant cluster.
    """

    n_words: int
    n_unique_words: int
    n_nondecodable_words: int
    n_unique_nondecodable_words: int
    unique_nondecodable_words: list[str]
    failure_causes: dict[str, int]


def score_text_decodability(
    text: str,
    student_knowledge: dict[str, Any],
    config: dict,
    language: str,
    *,
    score_column: str,
) -> TextDecodabilityResult:
    """Split ``text`` into words and score each one for decodability.

    A word is treated as non-decodable when its aggregated score is exactly
    0.0, matching the convention of the `decodability` package's curriculum
    aggregators (which are binary).

    Each distinct word is scored once and the verdict mapped back onto every
    occurrence of it: the row-wise aggregations are documented as slow, and a
    text for an early reader repeats words heavily.

    An empty text (or one with no extractable words) returns an all-zero
    result; the caller decides what score that earns.
    """
    objs = DECODABILITY_REGISTRY[language]
    words = objs["extract_words"](text)
    if not words:
        return TextDecodabilityResult(0, 0, 0, 0, [], {})

    scoring_methods, aggregations = resolve_decodability_config(config, language)
    if score_column not in aggregations:
        raise ValueError(
            f"score_column '{score_column}' is not one of the config's aggregation "
            f"outputs: {sorted(aggregations)}"
        )

    student_knowledge_model = objs["student_knowledge_model"].model_validate(
        student_knowledge
    )

    # TODO: push out any normalization such as lowercasing or deduplication
    # to the decodability package in the future
    lowered = [word.lower() for word in words]
    unique_words = sorted(set(lowered))

    scores = annotate_words(
        unique_words,
        scoring_methods=scoring_methods,
        student_knowledge=student_knowledge_model,
        aggregations=aggregations,
    )

    nondecodable_mask = scores[score_column] == 0.0
    unique_nondecodable_words = sorted(scores["word"][nondecodable_mask].tolist())
    nondecodable_word_set = set(unique_nondecodable_words)

    word_counts = Counter(lowered)
    n_nondecodable_words = sum(word_counts[word] for word in nondecodable_word_set)

    # Attribute each non-decodable word occurrence to every scorer that
    # rejected it. A
    # word can fail more than one scorer, so these counts need not sum to
    # n_nondecodable_words.
    scorer_columns = [
        column
        for column in scores.columns
        if column != "word" and column not in aggregations
    ]
    failure_causes = {
        column: sum(
            word_counts[word]
            for word in scores["word"][nondecodable_mask & (scores[column] == 0.0)]
        )
        for column in scorer_columns
    }

    return TextDecodabilityResult(
        n_words=len(words),
        n_unique_words=len(unique_words),
        n_nondecodable_words=n_nondecodable_words,
        n_unique_nondecodable_words=len(unique_nondecodable_words),
        unique_nondecodable_words=unique_nondecodable_words,
        failure_causes=failure_causes,
    )
