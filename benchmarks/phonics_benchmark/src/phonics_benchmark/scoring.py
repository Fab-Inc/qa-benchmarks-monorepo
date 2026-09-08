"""Score model-generated texts on how decodable they are for the target student.

There is no gold answer here: the deterministic `decodability` checker *is* the
ground truth. Each item is scored on the share of its words the student could
actually read.
"""

import math

import pandas as pd

from phonics_benchmark.labeling import (
    get_decodability_config,
    score_text_decodability,
)
from phonics_benchmark.utils import require_columns

# The per-item score is (1 - nondecodable_rate) ** REFERENCE_LENGTH, which reads
# as "the probability that a randomly drawn REFERENCE_LENGTH-word passage from
# this text is fully decodable" — a strict pass rate normalised to a reference
# sentence length.
#
# Normalising is what makes the metric measure the model rather than the text
# length. Scoring raw error counts instead (e.g. -0.25 per bad word, or a strict
# all-or-nothing pass) makes an identical model score 0.92 on a 15-word text and
# 0.19 on a 200-word one, and rewards writing short.
#
# 20 is chosen as a typical sentence length. Raising it makes the metric harsher
# and lowering it more forgiving; the ranking of models is unaffected.
REFERENCE_LENGTH = 20

# Prefix for the per-scorer failure-attribution columns. The scorer names are
# language-specific (for Kiswahili they separate an untaught grapheme from an
# untaught consonant cluster), so these columns are discovered by prefix rather
# than hard-coded.
FAILURE_PREFIX = "n_fail_"


def graded_score(n_words: int, n_nondecodable_words: int) -> float:
    """Return the length-normalised decodability score for one text.

    Returns 0.0 for an empty text: a model that writes nothing has not written
    a decodable text.
    """
    if n_words <= 0:
        return 0.0
    nondecodable_rate = n_nondecodable_words / n_words
    return (1.0 - nondecodable_rate) ** REFERENCE_LENGTH


@require_columns(["generated_text", "delimiter_found", "student_knowledge"])
def add_per_item_decodability(results: pd.DataFrame, language: str) -> pd.DataFrame:
    """Add per-item decodability counts, rates and the graded score.

    Expects ``generated_text`` — the model's raw ``output`` already stripped of
    commentary by ``parsing.extract_generated_text`` — and the item's
    ``student_knowledge``.

    A row whose ``delimiter_found`` is False is a *format failure*: it is scored
    0.0 rather than dropped, and its decodability columns are left NaN. Grading
    the raw response instead would charge the model for its preamble, while
    dropping the row would let a model dodge hard items by malformatting.
    """
    results = results.copy()

    config, score_column = get_decodability_config(language)
    failure_columns = [f"{FAILURE_PREFIX}{name}" for name in config["scoring_methods"]]

    def score_row(row: pd.Series) -> pd.Series:
        blank = dict.fromkeys(failure_columns, math.nan)

        if not row["delimiter_found"]:
            return pd.Series(
                {
                    "n_words": math.nan,
                    "n_unique_words": math.nan,
                    "unique_word_ratio": math.nan,
                    "n_nondecodable_words": math.nan,
                    "n_unique_nondecodable_words": math.nan,
                    "unique_nondecodable_words": [],
                    "word_decodable_rate": math.nan,
                    "unique_word_decodable_rate": math.nan,
                    "graded_score": 0.0,
                    "strict_pass": False,
                    "format_failure": True,
                    "under_reference_length": False,
                    **blank,
                }
            )

        result = score_text_decodability(
            row["generated_text"],
            row["student_knowledge"],
            config,
            language,
            score_column=score_column,
        )

        if result.n_words == 0:
            # An empty text scores zero, but its rates are undefined rather
            # than perfect — averaging in a 1.0 rate would flatter the model.
            return pd.Series(
                {
                    "n_words": 0,
                    "n_unique_words": 0,
                    "unique_word_ratio": math.nan,
                    "n_nondecodable_words": 0,
                    "n_unique_nondecodable_words": 0,
                    "unique_nondecodable_words": [],
                    "word_decodable_rate": math.nan,
                    "unique_word_decodable_rate": math.nan,
                    "graded_score": 0.0,
                    "strict_pass": False,
                    "format_failure": False,
                    "under_reference_length": True,
                    **blank,
                }
            )

        return pd.Series(
            {
                "n_words": result.n_words,
                "n_unique_words": result.n_unique_words,
                "unique_word_ratio": result.n_unique_words / result.n_words,
                "n_nondecodable_words": result.n_nondecodable_words,
                "n_unique_nondecodable_words": result.n_unique_nondecodable_words,
                "unique_nondecodable_words": result.unique_nondecodable_words,
                "word_decodable_rate": 1.0
                - result.n_nondecodable_words / result.n_words,
                "unique_word_decodable_rate": 1.0
                - result.n_unique_nondecodable_words / result.n_unique_words,
                "graded_score": graded_score(
                    result.n_words, result.n_nondecodable_words
                ),
                "strict_pass": result.n_nondecodable_words == 0,
                "format_failure": False,
                "under_reference_length": result.n_words < REFERENCE_LENGTH,
                **{
                    f"{FAILURE_PREFIX}{name}": result.failure_causes.get(name, math.nan)
                    for name in config["scoring_methods"]
                },
            }
        )

    return pd.concat([results, results.apply(score_row, axis=1)], axis=1)


@require_columns(
    [
        "model",
        "benchmark_item_id",
        "graded_score",
        "strict_pass",
        "format_failure",
        "n_words",
        "n_nondecodable_words",
        "n_unique_words",
        "n_unique_nondecodable_words",
        "unique_word_ratio",
        "under_reference_length",
    ]
)
def summarize_by_model(results: pd.DataFrame) -> pd.DataFrame:
    """Per-model headline metrics plus diagnostics.

    ``graded_score`` and ``strict_pass_rate`` are computed over *all* items,
    counting a format failure as a zero. The pooled word rates and the
    length diagnostics are computed over scorable items only, so a formatting
    problem stays visible as itself rather than masquerading as poor phonics.

    Metrics whose denominator is zero are NaN, never 0 or 1.
    """
    grouped = results.groupby("model")
    models = grouped.size().index

    scorable = results[~results["format_failure"]]
    scorable_grouped = scorable.groupby("model")

    def pooled_rate(numerator: str, denominator: str) -> pd.Series:
        total_bad = scorable_grouped[numerator].sum().reindex(models)
        total = scorable_grouped[denominator].sum().reindex(models)
        return (1.0 - total_bad / total).where(total > 0)

    summary = pd.DataFrame(
        {
            "n_items": grouped["benchmark_item_id"].count(),
            "graded_score": grouped["graded_score"].mean(),
            "strict_pass_rate": grouped["strict_pass"].mean(),
            "word_decodable_rate": pooled_rate("n_nondecodable_words", "n_words"),
            "unique_word_decodable_rate": pooled_rate(
                "n_unique_nondecodable_words", "n_unique_words"
            ),
            "format_failure_rate": grouped["format_failure"].mean(),
            "mean_n_words": scorable_grouped["n_words"].mean().reindex(models),
            "mean_unique_word_ratio": scorable_grouped["unique_word_ratio"]
            .mean()
            .reindex(models),
            "under_reference_length_rate": scorable_grouped["under_reference_length"]
            .mean()
            .reindex(models),
        }
    )

    total_words = scorable_grouped["n_words"].sum().reindex(models)
    total_nondecodable = scorable_grouped["n_nondecodable_words"].sum().reindex(models)
    summary["nondecodable_per_100_words"] = (
        100.0 * total_nondecodable / total_words
    ).where(total_words > 0)

    # Pooled failure attribution, e.g. untaught grapheme vs untaught cluster.
    for column in results.columns:
        if column.startswith(FAILURE_PREFIX):
            summary[f"pooled_{column}"] = scorable_grouped[column].sum().reindex(models)

    return summary.rename_axis("model_name").reset_index()


def _summarize_by(results: pd.DataFrame, dimension: str) -> pd.DataFrame:
    return (
        results.groupby(["model", dimension])
        .agg(
            graded_score=("graded_score", "mean"),
            strict_pass_rate=("strict_pass", "mean"),
            format_failure_rate=("format_failure", "mean"),
            n_items=("benchmark_item_id", "count"),
        )
        .reset_index()
    )


@require_columns(["model", "curriculum_phase", "graded_score", "strict_pass"])
def summarize_by_phase(results: pd.DataFrame) -> pd.DataFrame:
    """Per-model score by student curriculum phase.

    The most informative breakdown in the benchmark: a profile that teaches four
    graphemes leaves almost no writable vocabulary, while a full inventory
    leaves most of the language available.
    """
    return _summarize_by(results, "curriculum_phase")


@require_columns(["model", "scenario_id", "graded_score", "strict_pass"])
def summarize_by_scenario(results: pd.DataFrame) -> pd.DataFrame:
    """Per-model score by generation scenario (story, dialogue, ...)."""
    return _summarize_by(results, "scenario_id")
