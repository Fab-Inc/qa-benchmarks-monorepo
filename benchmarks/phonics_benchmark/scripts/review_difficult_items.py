"""Review difficult items for decodability assessment benchmark.

We look for items that all of the top 10 models get wrong, and for items that
are hard and discriminate poorly: few models get them right, and the ones that
do are not the strong ones (see ``phonics_benchmark.item_statistics``). For
the latter, every model's prediction is written out, right and wrong, for
inspection by hand.

We also look the other way, for items and item-words that are too easy to tell
models apart: those no model gets wrong, and those none of the top 10 gets
wrong. Both are reported with the counts behind them, so an entry the leaders
all get right while a weaker model misses it can be told apart from one nobody
misses at all.

Prerequisite:
    Latest results of decodability assessment benchmark results downloaded locally
    e.g. the result of the following command:

    ```bash
    uv run --frozen --package phonics-benchmark python \
    benchmarks/phonics_benchmark/scripts/download_unified_leaderboard_runs.py \
    --root outputs/phonics-gold-leaderboard-20260922-from-gcs
    ```

Usage:
    uv run --frozen --package phonics-benchmark python \
        benchmarks/phonics_benchmark/scripts/review_difficult_items.py \
        --language <language> \
        --target <target_directory>


    By default, --target is set to
    "outputs/phonics-gold-leaderboard-20260922-from-gcs/reports/"

    Hard, low-discrimination items are those with at most
    --max-proportion-correct of models right (and at least one) and a mean
    rank of the models that got them right of at most --max-mean-rank, ranking
    models by adjusted balanced accuracy.

Saves results in <target_directory>/<language>/difficult_items/, including
too_easy_items.csv and too_easy_item_words.csv for the easy end.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from decodability.kiswahili.segment import SYLLABIC_NASAL_REVIEWED_WORDS
from phonics_benchmark.item_statistics import (
    STATUS_INTERIOR,
    simple_difficulty_and_discrimination,
)

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(
    description="Review difficult items for decodability assessment benchmark."
)
parser.add_argument(
    "--language",
    type=str,
    default="english",
    help="Language of the phonics benchmark (default: english)",
)
parser.add_argument(
    "--target",
    type=str,
    default="outputs/phonics-gold-leaderboard-20260922-from-gcs/reports/",
    help="Target directory to save results, under /<language>/difficult_items/",
)
parser.add_argument(
    "--max-proportion-correct",
    type=float,
    default=0.2,
    help="Hardest items to review: at most this share of models right (default: 0.2)",
)
parser.add_argument(
    "--max-mean-rank",
    type=float,
    default=0.7,
    help=(
        "Least discriminating items to review: mean rank of the models that got "
        "them right at most this, 0 weakest to 1 strongest (default: 0.7)"
    ),
)


ARGS = parser.parse_args()
LANGUAGE = ARGS.language
REPO_ROOT = (
    Path(__file__).resolve().parents[3] if "__file__" in globals() else Path.cwd()
)
BASE = REPO_ROOT / ARGS.target

REPORT_DIR = BASE / f"{LANGUAGE}"
TARGET_DIR = REPORT_DIR / "difficult_items"
TARGET_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Prepare data
# --------------------------------------------------------------------------
def load_results():
    """Item-level results for tiered models

    Only models the tier file places are kept. Dropping them here rather than
    at each call site keeps the model set consistent: groupby discards rows
    with a NaN key, so a tally keyed on hardware_tier would lose the untiered
    models while one keyed on model alone would keep them.
    """
    results = pd.read_csv(REPORT_DIR / "results_with_scoring.csv")
    tiers = pd.read_csv(
        BASE / f"hardware_tiers/pooled_f2/hardware_tiers_{LANGUAGE}.csv"
    )
    results = results.merge(tiers[["model", "hardware_tier"]], on="model", how="inner")
    results["vocab_set"] = results.vocab_set.apply(eval)
    for col in ["response_parsed", "nondecodable_words"]:
        results[col] = results[col].apply(lambda cell: {w.lower() for w in eval(cell)})
    return results


def build(results=None):
    """Explode items to words and attach decodability features.

    The two library calls have different keys, so they are deduplicated
    separately. decompose_word depends only on spelling, so it runs once per
    unique word. analyse_word_kiswahili also needs the student profile, which
    is fixed within a benchmark item, so it runs once per (word, profile) pair
    rather than once per row across all models.
    """
    df = load_results() if results is None else results

    item_words = (
        df[
            [
                "benchmark_item_id",
                "model",
                "hardware_tier",
                "vocab_set",
                "student_knowledge",
                "response_parsed",
                "nondecodable_words",
            ]
        ]
        .explode("vocab_set")
        .rename(columns={"vocab_set": "word"})
    )

    pairs = item_words[["word", "student_knowledge"]].drop_duplicates()
    item_words = item_words.merge(pairs, on=["word", "student_knowledge"])

    item_words["predicted"] = [
        w.lower() in r for w, r in zip(item_words.word, item_words.response_parsed)
    ]
    item_words["is_nondecodable"] = [
        w.lower() in n for w, n in zip(item_words.word, item_words.nondecodable_words)
    ]
    return item_words.drop(columns=["response_parsed", "nondecodable_words"])


def get_model_scores():
    """Adjusted balanced accuracy per model, the leaderboard's metric."""
    model_summary = pd.read_csv(REPORT_DIR / "model_summary_eligible.csv")
    return model_summary.set_index("model")["adjusted_balanced_accuracy"]


def get_top10_models():
    return get_model_scores().nlargest(10).index.tolist()


def select_hard_low_discrimination_items(
    results, model_scores, max_proportion_correct, max_mean_rank
):
    """Item statistics for the difficult, poorly discriminating items, where
    - difficult: low proportion of models that got it right
    - poorly discriminating: low mean rank among the models that got it right
    sorted in ascending order by mean rank and then proportion correct

    Excludes items that no model got right since they have no mean rank.
    """
    items = simple_difficulty_and_discrimination(
        results, model_scores=model_scores, item_metadata_columns=()
    )
    selected = items[
        (items["status"] == STATUS_INTERIOR)
        & (items["proportion_correct"] <= max_proportion_correct)
        & (items["mean_rank_of_correct"] <= max_mean_rank)
    ]
    return selected.sort_values(["mean_rank_of_correct", "proportion_correct"])


def predictions_for_items(results, items, model_scores):
    """Every model's prediction on each of ``items``, right and wrong.

    One row per item and model, with the item's statistics alongside so the
    file can be read on its own. Within an item, models run strongest first.
    """
    model_rank = (
        model_scores.rank(ascending=False, method="min")
        .astype(int)
        .rename("model_rank")
    )
    predictions = results[results["benchmark_item_id"].isin(items["benchmark_item_id"])]
    predictions = predictions.assign(
        model_rank=predictions["model"].map(model_rank),
        # Sorted lists rather than sets so the CSV reads the same on every run.
        predicted=predictions["response_parsed"].map(sorted),
        gold=predictions["nondecodable_words"].map(sorted),
        extra_words=[
            sorted(p - g)
            for p, g in zip(predictions.response_parsed, predictions.nondecodable_words)
        ],
        missed_words=[
            sorted(g - p)
            for p, g in zip(predictions.response_parsed, predictions.nondecodable_words)
        ],
    )
    columns = [
        "benchmark_item_id",
        "model",
        "model_rank",
        "hardware_tier",
        "perfect_match",
        "predicted",
        "gold",
        "extra_words",
        "missed_words",
        "student_knowledge",
        "content",
    ]
    item_columns = [
        "benchmark_item_id",
        "models_correct",
        "models_answered",
        "proportion_correct",
        "mean_rank_of_correct",
        "chance_rank_low",
        "chance_rank_high",
    ]
    merged = items[item_columns].merge(
        predictions[columns], on="benchmark_item_id", validate="one_to_many"
    )
    # Keep the items in selection order (weakest first), then models by rank.
    item_order = {item: i for i, item in enumerate(items["benchmark_item_id"])}
    merged["item_order"] = merged["benchmark_item_id"].map(item_order)
    return merged.sort_values(["item_order", "model_rank"]).drop(columns="item_order")


def _tally_correct(frame, keys, correct_column, models=None):
    """Answers correct and answers given, per group.

    Counted over the rows present: a model that never answered an item has no
    row, so it is left out of both counts rather than counted as wrong. That is
    the ``models_answered`` convention of ``item_statistics``, and the one the
    all-wrong tallies in ``__main__`` already follow.
    """
    if models is not None:
        frame = frame[frame["model"].isin(models)]
    grouped = frame.groupby(keys)[correct_column]
    return pd.DataFrame(
        {"models_correct": grouped.sum(), "models_answered": grouped.size()}
    )


def _select_too_easy(frame, keys, correct_column, top_models, metadata_columns):
    """Groups every model, or every top model, got right.

    The mirror of the all-wrong selections: a group is too easy when nobody
    gets it wrong. ``keys`` makes the group either an item (the item id) or an
    item-word (the item id and the word).

    One row per group, carrying both flags and the counts behind them. Keeping
    the two flags side by side separates the genuinely easy entries, which no
    model misses, from the ones that only look easy at the top and still
    discriminate lower down the leaderboard.
    """
    overall = _tally_correct(frame, keys, correct_column)
    # Reindexed on the full set, so a group no top model answered comes back as
    # NaN rather than being dropped or read as zero of zero correct.
    top = _tally_correct(frame, keys, correct_column, top_models).reindex(overall.index)

    selected = overall.assign(
        proportion_correct=overall["models_correct"] / overall["models_answered"],
        all_models_correct=overall["models_correct"] == overall["models_answered"],
        top_models_correct=top["models_correct"],
        top_models_answered=top["models_answered"],
        # A group no top model answered is not one they all got right.
        all_top_models_correct=(top["models_answered"] > 0)
        & (top["models_correct"] == top["models_answered"]),
    )
    # Filled only now that the flags are set, so the fill cannot turn an
    # unanswered group into a spurious zero-of-zero success.
    selected[["top_models_correct", "top_models_answered"]] = (
        selected[["top_models_correct", "top_models_answered"]].fillna(0).astype(int)
    )
    selected = selected[
        selected["all_models_correct"] | selected["all_top_models_correct"]
    ]

    metadata = frame[[*keys, *metadata_columns]].drop_duplicates(subset=keys)
    selected = selected.reset_index().merge(metadata, on=keys, validate="one_to_one")
    # Easiest first: the ones nobody misses, then by how close the rest came.
    return selected.sort_values(
        ["all_models_correct", "proportion_correct"], ascending=False
    )


def select_too_easy_items(results, top_models):
    """Benchmark items every model, or every top model, matched exactly."""
    too_easy = _select_too_easy(
        results,
        ["benchmark_item_id"],
        "perfect_match",
        top_models,
        [
            "student_knowledge",
            "num_total_words",
            "num_nondecodable_words",
            "nondecodable_words",
            "content",
        ],
    )
    # Sorted lists rather than sets so the CSV reads the same on every run.
    return too_easy.assign(
        nondecodable_words=too_easy["nondecodable_words"].map(sorted)
    )


def select_too_easy_item_words(words, top_models):
    """Words within an item every model, or every top model, labelled right.

    A decodable word nobody flags is a weaker signal than a nondecodable one
    nobody misses, so the gold label rides along for sorting out which is which.
    The student profile does not: it runs to a couple of kilobytes and would be
    repeated on every one of tens of thousands of rows. Join on the item id to
    get it back.
    """
    return _select_too_easy(
        words,
        ["benchmark_item_id", "word"],
        "correct",
        top_models,
        ["is_nondecodable"],
    )


if __name__ == "__main__":
    results = load_results()
    top10_models = get_top10_models()
    logging.info(f"Top 10 models: {top10_models}")

    # Hard, poorly discriminating items, with every model's prediction on them.
    model_scores = get_model_scores()
    hard_items = select_hard_low_discrimination_items(
        results, model_scores, ARGS.max_proportion_correct, ARGS.max_mean_rank
    )
    logging.info(
        f"number of items with proportion correct <= {ARGS.max_proportion_correct} "
        f"and mean rank of correct models <= {ARGS.max_mean_rank}: {len(hard_items)}"
    )
    hard_items.to_csv(TARGET_DIR / "hard_low_discrimination_items.csv", index=False)
    predictions_for_items(results, hard_items, model_scores).to_csv(
        TARGET_DIR / "hard_low_discrimination_predictions.csv", index=False
    )

    cols_to_drop = [
        "call_status",
        "error_message",
        "error_type",
        "language",
        "latency_seconds",
        "max_tokens",
        "config_id",
        "config_path",
        "config_snapshot_path",
        "prompt_name",
        "model_call_id",
        "output",
        "prompt",
        "result_id",
        "run_id",
        "success",
        "task",
        "timestamp_utc",
        "nondecodable_ratio",
        "num_nondecodable_words",
        "tokens_used_completion",
        "response_parsed",
        "parse_ok",
        "num_total_words",
        "vocab_set",
        "perfect_match",
        "tp_count",
        "fp_count",
        "fn_count",
        "tn_count",
        "has_actual_nondecodable",
        "has_predicted_nondecodable",
        "precision",
        "recall",
        "f1",
    ]

    all_wrong_items_mask = ~results.groupby("benchmark_item_id")["perfect_match"].any()
    all_wrong_items = all_wrong_items_mask[all_wrong_items_mask].index.tolist()
    all_wrong_results = results[results.benchmark_item_id.isin(all_wrong_items)]
    all_wrong_results = all_wrong_results.drop(columns=cols_to_drop)
    all_wrong_results.to_csv(TARGET_DIR / "all_wrong_results.csv", index=False)

    all_wrong_top10_models_items_mask = (
        ~results[results["model"].isin(top10_models)]
        .groupby("benchmark_item_id")["perfect_match"]
        .any()
    )
    all_wrong_top10_models_items = all_wrong_top10_models_items_mask[
        all_wrong_top10_models_items_mask
    ].index.tolist()
    logging.info(f"number of items all models get wrong: {len(all_wrong_items)}")
    logging.info(
        f"number of items all top 10 models get wrong: "
        f"{len(all_wrong_top10_models_items)}"
    )
    only_top10_models_wrong_items = set(all_wrong_top10_models_items) - set(
        all_wrong_items
    )
    logging.info(
        f"number of items only the top10 models get wrong: "
        f"{len(only_top10_models_wrong_items)}"
    )
    results[
        (results.benchmark_item_id.isin(only_top10_models_wrong_items))
        & results.model.isin(top10_models)
    ].drop(columns=cols_to_drop).to_csv(
        TARGET_DIR / "only_top10_models_wrong_items.csv", index=False
    )

    # The easy end of the same scale: items nobody gets wrong carry no
    # information about which model is better.
    too_easy_items = select_too_easy_items(results, top10_models)
    n_items = results["benchmark_item_id"].nunique()
    n_items_all_right = int(too_easy_items["all_models_correct"].sum())
    n_items_top10_right = int(too_easy_items["all_top_models_correct"].sum())
    logging.info(
        f"number of items all models get right: {n_items_all_right} of {n_items} "
        f"({n_items_all_right / n_items:.1%})"
    )
    logging.info(
        f"number of items top 10 models get right: {n_items_top10_right} of "
        f"{n_items} ({n_items_top10_right / n_items:.1%})"
    )
    too_easy_items.to_csv(TARGET_DIR / "too_easy_items.csv", index=False)

    # word level
    words = build(results)
    words["correct"] = words.predicted == words.is_nondecodable
    words["cm"] = "fn"
    words.loc[words.predicted & words.is_nondecodable, "cm"] = "tp"
    words.loc[~words.predicted & words.is_nondecodable, "cm"] = "fn"
    words.loc[~words.predicted & ~words.is_nondecodable, "cm"] = "tn"
    words.loc[words.predicted & ~words.is_nondecodable, "cm"] = "fp"
    all_wrong_item_word_mask = words.benchmark_item_id.isin(all_wrong_items)
    all_wrong_top10_models_items_mask = words.benchmark_item_id.isin(
        all_wrong_top10_models_items
    )
    fp_grouped = words.loc[all_wrong_item_word_mask & (words["cm"] == "fp")].groupby(
        ["benchmark_item_id", "student_knowledge", "word"]
    )
    fp_words_for_top10_models = (
        words.loc[
            all_wrong_top10_models_items_mask
            & (words["cm"] == "fp")
            & (words["model"].isin(top10_models))
        ]
        .groupby(["benchmark_item_id", "student_knowledge", "word"])["word"]
        .count()
    )

    fp_words_for_top10_models.to_csv(
        TARGET_DIR / "fp_words_top10_models.csv", index=True
    )

    fn_words_for_top10_models = (
        words.loc[
            all_wrong_top10_models_items_mask
            & (words["cm"] == "fn")
            & (words["model"].isin(top10_models))
        ]
        .groupby(["benchmark_item_id", "student_knowledge", "word"])["word"]
        .count()
    )
    fn_words_for_top10_models.to_csv(
        TARGET_DIR / "fn_words_top10_models.csv", index=True
    )

    if LANGUAGE == "kiswahili":
        fp_rate = (
            fp_words_for_top10_models.index.get_level_values("word")
            .isin(SYLLABIC_NASAL_REVIEWED_WORDS)
            .mean()
        )
        fn_rate = (
            fn_words_for_top10_models.index.get_level_values("word")
            .isin(SYLLABIC_NASAL_REVIEWED_WORDS)
            .mean()
        )
        logging.info(
            f"Proportion of fp words for top 10 models that are in "
            f"SYLLABIC_NASAL_REVIEWED_WORDS: {fp_rate}"
        )
        logging.info(
            f"Proportion of fn words for top 10 models that are in "
            f"SYLLABIC_NASAL_REVIEWED_WORDS: {fn_rate}"
        )

    all_wrong_words_mask = ~words.groupby(["benchmark_item_id", "word"])[
        "correct"
    ].any()
    all_wrong_words = all_wrong_words_mask[all_wrong_words_mask].index.tolist()
    logging.info(f"Words that all models got wrong: \n{all_wrong_words}")

    all_wrong_top10_models_words_mask = (
        ~words[words["model"].isin(top10_models)]
        .groupby(["benchmark_item_id", "word"])["correct"]
        .any()
    )
    all_wrong_top10_models_words = all_wrong_top10_models_words_mask[
        all_wrong_top10_models_words_mask
    ].index.tolist()
    logging.info(
        f"Words that all top 10 models got wrong: \n{all_wrong_top10_models_words}"
    )

    # Item-words at the easy end. Counts rather than the full lists: these sets
    # run to thousands of entries, so the detail goes to the CSV.
    too_easy_item_words = select_too_easy_item_words(words, top10_models)
    n_item_words = len(words[["benchmark_item_id", "word"]].drop_duplicates())
    n_words_all_right = int(too_easy_item_words["all_models_correct"].sum())
    n_words_top10_right = int(too_easy_item_words["all_top_models_correct"].sum())
    n_words_only_top10_right = int(
        (
            too_easy_item_words["all_top_models_correct"]
            & ~too_easy_item_words["all_models_correct"]
        ).sum()
    )
    logging.info(
        f"number of item-words all models get right: {n_words_all_right} of "
        f"{n_item_words} ({n_words_all_right / n_item_words:.1%})"
    )
    logging.info(
        f"number of item-words all top 10 models get right: "
        f"{n_words_top10_right} of {n_item_words} "
        f"({n_words_top10_right / n_item_words:.1%})"
    )
    logging.info(
        f"number of item-words only the top 10 models get right: "
        f"{n_words_only_top10_right}"
    )
    # Split by gold label: a decodable word nobody flags is near-free, while a
    # nondecodable one nobody misses is a real but exhausted item.
    all_right_words_by_label = too_easy_item_words.loc[
        too_easy_item_words["all_models_correct"], "is_nondecodable"
    ].value_counts()
    logging.info(
        "item-words all models get right, by whether the word is nondecodable:"
        f"\n{all_right_words_by_label.to_string()}"
    )
    too_easy_item_words.to_csv(TARGET_DIR / "too_easy_item_words.csv", index=False)
