"""Review difficult items for decodability assessment benchmark.

We look for items that all of the top 10 models get wrong.

Prerequisite:
    Latest results of decodability assessment benchmark results downloaded locally
    e.g. the result of the following command:

    ```bash
    uv run --frozen --package phonics-benchmark python \
    benchmarks/phonics_benchmark/scripts/download_unified_leaderboard_runs.py \
    --root outputs/phonics-gold-leaderboard-20260922-from-gcs
    ```

Usage:
    python benchmarks/phonics_benchmark/scripts/review_difficult_items.py \
        --language <language> \
        --target <target_directory>


    By default, --target is set to
    "outputs/phonics-gold-leaderboard-20260922-from-gcs/reports/"

Saves results in <target_directory>/<language>/difficult_items/
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

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


LANGUAGE = parser.parse_args().language
REPO_ROOT = (
    Path(__file__).resolve().parents[3] if "__file__" in globals() else Path.cwd()
)
BASE = REPO_ROOT / parser.parse_args().target

REPORT_DIR = BASE / f"{LANGUAGE}"
TARGET_DIR = REPORT_DIR / "difficult_items"
TARGET_DIR.mkdir(parents=True, exist_ok=True)

TO_UNGATE = [
    "akamsalimia",
    "akamtuma",
    "alimnunulia",
    "alimpeleka",
    "alimpenda",
    "hakumpenda",
    "humpeleka",
    "inamlinda",
    "kumchapa",
    "kumfukuza",
    "kumkaribisha",
    "mbunge",
    "mbwa",
    "mchana",
    "mchanganyiko",
    "mchele",
    "mchezo",
    "mchi",
    "mchwa",
    "mdogo",
    "mdomo",
    "mdudu",
    "mfalme",
    "mfano",
    "mfupi",
    "mgeni",
    "mgonjwa",
    "mheshimiwa",
    "mjini",
    "mjomba",
    "mkali",
    "mkate",
    "mkono",
    "mkononi",
    "mkubwa",
    "mkuu",
    "mlango",
    "mlingoti",
    "mmoja",
    "mpeketoni",
    "mpunga",
    "mraba",
    "mrefu",
    "msafiri",
    "msalani",
    "msichana",
    "msingi",
    "msituni",
    "mstatili",
    "mti",
    "mtihani",
    "mtini",
    "mtoto",
    "mtu",
    "mvua",
    "mzazi",
    "mzee",
    "nchi",
    "nne",
    "wamsalimie",
    "wanamjua",
    "zilizompatia",
    "nzi",
    "kuamka",
    "kiamshakinywa",
    "kuzungumza",
    "namna",
    "umri",
    "huteremsha",
    "nje",
]


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


def get_top10_models():
    model_summary = pd.read_csv(REPORT_DIR / "model_summary_eligible.csv")
    top10_models = (
        model_summary.sort_values(by="adjusted_balanced_accuracy", ascending=False)
        .head(10)["model"]
        .tolist()
    )
    return top10_models


if __name__ == "__main__":
    results = load_results()
    top10_models = get_top10_models()
    logging.info(f"Top 10 models: {top10_models}")

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
        # import from package once implemented
        # from decodability.kiswahili.ungate_syllabic_nasal import TO_UNGATE
        fp_rate = (
            fp_words_for_top10_models.index.get_level_values("word")
            .isin(TO_UNGATE)
            .mean()
        )
        fn_rate = (
            fn_words_for_top10_models.index.get_level_values("word")
            .isin(TO_UNGATE)
            .mean()
        )
        logging.info(
            f"Proportion of fp words for top 10 models that are in TO_UNGATE: {fp_rate}"
        )
        logging.info(
            f"Proportion of fn words for top 10 models that are in TO_UNGATE: {fn_rate}"
        )

    all_wrong_words_mask = ~words.groupby(["benchmark_item_id", "word"])[
        "correct"
    ].any()
    all_wrong_words = all_wrong_words_mask[all_wrong_words_mask].index.tolist()
    logging.info(f"Words that all models got wrong: \n{all_wrong_words}")

    all_wrong_top10_models_words_mask = ~words.groupby(["benchmark_item_id", "word"])[
        "correct"
    ].any()
    all_wrong_top10_models_words = all_wrong_top10_models_words_mask[
        all_wrong_top10_models_words_mask
    ].index.tolist()
    logging.info(
        f"Words that all top 10 models got wrong: \n{all_wrong_top10_models_words}"
    )
