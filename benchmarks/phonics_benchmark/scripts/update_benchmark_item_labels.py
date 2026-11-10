"""
Relabel benchmark items using the latest decodability rules for their language.

Each item's non-decodable words are recomputed with the same scoring methods and
aggregation used to build benchmark items (``DECODABILITY_CONFIGS`` in
``phonics_benchmark.labeling``), and compared with the existing labels.

It will log a report on

- how many items were re-labelled
- how many and which words were
  - removed (previously non-decodable but now decodable)
  - added (previously decodable but now non-decodable)

Usage:

    uv run --frozen --package phonics_benchmark \
        benchmarks/phonics_benchmark/scripts/update_benchmark_item_labels.py \
        <existing_items_path> \
        <target_items_path> \
        [--language sw]

The language defaults to the items' ``language_iso_code``. Saves the re-labelled
version of the existing items dataset to the target items path.
"""

import json
import logging
from collections import Counter
from functools import cache
from pathlib import Path

import pandas as pd
import typer
from decodability.kiswahili.segment import decompose_word
from phonics_benchmark.curation import get_nondecodable_word_analysis
from phonics_benchmark.data_io import convert_records_to_dataframe
from phonics_benchmark.labeling import (
    get_decodability_config,
    get_vocabulary_set,
    label_nondecodable_words,
)
from phonics_benchmark.utils import load_jsonl, require_single_language_iso_code

# Graphemes that can head a Kiswahili nasal boundary; matches the keys of
# _NASAL_CLUSTER_FOLLOWERS in decodability.kiswahili.segment
KISWAHILI_NASALS = frozenset({"m", "n"})


@cache  # called once per word per report line
def has_nasal_boundary(word: str) -> bool:
    """Whether the Kiswahili nasal rules judge any boundary in ``word``.

    The rules only run on a nasal grapheme followed by a consonant grapheme, so
    a word without one gets the same verdict with or without them.
    """
    graphemes = decompose_word(word)
    return any(
        current.symbol in KISWAHILI_NASALS and following.is_consonant
        for current, following in zip(graphemes, graphemes[1:])
    )


def log_kiswahili_nasal_report(
    removed_by_word: Counter[str],
    added_by_word: Counter[str],
    unchanged_by_word: Counter[str],
) -> None:
    """Split the changes by whether a word has a nasal boundary.

    Changes to words without one cannot come from the nasal rules, so a non-zero
    count there points at some other change in the decodability tool.
    """
    for label, is_nasal in [("Nasal-boundary", True), ("Non-nasal", False)]:
        removed = Counter(
            {
                w: n
                for w, n in removed_by_word.items()
                if has_nasal_boundary(w) == is_nasal
            }
        )
        added = Counter(
            {
                w: n
                for w, n in added_by_word.items()
                if has_nasal_boundary(w) == is_nasal
            }
        )
        unchanged = sum(
            n for w, n in unchanged_by_word.items() if has_nasal_boundary(w) == is_nasal
        )
        logging.info(
            f"{label} words: {removed.total()} removed ({len(removed)} distinct), "
            f"{added.total()} added ({len(added)} distinct), "
            f"{unchanged} occurrences unchanged"
        )
        if not is_nasal and (removed or added):
            logging.info(f"  Non-nasal removed: {dict(removed.most_common(15))}")
            logging.info(f"  Non-nasal added: {dict(added.most_common(15))}")


def save_as_jsonl(items: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item))
            f.write("\n")


@typer.run
def main(
    load_items_path: str,
    save_items_path: str,
    language: str | None = typer.Option(
        None,
        "--language-code",
        help="Language ISO code, e.g. 'sw' or 'en'. Defaults to the items' own.",
    ),
    log_level: str = "INFO",
) -> None:
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))

    assert load_items_path != save_items_path, "Load and save paths must be different"

    items = load_jsonl(Path(load_items_path))
    language = require_single_language_iso_code(
        (item.get("language_iso_code", "") for item in items),
        source="the benchmark items",
        expected=language,
    )
    config = get_decodability_config(language)
    logging.info(f"Loaded {len(items)} items from {load_items_path} ({language})")

    # Copy required data
    items_df = pd.DataFrame(
        {
            "content": [item["content"] for item in items],
            "student_knowledge": [item["student_knowledge"] for item in items],
            "num_nondecodable_words": [
                item["num_nondecodable_words"] for item in items
            ],
            "nondecodable_ratio": [item["nondecodable_ratio"] for item in items],
        }
    )
    items_df["vocab_set"] = get_vocabulary_set(items_df, language)

    # Report existing nondecodable word stats
    analysis_pre = get_nondecodable_word_analysis(pd.DataFrame(items))

    # Create new labels
    new_labels = label_nondecodable_words(
        items_df, config, language, score_column=config["score_column"]
    )

    # Count label changes
    n_items_with_removal = 0
    n_items_with_addition = 0
    n_items_changed = 0
    removed_by_word: Counter[str] = Counter()
    added_by_word: Counter[str] = Counter()
    unchanged_by_word: Counter[str] = Counter()

    for item, vocab_set, new_label in zip(
        items, items_df["vocab_set"], new_labels, strict=True
    ):
        old = set(item["nondecodable_words"])
        new = set(new_label)
        removed = sorted(old - new)
        added = sorted(new - old)
        unchanged_by_word.update(vocab_set - old.symmetric_difference(new))

        if removed:
            n_items_with_removal += 1
            removed_by_word.update(removed)
            logging.debug(
                f"Removed from {item['benchmark_item_id']}'s nondecodable words: "
                f"{removed}"
            )
        if added:
            n_items_with_addition += 1
            added_by_word.update(added)
            logging.debug(
                f"Added to {item['benchmark_item_id']}'s nondecodable words: {added}"
            )
        if removed or added:
            n_items_changed += 1
            # Sorted, matching how label_nondecodable_words writes new items
            item["nondecodable_words"] = new_label
            item["num_nondecodable_words"] = len(new_label)
            item["num_total_words"] = len(vocab_set)
            item["nondecodable_ratio"] = len(new_label) / len(vocab_set)

    # Save updated items to JSONL
    save_as_jsonl(items, Path(save_items_path))
    logging.info(f"Saved updated items to {save_items_path}")

    # Report label change statistics
    logging.info(
        f"Items with changed labels: {n_items_changed} / {len(items)} "
        f"({n_items_changed / len(items):.1%})"
    )
    logging.info(f"Items with removal: {n_items_with_removal} / {len(items)}")
    logging.info(
        f"Removed words: {removed_by_word.total()} occurrences, "
        f"{len(removed_by_word)} distinct"
    )
    logging.info(f"Items with addition: {n_items_with_addition} / {len(items)}")
    logging.info(
        f"Added words: {added_by_word.total()} occurrences, "
        f"{len(added_by_word)} distinct"
    )
    logging.info(f"Word occurrences left unchanged: {unchanged_by_word.total()}")
    logging.info(f"Removed, by word: {dict(removed_by_word.most_common())}")
    logging.info(f"Added, by word: {dict(added_by_word.most_common())}")

    if language == "sw":
        log_kiswahili_nasal_report(removed_by_word, added_by_word, unchanged_by_word)

    # Report new nondecodable word stats
    analysis_post = get_nondecodable_word_analysis(pd.DataFrame(items))
    report = analysis_post.copy()
    for key, post in analysis_post.items():
        if key == "Non-decodable ratio histogram":
            for subkey, post_val in post.items():
                pre = analysis_pre["Non-decodable ratio histogram"][subkey]
                report[key][subkey] = f"{pre:3d} -> {post_val:3d} ({post_val - pre:+})"
            continue
        else:
            pre = analysis_pre[key]
            report[key] = f"{pre:3d} -> {post:3d} ({post - pre:+})"
    logging.info(f"Analysis post-update:\n{json.dumps(analysis_post, indent=2)}")
