"""
A stop-gap solution for ungating syllabic nasals in benchmark items.

Given a list of words with syllabic nasals in our dataset, this script
will compare them against the student knowledge to determine which
words should be _decodable_ (i.e. dropped from nondecodable words list.)

It also double-checks every word to include any accidentally dropped
nondecodable words.

Usage:

    uv run --frozen --package phonics_benchmark \
        benchmarks/phonics_benchmark/scripts/drop_syllabic_nasals.py \
        <path_to_load_items> \
        <path_to_save_items>


"""

import json
import logging
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import pandas as pd
import typer
from decodability.kiswahili import KiswahiliStudentKnowledge
from decodability.kiswahili.analyse import is_decodable_kenya_tusome
from decodability.kiswahili.segment import (
    GraphemeSpan,
    decompose_word,
    get_grapheme_spans,
)
from decodability.kiswahili.ungate_syllabic_nasals import TO_UNGATE, ok_to_ungate
from phonics_benchmark.labeling import get_vocabulary_set


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_as_jsonl(items: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item))
            f.write("\n")


@typer.run
def main(load_items_path: str, save_items_path: str, log_level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))

    items = load_jsonl(Path(load_items_path))
    # get_vocabulary_set needs a frame and only reads `content`, but the raw dicts
    # stay the source of truth so untouched items round-trip unchanged
    vocab_sets = get_vocabulary_set(
        pd.DataFrame({"content": [item["content"] for item in items]}), "sw"
    ).tolist()
    logging.info(f"Loaded {len(items)} items from {load_items_path}")

    words_to_ungate = [decompose_word(word) for word in TO_UNGATE]

    n_items_with_removal = 0
    n_items_with_addition = 0
    n_removed_words = 0
    n_added_words = 0

    for item, vocab_set in zip(items, vocab_sets, strict=True):
        new_label = item["nondecodable_words"].copy()
        removed = []
        added = []
        profile = KiswahiliStudentKnowledge.model_validate(item["student_knowledge"])

        for word_in_text, word_to_ungate in product(vocab_set, words_to_ungate):
            if word_in_text == word_to_ungate.word:
                # The word is nondecodable if it fails the baseline decodability check
                # and is not eligible to be ungated based on the student's knowledge
                # profile.
                is_decodable_by_tool = is_decodable_kenya_tusome(word_in_text, profile)
                can_be_ungated = ok_to_ungate(word_to_ungate, profile)
                nondecodable = not is_decodable_by_tool and not can_be_ungated

                if nondecodable and word_in_text not in new_label:
                    # Add it (back) to the nondecodable words list
                    new_label.append(word_in_text)
                    added.append(word_in_text)
                elif not nondecodable and word_in_text in new_label:
                    # Remove it from the nondecodable words list
                    new_label.remove(word_in_text)
                    removed.append(word_in_text)

        if removed or added:
            if removed:
                n_items_with_removal += 1
                n_removed_words += len(removed)
                logging.info(
                    f"Removed from {item['benchmark_item_id']}'s nondecodable words: "
                    f"{removed}"
                )

            if added:
                n_items_with_addition += 1
                n_added_words += len(added)
                logging.info(
                    f"Added to {item['benchmark_item_id']}'s nondecodable words: "
                    f"{added}"
                )

            item["nondecodable_words"] = new_label
            item["num_nondecodable_words"] = len(new_label)
            item["nondecodable_ratio"] = len(new_label) / len(vocab_set)

    save_as_jsonl(items, Path(save_items_path))
    logging.info(f"Saved updated items to {save_items_path}")
    logging.info(f"Number of items with removal: {n_items_with_removal}")
    logging.info(f"Number of removed words: {n_removed_words}")
    logging.info(f"Number of items with addition: {n_items_with_addition}")
    logging.info(f"Number of added words: {n_added_words}")
