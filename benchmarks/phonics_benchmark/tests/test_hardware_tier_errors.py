"""Check word-length binning, pooled FP/FN rates and calibration binning."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = load("hardware_tier_errors")
by_length = load("plot_error_rate_by_word_length")
calibration = load("plot_calibration")


@pytest.mark.parametrize(
    "letters,label",
    [(1, "1-3"), (3, "1-3"), (4, "4"), (7, "7"), (8, "8+"), (15, "8+")],
)
def test_length_bin(letters, label):
    assert common.length_bin(letters) == label


def test_parse_word_set():
    assert common.parse_word_set("set()") == set()
    assert common.parse_word_set("{'a', 'bug'}") == {"a", "bug"}


def scored_row(model, tier, tp, fp, fn, tn):
    return {
        "model": model,
        "hardware_tier": tier,
        "benchmark_item_id": "item",
        "tp": set(tp),
        "fp": set(fp),
        "fn": set(fn),
        "tn": set(tn),
    }


def test_rates_pool_counts_across_models():
    # Model a: 1 of 2 short decodable words flagged. Model b: 0 of 4. Pooled 1/6,
    # not the 25% average of the two model rates.
    scored = pd.DataFrame(
        [
            scored_row("a", "Cloud", [], ["at"], [], ["up"]),
            scored_row("b", "Cloud", [], [], [], ["at", "up", "in", "on"]),
        ]
    )
    words = common.explode_words(scored)
    table = by_length.rate_by_length(words, "fp").set_index("length_bin")
    assert table.loc["1-3", "opportunities"] == 6
    assert table.loc["1-3", "errors"] == 1
    assert table.loc["1-3", "rate"] == pytest.approx(1 / 6)


def test_false_negative_rate_uses_nondecodable_words():
    scored = pd.DataFrame(
        [scored_row("a", "Cloud", ["jump"], [], ["looked", "saw"], ["at"])]
    )
    table = by_length.rate_by_length(common.explode_words(scored), "fn")
    table = table.set_index("length_bin")
    assert table.loc["4", "rate"] == 0  # jump found
    assert table.loc["6", "rate"] == 1  # looked missed
    assert table.loc["1-3", "rate"] == 1  # saw missed


def test_calibration_keeps_ratio_one_in_last_bin():
    scored = pd.DataFrame(
        [
            scored_row("a", "Cloud", ["x", "y"], [], [], []),  # actual 1.0
            scored_row("a", "Cloud", [], [], ["x"], ["y"] * 0 + ["z"]),  # actual 0.5
        ]
    )
    table = calibration.calibration_table(calibration.item_ratios(scored))
    assert set(table.actual_bin) == {5, 9}
    last = table[table.actual_bin.eq(9)].iloc[0]
    assert last.mean_actual == 1.0 and last.mean_predicted == 1.0


def test_ranking_phrase():
    assert (
        common.ranking_phrase("pooled_f2", "english")
        == "highest-pooled_f2 English models"
    )


def test_language_defaults_to_all_languages():
    import argparse

    parser = argparse.ArgumentParser()
    common.add_common_args(parser)
    args = parser.parse_args(["--root", "x"])
    assert args.language == ["english", "kiswahili"]
    assert parser.parse_args(["--root", "x", "--language", "english"]).language == [
        "english"
    ]
