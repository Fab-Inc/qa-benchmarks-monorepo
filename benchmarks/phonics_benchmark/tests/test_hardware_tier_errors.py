"""Check word-length binning, pooled FP/FN rates and calibration binning."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
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
        == "English models with the highest pooled F2"
    )
    assert (
        common.ranking_phrase("adjusted_balanced_accuracy", "kiswahili")
        == "Kiswahili models with the highest adjusted balanced accuracy"
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


def test_inputs_are_read_from_the_reports_folder(tmp_path):
    # Gold leaderboard runs keep everything under <root>/reports/.
    tiers = tmp_path / "reports" / "hardware_tiers" / "pooled_f2"
    tiers.mkdir(parents=True)
    pd.DataFrame(
        {
            "model": ["a", "b"],
            "language": ["english", "kiswahili"],
            "hardware_tier": ["Cloud", "Cloud"],
        }
    ).to_csv(tiers / "hardware_tiers_top10_models.csv", index=False)
    scoring = tmp_path / "reports" / "english"
    scoring.mkdir()
    pd.DataFrame(
        {
            "model": ["a", "a", "other"],
            "benchmark_item_id": ["i1", "i2", "i1"],
            "success": [True, True, True],
            "parse_ok": [True, False, True],
            "tp": ["{'jump'}", "set()", "set()"],
            "fp": ["set()", "set()", "set()"],
            "fn": ["{'saw'}", "set()", "set()"],
            "tn": ["{'at'}", "set()", "set()"],
        }
    ).to_csv(scoring / "results_with_scoring.csv", index=False)

    assert common.tier_dir(tmp_path, "pooled_f2", None) == tiers
    models = common.load_top_models(tmp_path, "english", 10, "pooled_f2", None)
    assert models.model.tolist() == ["a"]
    scored = common.load_scored_items(tmp_path, "english", models)
    # Unselected models and unparseable responses are dropped.
    assert scored.benchmark_item_id.tolist() == ["i1"]
    assert scored.tp.iloc[0] == {"jump"}


def test_bootstrap_totals_redraw_every_item_count():
    counts = np.ones((50, 2, 3))  # 50 items, each contributing 1 everywhere
    totals = common.bootstrap_totals(counts, n_bootstrap=20, seed=0)
    assert totals.shape == (20, 2, 3)
    assert (totals == 50).all()  # a replicate draws exactly 50 items


def test_bootstrap_totals_are_reproducible_for_a_seed():
    counts = np.random.default_rng(1).integers(0, 5, size=(30, 2, 2)).astype(float)
    first = common.bootstrap_totals(counts, 10, seed=3)
    assert np.array_equal(first, common.bootstrap_totals(counts, 10, seed=3))
    assert not np.array_equal(first, common.bootstrap_totals(counts, 10, seed=4))


def synthetic_words(n_items=200):
    """Cloud models flag ~25% of decodable 4-letter words, over many items."""
    rng = np.random.default_rng(0)
    rows = []
    for item in range(n_items):
        outcome = "fp" if rng.random() < 0.25 else "tn"
        rows.append(("m", "Cloud", f"i{item}", "word", outcome, 4, "4"))
    return pd.DataFrame(
        rows,
        columns=[
            "model",
            "hardware_tier",
            "benchmark_item_id",
            "word",
            "outcome",
            "letters",
            "length_bin",
        ],
    )


def test_rate_interval_brackets_the_point_estimate():
    words = synthetic_words()
    table = by_length.rate_by_length(words, "fp", n_bootstrap=500, seed=0)
    row = table.iloc[0]
    assert row.rate_ci_low <= row.rate <= row.rate_ci_high
    assert row.rate_ci_high - row.rate_ci_low < 0.2
    assert row.n_bootstrap == 500 and row.ci_level == 0.95


def test_no_bootstrap_columns_when_switched_off():
    table = by_length.rate_by_length(synthetic_words(), "fp", n_bootstrap=0)
    assert not {"rate_ci_low", "rate_ci_high", "n_bootstrap"} & set(table.columns)


def test_calibration_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    rows = [
        {
            "model": "m",
            "hardware_tier": "Cloud",
            "benchmark_item_id": f"i{i}",
            "actual": 0.25,
            "predicted": float(rng.choice([0.1, 0.2, 0.3, 0.4])),
        }
        for i in range(120)
    ]
    table = calibration.calibration_table(pd.DataFrame(rows), n_bootstrap=300, seed=0)
    row = table.iloc[0]
    assert row.mean_predicted_ci_low <= row.mean_predicted <= row.mean_predicted_ci_high
    assert row.mean_actual_ci_low == row.mean_actual_ci_high == 0.25
    plain = calibration.calibration_table(pd.DataFrame(rows))
    assert "mean_predicted_ci_low" not in plain.columns


def test_bootstrap_args_default_on_and_validated():
    import argparse

    parser = argparse.ArgumentParser()
    common.add_common_args(parser)
    args = parser.parse_args(["--root", "x"])
    assert (args.n_bootstrap, args.ci_level, args.seed) == (1000, 0.95, 0)
    bad = parser.parse_args(["--root", "x", "--ci-level", "1.5"])
    with pytest.raises(SystemExit):
        common.validate_bootstrap_args(parser, bad)
