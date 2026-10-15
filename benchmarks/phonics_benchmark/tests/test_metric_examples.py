"""Run the documented metric and plotting examples without writing reports."""

import importlib
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
examples = importlib.import_module("build_metric_examples")


@pytest.mark.parametrize(
    "example",
    [
        examples.metric_equations,
        examples.worked_cases,
        examples.pooled_and_missing,
        examples.bootstrap_plot_examples,
        examples.plotting_examples,
    ],
    ids=[
        "counts_and_equations",
        "single_items",
        "pooling_and_missing",
        "bootstrap_plots",
        "plots_and_eligibility",
    ],
)
def test_metric_examples(example):
    checks = examples.Checks()
    example(checks)
    assert checks.results
    assert not checks.failures, "\n".join(
        f"{name}: {details}" for name, _, details in checks.failures
    )


def test_wrong_expectation_keeps_failure_diagnostics(monkeypatch):
    name, prediction, counts, _, explanation = examples.CASES[0]
    changed = (name, prediction, counts, (0, 1, 1, 100), explanation)
    monkeypatch.setattr(examples, "CASES", [changed, *examples.CASES[1:]])
    checks = examples.Checks()
    examples.worked_cases(checks)
    assert len(checks.failures) == 1
    name, passed, details = checks.failures[0]
    assert name == "Perfect: four metrics"
    assert passed is False
    assert details


def test_metric_explanations_do_not_run_bootstrapping(monkeypatch):
    def unexpected_bootstrap(*args, **kwargs):
        pytest.fail("Bootstrap belongs in the plotting examples")

    monkeypatch.setattr(examples, "bootstrap_metric_cis", unexpected_bootstrap)
    monkeypatch.setattr(examples, "summarize_balanced_accuracy", unexpected_bootstrap)
    checks = examples.Checks()
    examples.metric_equations(checks)
    examples.worked_cases(checks)
    examples.pooled_and_missing(checks)
    assert not checks.failures


@pytest.mark.parametrize("renderer", ["f2", "adjusted_ba"])
def test_non_enclosing_intervals_keep_bounds_and_bar_alignment(renderer):
    # Both cases deliberately put one interval above its score and one below.
    # Keep these edge cases out of the introductory HTML figures.
    frame = examples.pd.DataFrame(
        {
            "display_name": ["Lower", "Upper"],
            "score": [0.4, 0.8],
            "ci_low": [0.45, 0.55],
            "ci_high": [0.65, 0.75],
            "n_items": [901, 1000],
            "expected_items": 1000,
            "provider": "Synthetic",
            "color": "#376b94",
            "n_bootstrap": 1000,
            "ci_level": 0.95,
        },
        index=[9, 3],
    )
    colors = {"Synthetic": "#376b94"}
    if renderer == "f2":
        fig = examples.plot_leaderboard(frame, colors, "synthetic")
        scores, lows, highs = [0.4, 0.8], [0.45, 0.55], [0.65, 0.75]
    else:
        frame["score"] = [-40, 80]
        frame["ci_low"] = [-30, 55]
        frame["ci_high"] = [-20, 75]
        fig = examples.plot_adjusted_leaderboard(
            frame, colors, "synthetic", 901, "Supplied regression-test bounds."
        )
        scores, lows, highs = [-0.4, 0.8], [-0.3, 0.55], [-0.2, 0.75]
    try:
        checks = examples.Checks()
        examples.check_artists(
            checks, fig, renderer, scores, lows, highs, ["Lower", "Upper"]
        )
        assert fig.axes[0].get_xlim()[0] <= min(*scores, *lows)
        assert not checks.failures, "\n".join(
            f"{name}: {details}" for name, _, details in checks.failures
        )
    finally:
        examples.plt.close(fig)
