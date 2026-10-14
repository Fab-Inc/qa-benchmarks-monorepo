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
    [examples.worked_cases, examples.pooled_and_missing, examples.plotting_examples],
    ids=["single_items", "pooling_and_missing", "plots_and_eligibility"],
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
