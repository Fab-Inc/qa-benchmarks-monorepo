"""Check the tier gates, strict coverage threshold and top-N aggregation."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/analyze_phonics_hardware_tiers.py"
SPEC = importlib.util.spec_from_file_location("phonics_hardware", SCRIPT)
hardware = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hardware)


@pytest.mark.parametrize(
    "count,denominator,expected",
    [
        (732, 915, False),
        (733, 915, True),
        (861, 1077, False),
        (862, 1077, True),
        (80, 100, False),
        (81, 100, True),
    ],
)
def test_strict_coverage(count, denominator, expected):
    assert hardware.passes_coverage(count, denominator) is expected


@pytest.mark.parametrize(
    "spec,tier,band",
    [
        (None, None, False),
        ({"open": False, "size": 1e9}, "Cloud", False),
        ({"open": False, "size": None}, "Cloud", False),
        ({"open": True, "size": None}, None, False),
        ({"open": True, "size": 3.2e9}, "Mobile + Tablet", False),
        ({"open": True, "size": 9.6e9}, "Mobile + Tablet", False),
        ({"open": True, "size": 9.6e9 + 1}, "Workstation", False),
        ({"open": True, "size": hardware.PRECISION_BAND[0]}, "Workstation", False),
        ({"open": True, "size": hardware.PRECISION_BAND[0] + 1}, "Cloud", True),
        ({"open": True, "size": hardware.PRECISION_BAND[1]}, "Cloud", True),
        ({"open": True, "size": hardware.PRECISION_BAND[1] + 1}, "Cloud", False),
        ({"open": True, "size": 120e9, "size_active": 3e9}, "Cloud", False),
    ],
)
def test_reference_hardware_gates(spec, tier, band):
    result = hardware.assign_hardware(spec)
    assert result["hardware_tier"] == tier
    assert result["precision_dependent"] is band


@pytest.mark.parametrize("metric", ["pooled_f1", "pooled_f2", "balanced_accuracy"])
def test_top_n_uses_available_models_and_equal_model_weights(metric):
    rows = [
        {
            "model": name,
            "language": "english",
            "hardware_tier": "Cloud",
            metric: score,
            "included": included,
            "precision_dependent": False,
            "placed_by": "access gate (closed weights)",
            "n_items": n,
        }
        for name, score, included, n in [
            ("low", 0.2, True, 1000),
            ("high", 0.9, True, 862),
            ("excluded-best", 1.0, False, 861),
        ]
    ]
    audit = pd.DataFrame(rows)
    medians, means, selected = hardware.summarize_tiers(audit, 10, metric)
    assert medians.iloc[0][f"median_{metric}"] == pytest.approx(0.55)
    assert means.iloc[0][f"top_n_mean_{metric}"] == pytest.approx(0.55)
    assert means.iloc[0].n_used == 2
    assert selected.model.tolist() == ["high", "low"]
    _, best, chosen = hardware.summarize_tiers(audit, 1, metric)
    assert best.iloc[0][f"top_n_mean_{metric}"] == pytest.approx(0.9)
    assert chosen.model.tolist() == ["high"]
