"""Check tier gates, the inclusive item-count threshold and top-N aggregation."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/analyze_phonics_hardware_tiers.py"
SPEC = importlib.util.spec_from_file_location("phonics_hardware", SCRIPT)
hardware = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hardware)


@pytest.mark.parametrize(
    "count,minimum,expected",
    [
        (699, 700, False),
        (700, 700, True),
        (701, 700, True),
        (499, 500, False),
        (500, 500, True),
        (501, 500, True),
    ],
)
def test_minimum_scored_items(count, minimum, expected):
    assert hardware.passes_coverage(count, minimum) is expected


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


@pytest.mark.parametrize("language,total", [("english", 991), ("kiswahili", 1000)])
@pytest.mark.parametrize("min_coverage", [None, 0.8])
def test_audit_threshold_and_workstation_exclusion(
    tmp_path, language, total, min_coverage
):
    run = tmp_path / language
    (run / "report").mkdir(parents=True)
    threshold = 700 if min_coverage is None else (793 if language == "english" else 801)
    rows = [
        ("below", threshold - 1),
        ("exact", threshold),
        ("above", threshold + 1),
        ("workstation", 900),
    ]
    coverage = pd.DataFrame(
        [
            {
                "model": model,
                "expected_items": total,
                "scorable_items": count,
                "response_rows": total,
            }
            for model, count in rows
        ]
    )
    coverage.to_csv(run / "coverage_by_model.csv", index=False)
    pd.DataFrame(
        [
            {"model_name": model, "n_items": count, "pooled_f2": 0.5}
            for model, count in rows
        ]
    ).to_csv(run / "report/summary_by_model.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": model,
                "pooled_f2": 0.5,
                "pooled_f2_ci_low": 0.4,
                "pooled_f2_ci_high": 0.6,
                "n_bootstrap": 1000,
                "ci_level": 0.95,
            }
            for model, _ in rows
        ]
    ).to_csv(run / "report/bootstrap_ci.csv", index=False)
    specs = {
        model: {
            "open": True,
            "size": 12e9 if model == "workstation" else 3e9,
            "display_name": model,
        }
        for model, _ in rows
    }
    audit = hardware.build_audit(run, specs, total, min_coverage=min_coverage)
    assert audit.minimum_scorable_items.eq(threshold).all()
    with pytest.raises(ValueError, match="denominator"):
        hardware.build_audit(run, specs, total + 1, min_coverage=min_coverage)
    assert audit.loc[audit.included, "model"].tolist() == ["exact", "above"]
    assert (
        audit.set_index("model").loc["below", "exclusion_reason"]
        == "below_minimum_scorable_items"
    )
    assert (
        audit.set_index("model").loc["workstation", "exclusion_reason"]
        == "tier_excluded_from_comparison"
    )


@pytest.mark.parametrize(
    "total,minimum,coverage,required",
    [
        (991, 700, 0.8, 793),
        (1000, 700, 0.8, 801),
        (5, 1, 0.8, 5),
        (1000, 900, 0.8, 900),
        (1000, 700, None, 700),
    ],
)
def test_strict_coverage_uses_target_size_and_respects_count_floor(
    total, minimum, coverage, required
):
    assert hardware.minimum_scorable_items(total, minimum, coverage) == required


@pytest.mark.parametrize("coverage", [-0.1, 1, float("nan"), float("inf")])
def test_invalid_coverage_rejected(coverage):
    with pytest.raises(ValueError, match="min_coverage"):
        hardware.minimum_scorable_items(1000, 700, coverage)
