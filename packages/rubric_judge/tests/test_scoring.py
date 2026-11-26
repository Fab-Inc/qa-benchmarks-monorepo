"""Tests for judge scoring: section scoring rules and final aggregation."""

from __future__ import annotations

import json

import pytest
from rubric_judge.scoring import (
    DEFAULT_EVIDENCE_RULES,
    deduction_score,
    iter_judgments,
    load_weights,
    rescore_file,
    score_judgment,
    score_section,
)


def _positive(criterion_id: str, response: str, **extra):
    return {
        "criterion_id": criterion_id,
        "criterion_type": "positive",
        "response": response,
        **extra,
    }


def _evidence(items: list[dict], scoring_rules=None, **extra):
    row = {
        "criterion_id": "ev",
        "criterion_type": "evidence_list",
        "response": None,
        "evidence": items,
        **extra,
    }
    if scoring_rules is not None:
        row["scoring_rules"] = scoring_rules
    return row


class TestScoreSectionVerdicts:
    def test_positive_yes_no_ratio(self):
        out = score_section(
            [_positive("a", "Yes"), _positive("b", "No"), _positive("c", "Yes")]
        )
        assert out["score"] == pytest.approx(100 * 2 / 3)
        assert (out["yes_count"], out["applicable_count"]) == (2, 3)

    def test_na_excluded_from_denominator(self):
        out = score_section([_positive("a", "Yes"), _positive("b", "N/A")])
        assert out["score"] == 100.0
        assert out["na_count"] == 1

    def test_gated_out_excluded(self):
        out = score_section(
            [_positive("a", "Yes"), _positive("b", "No", gated_out=True)]
        )
        assert out["score"] == 100.0
        assert out["gated_out_count"] == 1

    def test_no_applicable_gives_none(self):
        out = score_section([_positive("a", "N/A")])
        assert out["score"] is None

    def test_fail_condition_yes_zeroes_section(self):
        out = score_section(
            [
                _positive("a", "Yes"),
                {
                    "criterion_id": "unsafe",
                    "criterion_type": "fail_condition",
                    "response": "Yes",
                },
            ]
        )
        assert out["score"] == 0.0
        assert out["triggered_fail_conditions"] == ["unsafe"]

    def test_fail_condition_no_is_fine(self):
        out = score_section(
            [
                _positive("a", "No"),
                {
                    "criterion_id": "unsafe",
                    "criterion_type": "fail_condition",
                    "response": "No",
                },
            ]
        )
        assert out["score"] == 0.0  # positives decide; no fail triggered
        assert out["triggered_fail_conditions"] == []


class TestLessThanRules:
    """Canonical grammar: {<optional matcher>, less_than: N}."""

    def test_less_than_counts_all_items(self):
        row = _evidence([{"quote": "x"}], [{"less_than": 1}])
        assert score_section([row])["score"] == 0.0
        row = _evidence([], [{"less_than": 1}])
        assert score_section([row])["score"] == 100.0

    def test_less_than_allows_some_items(self):
        items = [{"quote": "a"}, {"quote": "b"}]
        assert score_section([_evidence(items, [{"less_than": 3}])])["score"] == 100.0
        assert score_section([_evidence(items, [{"less_than": 2}])])["score"] == 0.0

    def test_severity_in_less_than(self):
        # "no medium or high items" == severity_in [low] under the old grammar,
        # but stated positively: fewer than 1 medium/high item.
        rule = {"severity_in": ["medium", "high"], "less_than": 1}
        assert (
            score_section([_evidence([{"severity": "low"}], [rule])])["score"] == 100.0
        )
        assert (
            score_section([_evidence([{"severity": "medium"}], [rule])])["score"] == 0.0
        )
        assert score_section([_evidence([], [rule])])["score"] == 100.0

    def test_severity_in_less_than_counts_only_matching(self):
        items = [{"severity": "medium"}, {"severity": "low"}, {"severity": "low"}]
        rule = {"severity_in": ["medium", "high"], "less_than": 2}
        assert score_section([_evidence(items, [rule])])["score"] == 100.0
        items.append({"severity": "high"})
        assert score_section([_evidence(items, [rule])])["score"] == 0.0

    def test_where_less_than_generic_field(self):
        items = [{"kind": "joke"}, {"kind": "joke"}, {"kind": "surprise"}]
        rule = {"where": {"field": "kind", "in": "joke"}, "less_than": 3}
        assert score_section([_evidence(items, [rule])])["score"] == 100.0
        rule = {"where": {"field": "kind", "not": "joke"}, "less_than": 1}
        assert score_section([_evidence(items, [rule])])["score"] == 0.0

    def test_item_without_field_does_not_match(self):
        # a quote-only item is not a "medium/high" item
        rule = {"severity_in": ["medium", "high"], "less_than": 1}
        assert score_section([_evidence([{"quote": "x"}], [rule])])["score"] == 100.0

    def test_greater_than_counts_all_items(self):
        items = [{"quote": "a"}, {"quote": "b"}]
        assert (
            score_section([_evidence(items, [{"greater_than": 1}])])["score"] == 100.0
        )
        assert (
            score_section([_evidence([{"quote": "a"}], [{"greater_than": 1}])])["score"]
            == 0.0
        )
        assert score_section([_evidence([], [{"greater_than": 0}])])["score"] == 0.0

    def test_greater_than_with_matcher(self):
        rule = {"severity_in": ["low"], "greater_than": 1}
        items = [{"severity": "low"}, {"severity": "low"}]
        assert score_section([_evidence(items, [rule])])["score"] == 100.0
        items = [{"severity": "low"}, {"severity": "high"}]
        assert score_section([_evidence(items, [rule])])["score"] == 0.0

    def test_greater_than_where(self):
        rule = {"where": {"field": "kind", "in": "joke"}, "greater_than": 0}
        assert score_section([_evidence([{"kind": "joke"}], [rule])])["score"] == 100.0
        assert (
            score_section([_evidence([{"kind": "surprise"}], [rule])])["score"] == 0.0
        )

    def test_less_than_and_greater_than_form_a_range(self):
        # {less_than: 3, greater_than: 1} = exactly 2 matching items
        rule = {"less_than": 3, "greater_than": 1}
        two = [{"quote": "a"}, {"quote": "b"}]
        assert score_section([_evidence(two, [rule])])["score"] == 100.0
        assert score_section([_evidence(two[:1], [rule])])["score"] == 0.0
        assert score_section([_evidence(two + two[:1], [rule])])["score"] == 0.0

    def test_range_with_matcher(self):
        # 1-2 low items earns the point
        rule = {"severity_in": ["low"], "less_than": 3, "greater_than": 0}
        one_low = [{"severity": "low"}, {"severity": "high"}]
        assert score_section([_evidence(one_low, [rule])])["score"] == 100.0
        three_low = [{"severity": "low"}] * 3
        assert score_section([_evidence(three_low, [rule])])["score"] == 0.0
        assert score_section([_evidence([], [rule])])["score"] == 0.0


class TestEvidenceListRules:
    """Legacy rule forms, still understood for pre-``less_than`` rows."""

    def test_severity_in_all_pass(self):
        row = _evidence([{"severity": "low"}], [{"severity_in": ["low"]}])
        assert score_section([row])["score"] == 100.0

    def test_severity_in_rejects_higher(self):
        row = _evidence([{"severity": "high"}], [{"severity_in": ["low", "medium"]}])
        assert score_section([row])["score"] == 0.0

    def test_empty_list_passes(self):
        row = _evidence([], [{"severity_in": ["low"]}, {"max_items": 0}])
        assert score_section([row])["score"] == 100.0

    def test_items_without_severity_fail_severity_in(self):
        row = _evidence(
            [{"severity": "low"}, {"quote": "x"}], [{"severity_in": ["low"]}]
        )
        assert score_section([row])["score"] == 0.0

    def test_max_items(self):
        row = _evidence([{"severity": "low"}, {"severity": "low"}], [{"max_items": 1}])
        assert score_section([row])["score"] == 0.0

    def test_missing_scoring_rules_uses_default_fallback(self):
        # Legacy rows: two points — no medium/high, then no high.
        row = _evidence([{"severity": "low"}])
        assert score_section([row])["score"] == 100.0
        row = _evidence([{"severity": "medium"}])
        assert score_section([row])["score"] == 50.0
        row = _evidence([{"severity": "high"}])
        assert score_section([row])["score"] == 0.0
        assert len(DEFAULT_EVIDENCE_RULES) == 2

    def test_explicit_empty_rules_is_evidence_only(self):
        out = score_section([_evidence([{"severity": "high"}], [])])
        assert out["score"] is None  # contributes no points, not a zero
        assert out["applicable_count"] == 0

    def test_max_count_where(self):
        items = [{"severity": "medium"}, {"severity": "medium"}, {"severity": "low"}]
        rule = {
            "max_count_where": {
                "field": "severity",
                "in": ["medium", "high"],
                "count": 2,
            }
        }
        assert score_section([_evidence(items, [rule])])["score"] == 100.0
        items.append({"severity": "medium"})
        assert score_section([_evidence(items, [rule])])["score"] == 0.0

    def test_max_count_where_counts_only_matching(self):
        # "fewer than 3 medium/high" — low items don't count
        items = [{"severity": "medium"}] * 2 + [{"severity": "low"}] * 5
        rule = {
            "max_count_where": {
                "field": "severity",
                "in": ["medium", "high"],
                "count": 2,
            }
        }
        assert score_section([_evidence(items, [rule])])["score"] == 100.0


def _judgment(sections, metrics=None, weights=None, tags=None):
    return {
        "benchmark_item_id": "item#m",
        "model": "m",
        "sections": sections,
        "metrics": metrics or {},
        "section_weights": weights or {},
        "tags": tags or [],
    }


class TestScoreJudgment:
    def test_weighted_final(self):
        j = _judgment(
            {
                "A": [_positive("a", "Yes")],  # 100
                "B": [_positive("b", "No")],  # 0
            },
            weights={"A": 1.0, "B": 3.0},
        )
        out = score_judgment(j)
        assert out["final_score"] == pytest.approx((100 * 1 + 0 * 3) / 4)

    def test_none_sections_excluded(self):
        j = _judgment(
            {
                "A": [_positive("a", "Yes")],
                "B": [_positive("b", "N/A")],
            }
        )
        out = score_judgment(j)
        assert "B" in out["sections_excluded"]
        assert out["final_score"] == 100.0

    def test_zero_weight_excluded(self):
        j = _judgment(
            {
                "A": [_positive("a", "No")],  # 0 but weight 0
                "B": [_positive("b", "Yes")],
            },
            weights={"A": 0.0, "B": 1.0},
        )
        out = score_judgment(j)
        assert "A" in out["sections_excluded"]
        assert out["final_score"] == 100.0

    def test_weight_overrides_beat_snapshot(self):
        j = _judgment(
            {
                "A": [_positive("a", "Yes")],
                "B": [_positive("b", "No")],
            },
            weights={"A": 1.0, "B": 1.0},
        )
        out = score_judgment(j, weights={"A": 0.0, "B": 1.0})
        assert out["final_score"] == 0.0


class TestExtraSectionScorer:
    def test_extra_section_included_with_payload_weight(self):
        j = _judgment({"A": [_positive("a", "Yes")]})  # 100, weight 1
        out = score_judgment(
            j,
            extra_section_scorer=lambda row: (
                "Measures",
                50.0,
                {"weight": 2.0, "deterministic": True},
            ),
        )
        # A=100*w1 + Measures=50*w2 -> (100 + 100)/3
        assert out["final_score"] == pytest.approx(200 / 3)
        sec = out["section_scores"]["Measures"]
        assert sec["deterministic"] is True
        assert sec["weight"] == 2.0
        assert "Measures" in out["sections_included_in_final"]

    def test_extra_section_none_score_adds_nothing(self):
        j = _judgment({"A": [_positive("a", "Yes")]})
        out = score_judgment(
            j,
            extra_section_scorer=lambda row: ("Measures", None, {"weight": 2.0}),
        )
        assert "Measures" not in out["section_scores"]
        assert out["final_score"] == 100.0

    def test_extra_section_none_return_adds_nothing(self):
        j = _judgment({"A": [_positive("a", "Yes")]})
        out = score_judgment(j, extra_section_scorer=lambda row: None)
        assert out["final_score"] == 100.0

    def test_weights_override_extra_section_weight(self):
        j = _judgment({"A": [_positive("a", "Yes")]})
        scorer = lambda row: ("Measures", 0.0, {"weight": 2.0})  # noqa: E731
        out = score_judgment(j, weights={"Measures": 0.0}, extra_section_scorer=scorer)
        assert "Measures" in out["sections_excluded"]
        assert out["final_score"] == 100.0


class TestRescoreFile:
    def test_round_trip(self, tmp_path):
        row = _judgment({"A": [_positive("a", "Yes")]}, weights={"A": 1.0})
        path = tmp_path / "j.jsonl"
        path.write_text(json.dumps(row) + "\n\n" + json.dumps(row) + "\n")
        out = rescore_file(path)
        assert len(out) == 2
        assert out[0]["final_score"] == 100.0

    def test_score_kwargs_forwarded(self, tmp_path):
        row = _judgment({"A": [_positive("a", "Yes")]}, weights={"A": 1.0})
        path = tmp_path / "j.jsonl"
        path.write_text(json.dumps(row) + "\n")
        out = rescore_file(
            path,
            extra_section_scorer=lambda r: ("Extra", 0.0, {"weight": 1.0}),
        )
        assert out[0]["final_score"] == 50.0

    def test_load_weights(self, tmp_path):
        p = tmp_path / "w.yaml"
        p.write_text("Safety: 0\nCoherence: 2\n")
        assert load_weights(p) == {"Safety": 0.0, "Coherence": 2.0}

    def test_iter_judgments_skips_blank_lines(self, tmp_path):
        p = tmp_path / "j.jsonl"
        p.write_text('{"a": 1}\n\n{"a": 2}\n')
        assert [r["a"] for r in iter_judgments(p)] == [1, 2]


class TestCriterionWeights:
    def test_weighted_ratio(self):
        out = score_section(
            [_positive("a", "Yes"), _positive("b", "No")],
            criterion_weights={"a": 3.0, "b": 1.0},
        )
        assert out["score"] == pytest.approx(75.0)

    def test_stored_weight_counts(self):
        out = score_section([_positive("a", "Yes", weight=3.0), _positive("b", "No")])
        # a counts 3 yes / 4 applicable
        assert out["score"] == pytest.approx(75.0)

    def test_override_beats_stored_weight(self):
        out = score_section(
            [_positive("a", "Yes", weight=3.0), _positive("b", "No")],
            criterion_weights={"a": 1.0},
        )
        assert out["score"] == pytest.approx(50.0)

    def test_zero_weight_drops_criterion(self):
        out = score_section(
            [_positive("a", "Yes"), _positive("b", "No")],
            criterion_weights={"b": 0.0},
        )
        assert out["score"] == 100.0

    def test_weight_applies_to_evidence_rules(self):
        row = _evidence(
            [{"severity": "high"}],
            [{"severity_in": ["low"]}, {"severity_in": ["low", "medium"]}],
        )
        assert score_section([row])["score"] == 0.0
        out = score_section([_positive("p", "Yes"), row], criterion_weights={"ev": 0.0})
        assert out["score"] == 100.0


class TestEvidenceRuleOverrides:
    def test_override_replaces_stored_rules(self):
        row = _evidence([{"severity": "high"}], [{"severity_in": ["low"]}])
        assert score_section([row])["score"] == 0.0
        out = score_section(
            [row],
            evidence_rules={"ev": [{"severity_in": ["low", "medium", "high"]}]},
        )
        assert out["score"] == 100.0

    def test_override_empty_list_removes_points(self):
        row = _evidence([{"severity": "high"}], [{"severity_in": ["low"]}])
        out = score_section([_positive("p", "Yes"), row], evidence_rules={"ev": []})
        assert out["score"] == 100.0

    def test_override_beats_legacy_fallback(self):
        # row has no scoring_rules -> would fall back to DEFAULT_EVIDENCE_RULES
        row = _evidence([{"severity": "medium"}])
        assert score_section([row])["score"] == 50.0
        out = score_section([row], evidence_rules={"ev": [{"severity_in": ["medium"]}]})
        assert out["score"] == 100.0


class TestDeductionScore:
    def test_perfect_row_scores_100(self):
        j = _judgment({"A": [_positive("a", "Yes")]})
        assert deduction_score(j)["score"] == 100.0

    def test_severity_costs(self):
        j = _judgment(
            {
                "A": [
                    _evidence(
                        [{"severity": "low"}, {"severity": "high"}, {"quote": "x"}],
                        [{"max_items": 9}],
                    )
                ]
            }
        )
        out = deduction_score(
            j, severity_costs={"low": 1.0, "medium": 3.0, "high": 8.0, "none": 0.5}
        )
        assert out["score"] == pytest.approx(100 - (1 + 8 + 0.5))
        assert out["breakdown"]["A"]["items"] == {"low": 1, "high": 1, "none": 1}

    def test_criterion_no_and_fail_costs(self):
        j = _judgment(
            {
                "A": [
                    _positive("a", "No"),
                    {
                        "criterion_id": "f",
                        "criterion_type": "fail_condition",
                        "response": "Yes",
                    },
                ]
            }
        )
        out = deduction_score(j, criterion_no_cost=5.0, fail_condition_cost=20.0)
        assert out["score"] == pytest.approx(75.0)

    def test_clamped_at_min(self):
        j = _judgment({"A": [_evidence([{"severity": "high"}] * 20)]})
        out = deduction_score(j, severity_costs={"low": 1, "medium": 2, "high": 10})
        assert out["score"] == 0.0

    def test_na_and_gated_excluded(self):
        j = _judgment(
            {
                "A": [
                    _positive("a", "N/A"),
                    _positive("b", "No", gated_out=True),
                ]
            }
        )
        assert deduction_score(j, criterion_no_cost=50.0)["score"] == 100.0

    def test_per_section_cost_override(self):
        j = _judgment(
            {
                "A": [_evidence([{"severity": "high"}])],
                "B": [_evidence([{"severity": "high"}])],
            }
        )
        out = deduction_score(
            j,
            severity_costs={"low": 1, "medium": 2, "high": 10},
            per_section_costs={"B": {"high": 0.0}},
        )
        assert out["score"] == pytest.approx(90.0)
