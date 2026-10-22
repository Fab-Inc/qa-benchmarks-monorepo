"""Tests for judge scoring: section scoring rules and final aggregation."""

from __future__ import annotations

import json

import pytest
from decodable_text_generation.judge.scoring import (
    DEFAULT_EVIDENCE_RULES,
    score_judgment,
    score_section,
    text_measure_scores,
    text_measures_section_score,
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


class TestEvidenceListRules:
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
        # the lang_2 rule: "fewer than 3 medium/high" — low items don't count
        items = [{"severity": "medium"}] * 2 + [{"severity": "low"}] * 5
        rule = {
            "max_count_where": {
                "field": "severity",
                "in": ["medium", "high"],
                "count": 2,
            }
        }
        assert score_section([_evidence(items, [rule])])["score"] == 100.0


class TestTextMeasures:
    """Defaults from rubrics/scoring.yaml stages (S1 here)."""

    def test_decodable_rescaled_between_floor_and_ceiling(self):
        m = text_measure_scores({"pct_decodable_words": 85.0}, "S1")
        assert m["text_decodable"] == 0.0  # floor maps to 0
        m = text_measure_scores({"pct_decodable_words": 100.0}, "S1")
        assert m["text_decodable"] == 100.0
        m = text_measure_scores({"pct_decodable_words": 92.5}, "S1")
        assert m["text_decodable"] == 50.0

    def test_target_linear_to_cap(self):
        # S1 target_words_min is 2: 1 word -> 50, 2+ -> 100
        assert text_measure_scores({"n_target_words": 1}, "S1")["text_target"] == 50.0
        assert text_measure_scores({"n_target_words": 5}, "S1")["text_target"] == 100.0

    def test_missing_metrics_give_none(self):
        m = text_measure_scores({}, "S1")
        assert m == {"text_decodable": None, "text_target": None, "text_length": None}
        assert text_measures_section_score({}, "S1") is None

    def test_zero_target_words_gates_section(self):
        metrics = {"n_target_words": 0, "pct_decodable_words": 99.0, "n_words": 40}
        assert text_measures_section_score(metrics, "S1") == 0.0

    def test_under_hard_floor_gates_section(self):
        metrics = {
            "n_target_words": 2,
            "pct_decodable_words": 89.9,  # under DECODABLE_HARD_FLOOR = 90
            "n_words": 40,
        }
        assert text_measures_section_score(metrics, "S1") == 0.0
        metrics["pct_decodable_words"] = 90.0
        assert text_measures_section_score(metrics, "S1") != 0.0

    def test_under_half_min_length_gates_section(self):
        metrics = {
            "n_target_words": 2,
            "pct_decodable_words": 95.0,
            "n_words": 9,  # < 20/2
            "reference_words_min": 20,
        }
        assert text_measures_section_score(metrics, "S1") == 0.0
        metrics["n_words"] = 10
        assert text_measures_section_score(metrics, "S1") != 0.0

    def test_section_score_averages_present_measures(self):
        metrics = {
            "n_target_words": 2,
            "pct_decodable_words": 100.0,
            "length_score": 0.5,
        }
        assert text_measures_section_score(metrics, "S1") == round(
            (100.0 + 100.0 + 50.0) / 3, 2
        )


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

    def test_text_measures_included_with_weight_2(self):
        j = _judgment(
            {"A": [_positive("a", "Yes")]},  # 100, weight 1
            metrics={
                "n_target_words": 2,
                "pct_decodable_words": 100.0,
                "length_score": 1.0,
            },
            tags=["S1"],
        )
        out = score_judgment(j)
        # A=100*w1 + Text measures=100*w2 -> 100
        assert out["final_score"] == 100.0
        assert out["section_scores"]["Text measures"]["deterministic"] is True
        assert out["section_scores"]["Text measures"]["weight"] == 2.0
        assert "Text measures" in out["sections_included_in_final"]

    def test_gated_text_measures_drags_final(self):
        j = _judgment(
            {"A": [_positive("a", "Yes")]},  # 100, weight 1
            metrics={"n_target_words": 0, "pct_decodable_words": 100.0},
            tags=["S1"],
        )
        out = score_judgment(j)
        # text measures gated to 0 with weight 2: (100*1 + 0*2)/3
        assert out["final_score"] == pytest.approx(100 / 3)

    def test_story_level_from_tags(self):
        out = score_judgment(_judgment({}, tags=["S2", "other"]))
        assert out["story_level"] == "S2"


class TestRescoreFile:
    def test_round_trip(self, tmp_path):
        row = _judgment({"A": [_positive("a", "Yes")]}, weights={"A": 1.0})
        path = tmp_path / "j.jsonl"
        path.write_text(json.dumps(row) + "\n\n" + json.dumps(row) + "\n")
        from decodable_text_generation.judge.scoring import rescore_file

        out = rescore_file(path)
        assert len(out) == 2
        assert out[0]["final_score"] == 100.0

    def test_load_weights(self, tmp_path):
        from decodable_text_generation.judge.scoring import load_weights

        p = tmp_path / "w.yaml"
        p.write_text("Safety: 0\nCoherence: 2\n")
        assert load_weights(p) == {"Safety": 0.0, "Coherence": 2.0}
