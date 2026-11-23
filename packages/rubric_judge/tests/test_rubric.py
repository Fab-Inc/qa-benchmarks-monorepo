"""Tests for rubric loading, criterion gating and criterion validation."""

from __future__ import annotations

import pytest
import yaml
from rubric_judge.rubric import (
    Criterion,
    CriterionType,
    Section,
    criterion_applies,
    filter_section_criteria,
    load_rubric,
    rubric_sha256,
)

LEVEL_ORDER = ("S1", "S2", "S3")


class TestCriterionApplies:
    def test_null_gating_always_applies(self):
        assert criterion_applies(None, set())
        assert criterion_applies(None, {"S1"})

    def test_band_gating(self):
        assert criterion_applies("S2+", {"S2"}, LEVEL_ORDER)
        assert criterion_applies("S2+", {"S3"}, LEVEL_ORDER)
        assert not criterion_applies("S2+", {"S1"}, LEVEL_ORDER)
        assert criterion_applies("S1+", {"S1"}, LEVEL_ORDER)

    def test_band_gating_highest_present_level_wins(self):
        # a text tagged with multiple levels uses the highest band
        assert criterion_applies("S2+", {"S1", "S3"}, LEVEL_ORDER)

    def test_band_gating_needs_ordered_tags(self):
        # without ordered_tags, "+"/"-" gatings never match
        assert not criterion_applies("S2+", {"S2"})
        assert not criterion_applies("S2-", {"S2"})

    def test_below_band_gating(self):
        assert criterion_applies("S2-", {"S1"}, LEVEL_ORDER)
        assert criterion_applies("S2-", {"S2"}, LEVEL_ORDER)
        assert not criterion_applies("S2-", {"S3"}, LEVEL_ORDER)

    def test_band_gating_needs_an_ordered_tag(self):
        # a text with no ordered tag matches neither "+" nor "-"
        assert not criterion_applies("S2-", {"other"}, LEVEL_ORDER)
        assert not criterion_applies("S2+", {"other"}, LEVEL_ORDER)

    def test_negated_gating(self):
        assert criterion_applies("!poem", {"prose"})
        assert not criterion_applies("!poem", {"poem"})
        # works on any tag, ordered or not
        assert criterion_applies("!S1", {"S2"}, LEVEL_ORDER)

    def test_category_style_tags(self):
        # namespaced tags are plain tags — membership is a category match
        assert criterion_applies("genre:poem", {"genre:poem"})
        assert not criterion_applies("genre:poem", {"genre:prose"})

    def test_feature_tag_membership(self):
        assert criterion_applies("-ed taught", {"-ed taught"}, LEVEL_ORDER)
        assert criterion_applies("S2", {"S2"}, LEVEL_ORDER)
        assert not criterion_applies("-ed taught", {"S2"}, LEVEL_ORDER)

    def test_filter_section_criteria(self):
        section = Section(
            name="s",
            criteria=[
                Criterion(id="a", question="?", criterion_type=CriterionType.POSITIVE),
                Criterion(
                    id="b",
                    question="?",
                    criterion_type=CriterionType.POSITIVE,
                    gating="S3+",
                ),
            ],
        )
        assert [
            c.id for c in filter_section_criteria(section, {"S1"}, LEVEL_ORDER)
        ] == ["a"]
        assert [
            c.id for c in filter_section_criteria(section, {"S3"}, LEVEL_ORDER)
        ] == ["a", "b"]


class TestCriterionValidation:
    def test_evidence_list_requires_items_or_fields(self):
        with pytest.raises(ValueError, match="evidence_items"):
            Criterion(
                id="bad",
                question="?",
                criterion_type=CriterionType.EVIDENCE_LIST,
            )
        # either declaration satisfies the requirement
        Criterion(
            id="ok",
            question="?",
            criterion_type=CriterionType.EVIDENCE_LIST,
            evidence_items={"quote": "str"},
        )
        Criterion(
            id="ok2",
            question="?",
            criterion_type=CriterionType.EVIDENCE_LIST,
            evidence_fields={"about": "str"},
        )


class TestLoadRubric:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "r.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "MySection": {
                        "name": "My Section",
                        "weight": 2.0,
                        "criteria": [
                            {
                                "id": "c1",
                                "question": "q?",
                                "criterion_type": "positive",
                            }
                        ],
                    }
                }
            )
        )
        rubric = load_rubric(path)
        assert rubric["MySection"].weight == 2.0
        assert rubric["MySection"].criteria[0].id == "c1"

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_rubric(tmp_path / "nope.yaml")

    def test_non_mapping_rejected(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("- a\n- b\n")
        with pytest.raises(ValueError):
            load_rubric(path)

    def test_rubric_sha256_stable(self, tmp_path):
        p = tmp_path / "r.yaml"
        p.write_text("x")
        assert rubric_sha256(p) == rubric_sha256(p)
        p.write_text("y")
        before = rubric_sha256(p)
        p.write_text("x")
        assert rubric_sha256(p) != before
