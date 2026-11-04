"""Tests for rubric loading, level bands, criterion gating and tag derivation."""

from __future__ import annotations

import pytest
import yaml
from decodable_text_generation.judge.rubric import (
    Criterion,
    CriterionType,
    Section,
    criterion_applies,
    derive_tags,
    filter_section_criteria,
    lesson_for_profile,
    level_for_lesson,
    level_tag_for_profile,
    load_rubric,
    rubric_sha256,
)


class TestLevelBands:
    """UFLI stages from scoring.yaml: S1 = lessons 1-19, S2 = 20-83,
    S3 = 84+. Tusome weeks band separately: g1 wk01-06 = S1, wk07-17 = S2,
    wk18+ and any grade >=2 = S3."""

    @pytest.mark.parametrize(
        ("lesson", "expected"),
        [(1, "S1"), (19, "S1"), (20, "S2"), (83, "S2"), (84, "S3"), (128, "S3")],
    )
    def test_level_for_lesson(self, lesson, expected):
        assert level_for_lesson(lesson) == expected

    @pytest.mark.parametrize(
        ("profile", "expected"),
        [("ufli-en-l005", 5), ("ufli-en-l109", 109)],
    )
    def test_lesson_for_profile(self, profile, expected):
        assert lesson_for_profile(profile) == expected

    @pytest.mark.parametrize(
        "profile",
        ["ufli-en-j05", "tusome-ke-l005", "ufli-sw-l005", "nonsense", "ufli-en-5"],
    )
    def test_lesson_for_profile_out_of_scope(self, profile):
        assert lesson_for_profile(profile) is None

    def test_level_tag_for_profile(self):
        assert level_tag_for_profile("ufli-en-l016") == "S1"
        assert level_tag_for_profile("ufli-en-l067") == "S2"
        assert level_tag_for_profile("ufli-en-l109") == "S3"
        assert level_tag_for_profile("ufli-en-j05") is None

    def test_level_tag_for_profile_tusome(self):
        assert level_tag_for_profile("tusome-ke-g1-wk01") == "S1"
        assert level_tag_for_profile("tusome-ke-g1-wk06") == "S1"
        assert level_tag_for_profile("tusome-ke-g1-wk07") == "S2"
        assert level_tag_for_profile("tusome-ke-g1-wk17") == "S2"
        assert level_tag_for_profile("tusome-ke-g1-wk18") == "S3"
        assert level_tag_for_profile("tusome-ke-g1-wk30") == "S3"
        # grade-2 profiles follow the grade-1 sequence: always S3
        assert level_tag_for_profile("tusome-ke-g2-wk03") == "S3"
        assert level_tag_for_profile("tusome-ke-l005") is None


class TestCriterionApplies:
    def test_null_gating_always_applies(self):
        assert criterion_applies(None, set())
        assert criterion_applies(None, {"S1"})

    def test_band_gating(self):
        assert criterion_applies("S2+", {"S2"})
        assert criterion_applies("S2+", {"S3"})
        assert not criterion_applies("S2+", {"S1"})
        assert criterion_applies("S1+", {"S1"})

    def test_band_gating_highest_present_level_wins(self):
        # a story tagged with multiple levels uses the highest band
        assert criterion_applies("S2+", {"S1", "S3"})

    def test_feature_tag_membership(self):
        assert criterion_applies("-ed taught", {"-ed taught"})
        assert criterion_applies("S2", {"S2"})
        assert not criterion_applies("-ed taught", {"S2"})

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
        assert [c.id for c in filter_section_criteria(section, {"S1"})] == ["a"]
        assert [c.id for c in filter_section_criteria(section, {"S3"})] == ["a", "b"]


class TestDeriveTags:
    def test_level_tag_from_profile(self):
        assert derive_tags("ufli-en-l109") == {"S3"}
        assert derive_tags("tusome-ke-g1-wk10") == {"S2"}
        assert derive_tags("ufli-en-j05") == set()

    def test_ed_taught_from_affixes(self):
        tags = derive_tags("ufli-en-l109", {"affixes": ["-ed", "-s"]})
        assert "-ed taught" in tags
        tags = derive_tags("ufli-en-l005", {"affixes": ["-s"]})
        assert "-ed taught" not in tags

    def test_nondecodable_and_target_tags(self):
        assert "non-decodable words" in derive_tags("x", n_nondecodable_words=2)
        assert "non-decodable words" not in derive_tags("x", n_nondecodable_words=0)
        assert "target words" in derive_tags("x", n_target_words=1)
        assert "repeats-frame-3" in derive_tags("x", max_frame_repeats=3)
        assert "repeats-frame-3" not in derive_tags("x", max_frame_repeats=2)

    def test_target_knowledge_needs_populated_values(self):
        assert "target knowledge" in derive_tags(
            "x", target_knowledge={"gpcs": [["sh", "ʃ"]]}
        )
        # a dict of empty fields is not a populated target
        assert "target knowledge" not in derive_tags(
            "x", target_knowledge={"gpcs": [], "sight_words": []}
        )

    def test_extra_tags_merge(self):
        assert "custom" in derive_tags("x", extra_tags=["custom"])


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
