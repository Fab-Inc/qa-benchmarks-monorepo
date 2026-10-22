"""Tests for section-granular resume of judge runs."""

from __future__ import annotations

from decodable_text_generation.judge.resume import (
    merge_judgment,
    partition_sections,
    rubric_section_shas,
    section_sha,
)
from decodable_text_generation.judge.rubric import Criterion, CriterionType, Section


def _section(*criteria, weight=1.0):
    return Section(name="s", criteria=list(criteria), weight=weight)


def _crit(cid, question="q?", **kw):
    return Criterion(
        id=cid, question=question, criterion_type=CriterionType.POSITIVE, **kw
    )


class TestSectionSha:
    def test_stable(self):
        assert section_sha(_section(_crit("a"))) == section_sha(_section(_crit("a")))

    def test_question_change_alters_sha(self):
        a = section_sha(_section(_crit("a", "one?")))
        b = section_sha(_section(_crit("a", "two?")))
        assert a != b

    def test_unset_optional_fields_do_not_alter_sha(self):
        # exclude_none keeps an absent optional field out of the hash
        a = section_sha(_section(_crit("a")))
        assert section_sha(_section(_crit("a"))) == a


class TestPartitionSections:
    def test_matching_shas_reusable(self):
        rubric = {"A": _section(_crit("a")), "B": _section(_crit("b"))}
        row = {"section_shas": rubric_section_shas(rubric)}
        reusable, stale = partition_sections(row, rubric)
        assert reusable == {"A", "B"} and not stale

    def test_changed_section_is_stale(self):
        rubric = {"A": _section(_crit("a")), "B": _section(_crit("b"))}
        shas = rubric_section_shas(rubric)
        row = {"section_shas": shas}
        new_rubric = {"A": _section(_crit("a")), "B": _section(_crit("b", "new q?"))}
        reusable, stale = partition_sections(row, new_rubric)
        assert reusable == {"A"} and stale == {"B"}

    def test_missing_sha_is_stale(self):
        rubric = {"A": _section(_crit("a")), "B": _section(_crit("b"))}
        row = {"section_shas": {"A": rubric_section_shas(rubric)["A"]}}
        reusable, stale = partition_sections(row, rubric)
        assert reusable == {"A"} and stale == {"B"}


class TestLegacyRows:
    """Rows without section_shas fall back to criterion-shape matching."""

    def _row(self, sections):
        return {"sections": sections}

    def test_identical_criteria_reusable(self):
        rubric = {"A": _section(_crit("a", "keep?"))}
        row = self._row(
            {
                "A": [
                    {
                        "criterion_id": "a",
                        "question": "keep?",
                        "criterion_type": "positive",
                        "response": "Yes",
                    }
                ]
            }
        )
        reusable, _ = partition_sections(row, rubric)
        assert reusable == {"A"}

    def test_different_question_is_stale(self):
        rubric = {"A": _section(_crit("a", "new?"))}
        row = self._row(
            {
                "A": [
                    {
                        "criterion_id": "a",
                        "question": "old?",
                        "criterion_type": "positive",
                        "response": "Yes",
                    }
                ]
            }
        )
        _, stale = partition_sections(row, rubric)
        assert stale == {"A"}

    def test_evidence_criteria_never_reusable_legacy(self):
        # legacy rows predate evidence answers — never reuse them
        rubric = {"A": _section(_crit("a", "keep?", evidence=True))}
        row = self._row(
            {
                "A": [
                    {
                        "criterion_id": "a",
                        "question": "keep?",
                        "criterion_type": "positive",
                        "response": "Yes",
                    }
                ]
            }
        )
        _, stale = partition_sections(row, rubric)
        assert stale == {"A"}

    def test_extra_criterion_is_stale(self):
        rubric = {"A": _section(_crit("a"), _crit("b"))}
        row = self._row(
            {
                "A": [
                    {
                        "criterion_id": "a",
                        "question": "q?",
                        "criterion_type": "positive",
                    }
                ]
            }
        )
        _, stale = partition_sections(row, rubric)
        assert stale == {"A"}


class TestMergeJudgment:
    def test_reusable_sections_from_cached(self):
        rubric = {
            "A": _section(_crit("a"), weight=2.0),
            "B": _section(_crit("b"), weight=1.0),
        }
        cached = {"sections": {"A": [{"criterion_id": "a", "response": "Yes"}]}}
        fresh = {"sections": {"B": [{"criterion_id": "b", "response": "No"}]}}
        merged = merge_judgment(cached, fresh, rubric, {"A"})
        assert merged["sections"]["A"] == cached["sections"]["A"]
        assert merged["sections"]["B"] == fresh["sections"]["B"]
        assert merged["section_weights"] == {"A": 2.0, "B": 1.0}
        assert merged["reused_sections"] == ["A"]
