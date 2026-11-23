"""Tests for call-granular resume of judge runs."""

from __future__ import annotations

from rubric_judge.engine import CallAnswer, assemble_judgment
from rubric_judge.plan import JudgeItem, plan_judgment
from rubric_judge.resume import filter_calls, merge_judgment, partition_calls
from rubric_judge.rubric import Criterion, CriterionType, Section


def _section(*criteria, weight=1.0, **kw):
    return Section(name="s", criteria=list(criteria), weight=weight, **kw)


def _crit(cid, question="q?", criterion_type=CriterionType.POSITIVE, **kw):
    return Criterion(id=cid, question=question, criterion_type=criterion_type, **kw)


def _plan(rubric, **item_kw):
    return plan_judgment(JudgeItem(text="t", **item_kw), rubric=rubric)


class TestPartitionCalls:
    def test_matching_fingerprints_reusable(self):
        rubric = {"A": _section(_crit("a")), "B": _section(_crit("b"))}
        plan = _plan(rubric)
        fps = plan.call_fingerprints()
        reusable, stale = partition_calls({"call_shas": fps}, fps, plan)
        assert reusable == {"A", "B"} and not stale

    def test_changed_call_is_stale(self):
        old = {"A": _section(_crit("a")), "B": _section(_crit("b"))}
        new = {"A": _section(_crit("a")), "B": _section(_crit("b", "new q?"))}
        plan = _plan(new)
        row = {"call_shas": _plan(old).call_fingerprints()}
        reusable, stale = partition_calls(row, plan.call_fingerprints(), plan)
        assert reusable == {"A"} and stale == {"B"}

    def test_question_mode_partial_reuse(self):
        # one changed question re-judges only its own call
        old = {"A": _section(_crit("a1"), _crit("a2"), call_per="question")}
        new = {"A": _section(_crit("a1"), _crit("a2", "new q?"), call_per="question")}
        plan = _plan(new)
        row = {"call_shas": _plan(old).call_fingerprints()}
        reusable, stale = partition_calls(row, plan.call_fingerprints(), plan)
        assert reusable == {"A::a1"} and stale == {"A::a2"}

    def test_missing_sha_is_stale(self):
        rubric = {"A": _section(_crit("a")), "B": _section(_crit("b"))}
        plan = _plan(rubric)
        fps = plan.call_fingerprints()
        row = {"call_shas": {"A": fps["A"]}}
        reusable, stale = partition_calls(row, fps, plan)
        assert reusable == {"A"} and stale == {"B"}

    def test_item_specific_fingerprints(self):
        # placeholder values are per-item: same rubric, different fills,
        # different fingerprints — a row judged with one fill is stale
        rubric = {"A": _section(_crit("a", "In {language}?"))}
        plan = _plan(rubric, placeholders={"language": "Swahili"})
        row = {
            "call_shas": _plan(
                rubric, placeholders={"language": "English"}
            ).call_fingerprints()
        }
        reusable, stale = partition_calls(row, plan.call_fingerprints(), plan)
        assert stale == {"A"}


class TestLegacyRows:
    """Rows without call_shas fall back to criterion-shape matching."""

    def _row(self, sections):
        return {"sections": sections}

    def _pos_row(self, cid, question="q?"):
        return {
            "criterion_id": cid,
            "question": question,
            "criterion_type": "positive",
            "response": "Yes",
        }

    def test_identical_criteria_reusable(self):
        rubric = {"A": _section(_crit("a", "keep?"))}
        plan = _plan(rubric)
        row = self._row({"A": [self._pos_row("a", "keep?")]})
        reusable, _ = partition_calls(row, plan.call_fingerprints(), plan)
        assert reusable == {"A"}

    def test_different_question_is_stale(self):
        rubric = {"A": _section(_crit("a", "new?"))}
        plan = _plan(rubric)
        row = self._row({"A": [self._pos_row("a", "old?")]})
        _, stale = partition_calls(row, plan.call_fingerprints(), plan)
        assert stale == {"A"}

    def test_evidence_criteria_never_reusable_legacy(self):
        # legacy rows predate evidence answers — never reuse them
        rubric = {"A": _section(_crit("a", "keep?", evidence=True))}
        plan = _plan(rubric)
        row = self._row({"A": [self._pos_row("a", "keep?")]})
        _, stale = partition_calls(row, plan.call_fingerprints(), plan)
        assert stale == {"A"}

    def test_missing_criterion_row_is_stale(self):
        rubric = {"A": _section(_crit("a"), _crit("b"))}
        plan = _plan(rubric)
        row = self._row({"A": [self._pos_row("a")]})
        _, stale = partition_calls(row, plan.call_fingerprints(), plan)
        assert stale == {"A"}

    def test_question_mode_partial_legacy_reuse(self):
        rubric = {
            "A": _section(_crit("a", "keep?"), _crit("b", "new?"), call_per="question")
        }
        plan = _plan(rubric)
        row = self._row(
            {"A": [self._pos_row("a", "keep?"), self._pos_row("b", "old?")]}
        )
        reusable, stale = partition_calls(row, plan.call_fingerprints(), plan)
        assert reusable == {"A::a"} and stale == {"A::b"}


class TestFilterCalls:
    def test_keeps_only_named_calls(self):
        rubric = {"A": _section(_crit("a1"), _crit("a2"), call_per="question")}
        plan = _plan(rubric)
        sub = filter_calls(plan, {"A::a2"})
        assert [c.key for c in sub.calls] == ["A::a2"]
        # the rubric view stays intact for assembly
        assert [c.id for c in sub.sections[0].criteria] == ["a1", "a2"]


class TestMergeJudgment:
    def test_reusable_calls_take_cached_rows(self):
        rubric = {
            "A": _section(_crit("a1"), _crit("a2"), call_per="question", weight=2.0)
        }
        plan = _plan(rubric)
        cached = {
            "sections": {
                "A": [
                    {"criterion_id": "a1", "response": "Yes"},
                    {"criterion_id": "a2", "response": "Yes"},
                ]
            }
        }
        fresh = {
            "sections": {
                "A": [
                    {"criterion_id": "a1", "response": "N/A"},
                    {"criterion_id": "a2", "response": "No"},
                ]
            }
        }
        merged = merge_judgment(cached, fresh, plan, {"A::a1"})
        rows = {r["criterion_id"]: r["response"] for r in merged["sections"]["A"]}
        assert rows == {"a1": "Yes", "a2": "No"}
        assert merged["section_weights"] == {"A": 2.0}
        assert merged["reused_calls"] == ["A::a1"]

    def test_resume_roundtrip(self):
        # partition -> filter -> assemble -> merge, end to end
        old_rubric = {"A": _section(_crit("a1"), _crit("a2"), call_per="question")}
        cached = {
            "call_shas": _plan(old_rubric).call_fingerprints(),
            "sections": {
                "A": [
                    {
                        "criterion_id": "a1",
                        "question": "q?",
                        "criterion_type": "positive",
                        "response": "Yes",
                    },
                    {
                        "criterion_id": "a2",
                        "question": "q?",
                        "criterion_type": "positive",
                        "response": "Yes",
                    },
                ]
            },
        }
        new_rubric = {
            "A": _section(_crit("a1"), _crit("a2", "new?"), call_per="question")
        }
        plan = _plan(new_rubric)
        reusable, stale = partition_calls(cached, plan.call_fingerprints(), plan)
        assert stale == {"A::a2"}

        sub = filter_calls(plan, stale)
        answers = {c.key: CallAnswer(key=c.key, data={"a2": "No"}) for c in sub.calls}
        fresh = assemble_judgment(sub, answers)
        merged = merge_judgment(cached, fresh, plan, reusable)

        rows = {r["criterion_id"]: r["response"] for r in merged["sections"]["A"]}
        assert rows == {"a1": "Yes", "a2": "No"}
