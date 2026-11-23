"""Tests for plan_judgment: call structure and prompt fingerprints."""

from __future__ import annotations

from rubric_judge.plan import JudgeItem, plan_judgment
from rubric_judge.rubric import Criterion, CriterionType, Section


def _crit(cid, question="q?", criterion_type=CriterionType.POSITIVE, **kw):
    return Criterion(id=cid, question=question, criterion_type=criterion_type, **kw)


def _section(*criteria, **kw):
    return Section(name="s", criteria=list(criteria), **kw)


def _plan(rubric, **item_kw):
    return plan_judgment(JudgeItem(text="a text", **item_kw), rubric=rubric)


class TestCallStructure:
    def test_one_call_per_section_by_default(self):
        plan = _plan({"A": _section(_crit("a1"), _crit("a2"))})
        assert [c.kind for c in plan.calls] == ["section"]
        assert plan.calls[0].criterion_ids == ("a1", "a2")

    def test_question_mode_one_call_per_criterion(self):
        plan = _plan({"A": _section(_crit("a1"), _crit("a2"), call_per="question")})
        assert [c.key for c in plan.calls] == ["A::a1", "A::a2"]
        assert all(c.kind == "question" for c in plan.calls)

    def test_gated_out_criteria_produce_no_call(self):
        plan = plan_judgment(
            JudgeItem(text="t", tags=frozenset({"S1"})),
            rubric={"A": _section(_crit("a1"), _crit("a2", gating="S2+"))},
            ordered_tags=("S1", "S2"),
        )
        assert [c.criterion_ids for c in plan.calls] == [("a1",)]
        assert plan.sections[0].gated_out == frozenset({"a2"})

    def test_fully_gated_section_produces_no_calls(self):
        plan = plan_judgment(
            JudgeItem(text="t", tags=frozenset({"S1"})),
            rubric={"A": _section(_crit("a2", gating="S2+"))},
            ordered_tags=("S1", "S2"),
        )
        assert plan.calls == []

    def test_prompts_contain_rendered_text(self):
        plan = plan_judgment(
            JudgeItem(
                text="the story",
                tags=frozenset(),
                placeholders={"language": "Swahili"},
                context="Student context.",
            ),
            rubric={
                "A": _section(_crit("a1", "Is it in {language}?")),
            },
            subject="a story",
        )
        prompt = plan.calls[0].prompt
        assert "the story" in prompt
        assert "Is it in Swahili?" in prompt
        assert "Student context." in prompt
        assert "a story" in prompt

    def test_prompt_template_replaces_boilerplate(self):
        plan = plan_judgment(
            JudgeItem(text="the answer", context="q: what?"),
            rubric={
                "A": _section(
                    _crit("a1"),
                    prompt_template=(
                        "Question: {context}\nAnswer: {text}\n"
                        "Criteria:\n{criteria}\n{instructions}"
                    ),
                )
            },
        )
        prompt = plan.calls[0].prompt
        assert "TEXT TO EVALUATE" not in prompt
        assert prompt.startswith("Question: q: what?\nAnswer: the answer\n")
        assert "1. [a1] q?" in prompt
        assert 'respond with exactly "Yes" or "No"' in prompt
        # schema machinery unchanged
        assert "a1" in plan.calls[0].schema_json["properties"]

    def test_prompt_template_unfilled_slot_raises(self):
        import pytest

        with pytest.raises(ValueError, match="prompt_template.*unfilled"):
            plan_judgment(
                JudgeItem(text="t", placeholders={}),
                rubric={"A": _section(_crit("a1"), prompt_template="Q {missing}")},
            )

    def test_inputs_rows(self):
        plan = _plan({"A": _section(_crit("a1"))})
        rows = plan.inputs_rows("item-1")
        assert rows == [
            {
                "benchmark_item_id": "item-1::A",
                "prompt": plan.calls[0].prompt,
                "system_prompt": None,
            }
        ]

    def test_system_prompt_on_calls_and_rows(self):
        plan = plan_judgment(
            JudgeItem(text="t", placeholders={"language": "Swahili"}),
            rubric={"A": _section(_crit("a1"))},
            system_prompt="You evaluate {language} texts.",
        )
        assert plan.calls[0].system_prompt == "You evaluate Swahili texts."
        assert plan.inputs_rows("i")[0]["system_prompt"] == (
            "You evaluate Swahili texts."
        )


class TestFingerprints:
    """Fingerprint = what the model is asked, not how it's scored."""

    def test_stable(self):
        rubric = {"A": _section(_crit("a"))}
        assert _plan(rubric).call_fingerprints() == _plan(rubric).call_fingerprints()

    def test_question_change_invalidates(self):
        a = _plan({"A": _section(_crit("a", "one?"))}).call_fingerprints()
        b = _plan({"A": _section(_crit("a", "two?"))}).call_fingerprints()
        assert a["A"] != b["A"]

    def test_scoring_changes_do_not_invalidate(self):
        def ev(rules):
            return _crit(
                "a",
                criterion_type=CriterionType.EVIDENCE_LIST,
                evidence_items={"quote": "str"},
                scoring_rules=rules,
            )

        none = _plan({"A": _section(ev(None))}).call_fingerprints()
        some = _plan({"A": _section(ev([{"less_than": 1}]))}).call_fingerprints()
        more = _plan({"A": _section(ev([{"less_than": 3}]))}).call_fingerprints()
        assert none == some == more
        light = _plan({"A": _section(_crit("a", weight=1.0), weight=1.0)})
        heavy = _plan({"A": _section(_crit("a", weight=5.0), weight=3.0)})
        assert light.call_fingerprints() == heavy.call_fingerprints()

    def test_placeholder_values_invalidate(self):
        rubric = {"A": _section(_crit("a", "In {language}?"))}
        a = _plan(rubric, placeholders={"language": "English"}).call_fingerprints()
        b = _plan(rubric, placeholders={"language": "Swahili"}).call_fingerprints()
        assert a["A"] != b["A"]

    def test_context_and_subject_invalidate(self):
        rubric = {"A": _section(_crit("a"))}
        a = _plan(rubric, context="ctx").call_fingerprints()
        b = _plan(rubric).call_fingerprints()
        assert a["A"] != b["A"]
        c = plan_judgment(
            JudgeItem(text="t"), rubric=rubric, subject="a poem"
        ).call_fingerprints()
        assert a["A"] != c["A"]

    def test_gating_outcome_invalidate(self):
        rubric = {"A": _section(_crit("a1"), _crit("a2", gating="S2+"))}
        a = plan_judgment(
            JudgeItem(text="t", tags=frozenset({"S1"})),
            rubric=rubric,
            ordered_tags=("S1", "S2"),
        ).call_fingerprints()
        b = plan_judgment(
            JudgeItem(text="t", tags=frozenset({"S2"})),
            rubric=rubric,
            ordered_tags=("S1", "S2"),
        ).call_fingerprints()
        assert a["A"] != b["A"]

    def test_system_prompt_invalidate(self):
        rubric = {"A": _section(_crit("a"))}
        a = _plan(rubric).call_fingerprints()
        b = plan_judgment(
            JudgeItem(text="t"), rubric=rubric, system_prompt="You are a judge."
        ).call_fingerprints()
        assert a["A"] != b["A"]

    def test_call_per_invalidate(self):
        a = _plan({"A": _section(_crit("a1"), _crit("a2"))}).call_fingerprints()
        b = _plan(
            {"A": _section(_crit("a1"), _crit("a2"), call_per="question")}
        ).call_fingerprints()
        assert a != b

    def test_question_mode_one_change_invalidates_only_its_call(self):
        a = _plan(
            {"A": _section(_crit("a1"), _crit("a2"), call_per="question")}
        ).call_fingerprints()
        b = _plan(
            {"A": _section(_crit("a1"), _crit("a2", "new?"), call_per="question")}
        ).call_fingerprints()
        assert a["A::a1"] == b["A::a1"]
        assert a["A::a2"] != b["A::a2"]

    def test_schema_change_invalidate(self):
        rubric_a = {
            "A": _section(
                _crit(
                    "a",
                    criterion_type=CriterionType.EVIDENCE_LIST,
                    evidence_items={"quote": "str", "severity": ["low", "high"]},
                )
            )
        }
        rubric_b = {
            "A": _section(
                _crit(
                    "a",
                    criterion_type=CriterionType.EVIDENCE_LIST,
                    evidence_items={"quote": "str", "severity": ["low"]},
                )
            )
        }
        assert (
            _plan(rubric_a).call_fingerprints() != _plan(rubric_b).call_fingerprints()
        )
