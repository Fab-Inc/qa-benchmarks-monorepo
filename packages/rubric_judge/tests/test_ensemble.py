"""Tests for ensemble judging: execute_ensemble, merge_ensemble, evaluate_ensemble."""

from __future__ import annotations

import json

from rubric_judge.engine import (
    CallAnswer,
    assemble_ensemble,
    evaluate_ensemble,
    merge_ensemble,
)
from rubric_judge.plan import JudgeItem, plan_judgment
from rubric_judge.rubric import Criterion, CriterionType, Section


def _crit(cid, question="q?", criterion_type=CriterionType.POSITIVE, **kw):
    return Criterion(id=cid, question=question, criterion_type=criterion_type, **kw)


def _section(*criteria, **kw):
    return Section(name="s", criteria=list(criteria), **kw)


def _judge_llm(response):
    """A JudgeLLM stub returning the same fields for every call."""

    def llm(prompt, *, model, max_tokens, response_schema, **kw):
        return {"response_text": json.dumps(response), "tokens_used": 10}

    return llm


def _yes_llm(prompt, *, model, max_tokens, response_schema, **kw):
    return {
        "response_text": json.dumps({f: "Yes" for f in response_schema.model_fields}),
        "tokens_used": 10,
    }


class TestEvaluateEnsemble:
    def test_majority_vote(self):
        rubric = {"A": _section(_crit("a1"), _crit("a2"))}
        judges = {
            "j1": _judge_llm({"a1": "Yes", "a2": "No"}),
            "j2": _yes_llm,
            "j3": _yes_llm,
        }
        out = evaluate_ensemble(
            JudgeItem(text="t"),
            rubric=rubric,
            judges=judges,
            verbose=False,
        )
        rows = {r["criterion_id"]: r for r in out["sections"]["A"]}
        assert rows["a1"]["response"] == "Yes"  # unanimous
        assert rows["a2"]["response"] == "Yes"  # 2/3 majority
        assert rows["a2"]["votes"] == {"j1": "No", "j2": "Yes", "j3": "Yes"}
        assert rows["a2"]["agreement"] == 2 / 3
        assert out["judges"] == ["j1", "j2", "j3"]

    def test_tie_is_na(self):
        rubric = {"A": _section(_crit("a1"))}
        judges = {
            "j1": _judge_llm({"a1": "Yes"}),
            "j2": _judge_llm({"a1": "No"}),
        }
        out = evaluate_ensemble(
            JudgeItem(text="t"), rubric=rubric, judges=judges, verbose=False
        )
        row = out["sections"]["A"][0]
        assert row["response"] == "N/A" and row["tie"] is True
        assert row["agreement"] == 0.5

    def test_single_responder_keeps_primary_row(self):
        rubric = {"A": _section(_crit("a1", allows_na=True))}
        judges = {
            "j1": _judge_llm({"a1": "Yes"}),
            "j2": _judge_llm({"a1": "N/A"}),
        }
        out = evaluate_ensemble(
            JudgeItem(text="t"), rubric=rubric, judges=judges, verbose=False
        )
        row = out["sections"]["A"][0]
        assert row["response"] == "Yes" and "votes" not in row

    def test_all_na_is_na(self):
        rubric = {"A": _section(_crit("a1", allows_na=True))}
        judges = {f"j{i}": _judge_llm({"a1": "N/A"}) for i in range(3)}
        out = evaluate_ensemble(
            JudgeItem(text="t"), rubric=rubric, judges=judges, verbose=False
        )
        assert out["sections"]["A"][0]["response"] == "N/A"

    def test_token_usage_records_judge(self):
        rubric = {"A": _section(_crit("a1"))}
        judges = {"j1": _yes_llm, "j2": _yes_llm}
        out = evaluate_ensemble(
            JudgeItem(text="t"), rubric=rubric, judges=judges, verbose=False
        )
        assert {u["judge"] for u in out["token_usage"]} == {"j1", "j2"}

    def test_system_prompt_passed_to_llm(self):
        seen = []

        def llm(prompt, *, model, max_tokens, response_schema, system_prompt=None):
            seen.append(system_prompt)
            return {
                "response_text": json.dumps(
                    {f: "Yes" for f in response_schema.model_fields}
                )
            }

        evaluate_ensemble(
            JudgeItem(text="t"),
            rubric={"A": _section(_crit("a1"))},
            judges={"j1": llm},
            system_prompt="You are a judge.",
            verbose=False,
        )
        assert seen == ["You are a judge."]


def _judgment(response, section="A", cid="a1", failed=None):
    row = {
        "criterion_id": cid,
        "question": "q?",
        "criterion_type": "positive",
        "response": response,
    }
    return {
        "sections": {section: [row]},
        "failed_sections": list(failed or []),
        "token_usage": [{"call": "section 'A'", "tokens_used": 1}],
        "tags": [],
        "section_weights": {section: 1.0},
    }


class TestMergeEnsemble:
    def test_donor_row_follows_verdict(self):
        # primary votes against the majority: fields come from a majority judge
        judgments = {
            "j1": _judgment("No"),
            "j2": _judgment("Yes"),
            "j3": _judgment("Yes"),
        }
        judgments["j2"]["sections"]["A"][0]["justification"] = "j2 reason"
        merged = merge_ensemble(judgments)
        row = merged["sections"]["A"][0]
        assert row["response"] == "Yes"
        assert row["justification"] == "j2 reason"

    def test_failed_only_when_all_judges_fail(self):
        a = _judgment("N/A", failed=["A"])
        b = _judgment("Yes")
        c = _judgment("Yes")
        merged = merge_ensemble({"j1": a, "j2": b, "j3": c})
        assert merged["failed_sections"] == []
        all_failed = merge_ensemble(
            {
                "j1": a,
                "j2": _judgment("N/A", failed=["A"]),
                "j3": _judgment("N/A", failed=["A"]),
            }
        )
        assert all_failed["failed_sections"] == ["A"]

    def test_assemble_ensemble_wraps(self):
        rubric = {"A": _section(_crit("a1"))}
        plan = plan_judgment(JudgeItem(text="t"), rubric=rubric)
        answers = {
            j: {c.key: CallAnswer(key=c.key, data={"a1": "Yes"}) for c in plan.calls}
            for j in ("j1", "j2")
        }
        merged = assemble_ensemble(plan, answers)
        assert merged["sections"]["A"][0]["response"] == "Yes"
        assert merged["judges"] == ["j1", "j2"]
