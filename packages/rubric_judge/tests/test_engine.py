"""Tests for the judge engine with a stubbed LLM backend."""

from __future__ import annotations

import json

import pytest
from rubric_judge.engine import evaluate_text
from rubric_judge.rubric import Criterion, CriterionType, Section


def _crit(cid, question="q?", **kw):
    return Criterion(
        id=cid, question=question, criterion_type=CriterionType.POSITIVE, **kw
    )


def _section(*criteria, **kw):
    return Section(name="s", criteria=list(criteria), **kw)


def _stub_llm(responses: dict[frozenset, dict], calls: list | None = None):
    """An ``llm`` callable: maps the schema's field ids to a canned response.

    ``responses`` maps frozensets of criterion ids to the JSON dict the
    stub returns for that call. Failing ids (``None`` value) raise.
    """
    prompts = calls if calls is not None else []

    def llm(prompt, *, model, max_tokens, response_schema):
        prompts.append(prompt)
        fields = frozenset(response_schema.model_fields)
        resp = responses[fields]
        if resp is None:
            raise RuntimeError("stub failure")
        return {"response_text": json.dumps(resp), "tokens_used": 10}

    return llm


class TestCallPer:
    def test_section_mode_makes_one_call(self):
        calls = []
        rubric = {
            "A": _section(_crit("a1"), _crit("a2"), call_per="section"),
            "B": _section(_crit("b1"), call_per="question"),
        }
        llm = _stub_llm(
            {
                frozenset({"a1", "a2"}): {"a1": "Yes", "a2": "No"},
                frozenset({"b1"}): {"b1": "Yes"},
            },
            calls,
        )
        out = evaluate_text(
            "text",
            rubric=rubric,
            tags=set(),
            judge_model="m",
            llm=llm,
            verbose=False,
        )
        assert len(calls) == 2
        responses = {r["criterion_id"]: r["response"] for r in out["sections"]["A"]}
        assert responses == {"a1": "Yes", "a2": "No"}
        assert not out["failed_sections"]
        assert len(out["token_usage"]) == 2

    def test_question_mode_makes_one_call_per_criterion(self):
        calls = []
        rubric = {"A": _section(_crit("a1"), _crit("a2"), call_per="question")}
        llm = _stub_llm(
            {
                frozenset({"a1"}): {"a1": "Yes"},
                frozenset({"a2"}): {"a2": "No"},
            },
            calls,
        )
        out = evaluate_text(
            "text",
            rubric=rubric,
            tags=set(),
            judge_model="m",
            llm=llm,
            verbose=False,
        )
        assert len(calls) == 2
        # each prompt carries exactly one criterion's question line
        assert sum("[a1]" in p for p in calls) == 1
        assert sum("[a2]" in p for p in calls) == 1
        responses = {r["criterion_id"]: r["response"] for r in out["sections"]["A"]}
        assert responses == {"a1": "Yes", "a2": "No"}
        assert len(out["token_usage"]) == 2

    def test_question_mode_partial_failure(self, monkeypatch):
        monkeypatch.setattr("rubric_judge.engine.time.sleep", lambda s: None)
        rubric = {"A": _section(_crit("a1"), _crit("a2"), call_per="question")}
        llm = _stub_llm(
            {
                frozenset({"a1"}): {"a1": "Yes"},
                frozenset({"a2"}): None,  # raises
            }
        )
        out = evaluate_text(
            "text",
            rubric=rubric,
            tags=set(),
            judge_model="m",
            llm=llm,
            verbose=False,
        )
        responses = {r["criterion_id"]: r["response"] for r in out["sections"]["A"]}
        assert responses == {"a1": "Yes", "a2": "N/A"}
        assert out["failed_sections"] == ["A"]

    def test_criterion_weight_stored_on_row(self):
        rubric = {
            "A": _section(_crit("a1", weight=2.0), _crit("a2"), call_per="section")
        }
        llm = _stub_llm({frozenset({"a1", "a2"}): {"a1": "Yes", "a2": "No"}})
        out = evaluate_text(
            "text",
            rubric=rubric,
            tags=set(),
            judge_model="m",
            llm=llm,
            verbose=False,
        )
        rows = {r["criterion_id"]: r for r in out["sections"]["A"]}
        assert rows["a1"]["weight"] == 2.0
        assert "weight" not in rows["a2"]  # default 1.0 stays implicit

    def test_placeholders_fill_rubric_text(self):
        calls = []
        rubric = {
            "A": _section(
                _crit("a1", "Is the text in {language}?"),
                section_preface="Judge {language} fluency.",
            )
        }
        llm = _stub_llm({frozenset({"a1"}): {"a1": "Yes"}}, calls)
        out = evaluate_text(
            "text",
            rubric=rubric,
            tags=set(),
            judge_model="m",
            llm=llm,
            placeholders={"language": "Swahili"},
            verbose=False,
        )
        full = "\n".join(calls)
        assert "{language}" not in full
        assert "Is the text in Swahili?" in full  # criterion question
        assert "Judge Swahili fluency." in full  # section preface
        # the stored row keeps the rendered question
        assert out["sections"]["A"][0]["question"] == "Is the text in Swahili?"

    def test_unfilled_placeholder_raises(self):
        rubric = {"A": _section(_crit("a1", "Is the text in {language}?"))}
        llm = _stub_llm({frozenset({"a1"}): {"a1": "Yes"}})
        with pytest.raises(ValueError, match="unfilled.*language"):
            evaluate_text(
                "text",
                rubric=rubric,
                tags=set(),
                judge_model="m",
                llm=llm,
                placeholders={},
                verbose=False,
            )

    def test_unfilled_placeholder_in_section_preface_raises(self):
        rubric = {
            "A": _section(_crit("a1"), section_preface="Judge {kind}."),
        }
        llm = _stub_llm({frozenset({"a1"}): {"a1": "Yes"}})
        with pytest.raises(ValueError, match="unfilled.*kind"):
            evaluate_text(
                "text",
                rubric=rubric,
                tags=set(),
                judge_model="m",
                llm=llm,
                placeholders={},
                verbose=False,
            )

    def test_escaped_braces_render_literally(self):
        calls = []
        rubric = {"A": _section(_crit("a1", "Does it show {{result}} like {fmt}?"))}
        llm = _stub_llm({frozenset({"a1"}): {"a1": "Yes"}}, calls)
        evaluate_text(
            "text",
            rubric=rubric,
            tags=set(),
            judge_model="m",
            llm=llm,
            placeholders={"fmt": "JSON"},
            verbose=False,
        )
        assert "Does it show {result} like JSON?" in calls[0]

    def test_no_placeholders_leaves_text_verbatim(self):
        calls = []
        rubric = {"A": _section(_crit("a1", "Is {anything} true?"))}
        llm = _stub_llm({frozenset({"a1"}): {"a1": "Yes"}}, calls)
        out = evaluate_text(
            "text",
            rubric=rubric,
            tags=set(),
            judge_model="m",
            llm=llm,
            verbose=False,
        )
        assert "Is {anything} true?" in calls[0]
        assert out["sections"]["A"][0]["question"] == "Is {anything} true?"

    def test_gated_out_criteria_get_no_call(self):
        calls = []
        rubric = {
            "A": _section(_crit("a1"), _crit("a2", gating="S2+"), call_per="question")
        }
        llm = _stub_llm({frozenset({"a1"}): {"a1": "Yes"}}, calls)
        out = evaluate_text(
            "text",
            rubric=rubric,
            tags={"S1"},
            judge_model="m",
            llm=llm,
            ordered_tags=("S1", "S2"),
            verbose=False,
        )
        assert len(calls) == 1
        responses = {r["criterion_id"]: r for r in out["sections"]["A"]}
        assert responses["a1"]["response"] == "Yes"
        assert responses["a2"]["gated_out"] is True
