"""Tests for the LLM calling and response logging wrapper."""

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from phonics_benchmark import llm
from phonics_benchmark.prompts import build_decodability_prompt


def _prompt_for(item: dict[str, Any]) -> str:
    return build_decodability_prompt(
        {
            "benchmark_item_id": item["benchmark_item_id"],
            "content": item["content"],
            "student_knowledge": item["student_knowledge"],
        },
        "sw",
    )


def test_call_llm_returns_metadata_and_logs_response(
    tusome_stage_11_item, tmp_path, monkeypatch
):
    prompt = _prompt_for(tusome_stage_11_item)
    response_text = json.dumps(
        tusome_stage_11_item["nondecodable_words"], ensure_ascii=False
    )
    caller = Mock()
    caller.call.return_value = SimpleNamespace(
        Message=response_text,
        TokensUsedCompletion=17,
    )
    get_caller = Mock(return_value=caller)
    monkeypatch.setattr(llm, "get_caller", get_caller)
    monkeypatch.setattr(llm.time, "monotonic", Mock(side_effect=[10.0, 11.2345]))

    record = llm.call_llm(
        prompt,
        model="test-model",
        system_prompt="Return only a JSON array.",
        max_tokens=321,
        log_base_dir=tmp_path,
        tag=tusome_stage_11_item["benchmark_item_id"],
    )

    messages = caller.call.call_args.args[0]
    get_caller.assert_called_once_with("test-model")
    assert [(message.Role, message.Message) for message in messages] == [
        ("system", "Return only a JSON array."),
        ("user", prompt),
    ]
    assert caller.call.call_args.kwargs == {
        "max_tokens": 321,
        "response_schema": None,
    }
    assert record["model"] == "test-model"
    assert record["response_text"] == response_text
    assert record["tokens_used_completion"] == 17
    assert record["latency_seconds"] == 1.235

    log_files = list((tmp_path / "llm_log").glob("*.json"))
    assert len(log_files) == 1
    assert tusome_stage_11_item["benchmark_item_id"] in log_files[0].name
    assert json.loads(log_files[0].read_text()) == record


def test_call_llm_bypass_does_not_create_a_caller(monkeypatch):
    def fail_if_called(model: str):
        pytest.fail(f"get_caller unexpectedly called for {model}")

    monkeypatch.setattr(llm, "get_caller", fail_if_called)

    assert llm.call_llm("unused", model="test-model", bypass_llm=True) == {
        "response_text": "BYPASS"
    }


@pytest.mark.parametrize(
    ("response", "expected_tokens"),
    [
        (None, None),
        (SimpleNamespace(Message="", TokensUsedCompletion=0), 0),
    ],
)
def test_call_llm_returns_empty_responses(
    response,
    expected_tokens,
    tusome_stage_11_item,
    monkeypatch,
):
    caller = Mock()
    caller.call.return_value = response
    monkeypatch.setattr(llm, "get_caller", lambda model: caller)

    record = llm.call_llm(
        _prompt_for(tusome_stage_11_item),
        model="test-model",
        tag=tusome_stage_11_item["benchmark_item_id"],
    )

    assert record["response_text"] == ""
    assert record["tokens_used_completion"] == expected_tokens
