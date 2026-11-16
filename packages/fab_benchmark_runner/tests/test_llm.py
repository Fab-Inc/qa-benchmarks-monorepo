"""The FDLLM call wrapper and default evaluator."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fab_benchmark_runner import Benchmark, load_run_config
from fab_benchmark_runner.llm import call_llm, evaluate_item_with_fdllm


def test_call_llm_sends_system_and_user_messages() -> None:
    caller = MagicMock()
    caller.call.return_value = SimpleNamespace(
        Message="answer",
        TokensUsed=30,
        TokensUsedCompletion=10,
        TokensUsedReasoning=4,
        ResponseModel="model-a-2026",
    )

    with patch("fdllm.get_caller", return_value=caller) as get_caller:
        record = call_llm(
            "prompt", model="model-a", max_tokens=99, system_prompt="Be brief."
        )

    get_caller.assert_called_once_with("model-a")
    messages = caller.call.call_args.args[0]
    assert [(m.Role, m.Message) for m in messages] == [
        ("system", "Be brief."),
        ("user", "prompt"),
    ]
    assert caller.call.call_args.kwargs == {"max_tokens": 99}
    assert record["response_text"] == "answer"
    assert record["tokens_used"] == 30
    assert record["tokens_used_completion"] == 10
    assert record["tokens_used_reasoning"] == 4
    assert record["model_version"] == "model-a-2026"
    assert record["max_tokens"] == 99
    assert record["latency_seconds"] >= 0


def test_call_llm_treats_a_missing_response_as_empty() -> None:
    caller = MagicMock()
    caller.call.return_value = None

    with patch("fdllm.get_caller", return_value=caller):
        record = call_llm("prompt", model="model-a", max_tokens=5)

    assert [m.Role for m in caller.call.call_args.args[0]] == ["user"]
    assert record["response_text"] == ""
    assert record["tokens_used"] is None


def test_default_evaluator_uses_run_settings(
    benchmark: Benchmark, write_run: Callable[..., Path]
) -> None:
    config = load_run_config(write_run(max_tokens=321), benchmark)
    config = replace(config, benchmark=replace(benchmark, system_prompt="System."))
    record = {
        "response_text": '["cat"]',
        "max_tokens": 321,
        "tokens_used": 12,
        "tokens_used_completion": 7,
        "tokens_used_reasoning": None,
        "model_version": None,
        "timestamp_utc": "20261116T120000_000000Z",
        "latency_seconds": 1.25,
    }

    with patch(
        "fab_benchmark_runner.llm.call_llm", return_value=record
    ) as call_llm_mock:
        payload = evaluate_item_with_fdllm(
            config, "model-a", {"benchmark_item_id": "item-1"}, "rendered"
        )

    call_llm_mock.assert_called_once_with(
        "rendered", model="model-a", max_tokens=321, system_prompt="System."
    )
    assert payload == {
        "output": '["cat"]',
        "max_tokens": 321,
        "tokens_used": 12,
        "tokens_used_completion": 7,
        "tokens_used_reasoning": None,
        "model_version": None,
        "timestamp_utc": "20261116T120000_000000Z",
        "latency_seconds": 1.25,
    }
