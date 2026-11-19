"""The FDLLM call wrapper and the default way runs call models."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from fab_benchmark_runner import ConfigError, ImageRef, ModelInput
from fab_benchmark_runner.llm import call_llm, call_model_with_fdllm


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


def test_runs_call_fdllm_with_the_stored_input_and_the_experiments_limit() -> None:
    record = {
        "response_text": "B",
        "max_tokens": 321,
        "tokens_used": 12,
        "tokens_used_completion": 7,
        "tokens_used_reasoning": None,
        "model_version": None,
        "timestamp_utc": "20261116T120000_000000Z",
        "latency_seconds": 1.25,
    }
    model_input = ModelInput("mcq-1", "Which?", "Answer with one letter.")

    with patch("fab_benchmark_runner.llm.call_llm", return_value=record) as call:
        payload = call_model_with_fdllm("model-a", model_input, 321)

    call.assert_called_once_with(
        "Which?",
        model="model-a",
        max_tokens=321,
        system_prompt="Answer with one letter.",
    )
    assert payload == {
        "output": "B",
        "tokens_used": 12,
        "tokens_used_completion": 7,
        "tokens_used_reasoning": None,
        "model_version": None,
        "timestamp_utc": "20261116T120000_000000Z",
        "latency_seconds": 1.25,
    }


def test_images_and_unlimited_calls_are_refused_until_they_are_supported() -> None:
    with_image = ModelInput(
        "v-1", "Which?", images=(ImageRef("gs://b/1.jpg", "high", "a" * 64),)
    )

    with pytest.raises(ConfigError, match="Images are not sent yet"):
        call_model_with_fdllm("model-a", with_image, 100)
    with pytest.raises(ConfigError, match="without a token limit"):
        call_model_with_fdllm("model-a", ModelInput("t-1", "Which?"), None)
