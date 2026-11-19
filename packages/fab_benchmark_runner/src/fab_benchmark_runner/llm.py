"""Calling models through FDLLM."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from fab_benchmark_runner.errors import ConfigError
from fab_benchmark_runner.inputs import ModelInput


def call_llm(
    prompt: str,
    *,
    model: str,
    max_tokens: int,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Send one text prompt to a registered model and return the call record."""

    from fdllm import LLMMessage, get_caller

    caller = get_caller(model)
    messages = []
    if system_prompt:
        messages.append(LLMMessage(Role="system", Message=system_prompt))
    messages.append(LLMMessage(Role="user", Message=prompt))
    started = time.monotonic()
    response = caller.call(messages, max_tokens=max_tokens)
    latency_seconds = round(time.monotonic() - started, 3)

    return {
        "response_text": (response.Message or "") if response is not None else "",
        "max_tokens": max_tokens,
        "tokens_used": getattr(response, "TokensUsed", None),
        "tokens_used_completion": getattr(response, "TokensUsedCompletion", None),
        "tokens_used_reasoning": getattr(response, "TokensUsedReasoning", None),
        "model_version": getattr(response, "ResponseModel", None),
        "timestamp_utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ"),
        "latency_seconds": latency_seconds,
    }


def call_model_with_fdllm(
    model_name: str, model_input: ModelInput, max_tokens: int | None
) -> dict[str, Any]:
    """The default way a run calls a model: one FDLLM call per stored input.

    Returns the row fields of the response; the runner adds the input and IDs.
    """

    if model_input.images:
        raise ConfigError("Images are not sent yet")
    if max_tokens is None:
        raise ConfigError("Runs without a token limit are not supported yet")
    record = call_llm(
        model_input.prompt,
        model=model_name,
        max_tokens=max_tokens,
        system_prompt=model_input.system_prompt,
    )
    return {
        "output": record["response_text"],
        "tokens_used": record["tokens_used"],
        "tokens_used_completion": record["tokens_used_completion"],
        "tokens_used_reasoning": record["tokens_used_reasoning"],
        "model_version": record["model_version"],
        "timestamp_utc": record["timestamp_utc"],
        "latency_seconds": record["latency_seconds"],
    }
