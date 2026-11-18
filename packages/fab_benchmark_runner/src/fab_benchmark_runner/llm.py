"""Calling models through FDLLM."""

from __future__ import annotations

import time
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from fab_benchmark_runner.config import RunConfig
from fab_benchmark_runner.request import PreparedImage, PreparedRequest


def call_llm(
    prompt: str,
    *,
    model: str,
    max_tokens: int,
    system_prompt: str | None = None,
    images: tuple[PreparedImage, ...] = (),
) -> dict[str, Any]:
    """Send one prepared user request to a model and return the call record."""

    from fdllm import LLMMessage, get_caller

    caller = get_caller(model)
    messages = []
    if system_prompt:
        messages.append(LLMMessage(Role="system", Message=system_prompt))
    messages.append(
        LLMMessage(
            Role="user",
            Message=prompt,
            Images=[image.to_llm_image() for image in images] or None,
        )
    )
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


def evaluate_item_with_fdllm(
    config: RunConfig,
    model_name: str,
    item: Mapping[str, Any],
    request: PreparedRequest,
) -> dict[str, Any]:
    """The default evaluator: call the model and return the fields for its row.

    ``item`` is unused because the prepared request carries everything the model sees.
    """

    record = call_llm(
        request.prompt,
        model=model_name,
        max_tokens=config.effective_max_tokens,
        system_prompt=config.benchmark.system_prompt,
        images=request.images,
    )
    return {
        "output": record["response_text"],
        "max_tokens": record["max_tokens"],
        "tokens_used": record["tokens_used"],
        "tokens_used_completion": record["tokens_used_completion"],
        "tokens_used_reasoning": record["tokens_used_reasoning"],
        "model_version": record["model_version"],
        "timestamp_utc": record["timestamp_utc"],
        "latency_seconds": record["latency_seconds"],
    }
