"""Calling models through FDLLM."""

from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from PIL import Image

from fab_benchmark_runner.inputs import ImageRef, ModelInput


def call_llm(
    prompt: str,
    *,
    model: str,
    max_tokens: int | None,
    system_prompt: str | None = None,
    images: Sequence[ImageRef] = (),
) -> dict[str, Any]:
    """Send one prompt, with any images, to a registered model; return the record.

    Images go into the one user message, in order, each at a local path. Each call
    builds its own FDLLM images, which resize as they are built, so calls running
    in parallel share nothing. With ``max_tokens`` None, FDLLM asks for what the
    model's context window leaves after the input, capped by the registry's
    ``Token_Limit_Completion``.
    """

    from fdllm import LLMMessage, get_caller

    caller = get_caller(model)
    messages = []
    if system_prompt:
        messages.append(LLMMessage(Role="system", Message=system_prompt))
    messages.append(
        LLMMessage(
            Role="user",
            Message=prompt,
            Images=[_llm_image(image) for image in images] or None,
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


def call_model_with_fdllm(
    model_name: str, model_input: ModelInput, max_tokens: int | None
) -> dict[str, Any]:
    """The default way a run calls a model: one FDLLM call per stored input.

    The runner hands over each image at a checked local copy. Returns the row
    fields of the response; the runner adds the input and IDs.
    """

    record = call_llm(
        model_input.prompt,
        model=model_name,
        max_tokens=max_tokens,
        system_prompt=model_input.system_prompt,
        images=model_input.images,
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


def _llm_image(image: ImageRef) -> Any:
    """Load an image from its local file into a new FDLLM image.

    FDLLM resizes an image only when it is built with one, so the file's pixels go
    in at construction, never assigned afterwards.
    """

    from fdllm import LLMImage

    with Image.open(image.uri) as source:
        source.load()
        picture = source.copy()
    return LLMImage(Img=picture, Detail=image.detail)
