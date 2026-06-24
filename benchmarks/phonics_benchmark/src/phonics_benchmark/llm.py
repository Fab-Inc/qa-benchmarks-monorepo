"""Wrapper for calling LLMs via fdllm, with response logging & truncation detection."""

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fdllm import LLMMessage, get_caller
from pydantic import BaseModel


def call_llm(
    prompt: str,
    model: str,
    system_prompt: str | None = None,
    max_tokens: int = 10000,
    bypass_llm: bool = False,
    should_log: bool = True,
    log_base_dir: Path | None = None,
    tag: str = "",
    response_schema: type[BaseModel] | None = None,
) -> dict[str, Any]:
    """Send a prompt to an LLM and return the response."""
    if bypass_llm:
        return {"response_text": "BYPASS"}

    caller = get_caller(model)
    messages: list[LLMMessage] = []
    if system_prompt:
        messages.append(LLMMessage(Role="system", Message=system_prompt))
    messages.append(LLMMessage(Role="user", Message=prompt))
    t0 = time.monotonic()
    outmsg: LLMMessage = caller.call(
        messages,
        max_tokens=max_tokens,
        response_schema=response_schema,
    )
    duration_seconds = round(time.monotonic() - t0, 3)

    if not outmsg or not outmsg.Message:
        raise ValueError(f"LLM response empty for prompt {tag}")

    may_be_truncated = _check_if_llm_response_truncated(outmsg, max_tokens, tag)

    response_text = outmsg.Message

    timestamp_utc: str = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")

    record = {
        "model": model,
        "system_prompt": system_prompt,
        "prompt": prompt,
        "timestamp_utc": timestamp_utc,
        "max_tokens": max_tokens,
        "tokens_used": outmsg.TokensUsed,
        "tokens_used_completion": outmsg.TokensUsedCompletion,
        "tokens_used_reasoning": outmsg.TokensUsedReasoning,
        "may_be_truncated": may_be_truncated,
        "duration_seconds": duration_seconds,
        "response_text": response_text,
    }

    if should_log and log_base_dir:
        _log_llm_response(log_base_dir, record, timestamp_utc, tag)

    return record


def _check_if_llm_response_truncated(
    outmsg: LLMMessage, max_tokens: int, tag: str
) -> bool | None:
    """Checks for possible truncation in LLM output by comparing tokens used to max."""
    if not outmsg.TokensUsed:
        print(f"WARNING: LLM token counts cannot be determined for prompt {tag}")
        return None
    # Input tokens might be included in max_tokens limit and tokens_used
    elif outmsg.TokensUsed >= (max_tokens * 0.9):  # conservative margin of error
        print(f"WARNING: possible truncated output for prompt {tag}")
        return True
    else:
        return False


def _log_llm_response(
    log_base_dir: Path, result: dict, timestamp: str, tag: str
) -> None:
    llm_log_dir: Path = log_base_dir / "llm_log"
    llm_log_dir.mkdir(exist_ok=True)

    output_file = llm_log_dir / f"llm_response_{timestamp}_{tag}.json"
    output_file.write_text(json.dumps(result, indent=2))
