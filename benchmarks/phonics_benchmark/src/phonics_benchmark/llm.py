"""Wrapper for calling LLMs via fdllm, with response logging & truncation detection."""

import json
import time
from datetime import UTC, datetime
from functools import cache
from importlib.resources import files
from pathlib import Path

import yaml
from fdllm import LLMMessage, get_caller
from pydantic import BaseModel


@cache
def _load_prompts() -> dict[str, str]:
    """Load and cache prompt templates from the packaged prompts.yaml."""
    text = (
        files("phonics_benchmark").joinpath("prompts.yaml").read_text(encoding="utf-8")
    )
    return yaml.safe_load(text)


def get_prompt(name: str, **kwargs: object) -> str:
    """Return the named prompt template, formatted with the given fields."""
    prompts = _load_prompts()
    try:
        template = prompts[name]
    except KeyError:
        raise KeyError(
            f"Unknown prompt '{name}'; available: {sorted(prompts)}"
        ) from None
    return template.format(**kwargs)


def call_llm(
    prompt: str,
    model: str,
    system_prompt: str | None = None,
    max_tokens: int = 10000,
    bypass_llm: bool = False,
    should_log: bool = True,
    log_path: Path | None = None,
    tag: str = "",
    response_schema: type[BaseModel] | None = None,
) -> str:
    """Send a prompt to an LLM and return the response."""
    if bypass_llm:
        return "BYPASS"

    caller = get_caller(model)
    message = LLMMessage(Role="user", Message=prompt)
    t0 = time.monotonic()
    outmsg: LLMMessage = caller.call(
        message,
        max_tokens=max_tokens,
        response_schema=response_schema,
    )
    duration_seconds = round(time.monotonic() - t0, 3)

    if not outmsg or not outmsg.Message:
        raise ValueError(f"LLM response empty for prompt {tag}")

    was_truncated = _check_if_llm_response_truncated(outmsg, max_tokens, tag)

    response_text = outmsg.Message

    if should_log and log_path:
        timestamp: str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")

        result = {
            "model": model,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "timestamp_utc": timestamp,
            "max_tokens": max_tokens,
            "tokens_used": outmsg.TokensUsed,
            "tokens_used_completion": outmsg.TokensUsedCompletion,
            "tokens_used_reasoning": outmsg.TokensUsedReasoning,
            "was_truncated": was_truncated,
            "duration_seconds": duration_seconds,
            "response": response_text,
        }

        _log_llm_response(log_path, result, timestamp, tag)

    return response_text


def _check_if_llm_response_truncated(
    outmsg: LLMMessage, max_tokens: int, tag: str
) -> bool | None:
    """Checks for possible truncation in LLM output by comparing tokens used to max."""
    if not outmsg.TokensUsed:
        print(f"WARNING: LLM token counts cannot be determined for prompt {tag}")
        return None
    # Gemini seemingly includes input prompt tokens in max_tokens limit and tokens_used
    elif outmsg.TokensUsed >= (max_tokens * 0.9):  # conservative margin of error
        print(f"WARNING: possible truncated output for prompt {tag}")
        return True
    else:
        return False


def _log_llm_response(log_path: Path, result: dict, timestamp: str, tag: str) -> None:
    llm_log_dir: Path = log_path / "llm_log"
    llm_log_dir.mkdir(exist_ok=True, parents=True)

    output_file = llm_log_dir / f"llm_response_{timestamp}_{tag}.json"
    output_file.write_text(json.dumps(result, indent=2))
