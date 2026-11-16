"""Stable identifiers for model calls, results and run configs.

These functions decide whether an existing result can be reused, so their output
must not change by accident: tests pin known values.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from hashlib import sha256
from typing import Any

RESULT_ID_LENGTH = 24


def canonical_json(value: Any, *, indent: int | None = None) -> str:
    """Serialize with sorted keys and unescaped Unicode, as every hash expects."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=indent)


def sha256_json(value: Any) -> str:
    """Return the SHA-256 hex digest of a value's canonical JSON."""

    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def model_call_id(
    *,
    benchmark_item_id: str,
    model: str,
    model_config: Mapping[str, Any],
    prompt: str,
    system_prompt: str | None,
    max_tokens: int,
) -> str:
    """Identify one model call by everything that shapes its response.

    ``model_config`` is the sanitized configuration FDLLM registered for the model.
    ``max_tokens`` is the effective limit actually sent to the provider.
    """

    return sha256_json(
        {
            "benchmark_item_id": benchmark_item_id,
            "generation_parameters": {"max_tokens": max_tokens},
            "model": model,
            "model_config": model_config,
            "prompt": prompt,
            "system_prompt": system_prompt,
        }
    )


def result_id(model_call_id: str) -> str:
    """Identify the stored result of one model call."""

    return sha256_json({"model_call_id": model_call_id})[:RESULT_ID_LENGTH]
