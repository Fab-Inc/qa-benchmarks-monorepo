"""Stable identifiers for model calls, results and run configs.

These functions decide whether an existing result can be reused, so their output
must not change by accident: tests pin known values.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import Any

RESULT_ID_LENGTH = 24
IMAGE_DETAILS = frozenset({"low", "high"})
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


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
    max_tokens: int | None,
    images: Sequence[Mapping[str, Any]] = (),
) -> str:
    """Identify one model call by everything that shapes its response.

    ``model_config`` is the sanitized configuration FDLLM registered for the model.
    ``max_tokens`` is the limit sent to the provider; None means no limit.
    ``images`` are the item's image references in the order they are sent. Only each
    image's file hash and detail level count, so where an image is stored doesn't
    change the ID. A call without images hashes exactly as before images existed.
    """

    payload: dict[str, Any] = {
        "benchmark_item_id": benchmark_item_id,
        "generation_parameters": {"max_tokens": max_tokens},
        "model": model,
        "model_config": model_config,
        "prompt": prompt,
        "system_prompt": system_prompt,
    }
    if images:
        payload["images"] = [image_identity(image) for image in images]
    return sha256_json(payload)


def image_identity(image: Mapping[str, Any]) -> dict[str, str]:
    """Return the part of an image reference that identifies what a model receives.

    Raises ValueError for a detail level other than low or high, or a hash that isn't
    64 lowercase hex characters.
    """

    detail = image.get("detail")
    digest = image.get("sha256")
    if detail not in IMAGE_DETAILS:
        raise ValueError(f"Image detail must be 'low' or 'high', not {detail!r}")
    if not isinstance(digest, str) or not _SHA256_HEX.fullmatch(digest):
        raise ValueError(
            f"Image sha256 must be 64 lowercase hex characters, not {digest!r}"
        )
    return {"detail": detail, "sha256": digest}


def result_id(model_call_id: str) -> str:
    """Identify the stored result of one model call."""

    return sha256_json({"model_call_id": model_call_id})[:RESULT_ID_LENGTH]
