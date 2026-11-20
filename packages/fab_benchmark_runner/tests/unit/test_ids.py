"""Pinned ID values: these must not change, or existing results stop matching.

Call IDs were the phonics runner's until max_tokens left them on 2026-11-19; result
file names still are.
"""

from __future__ import annotations

import json
from hashlib import sha256

import pytest

from fab_benchmark_runner import model_call_id, model_results_filename, result_id
from fab_benchmark_runner.registry import sanitize_model_config

FIREWORKS_CONFIG = {
    "Api_Interface": "OpenAI",
    "Api_Key_Env_Var": "FIREWORKS_API_KEY",
    "Max_Tokens_Arg_Name": "max_tokens",
    "Token_Window": 128000,
    "Client_Args": {
        "base_url": "https://api.fireworks.ai/inference/v1",
        "api_key": "secret-value",
    },
    "group": "Fireworks",
}
SWAHILI_PROMPT = "Soma: Mama analima shamba. Je, ni rahisi?"


def test_text_call_id_is_pinned() -> None:
    """Re-pinned on 2026-11-19, when max_tokens left call IDs.

    Until then these IDs matched the phonics runner's. Phonics and text generation
    results need their IDs recomputed before they can move to this package.
    """

    model_config = sanitize_model_config(FIREWORKS_CONFIG)
    call_id = model_call_id(
        benchmark_item_id="item-1",
        model="fw-llama-v3p2-1b-instruct",
        model_config=model_config,
        prompt=SWAHILI_PROMPT,
        system_prompt=None,
    )
    payload = {
        "benchmark_item_id": "item-1",
        "model": "fw-llama-v3p2-1b-instruct",
        "model_config": model_config,
        "prompt": SWAHILI_PROMPT,
        "system_prompt": None,
    }

    assert call_id == "72a62fa2640945d4ed2325a35b4fcd33eaab7a1f57afecfde8037d6ef3f19fe8"
    assert (
        call_id
        == sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
    )
    assert result_id(call_id) == "8ac3fa6eec8e7803187de266"


def test_an_empty_image_list_keeps_the_text_only_id() -> None:
    call_id = model_call_id(
        benchmark_item_id="item-1",
        model="fw-llama-v3p2-1b-instruct",
        model_config=sanitize_model_config(FIREWORKS_CONFIG),
        prompt=SWAHILI_PROMPT,
        system_prompt=None,
        images=(),
    )

    assert call_id == "72a62fa2640945d4ed2325a35b4fcd33eaab7a1f57afecfde8037d6ef3f19fe8"


IMAGE_SHA256 = sha256(b"image-1").hexdigest()
IMAGE_CALL = {
    "benchmark_item_id": "visual-1",
    "model": "model-a",
    "model_config": {},
    "prompt": "Which shape is shown?",
    "system_prompt": "Answer with one word.",
}


def test_image_call_id_is_pinned() -> None:
    """Pinned when images were added to call IDs (2026-11-18), re-pinned without
    max_tokens (2026-11-19)."""

    call_id = model_call_id(
        **IMAGE_CALL, images=[{"detail": "high", "sha256": IMAGE_SHA256}]
    )
    payload = {
        "benchmark_item_id": "visual-1",
        "images": [{"detail": "high", "sha256": IMAGE_SHA256}],
        "model": "model-a",
        "model_config": {},
        "prompt": "Which shape is shown?",
        "system_prompt": "Answer with one word.",
    }

    assert call_id == "5ce8a30909deff8b84e5ac137698bb3bef949b59efa06b594e1e8359ba4d4bae"
    assert (
        call_id
        == sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
    )
    assert result_id(call_id) == "5bf97359ae9bdef04aa46434"


def test_where_an_image_is_stored_does_not_change_the_id() -> None:
    bare = model_call_id(
        **IMAGE_CALL, images=[{"detail": "high", "sha256": IMAGE_SHA256}]
    )
    referenced = model_call_id(
        **IMAGE_CALL,
        images=[
            {
                "detail": "high",
                "sha256": IMAGE_SHA256,
                "uri": "gs://bucket/mathv/images/1.jpg",
            }
        ],
    )
    copied = model_call_id(
        **IMAGE_CALL,
        images=[{"detail": "high", "sha256": IMAGE_SHA256, "uri": "images/1.jpg"}],
    )

    assert bare == referenced == copied


def test_image_content_detail_and_order_change_the_id() -> None:
    first = {"detail": "high", "sha256": IMAGE_SHA256}
    second = {"detail": "low", "sha256": sha256(b"image-2").hexdigest()}

    ids = {
        model_call_id(**IMAGE_CALL),
        model_call_id(**IMAGE_CALL, images=[first]),
        model_call_id(**IMAGE_CALL, images=[first, second]),
        model_call_id(**IMAGE_CALL, images=[second, first]),
        model_call_id(**IMAGE_CALL, images=[{**first, "detail": "low"}, second]),
        model_call_id(
            **IMAGE_CALL,
            images=[{**first, "sha256": sha256(b"image-3").hexdigest()}, second],
        ),
    }

    assert len(ids) == 6


@pytest.mark.parametrize(
    ("image", "message"),
    [
        ({"detail": "auto", "sha256": IMAGE_SHA256}, "detail must be"),
        ({"sha256": IMAGE_SHA256}, "detail must be"),
        ({"detail": "high", "sha256": IMAGE_SHA256.upper()}, "sha256 must be"),
        ({"detail": "high", "sha256": IMAGE_SHA256[:63]}, "sha256 must be"),
        ({"detail": "high"}, "sha256 must be"),
    ],
)
def test_malformed_image_references_are_refused(
    image: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        model_call_id(**IMAGE_CALL, images=[image])


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("gpt-4o", "responses_gpt-4o.jsonl"),
        (
            "openrouter/qwen/qwen3-8b",
            "responses_openrouter_qwen_qwen3-8b-838cd6bc.jsonl",
        ),
        ("a b", "responses_a_b-c8687a08.jsonl"),
    ],
)
def test_result_filenames_match_the_phonics_runner(
    model_name: str, expected: str
) -> None:
    assert model_results_filename(model_name) == expected
