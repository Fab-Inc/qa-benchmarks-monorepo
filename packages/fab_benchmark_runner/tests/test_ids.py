"""Pinned ID values: these must not change, or existing results stop matching.

The expected values were computed with the phonics benchmark's own functions
(phonics_benchmark.runner._model_call_id/_result_id and
phonics_benchmark.config.config_identity_sha256) on main at 8b28132, so runs written
by the phonics runner resume unchanged under this package.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from fab_benchmark_runner import (
    Benchmark,
    ConfigError,
    config_identity,
    config_identity_sha256,
    execute_run,
    load_run_config,
    model_call_id,
    model_results_filename,
    result_id,
)
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
# The phonics config ID of _write_phonics_like_config with default max_tokens.
PHONICS_CONFIG_ID = "c0013b2a1bd7bc1fdee962da0dccb4010615a278721596e60f644ca472d25700"


@dataclass(frozen=True)
class PhonicsLikeParams:
    language: str
    prompt_name: str = "decodability_evaluation"


PHONICS_LIKE = Benchmark(
    name="phonics-like",
    params_type=PhonicsLikeParams,
    build_prompt=lambda item, _params: f"Soma: {item['benchmark_item_id']}",
    identity=lambda params: {
        "language": params.language,
        "task": "decodability",
        "prompt_name": params.prompt_name,
    },
)


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
    ("max_tokens", "expected_config_id"),
    [
        (None, "c0013b2a1bd7bc1fdee962da0dccb4010615a278721596e60f644ca472d25700"),
        (2048, "ffae7b55a6cc62e0cdb064d4e946291b751c2a99d055e21b8121edc2dccdcf57"),
    ],
)
def test_config_id_matches_the_phonics_runner(
    tmp_path: Path, max_tokens: int | None, expected_config_id: str
) -> None:
    config = load_run_config(
        _write_phonics_like_config(tmp_path, max_tokens=max_tokens), PHONICS_LIKE
    )

    assert config_identity(config) == {
        "language": "sw",
        "task": "decodability",
        "prompt_name": "decodability_evaluation_clusters",
        "max_tokens": max_tokens or 10_000,
    }
    assert config_identity_sha256(config) == expected_config_id


def test_runner_rows_carry_pinned_ids(
    tmp_path: Path,
    read_rows: Callable[..., list[dict[str, Any]]],
) -> None:
    """A whole run, with an injected evaluator and so empty model configs.

    The call IDs were the phonics runner's until max_tokens left them (2026-11-19);
    the config ID still is.
    """

    config = load_run_config(_write_phonics_like_config(tmp_path), PHONICS_LIKE)

    output = execute_run(config, evaluator=lambda *_args: {"output": "[]"})

    [row] = read_rows(output["results_paths"].values())
    assert row["model_call_id"] == (
        "0812464e4b3b4b8d6706099ef30bada6f56dfef1413bf125e3d5b6a079fdec22"
    )
    assert row["result_id"] == "354db67328a229c3c6588e02"
    assert row["config_id"] == (
        "c0013b2a1bd7bc1fdee962da0dccb4010615a278721596e60f644ca472d25700"
    )
    assert (row["language"], row["task"], row["prompt_name"]) == (
        "sw",
        "decodability",
        "decodability_evaluation_clusters",
    )


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


def _write_phonics_like_config(
    tmp_path: Path, *, max_tokens: int | None = None
) -> Path:
    (tmp_path / "items.jsonl").write_text(
        json.dumps({"benchmark_item_id": "item-1", "content": "Mama."}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "custom_models.yaml").write_text(
        json.dumps({"provider-a": {"models": {"model-a": {}}}}), encoding="utf-8"
    )
    config: dict[str, Any] = {
        "run_id": "golden",
        "language": "sw",
        "prompt_name": "decodability_evaluation_clusters",
        "items_path": "items.jsonl",
        "output_dir": "outputs",
        "models": ["model-a"],
        "custom_models_path": "custom_models.yaml",
    }
    if max_tokens is not None:
        config["max_tokens"] = max_tokens
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def test_a_phonics_run_folder_is_extended_without_new_calls(
    tmp_path: Path,
    read_rows: Callable[..., list[dict[str, Any]]],
) -> None:
    """A folder the phonics runner wrote: its result is reused, its record migrated.

    The row carries IDs already converted to this package's, as phonics results must
    be before they can be reused here.
    """

    config = load_run_config(_write_phonics_like_config(tmp_path), PHONICS_LIKE)
    run_dir = config.run_dir
    (run_dir / "responses").mkdir(parents=True)
    phonics_row = {
        "benchmark_item_id": "item-1",
        "call_status": "success",
        "config_id": PHONICS_CONFIG_ID,
        "model": "model-a",
        "model_call_id": (
            "0812464e4b3b4b8d6706099ef30bada6f56dfef1413bf125e3d5b6a079fdec22"
        ),
        "output": "[]",
        "result_id": "354db67328a229c3c6588e02",
        "success": True,
    }
    (run_dir / "responses" / "responses_model-a.jsonl").write_text(
        json.dumps(phonics_row) + "\n", encoding="utf-8"
    )
    items_sha256 = sha256((tmp_path / "items.jsonl").read_bytes()).hexdigest()
    phonics_snapshot = {
        "config_id": phonics_row["config_id"],
        "raw_config": {"language": "sw"},
        "resolved_config": {"items_sha256": items_sha256},
        "model_configs": {},
    }
    (run_dir / "config.json").write_text(json.dumps(phonics_snapshot), encoding="utf-8")
    (run_dir / "log.json").write_text('{"model-a": {}}\n', encoding="utf-8")

    def fail(*_args) -> dict:
        pytest.fail("the phonics result must be reused")

    output = execute_run(config, evaluator=fail)

    assert (output["executed_count"], output["skipped_count"]) == (0, 1)
    assert read_rows(output["results_paths"].values()) == [phonics_row]
    record = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert record["config_id"] == phonics_row["config_id"]
    assert record["items_sha256"] == items_sha256
    assert "resolved_config" not in record
    assert (run_dir / "log.json").read_text(encoding="utf-8") == '{"model-a": {}}\n'


def test_a_phonics_run_folder_of_another_run_is_refused(tmp_path: Path) -> None:
    config = load_run_config(_write_phonics_like_config(tmp_path), PHONICS_LIKE)
    config.run_dir.mkdir(parents=True)
    (config.run_dir / "config.json").write_text(
        json.dumps(
            {
                "config_id": PHONICS_CONFIG_ID,
                "resolved_config": {"items_sha256": "other-items"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="items_sha256 does not match"):
        execute_run(config, evaluator=lambda *_args: {"output": "[]"})
