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
from pathlib import Path
from typing import Any

import pytest

from fab_benchmark_runner import (
    Benchmark,
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


@pytest.mark.parametrize(
    ("max_tokens", "expected_model_call_id", "expected_result_id"),
    [
        (
            10_000,
            "c5b9aba56bb7bf59bd475f3227e5710f814cab3bbca3025720ca224a16cd97f5",
            "60a446cdd97b26cc47693999",
        ),
        (
            2048,
            "a9fc73c9d11230de2b3e6c97016bf72a3f409f63a57b914db236992084fd67f5",
            "0a0d7300364ef3469bbd71c0",
        ),
    ],
)
def test_call_ids_match_the_phonics_runner(
    max_tokens: int, expected_model_call_id: str, expected_result_id: str
) -> None:
    call_id = model_call_id(
        benchmark_item_id="item-1",
        model="fw-llama-v3p2-1b-instruct",
        model_config=sanitize_model_config(FIREWORKS_CONFIG),
        prompt=SWAHILI_PROMPT,
        system_prompt=None,
        max_tokens=max_tokens,
    )

    assert call_id == expected_model_call_id
    assert result_id(call_id) == expected_result_id


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


def test_runner_rows_carry_the_phonics_ids(
    tmp_path: Path,
    read_rows: Callable[..., list[dict[str, Any]]],
) -> None:
    """A whole run, with an injected evaluator and so empty model configs."""

    config = load_run_config(_write_phonics_like_config(tmp_path), PHONICS_LIKE)

    output = execute_run(config, evaluator=lambda *_args: {"output": "[]"})

    [row] = read_rows(output["results_paths"].values())
    assert row["model_call_id"] == (
        "5ec1fa19627f1a22d0ff5b4a3b1f8de7e5e4614861de238eb65ca01cbd2536be"
    )
    assert row["result_id"] == "59b5b241008b0c6d0c6c3cc3"
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
