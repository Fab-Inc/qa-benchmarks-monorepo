"""Run models against prepared benchmark items and score their responses.

Calls are fanned out with asyncio (fdllm handles retry/backoff for rate limits;
we add the concurrency cap it lacks). Scoring is deliberately crude (exact set
match against the labelled non-decodable words) — richer scoring is out of scope
for now. ``run_benchmark_async`` exposes an ``on_result`` hook so a runner can
stream results to disk, and a ``completed_keys`` set so an interrupted run can be
resumed without re-calling model/item pairs already done.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from decodability import get_decodability_objects

from phonics_benchmark.llm import call_llm, call_llm_async
from phonics_benchmark.parsing import parse_word_list
from phonics_benchmark.prompts import get_prompt


def build_decodability_prompt(
    benchmark_item: Any,
    language: str,
    *,
    prompt_name: str = "decodability_evaluation",
    **extra_params: Any,
) -> str:
    """Render a decodability prompt for a prepared benchmark item.

    ``prompt_name`` selects the template in prompts.yaml; ``extra_params`` are
    forwarded to the template for prompt variants that need more placeholders.
    """
    objs = get_decodability_objects(language)
    student_knowledge = objs.student_knowledge_model.model_validate(
        benchmark_item["student_knowledge"]
    )
    return get_prompt(
        prompt_name,
        language=objs.name,
        text=benchmark_item["content"],
        student_knowledge=student_knowledge,
        **extra_params,
    )


def submit_benchmark_item_to_model(
    model: str,
    benchmark_item: Any,
    language: str,
    *,
    prompt_name: str = "decodability_evaluation",
    max_tokens: int = 30000,
    bypass_llm: bool = False,
    **prompt_params: Any,
) -> dict[str, Any]:
    """Submit a single benchmark item to a model via fdllm (synchronous)."""
    prompt = build_decodability_prompt(
        benchmark_item, language, prompt_name=prompt_name, **prompt_params
    )
    call_record = call_llm(
        prompt,
        model=model,
        max_tokens=max_tokens,
        tag=str(benchmark_item["benchmark_item_id"]),
        bypass_llm=bypass_llm,
    )
    call_record.setdefault("model", model)
    call_record.setdefault("prompt", prompt)
    return call_record


def evaluate_response(
    call_record: dict[str, Any], benchmark_item: Any
) -> dict[str, Any]:
    """Parse and score a model response into a flat result record."""
    response_raw: str = call_record["response_text"]
    response_parsed, response_parse_ok = parse_word_list(response_raw)
    response_parsed = [word.lower() for word in response_parsed]
    if not response_parse_ok:
        # TODO: maybe retry the LLM call instead of just recording the failure
        print(
            f"WARNING: response for {benchmark_item['benchmark_item_id']} "
            f"did not match expected pattern"
        )

    correct_answer_parsed = [
        word.lower() for word in benchmark_item["non_decodable_words"]
    ]

    # Perfect one-to-one correspondence ignoring order (duplicates must match too).
    perfect_match = Counter(response_parsed) == Counter(correct_answer_parsed)

    return {
        "model_name": call_record.get("model"),
        "benchmark_item_id": benchmark_item["benchmark_item_id"],
        "student_knowledge_id": benchmark_item["student_knowledge_id"],
        "prompt": call_record.get("prompt"),
        "response_raw": response_raw,
        "response_parsed": response_parsed,
        "response_parse_ok": response_parse_ok,
        "correct_answer": correct_answer_parsed,
        "perfect_match": perfect_match,
        "max_tokens": call_record.get("max_tokens"),
        "tokens_used_completion": call_record.get("tokens_used_completion"),
        "latency_seconds": call_record.get("latency_seconds"),
        "timestamp_utc": call_record.get("timestamp_utc"),
    }


def load_completed_keys(path: Path) -> set[tuple[str, str]]:
    """Read the (model_name, benchmark_item_id) pairs already in a results JSONL.

    Returns an empty set if the file does not exist, so it is safe to call before
    a fresh run. Use the result as ``completed_keys`` to resume without repeating
    calls that already succeeded.
    """
    completed: set[tuple[str, str]] = set()
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            model_name = record.get("model_name")
            benchmark_item_id = record.get("benchmark_item_id")
            if model_name is not None and benchmark_item_id is not None:
                completed.add((model_name, benchmark_item_id))
    return completed


async def _submit_and_evaluate_async(
    model: str,
    benchmark_item: Any,
    language: str,
    *,
    prompt_name: str,
    max_tokens: int,
    bypass_llm: bool,
) -> dict[str, Any]:
    prompt = build_decodability_prompt(
        benchmark_item, language, prompt_name=prompt_name
    )
    call_record = await call_llm_async(
        prompt,
        model=model,
        max_tokens=max_tokens,
        tag=str(benchmark_item["benchmark_item_id"]),
        bypass_llm=bypass_llm,
    )
    call_record.setdefault("model", model)
    call_record.setdefault("prompt", prompt)
    return evaluate_response(call_record, benchmark_item)


async def run_benchmark_async(
    benchmark_items: pd.DataFrame,
    models: list[str],
    language: str,
    *,
    prompt_name: str = "decodability_evaluation",
    max_tokens: int = 30000,
    max_rows_per_model: int = 0,
    max_concurrency: int = 10,
    bypass_llm: bool = False,
    completed_keys: set[tuple[str, str]] | None = None,
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run every model against every benchmark item and score each response.

    Models are processed one at a time (rate limits are per-provider); each
    model's items are fanned out concurrently, capped at ``max_concurrency``.
    ``max_rows_per_model`` caps rows per model for quick tests (0 = all rows).
    ``completed_keys`` holds (model, benchmark_item_id) pairs to skip (for
    resuming). ``on_result`` is called with each result as it completes (e.g. to
    append to a JSONL crash-fallback). A call that raises is logged and dropped
    (no result recorded) so a later resume retries it. Returns all new results,
    in completion order.
    """
    completed_keys = completed_keys or set()
    semaphore = asyncio.Semaphore(max_concurrency)
    results: list[dict[str, Any]] = []

    async def worker(model: str, benchmark_item: Any) -> dict[str, Any]:
        async with semaphore:
            return await _submit_and_evaluate_async(
                model,
                benchmark_item,
                language,
                prompt_name=prompt_name,
                max_tokens=max_tokens,
                bypass_llm=bypass_llm,
            )

    for model in models:
        rows = (
            benchmark_items
            if max_rows_per_model == 0
            else benchmark_items.head(max_rows_per_model)
        )
        tasks = []
        skipped = 0
        for _, benchmark_item in rows.iterrows():
            if (model, benchmark_item["benchmark_item_id"]) in completed_keys:
                skipped += 1
                continue
            tasks.append(asyncio.create_task(worker(model, benchmark_item)))

        print(
            f"Running decodability benchmark for model: {model} "
            f"({len(tasks)} to run, {skipped} already complete)"
        )
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
            except Exception as exc:  # noqa: BLE001 - log and continue so resume retries
                print(f"WARNING: call failed for model {model}: {exc!r}")
                continue
            results.append(result)
            if on_result is not None:
                on_result(result)

    return results


def run_benchmark(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    """Synchronous wrapper around :func:`run_benchmark_async`."""
    return asyncio.run(run_benchmark_async(*args, **kwargs))
