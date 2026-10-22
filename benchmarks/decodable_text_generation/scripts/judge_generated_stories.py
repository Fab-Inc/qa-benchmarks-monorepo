"""Judge model-generated curriculum stories against the decodable-text rubric.

For each (profile, model) pair in the generation-run responses files, runs
``evaluate_text`` and appends a judgment row to
``outputs/judge-generated-en/judgments_<judge>.jsonl`` — resumable: a written
row whose rubric hash still matches is skipped on rerun, so running a subset
(``--profiles``) first means those judgments are already cached for the full
run later.

Usage:
    uv run --package decodable-text-generation python \
        benchmarks/decodable_text_generation/scripts/judge_generated_stories.py \
        --judges gpt-4.1-nano --profiles ufli-en-l016
"""

import json
import re
import sys
from pathlib import Path

import typer
from decodable_text_generation.judge.engine import evaluate_text
from decodable_text_generation.judge.metrics import story_metrics
from decodable_text_generation.judge.resume import (
    merge_judgment,
    partition_sections,
    rubric_section_shas,
)
from decodable_text_generation.judge.rubric import (
    derive_tags,
    load_rubric,
    packaged_rubric_path,
    rubric_sha256,
)
from fdllm import register_models

BENCH_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = BENCH_ROOT / "data"
DEFAULT_STORIES = (
    DEFAULT_DATA_ROOT / "curriculum_aligned" / "english" / "human_stories.tsv"
)
DEFAULT_RESPONSES = BENCH_ROOT / "outputs" / "en-generation-curriculum" / "responses"
DEFAULT_CONFIG = BENCH_ROOT / "configs" / "en-curriculum.json"
DEFAULT_OUTDIR = BENCH_ROOT / "outputs" / "judge-generated-en"
CUSTOM_MODELS = Path(__file__).resolve().parents[3] / "custom_models.yaml"
TEXT_TAG_RE = re.compile(r"<text>(.*?)</text>", re.DOTALL)


def story_body(output: str) -> str:
    m = TEXT_TAG_RE.search(output)
    return m.group(1).strip() if m else output.strip()


def input_tokens(model: str, call: dict) -> int | None:
    """Prompt tokens; Google reports thoughts separately from candidates."""
    total, completion, reasoning = (
        call.get("tokens_used"),
        call.get("tokens_used_completion"),
        call.get("tokens_used_reasoning"),
    )
    if total is None or completion is None:
        return None
    extra = reasoning if model.startswith("gemini") and reasoning else 0
    return total - completion - extra


def judge_stories(
    responses_dir: Path = DEFAULT_RESPONSES,
    stories_path: Path = DEFAULT_STORIES,
    config_path: Path = DEFAULT_CONFIG,
    data_root: Path = DEFAULT_DATA_ROOT,
    outdir: Path = DEFAULT_OUTDIR,
    judges: str = "gpt-4.1-nano",
    profiles: str | None = None,
    models: str | None = None,
    fresh: bool = False,
    reuse_from: Path | None = None,
) -> None:
    profiles_path = (
        data_root
        / "curriculum_aligned"
        / "english"
        / "student_knowledge_profiles.jsonl"
    )
    all_profiles = {
        p["student_knowledge_id"]: p
        for p in (
            json.loads(line)
            for line in profiles_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    config_models = json.loads(config_path.read_text())["models"]
    gen_models = [m.strip() for m in models.split(",")] if models else config_models

    # human reference stories: first row per profile
    stories: dict[str, dict[str, str]] = {}
    for line in stories_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            pid, story = line.split("\t", 1)
            stories.setdefault(pid, {}).setdefault("human", story)

    # (pid, model) -> story text from successful responses
    for model in gen_models:
        path = responses_dir / f"responses_{model}.jsonl"
        if not path.exists():
            print(f"warning: no responses file for {model}", file=sys.stderr)
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("call_status") == "success" and row.get("output"):
                pid = row["benchmark_item_id"].removesuffix("_story")
                if pid in all_profiles:
                    stories.setdefault(pid, {})[model] = story_body(row["output"])

    selected = [s.strip() for s in profiles.split(",")] if profiles else sorted(stories)
    missing = [p for p in selected if p not in all_profiles]
    if missing:
        raise typer.BadParameter(f"unknown profile ids: {missing}")
    # the human reference is judged alongside the model stories at every
    # selected profile ("--models" filters only the generated side)
    item_models = ["human"] + gen_models
    work = [
        (pid, m) for pid in selected for m in item_models if m in stories.get(pid, {})
    ]
    print(
        f"{len(work)} (profile, model) pairs across {len(selected)} profiles",
        file=sys.stderr,
    )

    register_models(str(CUSTOM_MODELS))
    rubric_path = packaged_rubric_path()
    rubric = load_rubric(rubric_path)
    rubric_sha = rubric_sha256(rubric_path)
    judge_models = [m.strip() for m in judges.split(",") if m.strip()]
    outdir.mkdir(parents=True, exist_ok=True)

    section_shas = rubric_section_shas(rubric)
    for judge_model in judge_models:
        out_path = outdir / f"judgments_{judge_model}.jsonl"
        # Latest cached row per item; sections may be partially reusable.
        # ``--reuse-from`` reads the cache from another run's file instead of
        # the output file, so each output always holds one run's judgments.
        cached_rows: dict[str, dict] = {}
        cache_source = reuse_from or out_path
        if not fresh and cache_source.exists():
            for line in cache_source.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    cached_rows[row["benchmark_item_id"]] = row
        pending = [
            (pid, m)
            for pid, m in work
            if not (
                (row := cached_rows.get(f"{pid}#{m}"))
                and not partition_sections(row, rubric)[1]
            )
        ]
        print(
            f"{judge_model}: {len(work) - len(pending)} cached, "
            f"{len(pending)} to judge -> {out_path}",
            file=sys.stderr,
            flush=True,
        )

        calls_seen = 0
        in_sum = out_sum = 0
        # One run per file: fresh/reuse-from runs rewrite the file rather
        # than appending a second set of judgments into it.
        write_mode = "w" if (fresh or reuse_from is not None) else "a"
        with out_path.open(write_mode, encoding="utf-8") as out:
            # In "w" mode every work item emits a row — fully cached items
            # are re-emitted with refreshed metrics, never silently dropped
            # (a truncated-then-empty file would lose the whole run).
            for i, (pid, gen_model) in enumerate(work):
                if write_mode == "a" and (pid, gen_model) not in pending:
                    continue
                profile = all_profiles[pid]
                story = stories[pid][gen_model]

                metrics = story_metrics(
                    story,
                    student_knowledge_id=pid,
                    student_knowledge=profile["student_knowledge"],
                    target_knowledge=profile["target_knowledge"],
                )
                tags = derive_tags(
                    pid,
                    profile["student_knowledge"],
                    n_nondecodable_words=metrics["n_nondecodable_words"],
                    target_knowledge=profile["target_knowledge"],
                    n_target_words=metrics.get("n_target_words") or 0,
                    max_frame_repeats=metrics.get("max_frame_repeats") or 0,
                )
                nondecodable = metrics["unique_nondecodable_words"]
                context = (
                    f"Student profile: {pid} (UFLI).\n"
                    "Words not decodable for this student: "
                    f"{', '.join(nondecodable) if nondecodable else 'none'}."
                )
                if metrics.get("target_description"):
                    context += (
                        "\nPractice goal for this lesson: "
                        f"{metrics['target_description']}"
                    )
                target_parts = (
                    [
                        f"{unit}: {', '.join(words)}"
                        for unit, words in (
                            metrics.get("target_gpc_words") or {}
                        ).items()
                    ]
                    + [
                        f"{rule} rule: {', '.join(words)}"
                        for rule, words in (
                            metrics.get("target_rule_words") or {}
                        ).items()
                    ]
                    + (metrics.get("target_sight_words_used") or [])
                )
                if target_parts:
                    context += (
                        "\nTarget units for this lesson and the words carrying them: "
                        + "; ".join(target_parts)
                        + "."
                    )
                target_word_list = sorted(
                    set().union(
                        *(metrics.get("target_gpc_words") or {}).values(),
                        *(metrics.get("target_rule_words") or {}).values(),
                    )
                    | set(metrics.get("target_sight_words_used") or [])
                )

                cached = cached_rows.get(f"{pid}#{gen_model}")
                reusable, stale = (
                    partition_sections(cached, rubric)
                    if cached and not fresh
                    else (set(), set(rubric))
                )
                if cached and not stale:
                    # Fully reusable: re-emit with refreshed metrics/tags.
                    judgment = dict(cached)
                    judgment["reused_sections"] = sorted(reusable)
                    judgment.update(
                        {
                            "metrics": metrics,
                            "tags": sorted(tags),
                            "rubric_sha": rubric_sha,
                            "section_shas": section_shas,
                            "section_weights": {
                                name: s.weight for name, s in rubric.items()
                            },
                        }
                    )
                    out.write(json.dumps(judgment) + "\n")
                    out.flush()
                    continue
                fresh_eval = evaluate_text(
                    story,
                    rubric={n: s for n, s in rubric.items() if n in stale},
                    tags=tags,
                    judge_model=judge_model,
                    context=context,
                    verbose=False,
                    placeholders={
                        "target_words": ", ".join(target_word_list),
                        "repetition_example": metrics.get("frame_example") or "",
                    },
                )
                judgment = (
                    merge_judgment(cached, fresh_eval, rubric, reusable)
                    if reusable
                    else fresh_eval
                )
                judgment.update(
                    {
                        "benchmark_item_id": f"{pid}#{gen_model}",
                        "profile_id": pid,
                        "model": gen_model,
                        "judge_model": judge_model,
                        "rubric_sha": rubric_sha,
                        "section_shas": section_shas,
                        "metrics": metrics,
                        "judge_error": "",
                    }
                )
                for call in judgment["token_usage"]:
                    call["input_tokens"] = input_tokens(judge_model, call)
                    in_sum += call["input_tokens"] or 0
                    out_sum += call["tokens_used_completion"] or 0
                calls_seen += len(judgment["token_usage"])
                out.write(json.dumps(judgment) + "\n")
                out.flush()
                print(
                    f"[{judge_model}] {pid}#{gen_model} {i + 1}/{len(pending)} "
                    f"judged ({len(judgment['token_usage'])} calls"
                    + (f", {len(reusable)} sections reused" if reusable else "")
                    + ")",
                    file=sys.stderr,
                    flush=True,
                )

        print(
            f"{judge_model}: {calls_seen} calls this run | "
            f"input tokens {in_sum:,} | output tokens {out_sum:,} | "
            f"~{in_sum // max(calls_seen, 1):,} in / "
            f"{out_sum // max(calls_seen, 1):,} out per call -> {out_path}"
        )


if __name__ == "__main__":
    typer.run(judge_stories)
