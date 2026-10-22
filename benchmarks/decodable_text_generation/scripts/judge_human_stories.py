"""Judge the human-written curriculum stories against the decodable-text rubric.

Runs ``evaluate_text`` on every row of ``human_stories.tsv`` for each judge
model and writes one judgment JSONL per judge to the output directory. Each
row carries the deterministic ``metrics`` (decodability, pattern spread,
target coverage) alongside the rubric ``sections``, and a ``token_usage``
list with the raw token counts of every judge call (one per rubric section,
plus any gate calls), so cost can be estimated offline.

Usage:
    uv run --package decodable-text-generation python \
        benchmarks/decodable_text_generation/scripts/judge_human_stories.py \
        --judges gemini-3.5-flash,claude-opus-5
"""

import csv
import json
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

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
CUSTOM_MODELS = Path(__file__).resolve().parents[3] / "custom_models.yaml"
DEFAULT_STORIES = (
    DEFAULT_DATA_ROOT / "curriculum_aligned" / "english" / "human_stories.tsv"
)
DEFAULT_OUTDIR = Path(__file__).resolve().parents[1] / "outputs" / "judge-human-en"


def input_tokens(model: str, call: dict) -> int | None:
    """Prompt tokens for one call.

    Anthropic/OpenAI fold reasoning into completion tokens (input = total -
    completion); Google reports thoughts separately from candidates, so
    reasoning must be subtracted too.
    """
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
    stories_path: Path = DEFAULT_STORIES,
    data_root: Path = DEFAULT_DATA_ROOT,
    outdir: Path = DEFAULT_OUTDIR,
    judges: str = "gemini-3.5-flash,claude-opus-5",
    limit: int | None = None,
    fresh: bool = False,
) -> None:
    rows = [
        row
        for row in csv.DictReader(stories_path.open(encoding="utf-8"), delimiter="\t")
    ]

    profiles_path = (
        data_root
        / "curriculum_aligned"
        / "english"
        / "student_knowledge_profiles.jsonl"
    )
    profiles = {
        p["student_knowledge_id"]: p
        for p in (
            json.loads(line)
            for line in profiles_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }

    # One story per profile: keep the first row for each profile id, drop
    # rows for lessons with no profile and extra rows beyond the first.
    seen: set[str] = set()
    deduped = []
    skipped_no_profile = skipped_dup = 0
    for row in rows:
        pid = row["profile_id"]
        if pid not in profiles:
            skipped_no_profile += 1
            continue
        if pid in seen:
            skipped_dup += 1
            continue
        seen.add(pid)
        deduped.append(row)
    print(
        f"{len(deduped)} stories (skipped {skipped_no_profile} rows with no "
        f"profile, {skipped_dup} extra rows for a profile)",
        file=sys.stderr,
    )
    rows = deduped
    if limit:
        rows = rows[:limit]

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
        cached_rows: dict[str, dict] = {}
        if out_path.exists() and not fresh:
            for line in out_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    cached_rows[row["benchmark_item_id"]] = row
        pending = [
            row
            for row in rows
            if not (
                (cached := cached_rows.get(f"{row['profile_id']}#human"))
                and not partition_sections(cached, rubric)[1]
            )
        ]
        n_cached = len(rows) - len(pending)
        print(
            f"{judge_model}: {n_cached} cached, {len(pending)} to judge -> {out_path}",
            file=sys.stderr,
            flush=True,
        )

        calls_seen = 0
        in_sum = out_sum = 0
        with out_path.open("a", encoding="utf-8") as out:
            for i, row in enumerate(pending):
                pid = row["profile_id"]
                story = row["story"]
                profile = profiles[pid]

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

                cached = cached_rows.get(f"{pid}#human")
                reusable, stale = (
                    partition_sections(cached, rubric)
                    if cached
                    else (set(), set(rubric))
                )
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
                        "benchmark_item_id": f"{pid}#human",
                        "profile_id": pid,
                        "story_index": i,
                        "model": "human",
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
                    f"[{judge_model}] {pid} story {i + 1}/{len(pending)} "
                    f"(+{n_cached} cached) judged "
                    f"({len(judgment['token_usage'])} calls"
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
