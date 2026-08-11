"""
Linguistic Knowledge Eval Module
Evaluates LLM performance on language knowledge questions.
Supports multiple_choice and short_form answer types.
"""

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from fdllm import get_caller
from fdllm.llmtypes import LLMMessage
from tqdm import tqdm

from linguistic_knowledge_benchmark.shared_utils import (
    BENCH_DIR,
    EVAL_MODELS,
    RATE_LIMITED_MODELS,
    english_language_name,
    run_dir,
)

CALL_STATUS_SUCCESS = "success"
CALL_STATUS_ERROR = "error"


def _build_system_prompts(language_name: str) -> dict[str, str]:
    """Return {answer_type: prompt}.

    The medium of evaluation is English for every language: questions are written
    in English and ask about the linguistics of ``language_name``. Every prompt
    is prefixed with that context so the model knows which language the question
    is about, regardless of answer type.
    """
    prefix = (
        "The following question is written in English and concerns the"
        f" linguistics of {english_language_name(language_name)}. "
    )
    return {
        "multiple_choice": (
            prefix + "Answer the multiple-choice question by selecting the letter"
            " corresponding to the correct answer (A, B, C or D)."
            "\nOnly provide the letter for your answer."
            "\nStop exactly after the letter."
        ),
        "short_form": (
            prefix + "Answer the question."
            "\nProvide only the answer in 1-3 words. Do not explain."
        ),
    }


def _build_user_message(question: dict) -> str:
    """Build the user message for a question, folding MCQ options into the text."""
    text = question["question"]
    if question.get("answer_type") == "multiple_choice" and question.get("options"):
        lines = [text, ""]
        for letter in ("A", "B", "C", "D"):
            if letter in question["options"]:
                lines.append(f"{letter}. {question['options'][letter]}")
        return "\n".join(lines)
    return text


def load_dataset(dataset_path: Path) -> list[dict]:
    """Load a Linguistic Knowledge evaluation dataset from a JSON file.

    Args:
        dataset_path: Path to the dataset JSON file.

    Returns:
        List of question dicts with keys: question_id, domain, question,
        answer_type, expected_answers, options (MCQ only), etc.
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    with open(dataset_path, encoding="utf-8") as f:
        return json.load(f)


def _responses_dir(language_name: str, run_id: str | None = None) -> Path:
    """Return the responses directory for a language/run.

    Always writes under runs/; omitting run_id uses the 'default' run.
    """
    return run_dir(language_name, run_id) / "responses"


def git_hash() -> str | None:
    """Return the short git hash of HEAD, or None (with a warning) if unavailable."""
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(BENCH_DIR),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception as e:
        print(f"WARNING: could not read git hash: {e}", file=sys.stderr)
        return None


def update_log(
    run_directory: Path, model_name: str, fields: dict[str, str | list[str] | None]
) -> None:
    """Merge-write model provenance fields into log.json; skip None values."""
    log_path = run_directory / "log.json"
    log_data: dict = {}
    if log_path.exists():
        with open(log_path, encoding="utf-8") as _f:
            log_data = json.load(_f)
    entry = log_data.setdefault(model_name, {})
    entry.update({k: v for k, v in fields.items() if v is not None})
    with open(log_path, "w", encoding="utf-8") as _f:
        json.dump(log_data, _f, indent=2)


def _question_result_row(
    question: dict,
    *,
    output: str,
    latency,
    total_tokens,
    completion_tokens,
    reasoning_tokens,
    call_status: str,
    error_type: str = "",
    error_message: str = "",
    retry_count: int = 0,
) -> dict:
    """Build a persisted stage-1 result row."""
    return {
        "question_id": question["question_id"],
        "domain": question["domain"],
        "answer_type": question["answer_type"],
        "expected_answers": "|".join(question["expected_answers"]),
        "output": output,
        "latency": latency,
        "total_tokens": total_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "call_status": call_status,
        "error_type": error_type,
        "error_message": error_message,
        "retry_count": retry_count,
    }


def _error_mask(results_df: pd.DataFrame) -> pd.Series:
    """Return a mask for rows with persisted model-call errors."""
    if "call_status" not in results_df.columns:
        return pd.Series([False] * len(results_df), index=results_df.index)
    return results_df["call_status"].eq(CALL_STATUS_ERROR)


def run_benchmark(
    model_name: str,
    *,
    language_name: str,
    dataset_path: Path,
    force: bool = False,
    retry_errors: bool = False,
    run_id: str | None = None,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Run the Linguistic Knowledge benchmark for a given model.

    Args:
        model_name: Model name (must match key in EVAL_MODELS / custom_models.yaml)
        language_name: Display name of the language (e.g. "Kiswahili", "Yoruba"),
            interpolated into the system prompts.
        dataset_path: Path to the dataset JSON file.
        force: Force re-run even if cached results exist
        retry_errors: Retry rows with persisted call_status="error" in an
            existing cache file
        run_id: Optional run ID; writes under
            cache/<language>/runs/<run_id>/responses/.
            Defaults to 'default' when not specified.
        output_dir: Directory to save results (default:
            cache/<language>/runs/<run_id>/responses)
    Returns:
        DataFrame with columns: question_id, domain, answer_type, expected_answers,
        output, latency, total_tokens, completion_tokens, reasoning_tokens,
        call_status, error_type, error_message, retry_count
    """
    if output_dir is None:
        output_dir = _responses_dir(language_name, run_id)
    output_dir = Path(output_dir)
    run_directory = output_dir.parent

    # Validate model name
    if model_name not in EVAL_MODELS:
        print(f"ERROR: Model '{model_name}' not found in list_models.yaml.")
        print(f"Available models: {', '.join(EVAL_MODELS.keys())}")
        raise ValueError(f"Invalid model: {model_name}")

    # Check cache
    cache_path = output_dir / f"responses_{model_name}.csv"
    if not force and cache_path.exists():
        cached_df = pd.read_csv(cache_path)
        cached_error_mask = _error_mask(cached_df)
        error_count = int(cached_error_mask.sum())

        if retry_errors and error_count == 0:
            print(f"No cached error rows for {model_name}; returning cached results.")
            return cached_df

        if not retry_errors:
            if error_count:
                print(
                    f"Cached results for {model_name} contain {error_count} "
                    "model-call error row(s). Use --retry-errors to retry them."
                )
            print(f"Results already exist for {model_name}. Use --force to re-run.")
            return cached_df
    elif retry_errors:
        raise FileNotFoundError(
            f"No cached results found for {model_name}; cannot use --retry-errors."
        )
    else:
        cached_df = None
        cached_error_mask = None

    # Load dataset and build prompts
    questions = load_dataset(dataset_path)
    system_prompts = _build_system_prompts(language_name)

    # Create caller
    try:
        caller = get_caller(model_name)
    except NotImplementedError:
        raise ValueError(
            f"Model '{model_name}' is in list_models.yaml but not "
            f"registered in custom_models.yaml."
        ) from None

    if retry_errors and cached_df is not None and cached_error_mask is not None:
        error_ids = set(cached_df.loc[cached_error_mask, "question_id"].astype(str))
        questions_to_run = [q for q in questions if str(q["question_id"]) in error_ids]
        results_by_id = {
            str(row["question_id"]): row.to_dict() for _, row in cached_df.iterrows()
        }
        desc = f"Retrying {model_name}"
    else:
        questions_to_run = questions
        results_by_id = {}
        desc = f"Running {model_name}"

    # Run benchmark
    for question in tqdm(questions_to_run, desc=desc):
        qid = str(question["question_id"])
        system_prompt = system_prompts[question["answer_type"]]
        prev_retry_count = results_by_id.get(qid, {}).get("retry_count", 0)
        next_retry_count = (
            ((prev_retry_count if isinstance(prev_retry_count, int) else 0) + 1)
            if retry_errors
            else 0
        )
        try:
            messages = [
                LLMMessage(Role="system", Message=system_prompt),
                LLMMessage(Role="user", Message=_build_user_message(question)),
            ]
            response = caller.call(messages, max_tokens=None)
            results_by_id[qid] = _question_result_row(
                question,
                output=response.Message,
                latency=response.Latency,
                total_tokens=response.TokensUsed,
                completion_tokens=response.TokensUsedCompletion,
                reasoning_tokens=response.TokensUsedReasoning,
                call_status=CALL_STATUS_SUCCESS,
                retry_count=next_retry_count,
            )
        except Exception as e:
            tqdm.write(f"ERROR on question {qid}: {e}", file=sys.stderr)
            results_by_id[qid] = _question_result_row(
                question,
                output="",
                latency=None,
                total_tokens=None,
                completion_tokens=None,
                reasoning_tokens=None,
                call_status=CALL_STATUS_ERROR,
                error_type=type(e).__name__,
                error_message=str(e),
                retry_count=next_retry_count,
            )

        if model_name in RATE_LIMITED_MODELS:
            time.sleep(5)

    results = [results_by_id[str(question["question_id"])] for question in questions]
    results_df = pd.DataFrame(results)
    results_df["retry_count"] = results_df["retry_count"].fillna(0).astype(int)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / f"responses_{model_name}.csv"
    results_df.to_csv(cache_path, index=False)
    prefix = "responses_retry" if retry_errors else "responses"
    update_log(
        run_directory,
        model_name,
        {
            f"{prefix}_timestamp-utc": datetime.now(UTC).isoformat(),
            f"{prefix}_git-hash": git_hash(),
        },
    )
    print(f"Results saved to {cache_path}")
    error_count = int(_error_mask(results_df).sum())
    if error_count > 0:
        print(
            f"{model_name} had {error_count} model-call error row(s). "
            f"Partial results were saved to {cache_path}; retry with --retry-errors."
        )
    return results_df
